"""One loaded model plus everything the pilot does with it: greedy answers, T=1 tries with a shared
prompt prefill, teacher-forced passes that record internal states, and steering hooks with
guaranteed removal.

Steering positions (same convention as the earlier study's SteeringHook "both"): during generation a
vector is added at the last prompt token and at every generated token; in a teacher-forced pass it is
added at the last prompt token and every continuation position, so a teacher-forced confidence under
steering mirrors what generation would see.
"""
import time

import numpy as np
import torch
from transformers import GenerationConfig

import model_utils as mu
from config import IDK_STRING, USER_TEMPLATE_BASE
from model_utils import batch_slices, decoder_layers, gather_logprobs, left_pad, position_ids_from_mask
from pilot import config as PC
from steering import SteeringHook, _OwnHook


# ----------------------------------------------------------------------------- hooks
def own_hooked_modules(model):
    layers = decoder_layers(model)
    return [model.model.embed_tokens] + list(layers) + [l.self_attn for l in layers] + [l.self_attn.o_proj for l in layers]


def assert_clean(model):
    for m in own_hooked_modules(model):
        for d in (m._forward_hooks, m._forward_pre_hooks):
            assert not any(isinstance(h, _OwnHook) for h in d.values()), "stale pilot hook left on the model"


class HookSet:
    """Registers (module, forward hook) pairs for the duration of a `with` block."""

    def __init__(self, model, specs=()):
        self.model, self.specs, self.handles = model, list(specs or []), []

    def __enter__(self):
        assert_clean(self.model)
        for module, hook in self.specs:
            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []
        assert_clean(self.model)
        return False

    @property
    def add_hooks(self):
        return [h for _, h in self.specs if getattr(h, "mode", None) == "add"]


class BlendHook(_OwnHook):
    """SQ3: at the prefill call only, the last prompt token's residual becomes
    (1 - beta) h + beta * target[row]. `targets` ([B, d]) is set per batch by the caller."""
    mode = "blend"

    def __init__(self, beta):
        self.beta = float(beta)
        self.targets = None
        self.calls = 0

    def __call__(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if h.shape[1] == 1 or self.targets is None:
            return None
        assert self.targets.shape[0] == h.shape[0], "blend targets / batch mismatch"
        t = self.targets.to(h.device, h.dtype)
        h = h.clone()
        h[:, -1, :] = (1.0 - self.beta) * h[:, -1, :] + self.beta * t
        self.calls += 1
        return (h,) + tuple(output[1:]) if is_tuple else h


class HeadCapture(_OwnHook):
    """Keeps a reference to every layer's o_proj input (concatenated head outputs) of the last forward."""

    def __init__(self, layers):
        self.layers, self.store, self.handles = layers, {}, []

    def _make(self, idx):
        def fn(module, args):
            self.store[idx] = args[0]
        return fn

    def register(self):
        for i, l in enumerate(self.layers):
            h = l.self_attn.o_proj.register_forward_pre_hook(self._make(i))
            self.handles.append(h)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles, self.store = [], {}


def resid_add(layer_module, vec):
    return [(layer_module, SteeringHook("add", vec, "both"))]


# ----------------------------------------------------------------------------- session
class Session:
    def __init__(self, spec, revision):
        self.spec = spec
        self.name = spec["name"]
        self.model, self.tok, self.fmt = mu.load_model(spec, revision)
        self.layers = decoder_layers(self.model)
        self.n_layers = len(self.layers)
        assert self.model.config.num_attention_heads == spec["n_heads"], "unexpected head count"
        self.plan = PC.layer_plan(spec)
        self.idk_cont = list(self.fmt.idk_ids) + [self.fmt.eot_id]
        self.sampler_mode = None

    # ------------------------------------------------------------ prompts
    def prompt(self, question, template=USER_TEMPLATE_BASE):
        return self.fmt.prompt(template.format(question=question))

    def user_prompt(self, text):
        return self.fmt.prompt(text)

    def answer_cont(self, token_ids, truncated):
        ids = list(token_ids)
        return ids if truncated else ids + [self.fmt.eot_id]

    # ------------------------------------------------------------ generation
    def greedy(self, prompts, specs=(), max_new_tokens=None, want_logprobs=False):
        if not prompts:
            return mu.GenOut([], [], np.zeros(0), np.zeros(0, int), np.zeros(0, bool))
        with HookSet(self.model, specs):
            return mu.generate(self.model, self.fmt, prompts, max_new_tokens or PC.QA_MAX_NEW_TOKENS,
                               want_logprobs=want_logprobs)

    def greedy_blend(self, prompts, layer, targets, beta, max_new_tokens=None):
        """Greedy answers with the SQ3 blend at `layer`; targets [N, d] row-aligned with prompts."""
        hook = BlendHook(beta)
        outs = []
        with HookSet(self.model, [(self.layers[layer], hook)]):
            for sl in batch_slices(len(prompts)):
                hook.targets = targets[sl]
                outs.append(mu.generate(self.model, self.fmt, prompts[sl], max_new_tokens or PC.QA_MAX_NEW_TOKENS,
                                        want_logprobs=False))
        hook.targets = None
        assert hook.calls == len(outs), "blend hook did not fire once per batch"
        return concat_gen(outs)

    @torch.no_grad()
    def _sample_shared(self, prompts, n, max_new_tokens, do_sample=True):
        """T=1 tries with the prompt prefilled once and its KV cache repeated n times (the prompt
        is ~85% of the compute when HF expands it per try). Returns per prompt [(text, ids, truncated)]."""
        stop = set(self.fmt.stop_ids)
        out_all = []
        for sl in batch_slices(len(prompts), PC.SAMPLE_BATCH_PROMPTS):
            ids, attn, T = left_pad(prompts[sl], self.fmt.pad_id)
            pre = self.model(input_ids=ids[:, :-1], attention_mask=attn[:, :-1],
                             position_ids=position_ids_from_mask(attn[:, :-1]), use_cache=True)
            cache = pre.past_key_values
            cache.batch_repeat_interleave(n)
            gc = GenerationConfig(do_sample=do_sample, temperature=1.0 if do_sample else None, top_p=1.0 if do_sample else None,
                                  top_k=0 if do_sample else None, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                                  eos_token_id=self.fmt.stop_ids, pad_token_id=self.fmt.pad_id,
                                  return_dict_in_generate=True)
            out = self.model.generate(input_ids=ids.repeat_interleave(n, 0), attention_mask=attn.repeat_interleave(n, 0),
                                      past_key_values=cache, generation_config=gc)
            toks, emitted = mu._split_generated(out.sequences[:, T:].tolist(), stop)
            for p in range(ids.shape[0]):
                out_all.append([(self.fmt.tok.decode(toks[k], skip_special_tokens=True).strip(), toks[k], not emitted[k])
                                for k in range(p * n, (p + 1) * n)])
            del pre, cache, out
        return out_all

    def sampler_self_test(self, prompts):
        """Greedy decoding through the shared-prefill path must give the plain greedy answers."""
        probe = prompts[:min(16, len(prompts))]
        shared = [rows[0][0] for rows in self._sample_shared(probe, 2, PC.QA_MAX_NEW_TOKENS, do_sample=False)]
        plain = self.greedy(probe).texts
        agree = float(np.mean([a == b for a, b in zip(shared, plain)])) if probe else 1.0
        self.sampler_mode = "shared_prefill" if agree >= 0.9 else "plain"
        print(f"SAMPLER: model={self.name} mode={self.sampler_mode} greedy_agreement={agree:.3f}")
        return agree

    def sample(self, prompts, n, seed, max_new_tokens=None):
        mnt = max_new_tokens or PC.QA_MAX_NEW_TOKENS
        if self.sampler_mode is None:
            self.sampler_self_test(prompts)
        torch.manual_seed(seed)
        if self.sampler_mode == "shared_prefill":
            return self._sample_shared(prompts, n, mnt)
        return mu.sample(self.model, self.fmt, prompts, n, mnt, seed)

    # ------------------------------------------------------------ teacher-forced passes
    @torch.no_grad()
    def tf(self, prompts, conts, layers=(), heads=False, specs=(), batch_size=None):
        """Forward prompt + continuation per row. Returns
        lp [N]          sum log p(continuation tokens) (float64);
        res_prompt      [N, len(layers), d] residual at the last prompt token (float16, CPU);
        res_ans         [N, len(layers), d] residual at the last answer token (the token before a final
                        end-of-turn; the last token when the continuation has none);
        heads           [N, n_layers, n_heads * head_dim] o_proj input at the last answer token (float16).
        `specs` (hooks) are applied at the last prompt token and every continuation position."""
        bs = batch_size or mu.BATCH_SIZE
        N, d, nL = len(prompts), self.spec["d_model"], len(layers)
        lp = np.zeros(N, dtype=np.float64)
        rp = torch.zeros((N, nL, d), dtype=torch.float16) if nL else None
        ra = torch.zeros((N, nL, d), dtype=torch.float16) if nL else None
        hv = (torch.zeros((N, self.n_layers, self.spec["n_heads"] * self.spec["head_dim"]), dtype=torch.float16)
              if heads else None)
        cap = HeadCapture(self.layers) if heads else None
        eot = self.fmt.eot_id
        with HookSet(self.model, specs) as hs:
            if cap:
                cap.register()
            try:
                for sl in batch_slices(N, bs):
                    P, C = prompts[sl], conts[sl]
                    B = len(P)
                    ids, attn, T = left_pad([p + c for p, c in zip(P, C)], self.fmt.pad_id)
                    Lc = [len(c) for c in C]
                    if hs.add_hooks:
                        m = torch.zeros((B, T), dtype=torch.bool, device=mu.DEVICE)
                        for i in range(B):
                            m[i, T - Lc[i] - 1:] = True
                        for h in hs.add_hooks:
                            h.position_mask = m
                    try:
                        out = self.model(input_ids=ids, attention_mask=attn, position_ids=position_ids_from_mask(attn),
                                         output_hidden_states=nL > 0, use_cache=False)
                    finally:
                        for h in hs.add_hooks:
                            h.position_mask = None
                    rows, cols, tg = [], [], []
                    for i in range(B):
                        for j in range(Lc[i]):
                            rows.append(i)
                            cols.append(T - Lc[i] - 1 + j)
                            tg.append(C[i][j])
                    lpb = np.zeros(B)
                    if rows:
                        v = gather_logprobs(out.logits, torch.tensor(rows, device=mu.DEVICE),
                                            torch.tensor(cols, device=mu.DEVICE),
                                            torch.tensor(tg, device=mu.DEVICE)).cpu().numpy()
                        np.add.at(lpb, np.array(rows), v)
                    lp[sl] = lpb
                    dev = out.logits.device
                    ar = torch.arange(B, device=dev)
                    p_last = torch.tensor([T - Lc[i] - 1 for i in range(B)], device=dev)
                    a_last = torch.tensor([T - 1 - (1 if (Lc[i] >= 1 and C[i][-1] == eot) else 0) for i in range(B)],
                                          device=dev)
                    for li, L in enumerate(layers):
                        H = out.hidden_states[L + 1]
                        rp[sl, li] = H[ar, p_last].to(torch.float16).cpu()
                        ra[sl, li] = H[ar, a_last].to(torch.float16).cpu()
                    if cap:
                        for l in range(self.n_layers):
                            hv[sl, l] = cap.store[l][ar, a_last].to(torch.float16).cpu()
                        cap.store = {}
                    del out
            finally:
                if cap:
                    cap.remove()
        return {"lp": lp, "res_prompt": rp, "res_ans": ra, "heads": hv}

    def last_prompt_resid(self, prompts, layers):
        """[N, len(layers), d] float32 residual at the last prompt token (no hooks)."""
        assert_clean(self.model)
        return mu.prompt_last_residuals(self.model, self.fmt, prompts, list(layers))

    def margin(self, prompts, conts, specs=()):
        """Confidence m = log P(answer + eot) - log P("I don't know." + eot), both teacher-forced in one
        call (same batching for every call on the same rows, so differences are not batch noise)."""
        n = len(prompts)
        out = self.tf(list(prompts) + list(prompts), list(conts) + [self.idk_cont] * n, specs=specs)
        return out["lp"][:n] - out["lp"][n:]

    # ------------------------------------------------------------ throughput calibration
    def calibrate(self, prompts):
        """Rates on this machine: greedy questions/s, teacher-forced sequences/s, sampled sequences/s."""
        warm = prompts[:mu.BATCH_SIZE]
        self.greedy(warm)
        self.tf(warm, [self.idk_cont] * len(warm))
        t = time.time()
        g = prompts[:2 * mu.BATCH_SIZE]
        self.greedy(g)
        r_gen = len(g) / (time.time() - t)
        t = time.time()
        f = (prompts * 4)[:4 * mu.BATCH_SIZE]
        self.tf(f, [self.idk_cont] * len(f))
        r_fwd = len(f) / (time.time() - t)
        self.sampler_self_test(prompts)
        s = prompts[:PC.SAMPLE_BATCH_PROMPTS]
        t = time.time()
        self.sample(s, PC.N_DIS_TRIES + PC.N_CON_TRIES, PC.SAMPLE_SEED + 999)
        r_samp = len(s) * (PC.N_DIS_TRIES + PC.N_CON_TRIES) / (time.time() - t)
        return {"gen": r_gen, "fwd": r_fwd, "samp": r_samp}


def concat_gen(outs):
    if not outs:
        return mu.GenOut([], [], np.zeros(0), np.zeros(0, int), np.zeros(0, bool))
    return mu.GenOut(sum((o.texts for o in outs), []), sum((o.token_ids for o in outs), []),
                     np.concatenate([o.ans_logprob for o in outs]), np.concatenate([o.n_tokens for o in outs]),
                     np.concatenate([o.truncated for o in outs]))


def unit(v):
    v = v.float()
    return v / v.norm().clamp(min=1e-12)


IDK_TEXT = IDK_STRING

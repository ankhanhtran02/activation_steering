"""Everything this experiment does with a loaded model, built on the SQ2 session (pilot/session.py): model
loading, chat template, prompt, teacher-forced state extraction (Session.tf) and its batching / padding
conventions are SQ2's, unchanged.

Batches: fixed size (config decoding.batch_size = 64, as SQ2), left padding, in the order given (question-id
order). Greedy outputs can depend on batch composition; this is why the draft A1 is generated once and reused.
"""
import contextlib
import gc
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GenerationConfig, LogitsProcessorList

import model_utils as mu
from model_utils import GenOut, batch_slices, left_pad, position_ids_from_mask
from pilot.session import HookSet, Session, concat_gen
from vesecs.cfg import depth_layer
from vesecs.hooks import Capture, DeltaHook, FirstStepBan, OverwriteLast, RowAddHook, SeededGumbel, SideOverwrite
from vesecs.io import stable_int


def _empty_gen():
    return GenOut([], [], np.zeros(0), np.zeros(0, int), np.zeros(0, bool))


class Engine:
    def __init__(self, spec, cfg):
        self.spec, self.cfg, self.name = spec, cfg, spec["name"]
        t = time.time()
        self.sess = Session(spec, spec["revision"])
        self.load_seconds = time.time() - t
        self.model, self.tok, self.fmt = self.sess.model, self.sess.tok, self.sess.fmt
        self.layers = self.sess.layers
        self.n_layers = len(self.layers)
        self.d = int(spec["d_model"])
        self.dev = mu.DEVICE
        dec = cfg["decoding"]
        self.bs = int(dec["batch_size"])
        self.max_new = int(dec["max_new_tokens"])
        self.eot = self.fmt.eot_id
        self.stop = set(self.fmt.stop_ids)
        self.idk_cont = self.sess.idk_cont

    # ------------------------------------------------------------ basics
    def layer(self, depth):
        return depth_layer(depth, self.n_layers)

    def prompts(self, questions):
        return [self.sess.prompt(q) for q in questions]

    def cont(self, token_ids, truncated):
        return self.sess.answer_cont(token_ids, truncated)

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=True).strip()

    def release(self):
        for k in ("sess", "model", "tok", "fmt", "layers"):
            setattr(self, k, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ generation
    @torch.no_grad()
    def _gen_batch(self, prompts, max_new, procs=None, hooks=()):
        ids, attn, T = left_pad(prompts, self.fmt.pad_id)
        gc_ = GenerationConfig(do_sample=False, max_new_tokens=max_new, temperature=None, top_p=None, top_k=None,
                               repetition_penalty=1.0, eos_token_id=self.fmt.stop_ids, pad_token_id=self.fmt.pad_id,
                               return_dict_in_generate=True)
        lp = LogitsProcessorList(procs(T)) if procs else None
        with HookSet(self.model, hooks):
            out = self.model.generate(input_ids=ids, attention_mask=attn, generation_config=gc_, logits_processor=lp)
        toks, emitted = mu._split_generated(out.sequences[:, T:].tolist(), self.stop)
        return GenOut([self.decode(t) for t in toks], toks, np.full(len(toks), np.nan),
                      np.array([len(t) for t in toks]), np.array([not e for e in emitted], dtype=bool))

    def generate(self, prompts, max_new=None, add=None, ban=None):
        """Greedy answers. add = (layer, vectors [N, d]): per-row vector added at the last prompt position and
        every generated position. ban = per-row lists of token ids forbidden at the first generated step."""
        if not prompts:
            return _empty_gen()
        outs = []
        for sl in batch_slices(len(prompts), self.bs):
            hooks, procs = (), None
            if add is not None:
                L, V = add
                hooks = [(self.layers[L], RowAddHook(torch.as_tensor(V[sl], dtype=torch.float32)))]
            if ban is not None:
                b = ban[sl]
                procs = lambda T, b=b: [FirstStepBan(b, T, self.dev)]
            outs.append(self._gen_batch(prompts[sl], max_new or self.max_new, procs, hooks))
        return concat_gen(outs)

    def forced(self, prompts, first_tokens, max_extra=None):
        """completion(q, c): first answer token forced to c, then greedy, at most max_new - 1 more tokens.
        token_ids include c; text = decode(c + generated)."""
        max_extra = max_extra or self.max_new - 1
        g = self.generate([p + [int(c)] for p, c in zip(prompts, first_tokens)], max_new=max_extra)
        ids = [[int(c)] + list(t) for c, t in zip(first_tokens, g.token_ids)]
        return GenOut([self.decode(t) for t in ids], ids, g.ans_logprob, np.array([len(t) for t in ids]), g.truncated)

    @torch.no_grad()
    def tries(self, prompts, qids, n, try_offset=1):
        """n T=1 tries per prompt; try j of question q uses seed hash(model, qid, j) (j = try_offset ...).
        SQ2's shared-prefill path when its self-test passes, else plain batches. Returns per prompt
        [(text, ids, truncated)] * n."""
        if self.sess.sampler_mode is None:
            self.sess.sampler_self_test(prompts)
        P = int(self.cfg["decoding"]["tries_prompts_per_batch"])
        out_all = []
        gc_ = GenerationConfig(do_sample=False, max_new_tokens=self.max_new, temperature=None, top_p=None, top_k=None,
                               repetition_penalty=1.0, eos_token_id=self.fmt.stop_ids, pad_token_id=self.fmt.pad_id,
                               return_dict_in_generate=True)
        for sl in batch_slices(len(prompts), P):
            seeds = [stable_int(self.name, q, try_offset + j) for q in qids[sl] for j in range(n)]
            ids, attn, T = left_pad(prompts[sl], self.fmt.pad_id)
            lp = LogitsProcessorList([SeededGumbel(seeds, self.dev)])
            if self.sess.sampler_mode == "shared_prefill":
                pre = self.model(input_ids=ids[:, :-1], attention_mask=attn[:, :-1],
                                 position_ids=position_ids_from_mask(attn[:, :-1]), use_cache=True)
                cache = pre.past_key_values
                cache.batch_repeat_interleave(n)
                out = self.model.generate(input_ids=ids.repeat_interleave(n, 0), attention_mask=attn.repeat_interleave(n, 0),
                                          past_key_values=cache, generation_config=gc_, logits_processor=lp)
                del pre, cache
            else:
                out = self.model.generate(input_ids=ids.repeat_interleave(n, 0), attention_mask=attn.repeat_interleave(n, 0),
                                          generation_config=gc_, logits_processor=lp)
            toks, emitted = mu._split_generated(out.sequences[:, T:].tolist(), self.stop)
            for p in range(ids.shape[0]):
                out_all.append([(self.decode(toks[k]), toks[k], not emitted[k]) for k in range(p * n, (p + 1) * n)])
            del out
        return out_all

    # ------------------------------------------------------------ teacher-forced passes
    def tf(self, prompts, conts, layers=()):
        """SQ2's extraction (Session.tf): lp of the continuation; residual after block L (hidden_states[L+1]) at
        the last prompt token (before) and the last answer token (after), float16 [N, nL, d]."""
        if not prompts:
            z = torch.zeros((0, len(layers), self.d), dtype=torch.float16)
            return {"lp": np.zeros(0), "res_prompt": z, "res_ans": z}
        return self.sess.tf(prompts, conts, layers=list(layers), batch_size=self.bs)

    def _positions(self, conts, T):
        Lc = [len(c) for c in conts]
        p_last = [T - l - 1 for l in Lc]
        a_last = [T - 1 - (1 if (l >= 1 and c[-1] == self.eot) else 0) for l, c in zip(Lc, conts)]
        return Lc, p_last, a_last

    @torch.no_grad()
    def steered_states(self, prompts, conts, layer, vectors, cap_layers):
        """Teacher-forced pass with the per-row vector added at the last prompt token and every continuation
        position (mirrors generation-time steering); states captured by own hooks registered after the
        steering hook, so a capture at the steered block sees the steered state. Returns {L: (before, after)}
        float32 [N, d]."""
        N = len(prompts)
        out = {L: (np.zeros((N, self.d), np.float32), np.zeros((N, self.d), np.float32)) for L in cap_layers}
        for sl in batch_slices(N, self.bs):
            P, Cn = prompts[sl], conts[sl]
            ids, attn, T = left_pad([p + c for p, c in zip(P, Cn)], self.fmt.pad_id)
            B = len(P)
            Lc, p_last, a_last = self._positions(Cn, T)
            m = torch.zeros((B, T), dtype=torch.bool, device=ids.device)
            for i in range(B):
                m[i, p_last[i]:] = True
            hook = RowAddHook(torch.as_tensor(vectors[sl], dtype=torch.float32))
            hook.position_mask = m
            caps = {L: Capture() for L in cap_layers}
            specs = [(self.layers[layer], hook)] + [(self.layers[L], caps[L]) for L in cap_layers]
            with HookSet(self.model, specs):
                self.model.model(input_ids=ids, attention_mask=attn, position_ids=position_ids_from_mask(attn),
                                 use_cache=False)
            ar = torch.arange(B, device=ids.device)
            pl, al = torch.tensor(p_last, device=ids.device), torch.tensor(a_last, device=ids.device)
            for L in cap_layers:
                out[L][0][sl] = caps[L].out[ar, pl].float().cpu().numpy()
                out[L][1][sl] = caps[L].out[ar, al].float().cpu().numpy()
        return out

    # ------------------------------------------------------------ VES pull vectors
    def _cont_logprob(self, ids, attn, conts, T):
        hs = self.model.model(input_ids=ids, attention_mask=attn, position_ids=position_ids_from_mask(attn),
                              use_cache=False).last_hidden_state
        rows, cols, tg = [], [], []
        for i, c in enumerate(conts):
            for j, t in enumerate(c):
                rows.append(i)
                cols.append(T - len(c) - 1 + j)
                tg.append(int(t))
        rows_t = torch.tensor(rows, device=ids.device)
        logits = self.model.lm_head(hs[rows_t, torch.tensor(cols, device=ids.device)]).float()
        lp_tok = F.log_softmax(logits, dim=-1).gather(-1, torch.tensor(tg, device=ids.device)[:, None])[:, 0]
        return torch.zeros(len(conts), device=ids.device, dtype=lp_tok.dtype).index_add(0, rows_t, lp_tok)

    def pull_vectors(self, prompts, conts, layers):
        """g_A = d L / d delta, delta added to the residual after block L at the last prompt token, L = sum of
        log-probs of the answer tokens and the end token (unsteered teacher-forced pass). delta is float32 and
        cast when added. One backward pass gives the gradient at every requested layer. Returns
        {L: float32 [N, d]}."""
        N = len(prompts)
        bs = int(self.cfg["decoding"]["backward_batch"])
        out = {L: np.zeros((N, self.d), np.float32) for L in layers}
        for sl in batch_slices(N, bs):
            P, Cn = prompts[sl], conts[sl]
            ids, attn, T = left_pad([p + c for p, c in zip(P, Cn)], self.fmt.pad_id)
            B = len(P)
            pos = torch.tensor([T - len(c) - 1 for c in Cn], device=ids.device)
            deltas = [torch.zeros((B, self.d), dtype=torch.float32, device=ids.device, requires_grad=True) for _ in layers]
            specs = [(self.layers[L], DeltaHook(dl, pos)) for L, dl in zip(layers, deltas)]
            with HookSet(self.model, specs), torch.enable_grad():
                lp = self._cont_logprob(ids, attn, Cn, T)
                grads = torch.autograd.grad(lp.sum(), deltas)
            for L, g in zip(layers, grads):
                out[L][sl] = g.detach().float().cpu().numpy()
            del lp, grads, deltas
        return out

    @torch.no_grad()
    def logprob_with_delta(self, prompts, conts, layer, deltas):
        """Per-row L with a fixed delta [N, d] added at the last prompt token after block `layer`."""
        N = len(prompts)
        out = np.zeros(N, np.float64)
        bs = int(self.cfg["decoding"]["backward_batch"])
        for sl in batch_slices(N, bs):
            P, Cn = prompts[sl], conts[sl]
            ids, attn, T = left_pad([p + c for p, c in zip(P, Cn)], self.fmt.pad_id)
            pos = torch.tensor([T - len(c) - 1 for c in Cn], device=ids.device)
            dl = torch.as_tensor(deltas[sl], dtype=torch.float32, device=ids.device)
            with HookSet(self.model, [(self.layers[layer], DeltaHook(dl, pos))]):
                out[sl] = self._cont_logprob(ids, attn, Cn, T).double().cpu().numpy()
        return out

    @contextlib.contextmanager
    def float32(self):
        """Temporarily run the model in float32 (gradient-check fallback). bf16 -> fp32 -> bf16 is exact."""
        dtype = next(self.model.parameters()).dtype
        if dtype == torch.float32:
            yield
            return
        self.model.float()
        try:
            yield
        finally:
            self.model.to(dtype)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------ ECS
    def embed_unit(self, token_ids):
        W = self.model.get_input_embeddings().weight
        e = W[torch.as_tensor(list(token_ids), device=W.device)].float()
        return e / e.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    @torch.no_grad()
    def lens(self, H, allowed, t_idk, k):
        """Logit lens: logits = W_U . norm_final(h). Returns (top over the full vocabulary [N], top-k allowed
        tokens [N, k]). H: [N, d] any dtype; allowed: bool [V] (False for special, non-alphanumeric, t_IDK)."""
        H = torch.as_tensor(H)
        dt = next(self.model.parameters()).dtype
        tops, cands = [], []
        allowed_t = torch.as_tensor(allowed, device=self.dev)
        for sl in batch_slices(H.shape[0], 256):
            lg = self.model.lm_head(self.model.model.norm(H[sl].to(self.dev, dt))).float()
            tops.append(lg.argmax(-1).cpu())
            V = min(lg.shape[-1], allowed_t.shape[0])
            masked = torch.full_like(lg, float("-inf"))
            masked[:, :V] = torch.where(allowed_t[:V], lg[:, :V], torch.tensor(float("-inf"), device=lg.device))
            cands.append(masked.topk(k, dim=-1).indices.cpu())
        return torch.cat(tops).numpy(), torch.cat(cands).numpy()

    @torch.no_grad()
    def echo(self, prompts, first_tokens, a, b, allowed, t_idk, k, rho_a, variants, placebo_token=None):
        """The one-pass echo, computed with the prompt's KV cache: each side row is one token at position T+1
        that attends to the prompt's keys and values (exactly the fused algorithm's side row). Per question:
        prefill the prompt once (h^a_T and the logit lens at block a), candidates C = top-k allowed lens tokens,
        slots = C + [f(A1)] if new; then one batched one-token step for every (slot, variant):
          ("exact", None):      the real token c through all blocks (ECS-exact);
          ("1p", beta):         state after block a overwritten with h^a_T + beta * rho_a * e_c (ECS-1P);
          ("placebo", beta):    same with a placebo token's embedding direction;
          ("identity", None):   state after block a written back unchanged (integrity check).
        Returns per question dicts and states {variant: float16 [n_rows, d]} with rows ordered (question, slot)."""
        out_q, rows_by_v = [], {v: [] for v in variants}
        nb = int(self.cfg["decoding"]["echo_batch_questions"])
        for sl in batch_slices(len(prompts), nb):
            P = prompts[sl]
            Bq = len(P)
            ids, attn, T = left_pad(P, self.fmt.pad_id)
            cap = Capture()
            with HookSet(self.model, [(self.layers[a], cap)]):
                pre = self.model(input_ids=ids, attention_mask=attn, position_ids=position_ids_from_mask(attn),
                                 use_cache=True)
            hT = cap.out[:, -1].clone()                         # [Bq, d] model dtype
            top_final = pre.logits[:, -1].float().argmax(-1).cpu().numpy()
            cache = pre.past_key_values
            del pre
            top_a, C = self.lens(hT, allowed, t_idk, k)
            slots = []
            for i in range(Bq):
                s = [int(c) for c in C[i]]
                f = first_tokens[sl.start + i]
                if f is not None and int(f) not in s:
                    s.append(int(f))
                slots.append(s)
            R = (k + 1) * len(variants)
            side_tok, mask, ident, targets, owner = [], [], [], [], []
            for i in range(Bq):
                rows = []
                for j, c in enumerate(slots[i]):
                    for v in variants:
                        rows.append((j, c, v))
                rows += [None] * (R - len(rows))
                for r in rows:
                    if r is None:
                        side_tok.append(slots[i][0])
                        mask.append(False)
                        ident.append(False)
                        targets.append(None)
                        owner.append(None)
                        continue
                    j, c, (kind, beta) = r
                    side_tok.append(c)
                    owner.append((i, j, (kind, beta)))
                    if kind in ("1p", "placebo"):
                        tok = c if kind == "1p" else int(placebo_token(sl.start + i, j))
                        tgt = hT[i].float() + float(beta) * float(rho_a) * self.embed_unit([tok])[0]
                        mask.append(True)
                        ident.append(False)
                        targets.append(tgt)
                    else:
                        mask.append(kind == "identity")
                        ident.append(kind == "identity")
                        targets.append(None)
            tg = torch.zeros((Bq * R, self.d), dtype=torch.float32, device=ids.device)
            for r, t in enumerate(targets):
                if t is not None:
                    tg[r] = t
            hook_a = SideOverwrite(tg, torch.tensor(mask, device=ids.device),
                                   torch.tensor(ident, device=ids.device) if any(ident) else None)
            cache.batch_repeat_interleave(R)
            side_ids = torch.tensor(side_tok, device=ids.device)[:, None]
            side_attn = torch.cat([attn.repeat_interleave(R, 0), torch.ones((Bq * R, 1), dtype=attn.dtype, device=ids.device)], 1)
            side_pos = position_ids_from_mask(side_attn)[:, -1:]
            capb = Capture()
            with HookSet(self.model, [(self.layers[a], hook_a), (self.layers[b], capb)]):
                self.model(input_ids=side_ids, attention_mask=side_attn, position_ids=side_pos, past_key_values=cache,
                           use_cache=True)
            hb = capb.out[:, 0].to(torch.float16).cpu().numpy()
            del cache
            for i in range(Bq):
                out_q.append({"top_a": int(top_a[i]), "top_final": int(top_final[i]), "C": [int(c) for c in C[i]],
                              "slots": slots[i]})
            for r, o in enumerate(owner):
                if o is not None:
                    i, j, v = o
                    rows_by_v[v].append(((sl.start + i, j), hb[r]))
        states = {}
        for v, lst in rows_by_v.items():
            lst.sort(key=lambda x: x[0])
            states[v] = np.stack([x[1] for x in lst]) if lst else np.zeros((0, self.d), np.float16)
        return out_q, states

    @torch.no_grad()
    def echo_full(self, prompts, tokens, a, b, fn):
        """Spec-literal simulation (section 6.2): forward prompt + c; a hook on block a's output sets position T+1
        to fn(position T); returns h^b at T+1 [N, d] float32. Used to check the KV-cache echo."""
        out = []
        for sl in batch_slices(len(prompts), self.bs):
            seqs = [p + [int(c)] for p, c in zip(prompts[sl], tokens[sl])]
            ids, attn, T = left_pad(seqs, self.fmt.pad_id)
            capb = Capture()
            with HookSet(self.model, [(self.layers[a], OverwriteLast(fn)), (self.layers[b], capb)]):
                self.model.model(input_ids=ids, attention_mask=attn, position_ids=position_ids_from_mask(attn),
                                 use_cache=False)
            out.append(capb.out[:, -1].float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.d), np.float32)

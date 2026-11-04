"""Model/tokenizer loading, chat template, batched greedy generation, teacher-forced passes and
activation caching.

Layer convention everywhere: layer L is the 0-indexed decoder block model.model.layers[L]; its
output residual is hidden_states[L + 1] (hidden_states[0] is the embedding output).
"""
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor,
                          LogitsProcessorList)

from config import BATCH_SIZE, IDK_STRING, IDK_VARIANTS, LLAMA_DATE_STRING, SAMPLE_BATCH_PROMPTS, SYSTEM_PROMPT

# The experiment runs on one CUDA GPU; STEER_DEVICE=cpu exists only for the CPU smoke test of the
# code paths (tiny random models), never for a reported run.
DEVICE = os.environ.get("STEER_DEVICE", "cuda")
LOGSOFTMAX_CHUNK = 1024


# ----------------------------------------------------------------------------- loading
def load_model(spec, revision):
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=revision, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None or "llama" in spec["name"]:  # plan: Llama pads with its eos token
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], revision=revision, dtype=torch.bfloat16, local_files_only=True)
    model.to(DEVICE).train(False)  # inference mode (dropout off)
    model.requires_grad_(False)
    assert len(model.model.layers) == spec["n_layers"], "unexpected layer count"
    assert model.config.hidden_size == spec["d_model"], "unexpected d_model"
    fmt = ChatFormatter(tok, spec["name"])
    # Replace the hub generation config so no sampling / repetition-penalty default leaks into
    # greedy decoding (newer transformers fill unset generate() arguments from it).
    model.generation_config = GenerationConfig(
        bos_token_id=tok.bos_token_id, eos_token_id=fmt.stop_ids, pad_token_id=tok.pad_token_id)
    return model, tok, fmt


def decoder_layers(model):
    return model.model.layers


# ----------------------------------------------------------------------------- chat template
class ChatFormatter:
    def __init__(self, tok, model_name):
        self.tok = tok
        self.is_llama = "llama" in model_name
        self.eot_id = self._find_eot()
        stop = {self.eot_id}
        if tok.eos_token_id is not None:
            stop.add(tok.eos_token_id)
        self.stop_ids = sorted(stop)
        # amendment A5: canonical reference "I don't know." (with the period the prompt asks for);
        # the margin's abstain term sums both written forms (see steering.SteeredQA.idk_logprob)
        self.idk_ids = tok(IDK_STRING, add_special_tokens=False).input_ids
        self.idk_variant_ids = [tok(s, add_special_tokens=False).input_ids for s in IDK_VARIANTS]
        assert self.idk_variant_ids[0] == self.idk_ids, "IDK_VARIANTS[0] must be IDK_STRING"
        self.pad_id = tok.pad_token_id

    def _render(self, messages, add_generation_prompt):
        kw = {"date_string": LLAMA_DATE_STRING} if self.is_llama else {}
        return self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=add_generation_prompt, **kw)

    def _find_eot(self):
        marker = "XQZJ"
        text = self._render([{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": "hi"},
                             {"role": "assistant", "content": marker}], False)
        after = text.split(marker, 1)[1]
        return self.tok(after, add_special_tokens=False).input_ids[0]

    def prompt(self, user, find=None):
        """Token ids of system + user turn + generation prompt. If `find` is given, also return the
        prompt-token indices overlapping the first occurrence of `find` inside the user turn."""
        text = self._render([{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": user}], True)
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=find is not None)
        ids = list(enc.input_ids)
        if find is None:
            return ids
        span_tokens = []
        u0 = text.find(user)
        s = text.find(find, u0) if u0 >= 0 else -1
        if s < 0 and u0 >= 0:
            s = text.lower().find(find.lower(), u0)
        if s >= 0:
            e = s + len(find)
            span_tokens = [k for k, (a, b) in enumerate(enc.offset_mapping) if a < e and b > s and k > 0]
        return ids, span_tokens


# ----------------------------------------------------------------------------- batching helpers
def batch_slices(n, bs=BATCH_SIZE):
    return [slice(i, min(i + bs, n)) for i in range(0, n, bs)]


def left_pad(seqs, pad_id):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, T - len(s):] = torch.tensor(s, dtype=torch.long)
        attn[i, T - len(s):] = 1
    return ids.to(DEVICE), attn.to(DEVICE), T


def position_ids_from_mask(attn):
    # Same convention as generate(): positions count real tokens only, so left padding does not
    # shift the rotary positions of a row.
    return (attn.cumsum(-1) - 1).clamp(min=0)


def gather_logprobs(logits, rows, cols, targets):
    """log softmax(logits[rows, cols])[targets] in float32, chunked over rows."""
    out = []
    for a in range(0, len(rows), LOGSOFTMAX_CHUNK):
        lg = logits[rows[a:a + LOGSOFTMAX_CHUNK], cols[a:a + LOGSOFTMAX_CHUNK]].float()
        out.append(F.log_softmax(lg, dim=-1).gather(-1, targets[a:a + LOGSOFTMAX_CHUNK, None])[:, 0])
    return torch.cat(out) if out else torch.zeros(0, device=logits.device)


# ----------------------------------------------------------------------------- generation
@dataclass
class GenOut:
    texts: list
    token_ids: list            # answer tokens without the stop token
    ans_logprob: np.ndarray    # sum log p(answer tokens + stop token if emitted); NaN if not requested
    n_tokens: np.ndarray
    truncated: np.ndarray      # hit max_new_tokens without end-of-turn

    def subset(self, idx):
        return GenOut([self.texts[i] for i in idx], [self.token_ids[i] for i in idx],
                      self.ans_logprob[idx], self.n_tokens[idx], self.truncated[idx])


class FirstTokenBan(LogitsProcessor):
    """Sets one token id per row to -inf at the first decode step only (-1 = no ban)."""

    def __init__(self, banned, prompt_len):
        self.banned = torch.as_tensor(banned, dtype=torch.long, device=DEVICE)
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == self.prompt_len:
            rows = (self.banned >= 0).nonzero(as_tuple=True)[0]
            scores = scores.clone()
            scores[rows, self.banned[rows]] = float("-inf")
        return scores


def _split_generated(gen_rows, stop_ids):
    toks, emitted = [], []
    for row in gen_rows:
        j = next((k for k, t in enumerate(row) if t in stop_ids), None)
        toks.append(row if j is None else row[:j])
        emitted.append(j is not None)
    return toks, emitted


@torch.no_grad()
def generate(model, fmt, prompts, max_new_tokens, want_logprobs=True, ban_first=None):
    """Greedy decoding in fixed batches of BATCH_SIZE (same composition for every condition).
    ban_first: optional per-prompt token id to forbid at the first decode step (-1 = none)."""
    stop = set(fmt.stop_ids)
    texts, tok_ids, lps, ntok, trunc = [], [], [], [], []
    for sl in batch_slices(len(prompts)):
        ids, attn, T = left_pad(prompts[sl], fmt.pad_id)
        gc = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, temperature=None,
                              top_p=None, top_k=None, repetition_penalty=1.0,
                              eos_token_id=fmt.stop_ids, pad_token_id=fmt.pad_id,
                              output_logits=want_logprobs, return_dict_in_generate=True)
        procs = LogitsProcessorList([FirstTokenBan(ban_first[sl], T)]) if ban_first is not None else None
        out = model.generate(input_ids=ids, attention_mask=attn, generation_config=gc,
                             logits_processor=procs)
        gen = out.sequences[:, T:]
        rows = gen.tolist()
        toks, emitted = _split_generated(rows, stop)
        if want_logprobs:
            step_lp = torch.stack([F.log_softmax(lg.float(), -1).gather(-1, gen[:, s, None])[:, 0]
                                   for s, lg in enumerate(out.logits)], 1).cpu().numpy()
        for i, (t, em) in enumerate(zip(toks, emitted)):
            n_scored = len(t) + (1 if em else 0)
            lps.append(float(step_lp[i, :n_scored].sum()) if want_logprobs else float("nan"))
            texts.append(fmt.tok.decode(t, skip_special_tokens=True).strip())
            tok_ids.append(t)
            ntok.append(len(t))
            trunc.append(not em)
    return GenOut(texts, tok_ids, np.array(lps), np.array(ntok), np.array(trunc, dtype=bool))


@torch.no_grad()
def sample(model, fmt, prompts, n_samples, max_new_tokens, seed):
    """Temperature-1.0 ancestral samples (top_p 1, top_k 0). Returns per prompt a list of
    (text, answer token ids, truncated)."""
    torch.manual_seed(seed)
    stop = set(fmt.stop_ids)
    out_all = []
    for sl in batch_slices(len(prompts), SAMPLE_BATCH_PROMPTS):
        ids, attn, T = left_pad(prompts[sl], fmt.pad_id)
        gc = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                              repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                              num_return_sequences=n_samples, eos_token_id=fmt.stop_ids,
                              pad_token_id=fmt.pad_id, return_dict_in_generate=True)
        out = model.generate(input_ids=ids, attention_mask=attn, generation_config=gc)
        toks, emitted = _split_generated(out.sequences[:, T:].tolist(), stop)
        for p in range(ids.shape[0]):
            out_all.append([(fmt.tok.decode(toks[k], skip_special_tokens=True).strip(), toks[k],
                             not emitted[k]) for k in range(p * n_samples, (p + 1) * n_samples)])
    return out_all


# ----------------------------------------------------------------------------- forward passes
def _set_masks(mask_hooks, mask):
    for h in mask_hooks or []:
        h.position_mask = mask


@torch.no_grad()
def teacher_force(model, fmt, prompts, conts, res_layers=None, res_ranges=None, mask_hooks=None,
                  batch_size=BATCH_SIZE):
    """Forward prompt + continuation.

    Returns per-row arrays of continuation-token log-probs. If res_layers is given, also returns
    residuals at those layers averaged over continuation positions res_ranges[i] = (a, b)
    (continuation coordinates, a = -1 means the last prompt token) and the residual at position
    b - 1. mask_hooks: registered SteeringHooks whose position mask is set to the last prompt token
    plus every continuation position.
    """
    lps, res_mean, res_last = [], [], []
    for sl in batch_slices(len(prompts), batch_size):
        P = [len(p) for p in prompts[sl]]
        seqs = [p + c for p, c in zip(prompts[sl], conts[sl])]
        ids, attn, T = left_pad(seqs, fmt.pad_id)
        B = len(seqs)
        off = [T - len(s) for s in seqs]
        if mask_hooks:
            m = torch.zeros((B, T), dtype=torch.bool, device=DEVICE)
            for i in range(B):
                m[i, off[i] + P[i] - 1:] = True
            _set_masks(mask_hooks, m)
        try:
            out = model(input_ids=ids, attention_mask=attn, position_ids=position_ids_from_mask(attn),
                        output_hidden_states=res_layers is not None, use_cache=False)
        finally:
            _set_masks(mask_hooks, None)
        rows, cols, tg, owner = [], [], [], []
        for i, c in enumerate(conts[sl]):
            for j, t in enumerate(c):
                rows.append(i)
                cols.append(off[i] + P[i] - 1 + j)
                tg.append(t)
                owner.append(i)
        if rows:
            lp = gather_logprobs(out.logits, torch.tensor(rows, device=DEVICE),
                                 torch.tensor(cols, device=DEVICE),
                                 torch.tensor(tg, device=DEVICE)).cpu().numpy()
        else:
            lp = np.zeros(0)
        owner = np.array(owner, dtype=np.int64)
        for i in range(B):
            lps.append(lp[owner == i])
        if res_layers is not None:
            hs = torch.stack([out.hidden_states[L + 1] for L in res_layers], 1)  # [B, nL, T, d]
            for i in range(B):
                a, b = res_ranges[sl.start + i]
                a_pos = off[i] + P[i] + a
                b_pos = off[i] + P[i] + b
                if b_pos <= a_pos:  # empty continuation span: fall back to the last prompt token
                    a_pos, b_pos = off[i] + P[i] - 1, off[i] + P[i]
                res_mean.append(hs[i, :, a_pos:b_pos].float().mean(1).cpu())
                res_last.append(hs[i, :, b_pos - 1].float().cpu())
        del out
    if res_layers is None:
        return lps
    return lps, torch.stack(res_mean), torch.stack(res_last)


@torch.no_grad()
def probe_pass(model, fmt, prompts, subj_spans, cand_layers, batch_size=BATCH_SIZE):
    """One forward over prompt + idk_string + end-of-turn per question.

    Returns: last-prompt-token residual after every block [N, n_layers, d] (float32, CPU), residual
    at the final idk token at cand_layers [N, n_cand, d], log P(idk + eot), and mean surprisal of
    the subject tokens in the user turn (NaN when the subject is not located)."""
    suffix = fmt.idk_ids + [fmt.eot_id]
    S = len(suffix)
    n_layers = len(decoder_layers(model))
    r_last, r_idk, lp_idk, surpr = [], [], [], []
    for sl in batch_slices(len(prompts), batch_size):
        seqs = [p + suffix for p in prompts[sl]]
        ids, attn, T = left_pad(seqs, fmt.pad_id)
        out = model(input_ids=ids, attention_mask=attn, position_ids=position_ids_from_mask(attn),
                    output_hidden_states=True, use_cache=False)
        p_last = T - S - 1
        hs = out.hidden_states
        r_last.append(torch.stack([hs[L + 1][:, p_last] for L in range(n_layers)], 1).float().cpu())
        r_idk.append(torch.stack([hs[L + 1][:, T - 2] for L in cand_layers], 1).float().cpu())
        B = ids.shape[0]
        cols = torch.arange(p_last, T - 1, device=DEVICE).repeat(B)
        rows = torch.arange(B, device=DEVICE).repeat_interleave(S)
        tg = ids[:, p_last + 1:].reshape(-1)
        lp_idk.append(gather_logprobs(out.logits, rows, cols, tg).view(B, S).sum(1).cpu())
        for i, span in enumerate(subj_spans[sl]):
            off = T - len(seqs[i])
            if not span:
                surpr.append(float("nan"))
                continue
            pos = torch.tensor([off + k for k in span], device=DEVICE)
            lp = gather_logprobs(out.logits, torch.full_like(pos, i), pos - 1, ids[i, pos])
            surpr.append(float(-lp.mean()))
        del out, hs
    return torch.cat(r_last), torch.cat(r_idk), torch.cat(lp_idk).numpy(), np.array(surpr)


@torch.no_grad()
def prompt_last_residuals(model, fmt, prompts, layers, batch_size=BATCH_SIZE):
    out_all = []
    for sl in batch_slices(len(prompts), batch_size):
        ids, attn, T = left_pad(prompts[sl], fmt.pad_id)
        out = model(input_ids=ids, attention_mask=attn, position_ids=position_ids_from_mask(attn),
                    output_hidden_states=True, use_cache=False)
        out_all.append(torch.stack([out.hidden_states[L + 1][:, -1] for L in layers], 1).float().cpu())
        del out
    return torch.cat(out_all)


@torch.no_grad()
def last_position_logits(model, fmt, prompts, batch_size=BATCH_SIZE):
    """Full next-token logits at the last prompt token (float32, CPU)."""
    out_all = []
    for sl in batch_slices(len(prompts), batch_size):
        ids, attn, T = left_pad(prompts[sl], fmt.pad_id)
        out = model(input_ids=ids, attention_mask=attn, position_ids=position_ids_from_mask(attn),
                    use_cache=False)
        out_all.append(out.logits[:, -1].float().cpu())
        del out
    return torch.cat(out_all)

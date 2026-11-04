"""Direction fitting, random and null directions, hook registration and removal.

Every direction is computed in float32 on CPU from cached residuals and unit-normalised; the hook
adds alpha * v_hat cast to the model dtype (bf16). Amendment A11: alpha = c * sigma_L(v), where
sigma_L(v) is the SD of the unsteered residual's projection on v_hat over PopQA train (the shared
site then fixes one absolute norm alpha* for every primary direction and the random draws).
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from model_utils import batch_slices, decoder_layers, left_pad, position_ids_from_mask


# ----------------------------------------------------------------------------- hooks
class _OwnHook:
    """Marker base class: every hook this project registers (checked by assert_no_hooks)."""


class SteeringHook(_OwnHook):
    """Forward hook on a decoder block (or embed_tokens for ablation).

    mode "add": adds `vector` ([d], float32, already scaled by alpha)
      - prefill call (seq len > 1): to column -1 (the last prompt token of every row, because of
        left padding) if positions in {"both", "prompt_only"};
      - decode call (seq len 1, KV cache): to the single new position if positions in
        {"both", "generated_only"};
      - teacher-forced call with `position_mask` [B, T] set: to every masked position.
    mode "ablate": h <- h - (h . r_hat) r_hat at all positions (`vector` is r_hat).
    mode "mean_ablate": h <- h - ((h . r_hat) - mu) r_hat at all positions (amendment A12), where mu
      is the mean projection on r_hat at this module over the harmless reference prompts.
    `calls` counts invocations by kind, so a hook that never fires is visible (amendment A5).
    """

    def __init__(self, mode, vector, positions="both", mean=0.0):
        assert mode in ("add", "ablate", "mean_ablate") and positions in ("both", "prompt_only", "generated_only")
        self.mode = mode
        self.vector = vector.detach().float().cpu()
        self.positions = positions
        self.mean = float(mean)
        self.position_mask = None
        self.calls = {"prefill": 0, "decode": 0, "masked": 0, "skipped": 0}
        self._cast = {}

    def _vec(self, h, dtype):
        key = (h.device, dtype)
        if key not in self._cast:
            self._cast[key] = self.vector.to(h.device).to(dtype)
        return self._cast[key]

    def __call__(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        if self.mode == "add":
            v = self._vec(h, h.dtype)
            if self.position_mask is not None:
                assert self.position_mask.shape == h.shape[:2], "position mask / sequence mismatch"
                self.calls["masked"] += 1
                h = h + self.position_mask[..., None].to(h.dtype) * v
            elif h.shape[1] > 1:
                if self.positions == "generated_only":
                    self.calls["skipped"] += 1
                    return None
                self.calls["prefill"] += 1
                h = h.clone()
                h[:, -1, :] = h[:, -1, :] + v
            else:
                if self.positions == "prompt_only":
                    self.calls["skipped"] += 1
                    return None
                self.calls["decode"] += 1
                h = h + v
        else:
            self.calls["prefill" if h.shape[1] > 1 else "decode"] += 1
            r = self._vec(h, torch.float32)
            hf = h.float()
            shift = (hf @ r)[..., None] - (self.mean if self.mode == "mean_ablate" else 0.0)
            h = (hf - shift * r).to(h.dtype)
        return (h,) + tuple(output[1:]) if is_tuple else h


class MeasureHook(_OwnHook):
    """Accumulates the mean projection of a module's output on r_hat over non-pad positions."""

    def __init__(self, r_hat):
        self.r = r_hat.detach().float().cpu()
        self.mask = None
        self.total, self.count = 0.0, 0

    def __call__(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        p = h.float() @ self.r.to(h.device)
        m = self.mask.to(p.dtype)
        self.total += float((p * m).sum())
        self.count += int(m.sum())
        return None


@dataclass
class Intervention:
    """A set of hooks. adds: list of (layer, scaled vector [d], positions). ablate: unit vector
    removed (directional) or set to its reference mean (mean) at embed_tokens and every decoder block
    output (ablate_layers=None) or only at the listed decoder blocks. Add-hooks are registered first,
    so on a shared block the addition happens before the ablation."""
    name: str
    adds: list = field(default_factory=list)
    ablate: torch.Tensor = None
    ablate_mode: str = "directional"
    ablate_layers: list = None
    ablate_means: dict = None          # {"embed": mu, L: mu} for mean ablation

    def hook_specs(self, model):
        layers = decoder_layers(model)
        specs = [(layers[L], SteeringHook("add", vec, pos)) for L, vec, pos in self.adds]
        if self.ablate is not None:
            mode = "mean_ablate" if self.ablate_mode == "mean" else "ablate"
            mu = self.ablate_means or {}
            if mode == "mean_ablate":
                assert mu, "mean ablation needs reference means"
            targets = ([("embed", model.model.embed_tokens)] + [(L, layers[L]) for L in range(len(layers))]
                       if self.ablate_layers is None else [(L, layers[L]) for L in self.ablate_layers])
            specs += [(mod, SteeringHook(mode, self.ablate, mean=mu.get(key, 0.0))) for key, mod in targets]
        return specs


def steer(name, layer, unit_dir, alpha, positions="both"):
    return Intervention(name, adds=[(layer, alpha * unit_dir.float(), positions)])


def hooked_modules(model):
    return [model.model.embed_tokens] + list(decoder_layers(model))


def assert_no_hooks(model):
    # Only this project's hooks count: recent transformers keep their own output-capturing hooks on
    # every decoder layer once output_hidden_states has been requested.
    for m in hooked_modules(model):
        stale = [h for h in m._forward_hooks.values() if isinstance(h, _OwnHook)]
        assert not stale, "stale steering hook left on the model"


@torch.no_grad()
def measure_mean_projections(model, fmt, prompts, r_hat, batch_size=64):
    """Mean projection of embed_tokens and every decoder-block output on r_hat over all non-pad
    positions of `prompts` (token-id lists). Returns {"embed": mu, 0: mu, 1: mu, ...}."""
    assert_no_hooks(model)
    layers = decoder_layers(model)
    mods = [("embed", model.model.embed_tokens)] + [(L, layers[L]) for L in range(len(layers))]
    hooks = {k: MeasureHook(r_hat) for k, _ in mods}
    handles = [m.register_forward_hook(hooks[k]) for k, m in mods]
    try:
        for sl in batch_slices(len(prompts), batch_size):
            ids, attn, _ = left_pad(prompts[sl], fmt.pad_id)
            for h in hooks.values():
                h.mask = attn
            model(input_ids=ids, attention_mask=attn, position_ids=position_ids_from_mask(attn), use_cache=False)
    finally:
        for hd in handles:
            hd.remove()
    assert_no_hooks(model)
    return {k: h.total / max(h.count, 1) for k, h in hooks.items()}


# ----------------------------------------------------------------------------- directions
def unit(v):
    v = v.float()
    n = v.norm()
    assert n > 0, "zero direction"
    return v / n


def cosine(a, b):
    return float(torch.dot(unit(a), unit(b)))


def diff_of_means(X, pos, neg):
    """X [N, d]; pos/neg boolean numpy masks. Returns mean(X[pos]) - mean(X[neg]) (float32)."""
    pos_t, neg_t = torch.as_tensor(pos), torch.as_tensor(neg)
    assert pos_t.sum() > 0 and neg_t.sum() > 0, "empty class in difference of means"
    return X[pos_t].float().mean(0) - X[neg_t].float().mean(0)


def mean_norms(resid_last):
    """rbar_L for every layer: mean L2 norm of residuals [N, n_layers, d] -> [n_layers]."""
    return resid_last.float().norm(dim=-1).mean(0)


def projection_sd(R, v):
    """sigma_L(v): SD over rows of R [N, d] of the projection on v_hat (amendment A11)."""
    return float((R.float() @ unit(v)).std(unbiased=True))


def direction_diagnostics(v, R):
    """Amendment A5 (Qwen layer-14 no-op): how concentrated v_hat is in a few residual coordinates,
    and whether those coordinates carry the residual's outlier / massive-activation mass."""
    v = unit(v)
    R = R.float()
    e = (v ** 2)
    top = torch.topk(e, 5)
    mu = R.mean(0)
    proj = R @ v
    ms = (R ** 2).mean(0)                       # per-dimension mean square of the residual
    share = ms / ms.sum()
    out = {"top1_energy": float(top.values[0]), "top5_energy": float(top.values.sum()),
           "top_dims": [int(i) for i in top.indices], "sigma": float(proj.std(unbiased=True)),
           "proj_mean": float(proj.mean()), "cos_mean_residual": float(torch.dot(v, unit(mu))) if mu.norm() > 0 else 0.0,
           "rbar": float(R.norm(dim=1).mean()), "resid_mean_norm": float(mu.norm()),
           "resid_centered_rms": float((R - mu).norm(dim=1).pow(2).mean().sqrt()),
           "resid_top1_dim_share": float(share.max()), "resid_top1_dim": int(share.argmax()),
           "dir_energy_in_resid_top5_dims": float(e[torch.topk(share, 5).indices].sum())}
    return out


def within_question_direction(H, qidx, labels):
    """H [M, d] output-level residuals; qidx [M] question index; labels [M] 1 correct / 0 wrong.
    D = mean over questions of (mean correct - mean wrong)."""
    diffs = []
    for q in np.unique(qidx):
        m = qidx == q
        c, w = m & (labels == 1), m & (labels == 0)
        if c.any() and w.any():
            diffs.append(H[torch.as_tensor(c)].mean(0) - H[torch.as_tensor(w)].mean(0))
    assert diffs, "no gray-zone question with both classes"
    return torch.stack(diffs).mean(0)


def between_question_direction(H, labels):
    return diff_of_means(H, labels == 1, labels == 0)


def permute_labels_within_question(qidx, labels, seed):
    rng = np.random.default_rng(seed)
    out = labels.copy()
    for q in np.unique(qidx):
        idx = np.flatnonzero(qidx == q)
        out[idx] = labels[idx][rng.permutation(len(idx))]
    return out


def bootstrap_questions(qidx, seed):
    """Question-level bootstrap of gray-zone outputs (D_wq refits, amendment A8). Returns row
    indices and new question ids (a question drawn twice counts as two questions)."""
    rng = np.random.default_rng(seed)
    uq = np.unique(qidx)
    qb = rng.choice(uq, len(uq), replace=True)
    rows = np.concatenate([np.flatnonzero(qidx == q) for q in qb])
    newq = np.concatenate([np.full((qidx == q).sum(), j) for j, q in enumerate(qb)])
    return rows, newq


def random_unit_directions(d, n, seed_base):
    """Isotropic Gaussian draws z ~ N(0, I_d) with torch.Generator(seed_base + k), CPU float32."""
    out = []
    for k in range(n):
        g = torch.Generator().manual_seed(seed_base + k)
        out.append(unit(torch.randn(d, generator=g, dtype=torch.float32)))
    return out


def covariance_shaped_direction(R, seed):
    """Amendment A9 null: z ~ N(0, Sigma_L) with Sigma_L the covariance of the train residuals R
    [N, d], drawn as Xc^T g / sqrt(N - 1), g ~ N(0, I_N); unit-normalised. Unlike an isotropic draw
    it lives in the subspace the residual actually varies in."""
    R = R.float()
    Xc = R - R.mean(0)
    g = torch.randn(R.shape[0], generator=torch.Generator().manual_seed(seed), dtype=torch.float32)
    return unit(Xc.T @ g / max(R.shape[0] - 1, 1) ** 0.5)


def shuffled_label_direction(R, pos, neg, seed):
    """Amendment A9 null: the known/unknown labels are permuted among the labelled rows (class sizes
    kept) and the difference of means is recomputed."""
    lab = np.flatnonzero(pos | neg)
    rng = np.random.default_rng(seed)
    perm = lab[rng.permutation(len(lab))]
    p2 = np.zeros_like(pos)
    n2 = np.zeros_like(neg)
    p2[perm[:int(pos.sum())]] = True
    n2[perm[int(pos.sum()):]] = True
    return unit(diff_of_means(R, p2, n2))


def random_cosine_null(v, n, seed):
    g = torch.Generator().manual_seed(seed)
    Z = torch.randn(n, v.numel(), generator=g)
    Z = Z / Z.norm(dim=1, keepdim=True)
    return (Z @ unit(v)).numpy()


# ----------------------------------------------------------------------------- hooked condition base
class SteeredQA:
    """Base for every hooked condition: register_hook / remove_hook around a generation or a
    teacher-forced pass, with removal guaranteed in `finally`."""

    def __init__(self, ctx):
        self.ctx = ctx
        self._handles, self._add_hooks, self._hooks = [], [], []
        self.last_hook_calls = {}

    def register_hook(self, iv):
        model = self.ctx.model
        assert_no_hooks(model)
        for module, hook in iv.hook_specs(model):
            self._handles.append(module.register_forward_hook(hook))
            self._hooks.append(hook)
            if hook.mode == "add":
                self._add_hooks.append(hook)

    def remove_hook(self):
        for hd in self._handles:
            hd.remove()
        calls = {}
        for h in self._hooks:
            for k, v in h.calls.items():
                calls[k] = calls.get(k, 0) + v
        self.last_hook_calls = calls
        self._handles, self._add_hooks, self._hooks = [], [], []
        assert_no_hooks(self.ctx.model)

    def answer(self, iv, prompts, max_new_tokens, ban_first=None, want_logprobs=True):
        from model_utils import generate
        self.register_hook(iv)
        try:
            return generate(self.ctx.model, self.ctx.fmt, prompts, max_new_tokens,
                            want_logprobs=want_logprobs, ban_first=ban_first)
        finally:
            self.remove_hook()

    def teacher_forced_logprob(self, iv, prompts, conts):
        """Sum log p(continuation) with the intervention applied at the last prompt token and every
        continuation position (mirrors generation-time steering)."""
        from model_utils import teacher_force
        self.register_hook(iv)
        try:
            lps = teacher_force(self.ctx.model, self.ctx.fmt, prompts, conts,
                                mask_hooks=self._add_hooks)
        finally:
            self.remove_hook()
        return np.array([float(x.sum()) for x in lps])

    def idk_logprob(self, iv, prompts):
        """log of [P(IDK variant 1 + eot) + P(IDK variant 2 + eot) + ...] under the intervention
        (amendment A5: the abstain term covers both written forms)."""
        fmt = self.ctx.fmt
        lp = np.stack([self.teacher_forced_logprob(iv, prompts, [ids + [fmt.eot_id]] * len(prompts))
                       for ids in fmt.idk_variant_ids], 1)
        return np.logaddexp.reduce(lp, axis=1)

    def steered_margin(self, iv, prompts, gen):
        """m = log P(own greedy answer + eot) [from generation scores under the hook]
               - log sum_v P(idk variant v + eot) [teacher-forced under the same hook]."""
        return gen.ans_logprob - self.idk_logprob(iv, prompts)

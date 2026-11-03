"""Direction fitting, norm-matched random directions, hook registration and removal.

Every direction is computed in float32 on CPU from cached residuals and unit-normalised; the hook
adds alpha * v_hat cast to the model dtype (bf16), with alpha = c * rbar_L where rbar_L is the mean
L2 norm of the residual after block L at the last prompt token over PopQA train (unsteered).
"""
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np
import torch

from model_utils import decoder_layers


# ----------------------------------------------------------------------------- hooks
class SteeringHook:
    """Forward hook on a decoder block (or embed_tokens for ablation).

    mode "add": adds `vector` ([d], float32, already scaled by alpha)
      - prefill call (seq len > 1): to column -1 (the last prompt token of every row, because of
        left padding) if positions in {"both", "prompt_only"};
      - decode call (seq len 1, KV cache): to the single new position if positions in
        {"both", "generated_only"};
      - teacher-forced call with `position_mask` [B, T] set: to every masked position.
    mode "ablate": h <- h - (h . r_hat) r_hat at all positions (`vector` is r_hat).
    """

    def __init__(self, mode, vector, positions="both"):
        assert mode in ("add", "ablate") and positions in ("both", "prompt_only", "generated_only")
        self.mode = mode
        self.vector = vector.detach().float().cpu()
        self.positions = positions
        self.position_mask = None
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
                h = h + self.position_mask[..., None].to(h.dtype) * v
            elif h.shape[1] > 1:
                if self.positions == "generated_only":
                    return None
                h = h.clone()
                h[:, -1, :] = h[:, -1, :] + v
            else:
                if self.positions == "prompt_only":
                    return None
                h = h + v
        else:
            r = self._vec(h, torch.float32)
            hf = h.float()
            h = (hf - (hf @ r)[..., None] * r).to(h.dtype)
        return (h,) + tuple(output[1:]) if is_tuple else h


@dataclass
class Intervention:
    """A set of hooks. adds: list of (layer, scaled vector [d], positions). ablate: unit vector
    removed from embed_tokens and every decoder block output. Add-hooks are registered first, so on
    a shared block the addition happens before the ablation."""
    name: str
    adds: list = field(default_factory=list)
    ablate: torch.Tensor = None

    def hook_specs(self, model):
        layers = decoder_layers(model)
        specs = [(layers[L], SteeringHook("add", vec, pos)) for L, vec, pos in self.adds]
        if self.ablate is not None:
            specs.append((model.model.embed_tokens, SteeringHook("ablate", self.ablate)))
            specs += [(layer, SteeringHook("ablate", self.ablate)) for layer in layers]
        return specs


def steer(name, layer, unit_dir, alpha, positions="both"):
    return Intervention(name, adds=[(layer, alpha * unit_dir.float(), positions)])


def hooked_modules(model):
    return [model.model.embed_tokens] + list(decoder_layers(model))


def assert_no_hooks(model):
    # Only steering hooks count: recent transformers keep their own output-capturing hooks on
    # every decoder layer once output_hidden_states has been requested.
    for m in hooked_modules(model):
        stale = [h for h in m._forward_hooks.values() if isinstance(h, SteeringHook)]
        assert not stale, "stale steering hook left on the model"


@contextmanager
def apply_intervention(model, iv):
    """Registers the intervention's hooks; yields the add-hooks (for teacher-forced position masks);
    always removes every hook in `finally` and re-checks that none is left."""
    assert_no_hooks(model)
    specs = [] if iv is None else iv.hook_specs(model)
    handles = []
    try:
        for module, hook in specs:
            handles.append(module.register_forward_hook(hook))
        yield [h for _, h in specs if h.mode == "add"]
    finally:
        for hd in handles:
            hd.remove()
    assert_no_hooks(model)


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


def random_unit_directions(d, n, seed_base):
    """Isotropic Gaussian draws z ~ N(0, I_d) with torch.Generator(seed_base + k), CPU float32."""
    out = []
    for k in range(n):
        g = torch.Generator().manual_seed(seed_base + k)
        out.append(unit(torch.randn(d, generator=g, dtype=torch.float32)))
    return out


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
        self._handles, self._add_hooks = [], []

    def register_hook(self, iv):
        model = self.ctx.model
        assert_no_hooks(model)
        for module, hook in iv.hook_specs(model):
            self._handles.append(module.register_forward_hook(hook))
            if hook.mode == "add":
                self._add_hooks.append(hook)

    def remove_hook(self):
        for hd in self._handles:
            hd.remove()
        self._handles, self._add_hooks = [], []
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

    def steered_margin(self, iv, prompts, gen):
        """m = log P(own greedy answer + eot) [from generation scores under the hook]
               - log P(idk_string + eot) [teacher-forced under the same hook]."""
        fmt = self.ctx.fmt
        lp_idk = self.teacher_forced_logprob(iv, prompts, [fmt.idk_ids + [fmt.eot_id]] * len(prompts))
        return gen.ans_logprob - lp_idk

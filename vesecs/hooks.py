"""Forward hooks on decoder blocks (spec section 12, "Hooks"). Every class derives from steering._OwnHook so
pilot.session.HookSet / assert_clean see them, and HookSet removes them in a `finally` block.

Decoder-block outputs are a tensor in transformers 5.x and a tuple (element 0 = hidden states) in 4.x;
both are handled.

  RowAddHook       add_vector: per-row vectors [B, d]; prefill call (seq len > 1): added at position -1 (the
                   last prompt token, rows are left-padded); decode call (seq len 1): added at position 0;
                   teacher-forced call with `position_mask` [B, T]: added at every masked position.
  DeltaHook        adds a (differentiable) per-row delta at one given position per row (VES pull vectors).
  Capture          stores the block output (detached) of the last call, or rows of it.
  SideOverwrite    ECS side rows (seq len 1): rows with mask True get their state replaced by targets[row].
  OverwriteLast    the spec's overwrite_last(layer, fn): out[:, -1] = fn(out[:, -2]) (full-sequence echo).
"""
import torch
from transformers import LogitsProcessor

from steering import _OwnHook


def _split(output):
    is_tuple = isinstance(output, tuple)
    return (output[0] if is_tuple else output), is_tuple


def _join(h, output, is_tuple):
    return (h,) + tuple(output[1:]) if is_tuple else h


class RowAddHook(_OwnHook):
    mode = "add"          # pilot.session.Session.tf sets position_mask on hooks with mode "add"

    def __init__(self, vectors=None):
        self.vectors = vectors
        self.position_mask = None
        self.calls = {"prefill": 0, "decode": 0, "masked": 0}

    def __call__(self, module, args, output):
        if self.vectors is None:
            return None
        h, is_tuple = _split(output)
        v = self.vectors.to(h.device, h.dtype)
        assert v.shape[0] == h.shape[0], "per-row steering vectors / batch mismatch"
        if self.position_mask is not None:
            assert self.position_mask.shape == h.shape[:2], "position mask / sequence mismatch"
            self.calls["masked"] += 1
            h = h + self.position_mask[..., None].to(h.dtype) * v[:, None, :]
        elif h.shape[1] > 1:
            self.calls["prefill"] += 1
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + v
        else:
            self.calls["decode"] += 1
            h = h + v[:, None, :]
        return _join(h, output, is_tuple)


class DeltaHook(_OwnHook):
    """h[i, pos[i]] += delta[i] (delta float32, cast to the model dtype; gradients flow back to delta)."""
    mode = "delta"

    def __init__(self, delta, pos):
        self.delta, self.pos = delta, pos

    def __call__(self, module, args, output):
        h, is_tuple = _split(output)
        rows = torch.arange(h.shape[0], device=h.device)
        h = h.index_put((rows, self.pos.to(h.device)), self.delta.to(h.device, h.dtype), accumulate=True)
        return _join(h, output, is_tuple)


class Capture(_OwnHook):
    mode = "capture"

    def __init__(self):
        self.out = None

    def __call__(self, module, args, output):
        h, _ = _split(output)
        self.out = h.detach()
        return None


class SideOverwrite(_OwnHook):
    """For the one-token ECS side step: rows with mask[row] True get targets[row] (in the model dtype); rows
    with identity[row] True write their own state back through the same path (integrity check 6)."""
    mode = "overwrite"

    def __init__(self, targets=None, mask=None, identity=None):
        self.targets, self.mask, self.identity = targets, mask, identity
        self.calls = 0

    def __call__(self, module, args, output):
        if self.targets is None:
            return None
        h, is_tuple = _split(output)
        assert h.shape[1] == 1 and h.shape[0] == self.targets.shape[0], "side-row overwrite expects [B, 1, d]"
        t = self.targets.to(h.device, h.dtype)
        if self.identity is not None:
            t = torch.where(self.identity.to(h.device)[:, None], h[:, 0], t)
        h = torch.where(self.mask.to(h.device)[:, None, None], t[:, None, :], h)
        self.calls += 1
        return _join(h, output, is_tuple)


class OverwriteLast(_OwnHook):
    """overwrite_last(layer, fn): out[:, -1] = fn(out[:, -2]) on full sequences (prompt + candidate token)."""
    mode = "overwrite_last"

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, module, args, output):
        h, is_tuple = _split(output)
        if h.shape[1] < 2:
            return None
        h = h.clone()
        h[:, -1] = self.fn(h[:, -2]).to(h.dtype)
        return _join(h, output, is_tuple)


# ----------------------------------------------------------------------------- logits processors
class SeededGumbel(LogitsProcessor):
    """T = 1 sampling with per-row seeds, run through greedy decoding: argmax(logits + Gumbel noise) is an
    exact draw from softmax(logits). Row r's noise comes from its own generator seeded by seeds[r] (= a hash
    of model, question id and try number), so a try does not depend on the other rows of its batch."""

    def __init__(self, seeds, device):
        self.gens = []
        for s in seeds:
            g = torch.Generator(device=device)
            g.manual_seed(int(s))
            self.gens.append(g)

    def __call__(self, input_ids, scores):
        V = scores.shape[-1]
        u = torch.stack([torch.rand(V, generator=g, device=scores.device) for g in self.gens])
        gumbel = -torch.log(-torch.log(u.clamp(min=1e-20)))
        return scores.float() + gumbel


class FirstStepBan(LogitsProcessor):
    """B7: sets -inf on each row's banned ids at the first generated step only."""

    def __init__(self, banned_lists, prompt_len, device):
        rows, cols = [], []
        for r, ids in enumerate(banned_lists):
            for t in ids:
                rows.append(r)
                cols.append(int(t))
        self.rows = torch.tensor(rows, dtype=torch.long, device=device)
        self.cols = torch.tensor(cols, dtype=torch.long, device=device)
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == self.prompt_len and len(self.rows):
            scores = scores.clone()
            scores[self.rows, self.cols] = float("-inf")
        return scores

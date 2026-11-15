"""Ridge probes (spec 4.4 and 6.5).

D, D_before, D_after follow the SQ2 detector recipe (pilot/probes.py::_fit_predict) unchanged: each feature
block (residual at one recorded depth, at the last prompt token = "before" / "prompt", or the last answer
token = "after" / "answer", or "both" = the two kernels summed) is standardised with the training rows' mean
and SD; linear kernel / d; centred labels; depth, moment and ridge strength (relative values x mean kernel
eigenvalue) chosen by closed-form leave-one-out AUROC on the training rows, ties keeping the first
candidate in SQ2's order. D_before and D_after restrict the moment. They share the kernels and the
leave-one-out table, which gives the same choices as three separate fits.

A fitted probe is stored in primal form (one weight vector per block): score(x) = sum_b ((x_b - mean_b) /
sd_b) . w_b + mean(y). This equals SQ2's dual-form scores (up to float rounding) and can score any new
(question, answer) pair. vesecs/tests.py checks the equality against pilot.probes._fit_predict.

E (echo detector): the same recipe on one vector (no depth or moment choice). When there are more rows than
dimensions, the leave-one-out quantities are computed in primal form (the same numbers, cheaper).
"""
from dataclasses import dataclass, field

import numpy as np
import torch

from pilot import config as PC
from pilot.metrics import safe_auroc

MOMENTS = ("prompt", "answer", "both")       # SQ2 names: prompt = before answering, answer = after
RESTRICT = {"D": MOMENTS, "D_before": ("prompt",), "D_after": ("answer",)}


def _t(x):
    return x if isinstance(x, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(x))


def _stats(A):
    """SQ2's _standardize statistics (float32, unbiased SD clamped at 1e-4)."""
    A = A.float()
    return A.mean(0), A.std(0).clamp(min=1e-4)


@dataclass
class Probe:
    layer: int                     # absolute layer (model.model.layers[layer])
    moment: str                    # prompt / answer / both / vector
    lam_rel: float
    lam: float
    ybar: float
    mean: dict = field(default_factory=dict)   # block -> [d] float32
    sd: dict = field(default_factory=dict)
    w: dict = field(default_factory=dict)      # block -> [d] float64
    info: dict = field(default_factory=dict)

    def blocks(self):
        return ("prompt", "answer") if self.moment == "both" else (self.moment,)

    def score(self, before=None, after=None, vector=None):
        """before / after / vector: [N, d] arrays (any float dtype) at this probe's layer."""
        src = {"prompt": before, "answer": after, "vector": vector}
        out = None
        for b in self.blocks():
            x = _t(src[b]).float()
            z = ((x - _t(self.mean[b])) / _t(self.sd[b])).double() @ _t(self.w[b])
            out = z if out is None else out + z
        return out.numpy() + self.ybar

    def to_npz(self):
        d = {"layer": self.layer, "moment": self.moment, "lam_rel": self.lam_rel, "lam": self.lam, "ybar": self.ybar}
        for b in self.blocks():
            d[f"mean_{b}"] = np.asarray(self.mean[b], dtype=np.float32)
            d[f"sd_{b}"] = np.asarray(self.sd[b], dtype=np.float32)
            d[f"w_{b}"] = np.asarray(self.w[b], dtype=np.float64)
        return d

    @classmethod
    def from_npz(cls, z, info=None):
        p = cls(int(z["layer"]), str(z["moment"]), float(z["lam_rel"]), float(z["lam"]), float(z["ybar"]), info=info or {})
        for b in p.blocks():
            p.mean[b], p.sd[b], p.w[b] = z[f"mean_{b}"], z[f"sd_{b}"], z[f"w_{b}"]
        return p


# ----------------------------------------------------------------------------- leave-one-out tables
def _loo_dual(K, y, lambdas):
    """Exactly SQ2's computation (pilot/probes.py lines 49-63). Returns [(auroc, lam_rel, lam)]."""
    yt = torch.tensor(y.astype(np.float64))
    yc = yt - float(yt.mean())
    s, U = torch.linalg.eigh(K)
    s = s.clamp(min=0)
    Uy = U.T @ yc
    U2 = U ** 2
    out = []
    for lam_rel in lambdas:
        lam = lam_rel * float(s.mean())
        f = s / (s + lam)
        yhat = U @ (f * Uy)
        h = U2 @ f
        loo = yc - (yc - yhat) / (1 - h).clamp(min=1e-6)
        out.append((safe_auroc(y, loo.numpy()), lam_rel, lam))
    return out


def _loo_primal(X, y, lambdas):
    """Same quantities as _loo_dual for K = X X^T (X float64 [n, p], n > p): the hat matrix is
    Z diag(1 / (s + lam)) Z^T with X^T X = V diag(s) V^T and Z = X V; mean kernel eigenvalue = sum(s) / n."""
    n = X.shape[0]
    yt = torch.tensor(y.astype(np.float64))
    yc = yt - float(yt.mean())
    s, V = torch.linalg.eigh(X.T @ X)
    s = s.clamp(min=0)
    Z = X @ V
    Zy = Z.T @ yc
    Z2 = Z ** 2
    mean_eig = float(s.sum()) / n
    out, cache = [], {}
    for lam_rel in lambdas:
        lam = lam_rel * mean_eig
        inv = 1.0 / (s + lam)
        yhat = Z @ (inv * Zy)
        h = Z2 @ inv
        loo = yc - (yc - yhat) / (1 - h).clamp(min=1e-6)
        out.append((safe_auroc(y, loo.numpy()), lam_rel, lam))
        cache[lam_rel] = V @ (inv * Zy)
    return out, cache


# ----------------------------------------------------------------------------- D family
def fit_family(Rp, Ra, y, rows, layers, restrict=None, lambdas=None):
    """Rp, Ra: [N, nL, d] float16 states (SQ2 shared.pt layout) at absolute `layers`; y: bool [N]; rows: training
    row indices. Returns {name: Probe} for the requested restrictions (default D, D_before, D_after)."""
    restrict = restrict or RESTRICT
    lambdas = lambdas or PC.RIDGE_LAMBDAS
    Rp, Ra = _t(Rp), _t(Ra)
    rows = np.asarray(rows)
    y_tr = np.asarray(y)[rows].astype(bool)
    n, nL, d = len(rows), Rp.shape[1], Rp.shape[2]
    if n < 10 or len(np.unique(y_tr)) < 2:
        return {}
    ridx = torch.as_tensor(rows)
    stats, std, Ktr = {}, {}, {}
    for li in range(nL):
        for pos, A in (("prompt", Rp), ("answer", Ra)):
            a_raw = A[ridx, li].float()
            mu, sd = _stats(a_raw)
            a = (a_raw - mu) / sd
            stats[(li, pos)] = (mu, sd)
            std[(li, pos)] = a
            Ktr[(li, pos)] = (a @ a.T).double() / d
        Ktr[(li, "both")] = Ktr[(li, "prompt")] + Ktr[(li, "answer")]
    wanted = set(m for ms in restrict.values() for m in ms)
    table = {}                                   # (li, moment) -> [(auroc, lam_rel, lam)]
    for li in range(nL):                         # SQ2's order: (li, prompt), (li, answer), (li, both)
        for pos in MOMENTS:
            if pos in wanted:
                table[(li, pos)] = _loo_dual(Ktr[(li, pos)], y_tr, lambdas)
    yt = torch.tensor(y_tr.astype(np.float64))
    ybar = float(yt.mean())
    yc = yt - ybar
    out = {}
    for name, moments in restrict.items():
        best = None
        for li in range(nL):
            for pos in MOMENTS:
                if pos not in moments:
                    continue
                for a, lam_rel, lam in table[(li, pos)]:
                    if np.isfinite(a) and (best is None or a > best[0]):
                        best = (a, (li, pos), lam_rel, lam)
        if best is None:
            continue
        a, (li, pos), lam_rel, lam = best
        K = Ktr[(li, pos)]
        alpha = torch.linalg.solve(K + lam * torch.eye(K.shape[0], dtype=K.dtype), yc)
        p = Probe(int(layers[li]), pos, float(lam_rel), float(lam), ybar,
                  info={"layer_index": int(li), "position": pos, "lambda_rel": float(lam_rel),
                        "loo_auroc_train": float(a), "n_train": int(n)})
        for b in p.blocks():
            mu, sd = stats[(li, b)]
            p.mean[b], p.sd[b] = mu.numpy(), sd.numpy()
            p.w[b] = ((std[(li, b)].double().T @ alpha) / d).numpy()
        out[name] = p
    return out


def score_family(probe, Rp, Ra, layers):
    """Score SQ2-layout states [N, nL, d] with a probe fitted by fit_family."""
    li = list(layers).index(probe.layer)
    Rp, Ra = _t(Rp), _t(Ra)
    return probe.score(before=Rp[:, li] if probe.moment in ("prompt", "both") else None,
                       after=Ra[:, li] if probe.moment in ("answer", "both") else None)


def crossfit_family(Rp, Ra, y, pop, folds, layers, restrict=None, lambdas=None):
    """Out-of-fold scores (SQ2 folds) for each restriction. Returns ({name: oof [N]}, {name: [fold infos]})."""
    restrict = restrict or RESTRICT
    oof = {k: np.full(len(y), np.nan) for k in restrict}
    infos = {k: [] for k in restrict}
    folds = np.asarray(folds)
    for k in sorted(set(folds[pop].tolist())):
        tr = np.flatnonzero(pop & (folds != k))
        te = np.flatnonzero(pop & (folds == k))
        fam = fit_family(Rp, Ra, y, tr, layers, restrict, lambdas)
        for name, p in fam.items():
            oof[name][te] = score_family(p, _t(Rp)[torch.as_tensor(te)], _t(Ra)[torch.as_tensor(te)], layers)
            infos[name].append({"fold": int(k), **p.info})
    return oof, infos


# ----------------------------------------------------------------------------- E (one vector)
def fit_vector(X, y, rows, layer, lambdas=None):
    """Ridge probe on one vector per row (X [N, d] float16/32), ridge strength by LOO AUROC (SQ2 grid)."""
    lambdas = lambdas or PC.RIDGE_LAMBDAS
    rows = np.asarray(rows)
    y_tr = np.asarray(y)[rows].astype(bool)
    n, d = len(rows), X.shape[1]
    if n < 10 or len(np.unique(y_tr)) < 2:
        return None
    a_raw = _t(X)[torch.as_tensor(rows)].float()
    mu, sd = _stats(a_raw)
    a = (a_raw - mu) / sd
    yt = torch.tensor(y_tr.astype(np.float64))
    ybar = float(yt.mean())
    if n <= d:
        K = (a @ a.T).double() / d
        table = _loo_dual(K, y_tr, lambdas)
    else:
        Xd = a.double() / np.sqrt(d)
        table, wcache = _loo_primal(Xd, y_tr, lambdas)
    best = None
    for auc, lam_rel, lam in table:
        if np.isfinite(auc) and (best is None or auc > best[0]):
            best = (auc, lam_rel, lam)
    if best is None:
        return None
    auc, lam_rel, lam = best
    if n <= d:
        alpha = torch.linalg.solve(K + lam * torch.eye(n, dtype=K.dtype), yt - ybar)
        w = (a.double().T @ alpha) / d
    else:
        w = wcache[lam_rel] / np.sqrt(d)
    p = Probe(int(layer), "vector", float(lam_rel), float(lam), ybar,
              info={"lambda_rel": float(lam_rel), "loo_auroc_train": float(auc), "n_train": int(n),
                    "form": "dual" if n <= d else "primal"})
    p.mean["vector"], p.sd["vector"], p.w["vector"] = mu.numpy(), sd.numpy(), w.numpy()
    return p


def crossfit_vector(X, y, pop, folds, layer, lambdas=None):
    """Out-of-fold scores; all rows of a question share its fold (folds given per row)."""
    oof = np.full(len(y), np.nan)
    folds = np.asarray(folds)
    for k in sorted(set(folds[pop].tolist())):
        tr = np.flatnonzero(pop & (folds != k))
        te = np.flatnonzero(pop & (folds == k))
        p = fit_vector(X, y, tr, layer, lambdas)
        if p is not None and len(te):
            oof[te] = p.score(vector=_t(X)[torch.as_tensor(te)])
    return oof

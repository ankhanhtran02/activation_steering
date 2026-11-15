"""The recoverability probe R (spec section 2.2) and the Platt scaling for RRA-EU.

R uses the SQ2 detector recipe unchanged (vesecs.detectors.fit_family = SQ2's pilot/probes.py ridge recipe in
primal form): ridge probe on the before state (last prompt token), the after state (last answer token) or both,
at one of the SQ2 depths; depth, moment and ridge strength chosen by leave-one-out AUROC on the training rows.
Label y2 = 1 if A2 is not null and graded right by the strict grader. Trained on all answered pool questions.
The diagnostic AUROC is cross-fitted with the SQ2 folds; the deployed R is fitted on the whole pool.

Platt scaling (RRA-EU): p = sigmoid(a s + b), fitted by maximum likelihood on cross-fitted pool scores only
(p1: the frozen D's cross-fitted pool scores vs A1 right; p2: R's cross-fitted pool scores vs y2).
"""
import numpy as np

from scoring import auroc
from vesecs.detectors import MOMENTS, Probe, crossfit_family, fit_family, score_family

RESTRICT_R = {"R": MOMENTS}


def fit_R(Rp, Ra, y2, answered, folds, layers):
    """Returns (deployed probe, cross-fitted pool scores [N] (NaN off the answered rows), per-fold choices)."""
    fam = fit_family(Rp, Ra, y2, np.flatnonzero(answered), layers, restrict=RESTRICT_R)
    if "R" not in fam:
        raise RuntimeError("R could not be fitted (fewer than 10 answered pool questions or one class only)")
    oof, infos = crossfit_family(Rp, Ra, y2, answered, folds, layers, restrict=RESTRICT_R)
    return fam["R"], oof["R"], infos["R"]


def shuffled_R_auroc(Rp, Ra, y2, answered, folds, layers, seed):
    """Integrity check 6: R trained on shuffled pool labels (permuted within the answered rows)."""
    rng = np.random.default_rng(seed)
    ysh = np.asarray(y2, bool).copy()
    idx = np.flatnonzero(answered)
    ysh[idx] = ysh[idx][rng.permutation(len(idx))]
    oof, _ = crossfit_family(Rp, Ra, ysh, answered, folds, layers, restrict=RESTRICT_R)
    ok = answered & np.isfinite(oof["R"])
    return float(auroc(ysh[ok], oof["R"][ok]))


def score(probe, Rp, Ra, layers):
    return score_family(probe, Rp, Ra, layers)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def platt_fit(s, y, ridge=1e-6, iters=100):
    """Maximum-likelihood logistic fit p = sigmoid(a s + b) (Newton steps on the standardised score; a tiny ridge
    keeps a separable sample finite). Returns {a, b, n, n_pos, converged}."""
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if len(s) < 2 or y.min() == y.max():
        return {"a": 0.0, "b": float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6))) if len(y) else 0.0,
                "n": int(len(s)), "n_pos": int(y.sum()), "converged": False}
    mu, sd = float(s.mean()), float(s.std()) or 1.0
    z = (s - mu) / sd
    w = np.zeros(2)
    X = np.column_stack([z, np.ones_like(z)])
    conv = False
    for _ in range(iters):
        p = _sigmoid(X @ w)
        g = X.T @ (p - y) + ridge * w
        H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(2)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            conv = True
            break
    a = w[0] / sd
    b = w[1] - w[0] * mu / sd
    return {"a": float(a), "b": float(b), "n": int(len(s)), "n_pos": int(y.sum()), "converged": conv}


def platt_apply(pl, s):
    s = np.asarray(s, dtype=np.float64)
    return _sigmoid(pl["a"] * s + pl["b"])


def calibration_curve(p, y, n_bins=10):
    """Quantile bins of p: [{p_mean, y_rate, n}] (empty bins dropped)."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(p)
    p, y = p[ok], y[ok]
    if not len(p):
        return []
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    out = []
    for b in range(max(len(edges) - 1, 1)):
        m = idx == b
        if m.any():
            out.append({"p_mean": float(p[m].mean()), "y_rate": float(y[m].mean()), "n": int(m.sum())})
    return out


__all__ = ["Probe", "fit_R", "shuffled_R_auroc", "score", "platt_fit", "platt_apply", "calibration_curve", "MOMENTS"]

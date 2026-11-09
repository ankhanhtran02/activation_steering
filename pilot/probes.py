"""Cross-fitted mistake detector on internal states (SQ2; also gates SQ1's "smart" steering and routes
SQ5's attainable pick).

For each of the 5 rotating folds, a linear ridge probe is trained on the other 4 folds and scores the
held-out fold, so every question gets a score from a probe that never saw it. Folds are grouped by
PopQA subject entity. Inside the training folds, the layer (5 recorded), the position (last prompt
token, last answer token, or both) and the ridge strength are chosen by leave-one-out AUROC, computed
in closed form from the kernel's eigendecomposition (no test-fold information is used). Questions of
the SQ2 screening extension are scored by a probe fitted the same way on the whole main pool; they are
never used for training.

Population: questions whose main answer is an answer (not "I don't know"); target: the main answer is
correct.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pilot import config as PC
from pilot.metrics import safe_auroc


def _standardize(Xtr, Xte):
    mu = Xtr.mean(0)
    sd = Xtr.std(0).clamp(min=1e-4)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _fit_predict(Rp_tr, Ra_tr, y_tr, Rp_te, Ra_te, lambdas):
    """Ridge probe trained on (Rp_tr, Ra_tr) [n, nL, d] with labels y_tr; config chosen by LOO AUROC on the
    training rows; returns scores for the test rows and the chosen config (None if untrainable)."""
    n, nL, d = Rp_tr.shape
    if n < 10 or len(np.unique(y_tr)) < 2 or Rp_te.shape[0] == 0:
        return None, None
    Ktr, Kte = {}, {}
    for li in range(nL):
        for pos, (A, B) in (("prompt", (Rp_tr, Rp_te)), ("answer", (Ra_tr, Ra_te))):
            a, b = _standardize(A[:, li].float(), B[:, li].float())
            Ktr[(li, pos)] = (a @ a.T).double() / d
            Kte[(li, pos)] = (b @ a.T).double() / d
        Ktr[(li, "both")] = Ktr[(li, "prompt")] + Ktr[(li, "answer")]
        Kte[(li, "both")] = Kte[(li, "prompt")] + Kte[(li, "answer")]
    yt = torch.tensor(y_tr.astype(np.float64))
    ybar = float(yt.mean())
    yc = yt - ybar
    best = None
    for key, K in Ktr.items():
        s, U = torch.linalg.eigh(K)
        s = s.clamp(min=0)
        Uy = U.T @ yc
        U2 = U ** 2
        for lam_rel in lambdas:
            lam = lam_rel * float(s.mean())
            f = s / (s + lam)
            yhat = U @ (f * Uy)
            h = U2 @ f
            loo = yc - (yc - yhat) / (1 - h).clamp(min=1e-6)
            a = safe_auroc(y_tr, loo.numpy())
            if np.isfinite(a) and (best is None or a > best[0]):
                best = (a, key, lam_rel, lam)
    if best is None:
        return None, None
    a, key, lam_rel, lam = best
    K = Ktr[key]
    alpha = torch.linalg.solve(K + lam * torch.eye(K.shape[0], dtype=K.dtype), yc)
    scores = (Kte[key] @ alpha).numpy() + ybar
    return scores, {"layer_index": key[0], "position": key[1], "lambda_rel": lam_rel, "loo_auroc_train": float(a),
                    "n_train": int(n), "n_test": int(Rp_te.shape[0])}


def crossfit_probe(res_prompt, res_ans, y, pop, folds, lambdas=None):
    lambdas = lambdas or PC.RIDGE_LAMBDAS
    oof = np.full(len(y), np.nan)
    chosen = []
    for k in range(PC.N_FOLDS):
        tr = np.flatnonzero(pop & (folds != k))
        te = np.flatnonzero(pop & (folds == k))
        s, info = _fit_predict(res_prompt[tr], res_ans[tr], y[tr], res_prompt[te], res_ans[te], lambdas)
        if s is None:
            chosen.append({"fold": k, "skipped": True})
            continue
        oof[te] = s
        chosen.append({"fold": k, **info})
    return oof, chosen


def full_probe(res_prompt, res_ans, y, pop, Rp_new, Ra_new, lambdas=None):
    """Probe trained on every population row of the main pool; scores new questions (the extension)."""
    tr = np.flatnonzero(pop)
    s, info = _fit_predict(res_prompt[tr], res_ans[tr], y[tr], Rp_new, Ra_new, lambdas or PC.RIDGE_LAMBDAS)
    return (np.full(Rp_new.shape[0], np.nan) if s is None else s), info


def _lr(F_tr, y_tr):
    sc = StandardScaler().fit(F_tr)
    return sc, LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(F_tr), y_tr)


def crossfit_lr(F, y, pop, folds):
    """Out-of-fold logistic-regression scores (standardised features, C = 1)."""
    oof = np.full(len(y), np.nan)
    ok = pop & np.all(np.isfinite(F), axis=1)
    for k in range(PC.N_FOLDS):
        tr = np.flatnonzero(ok & (folds != k))
        te = np.flatnonzero(ok & (folds == k))
        if len(te) == 0 or len(np.unique(y[tr])) < 2:
            continue
        sc, lr = _lr(F[tr], y[tr])
        oof[te] = lr.decision_function(sc.transform(F[te]))
    return oof


def fit_lr_score(F, y, pop, F_new):
    """Logistic regression trained on every usable main-pool row; scores new rows (NaN where undefined)."""
    ok = pop & np.all(np.isfinite(F), axis=1)
    out = np.full(len(F_new), np.nan)
    good = np.all(np.isfinite(F_new), axis=1)
    if ok.sum() < 2 or len(np.unique(y[ok])) < 2 or not good.any():
        return out
    sc, lr = _lr(F[ok], y[ok])
    out[good] = lr.decision_function(sc.transform(F_new[good]))
    return out


def output_features(sh):
    """Outputs-only predictors: confidence (margin, mean token log-prob) and answer consistency
    (agreement with the main answer, distinct answers, share of declines among the 8 separate tries)."""
    return np.column_stack([sh["margin"], sh["mean_tok_lp"], sh["agree"], sh["distinct"], sh["idk_frac"]])


OUTPUT_FEATURE_NAMES = ["margin", "mean_token_logprob", "agreement", "distinct_answers", "decline_share"]

"""Statistics shared by the sub-questions: AUROC within confidence strata, bootstrap, answer
equivalence / clustering, consistency features and majority votes."""
import numpy as np
from scipy.stats import rankdata, spearmanr

from pilot import config as PC
from scoring import A_, C_, W_, abstention_type, auroc, normalize_answer


def to_py(o):
    """JSON-safe copy (numpy scalars / arrays, NaN -> None)."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else f
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ----------------------------------------------------------------------------- AUROC variants
def decile_bins(x, n_bins=None):
    n_bins = n_bins or PC.SQ1_N_BINS
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return np.zeros(0, dtype=int)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1])
    return np.searchsorted(edges, x, side="right")


def stratified_auroc(score, label, strata):
    """AUROC of `score` for `label` using only pairs inside the same stratum (pooled over strata,
    weighted by the number of positive-negative pairs). 0.5 = no information beyond the strata."""
    score, label, strata = np.asarray(score, float), np.asarray(label, bool), np.asarray(strata)
    num, den = 0.0, 0.0
    for s in np.unique(strata):
        m = strata == s
        y, x = label[m], score[m]
        n1, n0 = int(y.sum()), int((~y).sum())
        if n1 == 0 or n0 == 0:
            continue
        r = rankdata(x)
        u = r[y].sum() - n1 * (n1 + 1) / 2
        num += u
        den += n1 * n0
    return float(num / den) if den else float("nan")


def residualize(v, X):
    """Least-squares residual of v on [1, X]: the part of v that no linear combination of the covariates
    explains (Frisch-Waugh). A score built from it cannot carry information that the covariates carry."""
    v = np.asarray(v, dtype=np.float64)
    A = np.column_stack([np.ones(len(v)), np.asarray(X, dtype=np.float64)])
    beta, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ beta


def safe_auroc(label, score):
    label = np.asarray(label, bool)
    if label.all() or (~label).all():
        return float("nan")
    return auroc(label, score)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).correlation)


def bootstrap_ci(stat_fn, n, B=None, seed=None, cluster=None):
    """Percentile CI of stat_fn(idx) over resamples of n units (or of clusters: `cluster` maps rows to
    cluster ids and whole clusters are resampled)."""
    B = B or PC.N_BOOT
    rng = np.random.default_rng(PC.BOOT_SEED if seed is None else seed)
    vals = []
    if cluster is None:
        for _ in range(B):
            vals.append(stat_fn(rng.integers(0, n, n)))
    else:
        cluster = np.asarray(cluster)
        uc = np.unique(cluster)
        rows_of = {c: np.flatnonzero(cluster == c) for c in uc}
        for _ in range(B):
            pick = uc[rng.integers(0, len(uc), len(uc))]
            vals.append(stat_fn(np.concatenate([rows_of[c] for c in pick])))
    v = np.array(vals, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 0.95 * B:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "n_valid": int(len(v))}
    return {"ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5)), "n_valid": int(len(v))}


# ----------------------------------------------------------------------------- answers
def is_idk(text):
    return abstention_type(text) is not None


def same_answer(a, b):
    """Two answers are the same if their normalised forms are equal or one contains the other as
    whole words. Abstentions only match abstentions."""
    ia, ib = is_idk(a), is_idk(b)
    if ia or ib:
        return ia and ib
    na, nb = normalize_answer(a), normalize_answer(b)
    if not na or not nb:
        return False
    return na == nb or f" {na} " in f" {nb} " or f" {nb} " in f" {na} "


def cluster_answers(texts):
    """Greedy clustering by same_answer; returns cluster id per text (in order of first appearance)."""
    reps, ids = [], []
    for t in texts:
        for k, r in enumerate(reps):
            if same_answer(t, r):
                ids.append(k)
                break
        else:
            reps.append(t)
            ids.append(len(reps) - 1)
    return ids, reps


def majority(texts):
    """Index of the representative text of the largest cluster; ties go to the cluster holding
    texts[0] (the canonical answer), then to the earliest cluster."""
    ids, reps = cluster_answers(texts)
    counts = np.bincount(ids)
    best = np.flatnonzero(counts == counts.max())
    k = ids[0] if ids[0] in best else int(best[0])
    return ids.index(k)


def consistency_features(greedy_texts, tries):
    """Per question: agreement of the separate tries with the main answer, number of distinct answers
    (as a fraction of tries) and the share of tries that decline."""
    agree, distinct, idk = [], [], []
    for g, ts in zip(greedy_texts, tries):
        agree.append(float(np.mean([same_answer(t, g) for t in ts])) if ts else np.nan)
        distinct.append(len(cluster_answers(ts)[1]) / max(len(ts), 1))
        idk.append(float(np.mean([is_idk(t) for t in ts])) if ts else np.nan)
    return np.array(agree), np.array(distinct), np.array(idk)


def transition_counts(before, after):
    """Fixes (not correct -> correct) and breaks (correct -> not correct)."""
    before, after = np.asarray(before), np.asarray(after)
    fixes = int(((before != C_) & (after == C_)).sum())
    breaks = int(((before == C_) & (after != C_)).sum())
    return {"fixes": fixes, "breaks": breaks, "net": fixes - breaks,
            "W_to_C": int(((before == W_) & (after == C_)).sum()), "A_to_C": int(((before == A_) & (after == C_)).sum()),
            "C_to_W": int(((before == C_) & (after == W_)).sum()), "C_to_A": int(((before == C_) & (after == A_)).sum()),
            "W_to_A": int(((before == W_) & (after == A_)).sum()),
            "acc_before": float(np.mean(before == C_)) if len(before) else float("nan"),
            "acc_after": float(np.mean(after == C_)) if len(after) else float("nan"), "n": int(len(before))}

"""Paired inference and table quantities (numpy/scipy only; shared by report.py, reanalyze.py and
audit.py). Pre-registration amendment 2026-11-04, sections A1 and A3.

Rules
- Effect sizes: paired bootstrap over questions (N_BOOTSTRAP resamples, BOOTSTRAP_SEED). Coverage
  matching and curve construction are redone inside every resample. A resample is valid when the
  statistic is finite. If fewer than MIN_VALID_RESAMPLE_FRAC of the resamples are valid, the CI is
  reported as "undefined"; it is never computed from the surviving resamples, which form a
  conditional (biased) subset.
- Tests: two-sided paired sign-flip permutation test on per-question score differences (the score is
  the registered abstention score, so the test is about the reported effect). For coverage-matched
  comparisons the test is conditional on the coverage matching of the full sample. Exact McNemar on
  correct / not-correct is kept only as a secondary test of accuracy; it does not test the score.
- Multiplicity: Holm within each registered family (H1 {H1a, H1c}, H2 {H2b, H2c, H2d}, H3 {H3a,
  H3c}, H4 {H4b, H4c, H4d}, generalisation {H1a, H4b, H4d on transfer sets}); every other paired row
  is Holm-corrected within its own exploratory family.
- AURC: both arms on one common coverage grid (1% steps from 1% to the smaller maximum coverage
  minus AURC_GRID_SLACK, fixed from the full sample and reused in every resample).
"""
import math

import numpy as np

from config import (AURC_GRID_SLACK, AURC_GRID_STEP, AURC_MIN_POINTS, BOOTSTRAP_SEED,
                    MIN_VALID_RESAMPLE_FRAC, N_BOOTSTRAP, N_PERMUTATIONS, PERMUTATION_SEED)
from scoring import A_, C_, W_, abstention_score, auroc, match_coverage, order_by_signal, score_vec

UNDEFINED = "undefined"

# Registered confirmatory families (plan: statistics.multiplicity). Rows not listed here are
# exploratory and Holm-corrected within their own family label.
CONFIRMATORY_FAMILIES = ("H1", "H2", "H3", "H4", "generalisation")


def is_num(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and math.isfinite(float(v))


# ----------------------------------------------------------------------------- bootstrap
_IDX_CACHE = {}


def bootstrap_indices(n, B=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    key = (n, B, seed)
    if key not in _IDX_CACHE:
        _IDX_CACHE[key] = np.random.default_rng(seed).integers(0, n, size=(B, n))
    return _IDX_CACHE[key]


def paired_bootstrap(stat_fn, n, B=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, alpha=0.05,
                     min_valid_frac=MIN_VALID_RESAMPLE_FRAC):
    """stat_fn(idx) -> float on one resample of question indices. Returns point-free summary of the
    resampling distribution with the valid-resample rule applied."""
    if n == 0:
        return undefined_boot("empty sample", 0, B)
    idx = bootstrap_indices(n, B, seed)
    vals = np.array([stat_fn(r) for r in idx], dtype=np.float64)
    ok = np.isfinite(vals)
    frac = float(ok.mean())
    if frac < min_valid_frac:
        return undefined_boot(f"only {frac:.1%} of resamples valid (< {min_valid_frac:.0%})", int(ok.sum()), B)
    v = vals[ok]
    return {"status": "defined", "defined": True, "n_resamples": B, "n_valid": int(ok.sum()), "valid_frac": frac,
            "boot_mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "ci_low": float(np.percentile(v, 100 * alpha / 2)), "ci_high": float(np.percentile(v, 100 * (1 - alpha / 2))),
            "ci90_low": float(np.percentile(v, 5)), "ci90_high": float(np.percentile(v, 95)),
            # two-sided bootstrap p for "statistic = 0" (secondary; the sign-flip test is primary)
            "p_boot": float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))}


def undefined_boot(reason, n_valid=0, B=N_BOOTSTRAP):
    nan = float("nan")
    return {"status": UNDEFINED, "defined": False, "reason": reason, "n_resamples": B, "n_valid": n_valid,
            "valid_frac": n_valid / B if B else nan, "boot_mean": nan, "sd": nan, "ci_low": nan, "ci_high": nan,
            "ci90_low": nan, "ci90_high": nan, "p_boot": nan}


# ----------------------------------------------------------------------------- permutation test
def sign_flip_test(d, n_perm=N_PERMUTATIONS, seed=PERMUTATION_SEED, chunk=1000):
    """Two-sided paired sign-flip permutation test of E[d] = 0 for per-question differences d.
    p = (1 + #{|mean(s * d)| >= |mean(d)|}) / (1 + n_perm). NaN if any d is not finite."""
    d = np.asarray(d, dtype=np.float64)
    if len(d) == 0 or not np.all(np.isfinite(d)):
        return float("nan")
    nz = d[d != 0]                       # zero differences do not change under a sign flip
    if len(nz) == 0:
        return 1.0
    obs = abs(nz.sum())
    rng = np.random.default_rng(seed)
    hits, done = 0, 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        signs = rng.integers(0, 2, size=(m, len(nz)), dtype=np.int8).astype(np.float64) * 2 - 1
        hits += int((np.abs(signs @ nz) >= obs - 1e-9).sum())
        done += m
    return (1 + hits) / (1 + n_perm)


def holm(pvals):
    """Holm step-down adjusted p-values; NaN entries stay NaN and do not count toward the family."""
    p = np.asarray(pvals, dtype=np.float64)
    adj = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if len(ok) == 0:
        return adj
    order = ok[np.argsort(p[ok])]
    running, m = 0.0, len(ok)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def holm_by_family(rows, p_key="p", family_key="family", out_key="holm_p"):
    fams = {}
    for i, r in enumerate(rows):
        fams.setdefault(r.get(family_key, "exploratory"), []).append(i)
    for fam, idx in fams.items():
        adj = holm([rows[i].get(p_key, float("nan")) for i in idx])
        for i, a in zip(idx, adj):
            rows[i][out_key] = float(a)
            rows[i]["holm_family_size"] = int(np.isfinite([rows[j].get(p_key, float("nan")) for j in idx]).sum())
    return rows


# ----------------------------------------------------------------------------- per-question paired vectors
def per_item_scores(states, lam=1.0):
    return 100.0 * score_vec(np.asarray(states), lam)


def matched_states(base, signal, tiebreak, k):
    return match_coverage(base, signal, tiebreak, k)


def coverage_matched_diff(X, base, signal, tiebreak, lam=1.0):
    """Per-question differences s(X) - s(threshold answering the k_X highest-signal items), or None
    when X answers more items than the base (surplus undefined)."""
    k = int((X != A_).sum())
    Y = match_coverage(base, signal, tiebreak, k)
    if Y is None:
        return None
    return per_item_scores(X, lam) - per_item_scores(Y, lam)


def surplus_stat(X, base, signal, tiebreak, lam=1.0):
    """Bootstrap statistic: surplus vs the coverage-matched threshold, matching redone per resample."""
    def f(idx):
        Y = match_coverage(base[idx], signal[idx], tiebreak[idx], int((X[idx] != A_).sum()))
        return float("nan") if Y is None else abstention_score(X[idx], lam) - abstention_score(Y, lam)
    return f


def surplus_point(X, base, signal, tiebreak, lam=1.0):
    k = int((X != A_).sum())
    Y = match_coverage(base, signal, tiebreak, k)
    return float("nan") if Y is None else abstention_score(X, lam) - abstention_score(Y, lam)


def surplus_per_answered(X, base, signal, tiebreak, lam=1.0):
    """Surplus expressed per answered item: 100 * [sum s(X) - sum s(Y_kX)] / k_X (points per answer)."""
    k = int((X != A_).sum())
    if k == 0:
        return float("nan")
    s = surplus_point(X, base, signal, tiebreak, lam)
    return s * len(X) / k if math.isfinite(s) else float("nan")


# ----------------------------------------------------------------------------- transitions
def net_transitions(base, cond, lam=1.0):
    """Gains (W->C, W->A, A->C) and losses (C->A, C->W, A->W) of `cond` relative to `base`, as item
    counts and as points of S_lambda (x100 / N). net_points equals S(cond) - S(base)."""
    base, cond = np.asarray(base), np.asarray(cond)
    n = len(base)
    delta = score_vec(cond, lam) - score_vec(base, lam)
    cell = lambda a, b: int(((base == a) & (cond == b)).sum())
    gains = {"W_to_C": cell(W_, C_), "W_to_A": cell(W_, A_), "A_to_C": cell(A_, C_)}
    losses = {"C_to_A": cell(C_, A_), "C_to_W": cell(C_, W_), "A_to_W": cell(A_, W_)}
    gains["items"], losses["items"] = sum(gains.values()), sum(losses.values())
    gains["points"] = float(100.0 * delta[delta > 0].sum() / n) if n else float("nan")
    losses["points"] = float(100.0 * delta[delta < 0].sum() / n) if n else float("nan")
    return {"gains": gains, "losses": losses, "net_items": gains["items"] - losses["items"],
            "net_points": float(100.0 * delta.mean()) if n else float("nan")}


# ----------------------------------------------------------------------------- AURC on a common grid
def max_coverage(states):
    return float((np.asarray(states) != A_).mean()) if len(states) else 0.0


def common_grid(cov_a, cov_b, step=AURC_GRID_STEP, slack=AURC_GRID_SLACK):
    top = math.floor(min(cov_a, cov_b) / step + 1e-9) * step - slack
    if top < step:
        return np.zeros(0)
    return np.round(np.arange(step, top + 1e-9, step), 4)


def risk_on_grid(states, conf, tiebreak, grid):
    """Selective risk W/(C+W) after answering the round(c * n) most confident answered items, for
    every c in grid. NaN where that many answered items do not exist."""
    n = len(states)
    ans = np.flatnonzero(states != A_)
    ranked = states[ans[order_by_signal(conf[ans], tiebreak[ans])]]
    cum_w = np.cumsum(ranked == W_)
    out = np.full(len(grid), np.nan)
    for j, c in enumerate(grid):
        k = max(1, int(round(c * n)))
        if k <= len(ranked):
            out[j] = cum_w[k - 1] / k
    return out


def _trapz(y, x):
    f = getattr(np, "trapezoid", None) or np.trapz
    return float(f(y, x))


def aurc_on_grid(states, conf, tiebreak, grid):
    if len(grid) < 2:
        return float("nan")
    r = risk_on_grid(states, conf, tiebreak, grid)
    if not np.all(np.isfinite(r)):
        return float("nan")
    return _trapz(r, grid) / (grid[-1] - grid[0])


def aurc_comparison(sa, ca, sb, cb, tiebreak, n_boot=True):
    """AURC(a) - AURC(b) on the common grid fixed from the full sample; bootstrap reuses that grid
    (a resample whose coverage cannot reach the grid top is invalid)."""
    grid = common_grid(max_coverage(sa), max_coverage(sb))
    out = {"grid_top": float(grid[-1]) if len(grid) else float("nan"), "n_grid_points": int(len(grid)),
           "coverage_a": max_coverage(sa), "coverage_b": max_coverage(sb)}
    if len(grid) < AURC_MIN_POINTS:
        out.update({"status": UNDEFINED, "reason": f"common coverage grid has {len(grid)} points (< {AURC_MIN_POINTS})",
                    "aurc_a": float("nan"), "aurc_b": float("nan"), "diff": float("nan"),
                    "boot": undefined_boot("common grid too short")})
        return out
    a, b = aurc_on_grid(sa, ca, tiebreak, grid), aurc_on_grid(sb, cb, tiebreak, grid)
    out.update({"status": "defined", "aurc_a": a, "aurc_b": b, "diff": a - b})
    if n_boot:
        out["boot"] = paired_bootstrap(lambda idx: aurc_on_grid(sa[idx], ca[idx], tiebreak[idx], grid)
                                       - aurc_on_grid(sb[idx], cb[idx], tiebreak[idx], grid), len(sa))
    return out


def selective_accuracy_at(states, conf, tiebreak, coverage):
    n = len(states)
    ans = np.flatnonzero(states != A_)
    k = int(round(coverage * n))
    if k == 0 or k > len(ans):
        return float("nan")
    top = states[ans[order_by_signal(conf[ans], tiebreak[ans])[:k]]]
    return float((top == C_).mean())


# ----------------------------------------------------------------------------- AUROC
def within_stratum_auroc(y, s, strata):
    """Canonical within-decile AUROC (amendment A6): probability that a randomly chosen (positive,
    negative) pair drawn from the SAME stratum is ranked correctly, ties counting one half. Equals the
    pair-weighted mean of per-stratum AUROCs over strata that contain both classes. Returns
    (value, per-stratum table)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    strata = np.asarray(strata)
    ok = np.isfinite(s)
    num, den, per = 0.0, 0.0, {}
    for g in np.unique(strata[ok]):
        m = ok & (strata == g)
        n1, n0 = int(y[m].sum()), int((~y[m]).sum())
        if n1 == 0 or n0 == 0:
            per[str(g)] = {"n_pos": n1, "n_neg": n0, "auroc": float("nan")}
            continue
        a = auroc(y[m], s[m])
        per[str(g)] = {"n_pos": n1, "n_neg": n0, "auroc": a}
        num += a * n1 * n0
        den += n1 * n0
    return (num / den if den else float("nan")), per


def auroc_with_ci(y, s, n=None):
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    point = auroc(y, s)
    boot = paired_bootstrap(lambda idx: auroc(y[idx], s[idx]), len(y))
    return point, boot


# ----------------------------------------------------------------------------- inter-rater agreement
def cohen_kappa(a, b, labels=("correct", "abstained", "wrong")):
    a, b = list(a), list(b)
    pairs = [(x, y) for x, y in zip(a, b) if x in labels and y in labels]
    n = len(pairs)
    if n == 0:
        return {"kappa": float("nan"), "n": 0, "agreement": float("nan")}
    po = sum(x == y for x, y in pairs) / n
    pe = sum((sum(x == l for x, _ in pairs) / n) * (sum(y == l for _, y in pairs) / n) for l in labels)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    conf = {f"{x}|{y}": sum(1 for p, q in pairs if p == x and q == y) for x in labels for y in labels}
    return {"kappa": float(k), "n": n, "agreement": float(po), "confusion": conf}


# ----------------------------------------------------------------------------- formatting helper
def fmt_stat(v, status=None):
    if status == UNDEFINED:
        return UNDEFINED
    if v is None or isinstance(v, str):
        return UNDEFINED if v in (None, UNDEFINED) else v
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    f = float(v)
    return "nan" if not math.isfinite(f) else f"{f:.4f}"

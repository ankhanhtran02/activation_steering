"""Statistics (spec section 8): utilities U_lambda per 100 questions, paired bootstrap CIs over test question
ids (2,000 resamples, both models' rows move together, percentile interval), pooled values = mean of the two
models' values, and the ECS gap closure G = (A_echo - A_bef) / (A_aft - A_bef)."""
import numpy as np

from scoring import A_, C_, W_, auroc


def score_vec(labels, lam):
    labels = np.asarray(labels)
    return np.where(labels == C_, 1.0, np.where(labels == W_, -float(lam), 0.0))


class Boot:
    """Resample indices over the common test question list (shared by every contrast and both models)."""

    def __init__(self, n, B, seed, min_valid_frac=0.95):
        self.n, self.B, self.min_valid = n, B, min_valid_frac
        self.idx = np.random.default_rng(seed).integers(0, n, size=(B, n)) if n else np.zeros((B, 0), int)

    def ci(self, vals):
        v = np.asarray(vals, dtype=np.float64)
        v = v[np.isfinite(v)]
        if len(v) < self.min_valid * self.B:
            return {"ci_low": float("nan"), "ci_high": float("nan"), "n_valid": int(len(v))}
        return {"ci_low": float(np.percentile(v, 2.5)), "ci_high": float(np.percentile(v, 97.5)), "n_valid": int(len(v))}

    def mean_contrast(self, per_model_vec, per_model_mask=None):
        """per_model_vec: {model: [n] per-question values (e.g. 100 * (s_method - s_B3))}; masks restrict each
        model to a stratum. Returns point estimates per model and pooled, with CIs."""
        models = list(per_model_vec)
        masks = per_model_mask or {m: np.ones(self.n, bool) for m in models}
        out = {"per_model": {}}
        pooled_pt, boots = [], []
        for m in models:
            x, mk = np.asarray(per_model_vec[m], dtype=np.float64), np.asarray(masks[m], dtype=bool)
            pt = float(x[mk].mean()) if mk.any() else float("nan")
            num = (x * mk)[self.idx].sum(1)
            den = mk[self.idx].sum(1).astype(np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                b = np.where(den > 0, num / np.maximum(den, 1), np.nan)
            out["per_model"][m] = {"value": pt, "n": int(mk.sum()), **self.ci(b)}
            pooled_pt.append(pt)
            boots.append(b)
        out["pooled"] = {"value": float(np.mean(pooled_pt)) if all(np.isfinite(pooled_pt)) else float("nan"),
                         **self.ci(np.mean(np.stack(boots), axis=0))}
        return out


def utility_diff_vec(lab_a, lab_b, lam=1.0):
    return 100.0 * (score_vec(lab_a, lam) - score_vec(lab_b, lam))


def gap_closure(y, s_bef, s_aft, s_echo, min_gap):
    """G for one model on answered drafts. Returns dict with the AUROCs and G (NaN if A_aft - A_bef < min_gap)."""
    a_bef, a_aft, a_echo = auroc(y, s_bef), auroc(y, s_aft), auroc(y, s_echo)
    gap = a_aft - a_bef
    G = (a_echo - a_bef) / gap if (np.isfinite(gap) and gap >= min_gap) else float("nan")
    return {"auroc_before": a_bef, "auroc_after": a_aft, "auroc_echo": a_echo, "gap": gap, "G": G,
            "defined": bool(np.isfinite(G))}


def gap_closure_boot(boot, per_model, min_gap):
    """per_model: {model: (answered mask [n], y [n], s_bef [n], s_aft [n], s_echo [n])} over the common test list.
    Pooled G = mean of the two models' G; the CI resamples question ids for both models together. In a
    resample, G is computed when the AUROC gap is positive (the min_gap rule decides definedness on the
    point estimate only)."""
    point, Gb = {}, []
    for m, (ans, y, sb, sa, se) in per_model.items():
        point[m] = gap_closure(y[ans], sb[ans], sa[ans], se[ans], min_gap)
    for r in boot.idx:
        vals = []
        for m, (ans, y, sb, sa, se) in per_model.items():
            q = r[ans[r]]
            ab, aa, ae = auroc(y[q], sb[q]), auroc(y[q], sa[q]), auroc(y[q], se[q])
            vals.append((ae - ab) / (aa - ab) if aa - ab > 0 else np.nan)
        Gb.append(np.mean(vals))
    defined = all(p["defined"] for p in point.values())
    pooled = float(np.mean([p["G"] for p in point.values()])) if defined else float("nan")
    return {"per_model": point, "pooled": {"value": pooled, "defined": defined,
                                           **(boot.ci(Gb) if defined else {"ci_low": float("nan"), "ci_high": float("nan")})}}


def net_fixes(lab_method, lab_ref):
    lab_method, lab_ref = np.asarray(lab_method), np.asarray(lab_ref)
    fixes = int(((lab_ref != C_) & (lab_method == C_)).sum())
    breaks = int(((lab_ref == C_) & (lab_method != C_)).sum())
    return {"fixes": fixes, "breaks": breaks, "net": fixes - breaks}


def risk_coverage(labels_by_tau):
    """[(tau, labels)] -> [{tau, coverage, risk}] (answered share, wrong / answered)."""
    out = []
    for tau, lab in labels_by_tau:
        lab = np.asarray(lab)
        ans = lab != A_
        out.append({"tau": tau, "coverage": float(ans.mean()) if len(lab) else float("nan"),
                    "risk": float((lab[ans] == W_).mean()) if ans.any() else float("nan")})
    return out

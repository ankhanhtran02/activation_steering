"""Corrected condition tables, paired comparisons and H1/H2c analyses on a ModelView (numpy only).

Used by report.py (live run, views built from ModelRunner objects) and by reanalyze.py (offline,
views built from saved per-item records), so a finished run and a new run are analysed by the same
code. Implements amendment 2026-11-04 sections A1 (statistics), A3 (always-abstain, net
transitions, surplus per answered item) and A7 (gated knockout, exploratory).
"""
import math
from collections import Counter

import numpy as np

import stats as S
from config import (COVERAGE_POINTS, H2B_MAX_TOP_TERCILE_DAMAGE, H2C_MAX_ABLATION_ABST_RISE, H2C_MIN_STEER_INCREASE,
                    LAMBDAS, MIN_VALID_RESAMPLE_FRAC, TOST_MARGIN)
from scoring import (A_, C_, W_, abstention_score, alias_match, mcnemar_exact, normalize_answer, summary_metrics,
                     transition_metrics)


class ModelView:
    """Per-model, per-question-set arrays in one fixed question order.

    outs[name] -> int8 states; conf[name] -> confidence used to rank that condition's answers;
    base = unsteered states; margin = unsteered margin; tau1 = tau_dev(lambda = 1) of the margin gate."""

    def __init__(self, model, dataset, qids, tiebreak, base, margin, outs, conf=None, texts=None, tau1=float("nan"),
                 tercile=None, decile=None, kstatus=None, relation=None, popularity=None, extras=None):
        self.model, self.dataset = model, dataset
        self.qids = list(qids)
        self.tiebreak = np.asarray(tiebreak, dtype=np.int64)
        self.base = np.asarray(base, dtype=np.int8)
        self.margin = np.asarray(margin, dtype=np.float64)
        self.outs = {k: np.asarray(v, dtype=np.int8) for k, v in outs.items()}
        self.conf = {k: np.asarray(v, dtype=np.float64) for k, v in (conf or {}).items()}
        self.texts = texts or {}
        self.tau1 = float(tau1)
        self.tercile, self.decile = tercile, decile
        self.kstatus, self.relation, self.popularity = kstatus, relation, popularity
        self.extras = extras or {}
        self.outs.setdefault("unsteered", self.base)
        self.outs.setdefault("always_abstain", np.full(len(self.base), A_, dtype=np.int8))
        self.conf.setdefault("unsteered", self.margin)

    @property
    def n(self):
        return len(self.base)

    def gate(self):
        return (self.base == A_) | (self.margin < self.tau1)


# ----------------------------------------------------------------------------- condition table
def condition_row(v, name):
    X = v.outs[name]
    m = summary_metrics(X)
    m.update(transition_metrics(v.base, X))
    m.update(S.net_transitions(v.base, X))
    k = int((X != A_).sum())
    m["answered"] = k
    m["unmatched_coverage"] = bool(k > int((v.base != A_).sum()))
    m["surplus_vs_threshold"] = {f"lam{lam:g}": S.surplus_point(X, v.base, v.margin, v.tiebreak, lam) for lam in LAMBDAS}
    m["surplus_per_answered"] = S.surplus_per_answered(X, v.base, v.margin, v.tiebreak)
    if v.popularity is not None:
        m["surplus_vs_popularity_gate"] = S.surplus_point(X, v.base, np.asarray(v.popularity, float), v.tiebreak)
    return m


def condition_table(v, names=None):
    return {n: condition_row(v, n) for n in (names or v.outs)}


# ----------------------------------------------------------------------------- paired rows
PAIRED_METHODS = ["knowledge_dir_steer", "read_dir_steer", "within_question_dir_steer", "refusal_dir_steer",
                  "margin_gated_steer_knowledge", "margin_gated_steer_knowledge_tuned", "margin_gated_steer_wq",
                  "dose_matched_placebo_force", "knowledge_dir_steer_within_relation",
                  "knowledge_dir_steer_between_relation", "within_question_dir_steer_own", "confidence_prompt_t050",
                  "majority_vote_k8", "best_of_k8", "confidence_threshold_margin", "popularity_gate",
                  "always_abstain"]


def family_of(method, baseline, dataset):
    """Registered confirmatory families (plan statistics.multiplicity); everything else exploratory.
    Final F2: H1c is the gated arm at its dev-tuned strength; the gated arm at c* (registered H1c,
    decided by construction) is exploratory."""
    transfer = dataset != "popqa"
    if method == "knowledge_dir_steer" and baseline == "confidence_threshold_margin":
        return "generalisation" if transfer else "H1"            # H1a
    if method == "margin_gated_steer_knowledge_tuned" and baseline == "confidence_threshold_margin" and not transfer:
        return "H1"                                               # H1c (final F2)
    if method in ("within_question_dir_steer", "margin_gated_steer_wq"):
        return "generalisation" if transfer else "H4"             # H4d
    return f"exploratory_{dataset}"


def _comparison(v, method, baseline):
    """(per-item difference vector or None, bootstrap statistic, matching label, reason)."""
    X = v.outs[method]
    sX = S.per_item_scores(X)
    if baseline == "unsteered":
        Y = v.base
        return sX - S.per_item_scores(Y), (lambda idx: abstention_score(X[idx]) - abstention_score(Y[idx])), Y, "unsteered", ""
    op = v.outs.get("confidence_threshold_margin")
    if method.startswith("margin_gated") or method == "confidence_threshold_margin":
        if op is None:
            return None, None, None, "operating_point", "threshold operating point missing"
        return sX - S.per_item_scores(op), (lambda idx: abstention_score(X[idx]) - abstention_score(op[idx])), op, \
            "operating_point", ""
    d = S.coverage_matched_diff(X, v.base, v.margin, v.tiebreak)
    if d is None:
        return None, None, None, "coverage_matched", f"{method} answers more items than unsteered"
    k = int((X != A_).sum())
    Y = S.matched_states(v.base, v.margin, v.tiebreak, k)
    return d, S.surplus_stat(X, v.base, v.margin, v.tiebreak), Y, "coverage_matched", ""


def paired_rows(views, dataset):
    """Per-model rows (inference units of the registered families) plus pooled rows (descriptive).
    All views share one question order (same test questions for every model)."""
    rows = []
    methods = [m for m in PAIRED_METHODS if all(m in v.outs for v in views)]
    for method in methods:
        for baseline in ("unsteered", "confidence_threshold_margin"):
            if method == baseline or (method == "confidence_threshold_margin" and baseline != "unsteered"):
                continue
            comps = {v.model: _comparison(v, method, baseline) for v in views}
            for v in views:
                d, stat, Y, matching, reason = comps[v.model]
                rows.append(_row(f"{method}@{v.model}", f"{baseline}@{v.model}", method, baseline, v.model, dataset,
                                 family_of(method, baseline, dataset), v.n, d, stat, matching, reason,
                                 v.outs[method], Y))
            # pooled: defined only if every model's comparison is defined (never silently drop a model)
            missing = [m for m, c in comps.items() if c[0] is None]
            if any(v.qids != views[0].qids for v in views):
                continue                                  # pooling needs one shared question order
            if missing:
                rows.append(_row(method, baseline, method, baseline, "pooled", dataset, f"pooled_{dataset}", views[0].n,
                                 None, None, comps[views[0].model][3],
                                 "undefined in " + ", ".join(missing) + ": " + "; ".join(comps[m][4] for m in missing),
                                 None, None))
                continue
            d_pool = np.mean([comps[v.model][0] for v in views], axis=0)
            stats_ = [comps[v.model][1] for v in views]
            stat_pool = lambda idx, stats_=stats_: float(np.mean([f(idx) for f in stats_]))
            X_all = np.concatenate([v.outs[method] for v in views])
            Y_all = np.concatenate([comps[v.model][2] for v in views])
            rows.append(_row(method, baseline, method, baseline, "pooled", dataset, f"pooled_{dataset}", views[0].n,
                             d_pool, stat_pool, comps[views[0].model][3], "", X_all, Y_all))
    if dataset != "popqa":
        for r in rows:
            r["method"], r["baseline"] = f"{dataset}:{r['method']}", f"{dataset}:{r['baseline']}"
    return rows


def finalize_paired(rows):
    """Holm within each family over every dataset at once (the generalisation family spans the
    transfer sets)."""
    return S.holm_by_family(rows, p_key="perm_p")


def _row(label_m, label_b, method, baseline, model, dataset, family, n, d, stat, matching, reason, X, Y):
    row = {"method": label_m, "baseline": label_b, "method_name": method, "baseline_name": baseline, "model": model,
           "dataset": dataset, "family": family, "matching": matching}
    if d is None:
        row.update({"status": S.UNDEFINED, "reason": reason, "mean_diff": float("nan"), "ci_low": float("nan"),
                    "ci_high": float("nan"), "perm_p": float("nan"), "mcnemar_p": float("nan"), "valid_frac": float("nan")})
        return row
    b = S.paired_bootstrap(stat, n)
    p_mc, _, _ = mcnemar_exact(X == C_, Y == C_) if X is not None else (float("nan"), 0, 0)
    # a comparison whose CI is undefined (< 95% valid resamples) is undefined as a whole: no p-value,
    # and it does not count toward its Holm family
    p = S.sign_flip_test(d) if b["defined"] else float("nan")
    row.update({"status": b["status"], "reason": b.get("reason", ""), "mean_diff": float(np.mean(d)),
                "ci_low": b["ci_low"], "ci_high": b["ci_high"], "ci90_low": b["ci90_low"], "ci90_high": b["ci90_high"],
                "boot_sd": b["sd"], "valid_frac": b["valid_frac"], "perm_p": p,
                "mcnemar_p": p_mc, "n_changed": int((np.asarray(d) != 0).sum())})
    return row


def format_paired(row):
    st = row["status"]
    f = lambda k: S.fmt_stat(row.get(k), st if k in ("ci_low", "ci_high") else None)
    return (f"PAIRED: {row['method']} vs {row['baseline']} mean_diff={f('mean_diff')} ci_low={f('ci_low')} "
            f"ci_high={f('ci_high')} mcnemar_p={f('mcnemar_p')} holm_p={f('holm_p')} perm_p={f('perm_p')} "
            f"holm_basis=perm_p family={row['family']} valid_frac={f('valid_frac')} status={st}"
            + (f" reason=\"{row['reason']}\"" if row.get("reason") else ""))


# ----------------------------------------------------------------------------- H1
def h1a_category(point, b):
    """Final F12: one outcome label per H1a estimate (priority order):
    superior      surplus >= +5 and 95% CI > 0 (the registered rejection of H1)
    equivalent    90% CI inside (-3, +3) (the registered TOST prediction)
    inferior      95% CI < 0 (steering selects worse than the threshold)
    non_superior  95% CI upper bound < +3 (no meaningful advantage, equivalence not shown)
    inconclusive  anything else"""
    if not b.get("defined"):
        return "undefined"
    if point >= 5 and b["ci_low"] > 0:
        return "superior"
    if b["ci90_low"] > -TOST_MARGIN and b["ci90_high"] < TOST_MARGIN:
        return "equivalent"
    if b["ci_high"] < 0:
        return "inferior"
    if b["ci_high"] < TOST_MARGIN:
        return "non_superior"
    return "inconclusive"


def h1a(v, name="knowledge_dir_steer"):
    X = v.outs[name]
    point = S.surplus_point(X, v.base, v.margin, v.tiebreak)
    d = S.coverage_matched_diff(X, v.base, v.margin, v.tiebreak)
    if d is None:
        return {"status": S.UNDEFINED, "reason": f"{name} answers more items than unsteered", "surplus": point,
                "category": "undefined"}
    fs = S.surplus_stat(X, v.base, v.margin, v.tiebreak)
    b = S.paired_bootstrap(fs, v.n)
    k = int((X != A_).sum())
    out = {"surplus": point, "surplus_per_answered": S.surplus_per_answered(X, v.base, v.margin, v.tiebreak),
           "answered": k, "coverage": k / v.n if v.n else float("nan"),
           "perm_p": S.sign_flip_test(d) if b["defined"] else float("nan"), **b}
    if b["defined"]:
        out.update({"tost_equivalent_3pts": bool(b["ci90_low"] > -TOST_MARGIN and b["ci90_high"] < TOST_MARGIN),
                    "pragmatist_surplus_ge_5": bool(point >= 5 and b["ci_low"] > 0),
                    "non_superior_3pts": bool(b["ci_high"] < TOST_MARGIN), "inferior": bool(b["ci_high"] < 0),
                    "underpowered_se_gt_1.6": bool(b["sd"] > 1.6)})
    out["category"] = h1a_category(point, b)
    # final F12: the same contrast on the answered-accuracy scale (percentage points): with both arms
    # answering k items, surplus * n / (2k) is the difference in accuracy among answered items
    if k > 0:
        acc = S.paired_bootstrap(lambda idx: fs(idx) * len(idx) / (2.0 * max(1, int((X[idx] != A_).sum()))), v.n)
        out["answered_accuracy_diff_pp"] = {"point": point * v.n / (2.0 * k), "ci_low": acc["ci_low"],
                                            "ci_high": acc["ci_high"], "ci90_low": acc["ci90_low"],
                                            "ci90_high": acc["ci90_high"], "status": acc["status"]}
    return out


def gated_vs_refusal(v, X, boot=True):
    """Gated steering composed from steered states X (steered where the margin gate fires, unsteered
    elsewhere) minus gated refusal (the threshold operating point)."""
    op = v.outs.get("confidence_threshold_margin")
    if op is None:
        return {"status": S.UNDEFINED, "reason": "threshold operating point missing"}
    G = np.where(v.gate(), X, v.base)
    d = S.per_item_scores(G) - S.per_item_scores(op)
    out = {"delta_vs_gated_refusal": float(d.mean()), "n_gated": int(v.gate().sum())}
    if boot:
        out.update(S.paired_bootstrap(lambda idx: abstention_score(G[idx]) - abstention_score(op[idx]), v.n))
        out["perm_p"] = S.sign_flip_test(d)
    return out


def h1b_dose_matched(v, arms, learned="knowledge_dir_steer"):
    """Final F3-F5. arms = {condition: matched_on_dev}. For every dose-matched arm: surplus of the
    learned direction minus the arm's surplus (each against its own coverage-matched threshold), with
    the paired bootstrap and TOST at +/- 3 points. The pre-specified placebo is the uniform push
    (dose_matched_placebo_force); the registered comparison is the best dev-matched random draw."""
    out = {"learned": learned, "learned_surplus": S.surplus_point(v.outs[learned], v.base, v.margin, v.tiebreak),
           "learned_abstention_rate": float((v.outs[learned] == A_).mean()), "arms": {}}
    for name, matched in arms.items():
        if name not in v.outs:
            continue
        sd = surplus_difference(v, learned, name)
        e = {"matched_on_dev": bool(matched), "abstention_rate": float((v.outs[name] == A_).mean()), **sd}
        if sd.get("defined"):
            e["tost_equivalent_3pts"] = bool(sd["ci90_low"] > -TOST_MARGIN and sd["ci90_high"] < TOST_MARGIN)
            e["learned_exceeds_by_5"] = bool(sd["diff"] >= 5 and sd["ci_low"] > 0)
        out["arms"][name] = e
    rnd = [n for n in out["arms"] if n.startswith("dose_matched_placebo_iso_") or n.startswith("dose_matched_placebo_cov_")]
    usable = [n for n in rnd if out["arms"][n]["matched_on_dev"] and out["arms"][n].get("defined")]
    out["n_random_arms"], out["n_random_matched_on_dev"] = len(rnd), len(usable)
    if usable:
        best = max(usable, key=lambda n: out["arms"][n]["surplus_b"])
        out["registered_best_matched_random"] = {"arm": best, **{k: out["arms"][best].get(k) for k in
                                                                 ("diff", "ci_low", "ci_high", "ci90_low", "ci90_high",
                                                                  "learned_exceeds_by_5", "tost_equivalent_3pts")}}
    else:
        out["registered_best_matched_random"] = {
            "status": S.UNDEFINED, "reason": "no random direction (sign-aligned isotropic or covariance-shaped) reached "
                                             "knowledge_dir_steer's dev abstention increase without degenerating: random "
                                             "directions cannot induce clean abstention (pre-registered branch: H1a decides)"}
    f = out["arms"].get("dose_matched_placebo_force")
    out["primary_uniform_push"] = (f if f is not None else {"status": S.UNDEFINED, "reason": "uniform push arm missing"})
    return out


def h1b(v, name="knowledge_dir_steer", draw_prefix="random_dir_placebo_s",
        cm_prefix="random_dir_placebo_coverage_matched_s"):
    learned = S.surplus_point(v.outs[name], v.base, v.margin, v.tiebreak)
    draws = sorted([k for k in v.outs if k.startswith(draw_prefix)], key=lambda k: int(k.rsplit("s", 1)[1]))
    out = {"learned_surplus": learned, "n_draws": len(draws)}
    if draws:
        sur = {d: S.surplus_point(v.outs[d], v.base, v.margin, v.tiebreak) for d in draws}
        ok = [d for d in draws if math.isfinite(sur[d])]
        frac = len(ok) / len(draws)
        lift = lambda X: transition_metrics(v.base, X)["selectivity_lift"]
        out.update({"n_draws_surplus_defined": len(ok), "draw_surplus": sur,
                    "lift_learned": lift(v.outs[name]), "lift_random_median": float(np.median([lift(v.outs[d]) for d in draws]))})
        if frac < MIN_VALID_RESAMPLE_FRAC:
            out["norm_matched"] = {"status": S.UNDEFINED, "reason": f"surplus undefined for {len(draws) - len(ok)} of "
                                   f"{len(draws)} draws (they answer more items than unsteered)"}
        elif not math.isfinite(learned):
            out["norm_matched"] = {"status": S.UNDEFINED, "reason": f"{name} surplus undefined"}
        else:
            vals = np.array([sur[d] for d in ok])
            out["norm_matched"] = {"status": "defined", "p95": float(np.percentile(vals, 95)),
                                   "learned_above_p95": bool(learned > np.percentile(vals, 95)),
                                   "empirical_p": (1 + int((vals >= learned).sum())) / (len(vals) + 1)}
    cm = sorted([k for k in v.outs if k.startswith(cm_prefix)], key=lambda k: int(k.rsplit("s", 1)[1]))
    if cm:
        matched = v.extras.get("cm_matched_on_dev", {})
        sur = {d: S.surplus_point(v.outs[d], v.base, v.margin, v.tiebreak) for d in cm}
        usable = [d for d in cm if matched.get(d) and math.isfinite(sur[d])]
        out["cm_draw_surplus"], out["cm_matched_on_dev"] = sur, {d: bool(matched.get(d)) for d in cm}
        if not usable:
            why = ("no coverage-matched draw reached the knowledge_dir_steer dev dose with <= 10% degeneration"
                   if not any(matched.get(d) for d in cm) else "every dose-matched draw answers more items than unsteered")
            out["coverage_matched"] = {"status": S.UNDEFINED, "reason": why}
        elif not math.isfinite(learned):
            out["coverage_matched"] = {"status": S.UNDEFINED, "reason": f"{name} surplus undefined"}
        else:
            best = max(usable, key=lambda d: sur[d])
            Xk, Xb = v.outs[name], v.outs[best]
            fk, fb = S.surplus_stat(Xk, v.base, v.margin, v.tiebreak), S.surplus_stat(Xb, v.base, v.margin, v.tiebreak)
            b = S.paired_bootstrap(lambda idx: fk(idx) - fb(idx), v.n)
            out["coverage_matched"] = {"best_draw": best, "learned_minus_best": learned - sur[best], **b}
    return out


def surplus_difference(v, a, b):
    """Surplus of arm a minus surplus of arm b (each vs its own coverage-matched threshold), paired
    bootstrap with the matching redone in every resample and the valid-resample rule."""
    sa = S.surplus_point(v.outs[a], v.base, v.margin, v.tiebreak)
    sb = S.surplus_point(v.outs[b], v.base, v.margin, v.tiebreak)
    if not (math.isfinite(sa) and math.isfinite(sb)):
        return {"status": S.UNDEFINED, "reason": f"surplus undefined for {a if not math.isfinite(sa) else b}",
                "surplus_a": sa, "surplus_b": sb}
    fa = S.surplus_stat(v.outs[a], v.base, v.margin, v.tiebreak)
    fb = S.surplus_stat(v.outs[b], v.base, v.margin, v.tiebreak)
    return {"diff": sa - sb, "surplus_a": sa, "surplus_b": sb, **S.paired_bootstrap(lambda idx: fa(idx) - fb(idx), v.n)}


def h1c(v, name="margin_gated_steer_knowledge"):
    op = v.outs.get("confidence_threshold_margin")
    if op is None or name not in v.outs:
        return {"status": S.UNDEFINED, "reason": "missing arm"}
    X, g = v.outs[name], v.gate()
    d = S.per_item_scores(X) - S.per_item_scores(op)
    off = np.abs(d[~g]).max() if (~g).any() else 0.0
    b = S.paired_bootstrap(lambda idx: abstention_score(X[idx]) - abstention_score(op[idx]), v.n)
    p_mc, bb, cc = mcnemar_exact((X == C_)[g], (op == C_)[g])
    return {"delta_vs_gated_refusal": float(d.mean()), "n_gated": int(g.sum()), "off_gate_max_abs_diff": float(off),
            "perm_p": S.sign_flip_test(d), "mcnemar_p_gated_secondary": p_mc, "discordant": [bb, cc], **b}


def h1d(v, name="knowledge_dir_steer"):
    X = v.outs[name]
    cx = v.conf.get(name)
    if cx is None or not np.isfinite(cx[X != A_]).all():
        return {"status": S.UNDEFINED, "reason": "steered confidence (steered margin) not available"}
    comp = S.aurc_comparison(X, cx, v.base, v.margin, v.tiebreak)
    sel = lambda st, cf: {f"{int(c * 100)}": S.selective_accuracy_at(st, cf, v.tiebreak, c) for c in COVERAGE_POINTS}
    comp["selective_accuracy_steered"] = sel(X, cx)
    comp["selective_accuracy_unsteered"] = sel(v.base, v.margin)
    if comp["status"] == "defined":
        top = comp["grid_top"]
        comp["selective_accuracy_at_grid_top"] = {"steered": S.selective_accuracy_at(X, cx, v.tiebreak, top),
                                                  "unsteered": S.selective_accuracy_at(v.base, v.margin, v.tiebreak, top)}
    # final F12: steered minus unsteered selective accuracy at each coverage point, paired bootstrap,
    # TOST at +/- 3 percentage points (the plan's "within 3 points at 70% coverage")
    tb = v.tiebreak
    comp["selective_accuracy_diff_pp"] = {}
    for c in COVERAGE_POINTS:
        f = lambda idx, c=c: 100.0 * (S.selective_accuracy_at(X[idx], cx[idx], tb[idx], c)
                                      - S.selective_accuracy_at(v.base[idx], v.margin[idx], tb[idx], c))
        pt = f(np.arange(v.n))
        if not math.isfinite(pt):
            comp["selective_accuracy_diff_pp"][f"{int(c * 100)}"] = {"status": S.UNDEFINED,
                                                                    "reason": "coverage not reachable"}
            continue
        b = S.paired_bootstrap(f, v.n)
        comp["selective_accuracy_diff_pp"][f"{int(c * 100)}"] = {
            "point": pt, **b, "tost_equivalent_3pp": bool(b["defined"] and b["ci90_low"] > -3 and b["ci90_high"] < 3)}
    return comp


def h1f(v, h1a_res, name="confidence_prompt_t050"):
    if name not in v.outs:
        return {"status": S.UNDEFINED, "reason": "confidence prompt missing"}
    s = S.surplus_point(v.outs[name], v.base, v.margin, v.tiebreak)
    if not math.isfinite(s):
        return {"status": S.UNDEFINED, "reason": f"{name} answers more items than unsteered", "surplus": s}
    if not h1a_res.get("defined"):
        return {"status": S.UNDEFINED, "reason": "knowledge_dir_steer surplus CI undefined", "surplus": s}
    return {"status": "defined", "surplus": s, "knowledge_surplus_ci": [h1a_res["ci_low"], h1a_res["ci_high"]],
            "inside_ci": bool(h1a_res["ci_low"] <= s <= h1a_res["ci_high"])}


# ----------------------------------------------------------------------------- H2b / H2c validity
def h2b(v, ablation="refusal_dir_ablation", positive_control_passed=None):
    if ablation not in v.outs or v.tercile is None:
        return {"status": S.UNDEFINED, "reason": "missing arm"}
    X, terc = v.outs[ablation], np.asarray(v.tercile)
    bot, top = terc == 0, terc == 2
    a0, a1 = float((v.base[bot] == A_).mean()), float((X[bot] == A_).mean())
    lost = (v.base == A_) & (X != A_)
    dmg = 100 * (float((X[top] == C_).mean()) - float((v.base[top] == C_).mean()))
    rise = 100 * (float((X == A_).mean()) - float((v.base == A_).mean()))
    d = (X[bot] == A_).astype(float) - (v.base[bot] == A_).astype(float)
    out = {"bottom_tercile_abstention_unsteered": a0, "bottom_tercile_abstention_ablated": a1,
           "relative_drop": (a0 - a1) / a0 if a0 > 0 else float("nan"),
           "lost_abstention_destination": {"C": int((lost & (X == C_)).sum()), "W": int((lost & (X == W_)).sum())},
           "top_tercile_accuracy_change_pts": dmg, "overall_abstention_change_pts": rise,
           "perm_p_bottom_tercile_abstention": S.sign_flip_test(d),
           "positive_control_passed": positive_control_passed,
           "ablation_damages_model": bool(abs(dmg) >= H2B_MAX_TOP_TERCILE_DAMAGE or rise > H2C_MAX_ABLATION_ABST_RISE)}
    out["status"] = "defined" if positive_control_passed else S.UNDEFINED
    if not positive_control_passed:
        out["reason"] = "refusal positive control failed"
    elif out["ablation_damages_model"]:
        out["note"] = ("the ablation changes top-tercile accuracy by >= 3 points or raises abstention: it damages the "
                       "model, so a missing abstention drop cannot be read as evidence against H2")
    return out


def h2c(v, steer="knowledge_dir_steer", ablation="refusal_dir_ablation",
        combo="refusal_dir_ablation_plus_knowledge_steer", positive_control_passed=None):
    if not all(k in v.outs for k in (steer, ablation, combo)):
        return {"status": S.UNDEFINED, "reason": "missing arm"}
    Xs, Xa, Xc = v.outs[steer], v.outs[ablation], v.outs[combo]

    def frac(idx):
        A = lambda X: float((X[idx] == A_).mean())
        inc = A(Xs) - A(v.base)
        rise = A(Xa) - A(v.base)
        if 100 * inc < H2C_MIN_STEER_INCREASE or 100 * rise > H2C_MAX_ABLATION_ABST_RISE:
            return float("nan")
        return 1 - (A(Xc) - A(Xa)) / inc
    all_idx = np.arange(v.n)
    A = lambda X: float((X == A_).mean())
    inc, rise = 100 * (A(Xs) - A(v.base)), 100 * (A(Xa) - A(v.base))
    out = {"steer_abstention_increase_pts": inc, "ablation_abstention_change_pts": rise,
           "combo_minus_ablation_pts": 100 * (A(Xc) - A(Xa)), "positive_control_passed": positive_control_passed}
    if not positive_control_passed:
        return {**out, "status": S.UNDEFINED, "reason": "refusal positive control failed"}
    if rise > H2C_MAX_ABLATION_ABST_RISE:
        return {**out, "status": S.UNDEFINED,
                "reason": f"ablation alone raises abstention by {rise:.1f} points (> {H2C_MAX_ABLATION_ABST_RISE:g}): "
                          "the ablated model is damaged and near the abstention ceiling, so the fraction removed is "
                          "not interpretable"}
    if inc < H2C_MIN_STEER_INCREASE:
        return {**out, "status": S.UNDEFINED, "reason": f"steering raises abstention by only {inc:.1f} points"}
    point = frac(all_idx)
    b = S.paired_bootstrap(frac, v.n)
    return {**out, "fraction_removed": point, **b, "predicted_ge_0.5": bool(b["defined"] and b["ci_low"] >= 0.5)
            if b["defined"] else None}


def _logit(p, n):
    p = min(max(p, 0.5 / n), 1 - 0.5 / n)
    return math.log(p / (1 - p))


def h2c_final(v, steer="knowledge_dir_steer", ablation="refusal_dir_ablation",
              combo="refusal_dir_ablation_plus_knowledge_steer", positive_control_passed=None):
    """Final F7. The registered fraction removed, 1 - [A(abl+steer) - A(abl)] / [A(steer) - A(unst)], is in
    absolute points, so an ablation that moves baseline abstention changes the headroom (run 2, Qwen:
    ablation alone 67% -> 38%, fraction -1.59). Validity is now two-sided (|ablation change| <= 5 points
    for the registered form) and two baseline-robust forms are pre-specified:
      conditional: on items answered by BOTH the unsteered and the ablated model, the share that steering
                   turns into abstentions with the ablation vs without; removed = 1 - ablated / intact;
      log-odds:    1 - [logit A(abl+steer) - logit A(abl)] / [logit A(steer) - logit A(unst)].
    Primary = registered when valid, else conditional."""
    reg = h2c(v, steer, ablation, combo, positive_control_passed)
    if not all(k in v.outs for k in (steer, ablation, combo)):
        return reg
    Xs, Xa, Xc = v.outs[steer], v.outs[ablation], v.outs[combo]
    rise = reg.get("ablation_abstention_change_pts", float("nan"))
    two_sided_ok = bool(math.isfinite(rise) and abs(rise) <= H2C_MAX_ABLATION_ABST_RISE)

    def cond(idx):
        S_ = (v.base[idx] != A_) & (Xa[idx] != A_)
        if S_.sum() < 10:
            return float("nan")
        intact = float((Xs[idx][S_] == A_).mean())
        abl = float((Xc[idx][S_] == A_).mean())
        return 1 - abl / intact if 100 * intact >= H2C_MIN_STEER_INCREASE else float("nan")

    def lo(idx):
        n = len(idx)
        A = lambda X: float((X[idx] == A_).mean())
        den = _logit(A(Xs), n) - _logit(A(v.base), n)
        if 100 * (A(Xs) - A(v.base)) < H2C_MIN_STEER_INCREASE or abs(den) < 1e-9:
            return float("nan")
        return 1 - (_logit(A(Xc), n) - _logit(A(Xa), n)) / den
    all_idx = np.arange(v.n)
    S_all = (v.base != A_) & (Xa != A_)
    out = {**reg, "two_sided_valid": two_sided_ok, "n_answered_by_both_baselines": int(S_all.sum()),
           "steer_induced_abstention_on_both_answered_intact": float((Xs[S_all] == A_).mean()) if S_all.any() else float("nan"),
           "steer_induced_abstention_on_both_answered_ablated": float((Xc[S_all] == A_).mean()) if S_all.any() else float("nan")}
    if not positive_control_passed:
        out.update({"primary": "untestable", "primary_value": float("nan")})
        return out
    bc, bl = S.paired_bootstrap(cond, v.n), S.paired_bootstrap(lo, v.n)
    out["fraction_removed_conditional"] = {"point": cond(all_idx), **bc}
    out["fraction_removed_logodds"] = {"point": lo(all_idx), **bl}
    if two_sided_ok and reg.get("status") == "defined":
        out.update({"primary": "registered", "primary_value": reg["fraction_removed"],
                    "primary_ci": [reg.get("ci_low"), reg.get("ci_high")]})
    else:
        out.update({"primary": "conditional", "primary_value": out["fraction_removed_conditional"]["point"],
                    "primary_ci": [bc["ci_low"], bc["ci_high"]]})
    return out


# ----------------------------------------------------------------------------- gated knockout (A7)
def relation_answer_prior(all_items_by_relation):
    """{relation: Counter(normalised first gold alias)} over every item of that relation."""
    return {r: Counter(normalize_answer(a) for a in firsts if normalize_answer(a)) for r, firsts in all_items_by_relation.items()}


def chance_second_guess(relation, aliases, first_alias, a1_text, prior):
    """Probability that a guess drawn from the relation's answer prior (leave-one-out, excluding the
    first answer's normalisation) matches a gold alias on word boundaries."""
    cnt = prior.get(relation)
    if not cnt:
        return float("nan")
    own, a1 = normalize_answer(first_alias), normalize_answer(a1_text)
    tot, hit = 0.0, 0.0
    for a, c in cnt.items():
        c = c - (1 if a == own else 0)
        if c <= 0 or a == a1:
            continue
        tot += c
        if alias_match(a, aliases):
            hit += c
    return hit / tot if tot else float("nan")


def gated_knockout(v, ko_name="first_token_ban_regenerate", aliases=None, relations=None, first_aliases=None,
                   prior=None):
    """Exploratory: on items the margin gate withdraws, answer with the forced second guess A2 instead
    of abstaining. delta vs gated refusal = mean over gated answered items of s(A2). A per-relation
    chance baseline gives the accuracy a second guess drawn from the relation's answer prior would
    reach on the same items."""
    if ko_name not in v.outs or "confidence_threshold_margin" not in v.outs:
        return {"status": S.UNDEFINED, "reason": "missing arm"}
    A2, op = v.outs[ko_name], v.outs["confidence_threshold_margin"]
    g_ans = v.gate() & (v.base != A_)
    G = op.copy()
    G[g_ans] = A2[g_ans]
    d = S.per_item_scores(G) - S.per_item_scores(op)
    b = S.paired_bootstrap(lambda idx: abstention_score(G[idx]) - abstention_score(op[idx]), v.n)
    out = {"label": "exploratory", "n_gated_answered": int(g_ans.sum()), "delta_vs_gated_refusal": float(d.mean()),
           "perm_p": S.sign_flip_test(d), **b,
           "A2_accuracy_on_gated": float((A2[g_ans] == C_).mean()) if g_ans.any() else float("nan"),
           "A2_abstention_on_gated": float((A2[g_ans] == A_).mean()) if g_ans.any() else float("nan"),
           "gated_A1_wrong": int((g_ans & (v.base == W_)).sum()), "gated_A1_correct": int((g_ans & (v.base == C_)).sum())}
    if aliases is not None and prior is not None and ko_name in v.texts and "unsteered" in v.texts:
        idx = np.flatnonzero(g_ans)
        ch = np.array([chance_second_guess(relations[i], aliases[i], first_aliases[i], v.texts["unsteered"][i], prior)
                       for i in idx])
        rel = np.array([relations[i] for i in idx])
        obs = (A2[idx] == C_).astype(float)
        per = {}
        for r in np.unique(rel):
            m = rel == r
            per[str(r)] = {"n": int(m.sum()), "A2_accuracy": float(obs[m].mean()),
                           "chance_accuracy": float(np.nanmean(ch[m])) if np.isfinite(ch[m]).any() else float("nan")}
        ok = np.isfinite(ch)
        out["chance_baseline"] = {"A2_accuracy": float(obs.mean()) if len(obs) else float("nan"),
                                  "chance_accuracy": float(ch[ok].mean()) if ok.any() else float("nan"),
                                  "A2_minus_chance": float((obs[ok] - ch[ok]).mean()) if ok.any() else float("nan"),
                                  "per_relation": per}
    return out

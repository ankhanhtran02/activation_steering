"""Go / stop rules (pilot/PREREGISTRATION.md section 3), as pure functions of the saved metrics, and the
grading-check gate. A signal is FINAL only after the hand check of 100 answers per model agrees with
the automatic grader on >= 95% of answered questions; before that it is PROVISIONAL, and if the check
fails the signals are WITHHELD until the grader is fixed and the saved answers are re-graded.

Signals: GO, STOP, BORDERLINE, INSUFFICIENT_DATA (SQ2: too few disagreement questions), NOT_RUN,
UNTESTABLE (a phase failed or a quantity is undefined).
"""
import csv
from pathlib import Path

import numpy as np

from pilot import config as PC


def _f(x):
    return float(x) if x is not None and np.isfinite(float(x)) else float("nan")


def _not_run(r, name):
    """NOT_RUN for a phase skipped by the budget ladder or a hard stop, UNTESTABLE for a failed one."""
    if r is None:
        return {"signal": "UNTESTABLE", "reason": f"{name} failed or did not start"}
    if "skipped" in r:
        return {"signal": "NOT_RUN", "reason": r["skipped"]}
    return None


def combine(per_model):
    sig = [s["signal"] for s in per_model.values() if s["signal"] not in ("NOT_RUN", "UNTESTABLE")]
    if not sig:
        return "UNTESTABLE"
    if all(s == "GO" for s in sig):
        return "GO"
    if all(s == "STOP" for s in sig):
        return "STOP"
    return "BORDERLINE"


# ----------------------------------------------------------------------------- SQ1
def sq1_model(r):
    if _not_run(r, "SQ1") or "cheap" not in r:
        return _not_run(r, "SQ1") or {"signal": "UNTESTABLE", "reason": "SQ1 incomplete"}
    c = r["cheap"]
    sp = _f(c.get("smart_probe", {}).get("c_auroc"))
    so = _f(c.get("smart_oracle", {}).get("c_auroc"))
    rmax = _f(r["random_c_auroc"]["max"])
    sp_adj = _f(c.get("smart_probe", {}).get("c_auroc_adjusted"))
    rmax_adj = _f(r.get("random_c_auroc_adjusted", {}).get("max"))
    ag = r["agreement"]
    sep_probe, sep_oracle, sep_adj = sp > rmax, so > rmax, sp_adj > rmax_adj
    agree = _f(ag["median_item_auroc"]) >= PC.SQ1_GO_ITEM_AUROC and bool(ag["real_smart_above_all_random"])
    d = {"smart_probe_c_auroc": sp, "smart_oracle_c_auroc": so, "random_max_c_auroc": rmax,
         "smart_probe_c_auroc_adjusted": sp_adj, "random_max_c_auroc_adjusted": rmax_adj,
         "separates": sep_probe, "separates_after_adjustment": sep_adj, "planted_control_separates": sep_oracle,
         "median_item_auroc": _f(ag["median_item_auroc"]), "real_smart_above_all_random": ag["real_smart_above_all_random"],
         "methods_c_auroc": {k: _f(c[k]["c_auroc"]) for k in ("caa", "iti", "push") if k in c},
         "methods_pct_vs_random": {k: _f(c[k]["pct_vs_random"]) for k in ("caa", "iti", "push") if k in c},
         "methods_c_auroc_adjusted": {k: _f(c[k].get("c_auroc_adjusted")) for k in ("caa", "iti", "push") if k in c},
         "methods_pct_vs_random_adjusted": {k: _f(c[k].get("pct_vs_random_adjusted")) for k in ("caa", "iti", "push")
                                            if k in c}}
    if sep_probe and sep_adj and agree:
        return {"signal": "GO", "reason": "cheap test separates the smart version from all random directions (also "
                                          "after adjusting for confidence shape, length and dataset) and predicts "
                                          "the real answers", **d}
    if sep_probe and not sep_adj:
        return {"signal": "BORDERLINE", "reason": "the separation disappears after adjusting for starting confidence, "
                                                  "answer length and dataset: suspected artifact", **d}
    if sep_probe:
        return {"signal": "BORDERLINE", "reason": "cheap test separates, but the real-answer check does not confirm it", **d}
    if sep_oracle:
        return {"signal": "BORDERLINE", "reason": "the planted control separates (the test has power) but the "
                                                  "detector-gated version does not: the detector adds nothing "
                                                  "within confidence strata, so the smart version is not a valid "
                                                  "positive control here", **d}
    return {"signal": "STOP", "reason": "the cheap test cannot tell the smart version (or the planted control) "
                                        "from random directions", **d}


# ----------------------------------------------------------------------------- SQ2
def sq2(r):
    if not r or "n_disagreement" not in r:
        return {"signal": "UNTESTABLE", "reason": (r or {}).get("error", "SQ2 did not run")}
    n = r["n_disagreement"]
    dlt, ci = _f(r.get("delta_pooled")), r.get("delta_pooled_ci", {})
    lo, hi = _f(ci.get("ci_low")), _f(ci.get("ci_high"))
    d = {"n_disagreement": n, "n_disagreement_main": r.get("n_disagreement_main"),
         "n_disagreement_extension": r.get("n_disagreement_extension"), "delta_auroc": dlt, "ci_low": lo, "ci_high": hi}
    if n < PC.SQ2_MIN_DISAGREE:
        return {"signal": "INSUFFICIENT_DATA", "reason": f"{n} disagreement questions < {PC.SQ2_MIN_DISAGREE}: "
                                                         "a full study needs a much larger pool", **d}
    if not np.isfinite(dlt):
        return {"signal": "UNTESTABLE", "reason": "AUROC gain undefined", **d}
    if dlt >= PC.SQ2_GO_DELTA and lo > 0:
        return {"signal": "GO", "reason": f"detector adds {dlt:.3f} AUROC (CI {lo:.3f} to {hi:.3f})", **d}
    if hi < PC.SQ2_GO_DELTA:
        return {"signal": "STOP", "reason": f"gain {dlt:.3f}, CI upper bound {hi:.3f} < {PC.SQ2_GO_DELTA}", **d}
    return {"signal": "BORDERLINE", "reason": f"gain {dlt:.3f}, CI {lo:.3f} to {hi:.3f}", **d}


# ----------------------------------------------------------------------------- SQ3
def _sq3_rule(n_wrong, n_fixable, it, pr, ct, n_test):
    R = n_fixable / n_wrong if n_wrong else float("nan")
    d = {"fixable_frac": R, "internal": it, "prob_avg": pr, "control": ct, "n_test": n_test}
    if not np.isfinite(R):
        return {"signal": "UNTESTABLE", "reason": "no wrong answers", **d}
    if R < PC.SQ3_MIN_FIXABLE:
        return {"signal": "STOP", "reason": f"little room: {R:.1%} of wrong answers fixable by rewording", **d}
    if pr["net"] - it["net"] >= PC.SQ3_CLEAR_LOSS * n_test:
        return {"signal": "STOP", "reason": "internal averaging clearly loses to probability averaging", **d}
    if ct["net"] >= it["net"]:
        return {"signal": "STOP", "reason": "the different-subject control does as well (smoothing, not the fact)", **d}
    if it["fixes"] >= pr["fixes"] and it["breaks"] <= it["fixes"]:
        return {"signal": "GO", "reason": f"{R:.1%} fixable; internal fixes {it['fixes']} >= probability averaging "
                                          f"{pr['fixes']}, breaks {it['breaks']}", **d}
    return {"signal": "BORDERLINE", "reason": "room exists but internal averaging neither clearly wins nor clearly loses", **d}


def sq3_model(r):
    if _not_run(r, "SQ3") or "test" not in r:
        return _not_run(r, "SQ3") or {"signal": "UNTESTABLE", "reason": "SQ3 incomplete"}
    t = r["test"]
    return _sq3_rule(r["room"]["n_wrong_original"], r["room"]["n_fixable_by_rewording"], t["internal_avg"], t["prob_avg"],
                     t["control_other_subject"], r["test_n"])


def sq3_pooled(per_model_results):
    rs = [r for r in per_model_results.values() if r and "test" in r]
    if not rs:
        return {"signal": "UNTESTABLE", "reason": "SQ3 did not run"}
    keys = ("fixes", "breaks", "net")
    agg = lambda m: {k: sum(r["test"][m][k] for r in rs) for k in keys}
    return _sq3_rule(sum(r["room"]["n_wrong_original"] for r in rs), sum(r["room"]["n_fixable_by_rewording"] for r in rs),
                     agg("internal_avg"), agg("prob_avg"), agg("control_other_subject"), sum(r["test_n"] for r in rs))


# ----------------------------------------------------------------------------- SQ4
def sq4(r):
    if not r or "skipped" in r:
        return {"signal": "NOT_RUN", "reason": (r or {}).get("skipped", "SQ4 did not run")}
    if "delta_surplus" not in r:
        return {"signal": "UNTESTABLE", "reason": "SQ4 failed"}
    dlt = _f(r["delta_surplus"])
    lo, hi = _f(r["delta_surplus_ci"]["ci_low"]), _f(r["delta_surplus_ci"]["ci_high"])
    d = {"delta_surplus_points": dlt, "ci_low": lo, "ci_high": hi}
    if dlt > 0 and lo > 0:
        return {"signal": "GO", "reason": "real-label adapter beats the confidence-label adapter", **d}
    if lo > -PC.SQ4_SESOI and hi < PC.SQ4_SESOI:
        return {"signal": "STOP", "reason": f"the two adapters match (CI inside +/-{PC.SQ4_SESOI} points)", **d}
    return {"signal": "BORDERLINE", "reason": "difference not resolved", **d}


# ----------------------------------------------------------------------------- SQ5
def sq5_from_gap(gap, ci_low):
    """gap = attainable pick - simple prompt (points)."""
    if not np.isfinite(gap):
        return "UNTESTABLE"
    if gap <= PC.SQ5_STOP_GAP:
        return "STOP"
    if gap >= PC.SQ5_GO_GAP and ci_low > 0:
        return "GO"
    return "BORDERLINE"


def sq5_model(r):
    if _not_run(r, "SQ5") or "gap_points" not in r:
        return _not_run(r, "SQ5") or {"signal": "UNTESTABLE", "reason": r.get("error", "SQ5 incomplete")}
    g, lo, hi = _f(r["gap_points"]), _f(r["gap_ci"]["ci_low"]), _f(r["gap_ci"]["ci_high"])
    return {"signal": sq5_from_gap(g, lo),
            "reason": f"simple prompt {g:.1f} points below the attainable pick (CI {lo:.1f} to {hi:.1f}); "
                      f"oracle upper bound {_f(r.get('oracle_gap_points_upper_bound')):.1f}",
            "gap_points": g, "ci_low": lo, "ci_high": hi,
            "steering_gain_over_simple_prompt": _f(r.get("steering", {}).get("gain_over_simple_prompt_points"))}


def sq5_pooled(per_model_results):
    rs = [r for r in per_model_results.values() if r and "gap_items" in r]
    if not rs:
        return {"signal": "UNTESTABLE", "reason": "SQ5 did not run"}
    G = np.array([r["gap_items"] for r in rs if len(r["gap_items"]) == len(rs[0]["gap_items"])])
    per_q = G.mean(0)                              # same SQ5 questions in the same order for every model
    g = float(per_q.mean())
    from pilot.metrics import bootstrap_ci
    ci = bootstrap_ci(lambda q: float(per_q[q].mean()), len(per_q))
    lo, hi = _f(ci["ci_low"]), _f(ci["ci_high"])
    return {"signal": sq5_from_gap(g, lo), "reason": f"mean gap to the attainable pick {g:.1f} points "
                                                     f"(CI {lo:.1f} to {hi:.1f}) over {len(G)} models",
            "gap_points": g, "ci_low": lo, "ci_high": hi}


# ----------------------------------------------------------------------------- grading gate
def grading_model(rows):
    """Two-stage plan (config): stage 1 decides unless it is ambiguous; then stage 2 rows are needed."""
    def disagreements(rs):
        return sum((r["human_label"] or "").strip().lower() != (r["auto_label"] or "").strip().lower() for r in rs)
    s1 = [r for r in rows if (r["stage"] or "").strip() == "1"]
    s2 = [r for r in rows if (r["stage"] or "").strip() == "2"]
    if not s1 or any(not (r["human_label"] or "").strip() for r in s1):
        return {"status": "PENDING", "stage": 1, "n_filled": sum(bool((r["human_label"] or "").strip()) for r in s1), "n": len(s1)}
    x1 = disagreements(s1)
    if x1 <= PC.GRADING_STAGE1_PASS_MAX:
        return {"status": "OK", "stage": 1, "disagreements": x1, "n": len(s1), "disagreement_rate": x1 / len(s1)}
    if x1 >= PC.GRADING_STAGE1_FAIL_MIN:
        return {"status": "FAIL", "stage": 1, "disagreements": x1, "n": len(s1), "disagreement_rate": x1 / len(s1)}
    if not s2 or any(not (r["human_label"] or "").strip() for r in s2):
        return {"status": "NEED_STAGE2", "stage": 1, "disagreements": x1, "n": len(s1),
                "n_stage2_filled": sum(bool((r["human_label"] or "").strip()) for r in s2), "n_stage2": len(s2)}
    x = x1 + disagreements(s2)
    ok = x <= round(PC.GRADING_MAX_DISAGREE * (len(s1) + len(s2)))     # 10 of 200
    return {"status": "OK" if ok else "FAIL", "stage": 2, "disagreements": x, "n": len(s1) + len(s2),
            "disagreement_rate": x / (len(s1) + len(s2))}


def grading_status(run_dir, model_names):
    out = {}
    for m in model_names:
        p = Path(run_dir) / f"grading_check_{m}.csv"
        out[m] = grading_model(list(csv.DictReader(open(p, encoding="utf-8")))) if p.exists() else {"status": "MISSING"}
    st = [v["status"] for v in out.values()]
    if "FAIL" in st:
        status = "FAIL"
    elif any(s in ("PENDING", "MISSING", "NEED_STAGE2") for s in st):
        status = "PENDING"
    else:
        status = "OK"
    return status, out


def rewording_status(run_dir):
    p = Path(run_dir) / "paraphrase_check.csv"
    if not p.exists():
        return {"status": "MISSING"}
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    filled = [r for r in rows if (r["meaning_preserved"] or "").strip()]
    if len(filled) < len(rows):
        return {"status": "PENDING", "n_filled": len(filled), "n": len(rows)}
    bad = [r for r in filled if (r["meaning_preserved"] or "").strip().lower() not in ("yes", "y", "1", "true")]
    return {"status": "OK" if not bad else "FAIL", "n": len(filled), "n_not_preserved": len(bad),
            "templates_not_preserved": sorted({(r["relation"], r["template"]) for r in bad})}


def all_signals(results, run_dir):
    models = results.get("models_run", [])
    per = results.get("per_model", {})
    sq1_per = {m: sq1_model(per.get(m, {}).get("sq1")) for m in models}
    sq3_per = {m: sq3_model(per.get(m, {}).get("sq3")) for m in models}
    sq5_per = {m: sq5_model(per.get(m, {}).get("sq5")) for m in models}
    sig = {
        "SQ1": {"signal": combine(sq1_per), "per_model": sq1_per},
        "SQ2": sq2(results.get("sq2")),
        "SQ3": {**sq3_pooled({m: per.get(m, {}).get("sq3") for m in models}), "per_model": sq3_per},
        "SQ4": sq4(results.get("sq4")),
        "SQ5": {**sq5_pooled({m: per.get(m, {}).get("sq5") for m in models}), "per_model": sq5_per},
    }
    grading, grading_detail = grading_status(run_dir, models)
    reword = rewording_status(run_dir)
    status = {"OK": "FINAL", "PENDING": "PROVISIONAL", "FAIL": "WITHHELD"}[grading]
    if reword["status"] == "FAIL":
        sig["SQ3"]["note"] = "rewording hand check found meaning changes; SQ3 needs those templates fixed"
    return {"status": status, "grading_check": {"status": grading, **grading_detail}, "rewording_check": reword,
            "signals": sig}

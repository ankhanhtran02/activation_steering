"""Final F12: the verdict table. One verdict per registered hypothesis and model, then across models
under the amended kill rule (preregistration_amendment_2026-11-07_final_run.md, section 4):

  REJECTED      both 3B families meet the hypothesis's rejection criterion
  SUPPORTED     both families meet its support criterion
  MIXED         the families disagree, or neither criterion is met in at least one family
  UNTESTABLE    the test cannot be run in at least one family (e.g. failed positive control)

A screening verdict is final and is acted on without a 7-9B confirmation (amended kill rule). Every
verdict is computed under the registered automatic labels (primary) and recomputed under the LLM-judge
labels; a verdict that changes under the judge labels is flagged label-sensitive, and the human
labels in judge_disagreements.csv decide it.

Criteria (copied from the amendment; thresholds are the plan's):
  H1  (confirmatory family H1a + H1c)
      REJECTED   H1a surplus >= +5 with 95% CI > 0 (pragmatist), or tuned gated steering beats gated
                 refusal with 95% CI > 0
      SUPPORTED  H1a 90% CI inside +/-3 ("equivalence"), or 95% CI upper < +3 ("non-superiority":
                 steering not better than its detector, the equivalence prediction missed), and no
                 dev-clean dose of the ladder is "superior", and H1c not rejected
  H1b (placebo) the dose-matched uniform push: SUPPORTED if learned - push is equivalent (TOST +/-3) or
                its 95% CI upper < +3; REJECTED if >= +5 with CI > 0; UNMATCHED if the push could not
                be dose-matched on dev
  H1d (TriviaQA, 70% coverage) SUPPORTED if steered - unsteered selective accuracy is equivalent at
                +/-3 pp; REJECTED if its 95% CI excludes the +/-3 pp band on one side
  H1e (mechanism, dev-clean grid) SUPPORTED if Spearman(m0, c_flip) >= 0.7, CV(gain) <= 0.35 and
                >= 80% monotone; REJECTED if CV > 0.7 or > 30% non-monotone
  H2  REJECTED   positive control passes AND bottom-tercile abstention drops < 15% relative AND the
                 primary H2c fraction removed <= 0.20 (steered abstention keeps >= 80%) AND benign
                 spillover < 2 points (the plan's conjunction)
      SUPPORTED  every H2 prediction holds (cos >= 0.3 with D_idk and D_ku; drop >= 50% with lost
                 abstentions mostly wrong and top-tercile accuracy within 3 points; H2c CI lower >= 0.5;
                 spillover >= +5 above the random 95th percentile)
  H3  (registered comparator popularity + surprisal, primary; A6 comparator secondary)
      REJECTED   the dev-selected internal signal keeps within-decile AUROC >= 0.65 and beats the
                 comparator by >= 0.05
      SUPPORTED  internal minus comparator <= 0.03, every internal within-decile AUROC < 0.60, and the
                 increment over popularity <= 0.03
  H4  (recovery half) REJECTED   knockout recovery < 10% in stored_outcompeted AND D_wq W->C not above
                 both nulls at the shared site nor at its own dose
      SUPPORTED  knockout >= 15% (stored_outcompeted), < 3% (missing), detection AUROC >= 0.70 and
                 >= 0.07 above A1; or D_wq above both nulls
  H4a (abstention steering only withdraws) SUPPORTED if W->C in stored_outcompeted < 5% and W->A >= 3x W->C
"""
import json
import math

from config import VERDICTS_PATH


def _f(x, default=float("nan")):
    try:
        x = float(x)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _ci_rejects_positive(h, key="delta_vs_gated_refusal"):
    return bool(h and h.get("defined") and _f(h.get("ci_low"), -1) > 0 and _f(h.get(key), -1) > 0)


# ----------------------------------------------------------------------------- H1
def h1_from(cat, h1c_rejects, sweep_cats):
    if cat in (None, "undefined"):
        return "UNTESTABLE", None
    if cat == "superior" or h1c_rejects:
        return "REJECTED", "steering beats its detector" if cat == "superior" else "gated steering beats gated refusal"
    if cat in ("equivalent", "non_superior", "inferior") and "superior" not in (sweep_cats or {}).values():
        return "SUPPORTED", "equivalence" if cat == "equivalent" else "non-superiority (equivalence prediction missed)"
    return "INCONCLUSIVE", None


def h1_model(a):
    h = a.get("hypotheses", {})
    sweep = h.get("H1a_sweep", {})
    cats = sweep.get("summary", {}).get("categories", {})
    cat = h.get("H1a", {}).get("category")
    h1c_rej = _ci_rejects_positive(h.get("H1c"))
    v, form = h1_from(cat, h1c_rej, cats)
    rep = a.get("seeds", {}).get("replication", {})
    out = {"verdict": v, "form": form, "H1a_category": cat, "H1a_surplus": h.get("H1a", {}).get("surplus"),
           "H1a_ci95": [h.get("H1a", {}).get("ci_low"), h.get("H1a", {}).get("ci_high")],
           "H1a_ci90": [h.get("H1a", {}).get("ci90_low"), h.get("H1a", {}).get("ci90_high")],
           "H1a_answered_accuracy_diff_pp": h.get("H1a", {}).get("answered_accuracy_diff_pp"),
           "H1c_tuned_delta": h.get("H1c", {}).get("delta_vs_gated_refusal"),
           "H1c_tuned_ci95": [h.get("H1c", {}).get("ci_low"), h.get("H1c", {}).get("ci_high")],
           "H1c_c_gate": h.get("H1c", {}).get("c_gate"), "H1c_rejects": h1c_rej,
           "sweep_categories": cats, "seed_replication": rep.get("H1a"), "seed_categories": rep.get("H1a_categories"),
           "labels": {}}
    for lab, res in a.get("label_sensitivity", {}).items():
        if not isinstance(res, dict) or "H1a" not in res:
            continue
        vv, ff = h1_from(res["H1a"].get("category"), _ci_rejects_positive(res.get("H1c")), res.get("H1a_sweep_categories"))
        out["labels"][lab] = {"verdict": vv, "form": ff, "H1a_category": res["H1a"].get("category"),
                              "H1a_surplus": res["H1a"].get("surplus")}
    return out


def h1b_model(a):
    hb = a.get("hypotheses", {}).get("H1b_final", {})
    push = hb.get("primary_uniform_push", {})
    reg = hb.get("registered_best_matched_random", {})
    if not push or push.get("status") == "undefined" or "diff" not in push:
        v = "UNTESTABLE"
    elif not push.get("matched_on_dev"):
        v = "UNMATCHED"
    elif push.get("learned_exceeds_by_5"):
        v = "REJECTED"
    elif push.get("tost_equivalent_3pts") or _f(push.get("ci_high"), 99) < 3:
        v = "SUPPORTED"
    else:
        v = "INCONCLUSIVE"
    arms = hb.get("arms", {})
    dec = {n: {"diff_vs_learned": arms[n].get("diff"), "ci95": [arms[n].get("ci_low"), arms[n].get("ci_high")],
               "matched_on_dev": arms[n].get("matched_on_dev"), "surplus": arms[n].get("surplus_b")}
           for n in ("knowledge_dir_steer_within_relation", "knowledge_dir_steer_between_relation") if n in arms}
    return {"verdict": v, "uniform_push": {k: push.get(k) for k in ("diff", "ci_low", "ci_high", "ci90_low", "ci90_high",
                                                                     "matched_on_dev", "surplus_a", "surplus_b")},
            "random_directions": reg if reg.get("status") == "undefined" else {k: reg.get(k) for k in ("arm", "diff", "ci_low", "ci_high")},
            "n_random_matched_on_dev": hb.get("n_random_matched_on_dev"), "n_random_arms": hb.get("n_random_arms"),
            "knowledge_decomposition": dec, "ku_decomposition_norms": hb.get("ku_decomposition"),
            "norm_matched_registered": a.get("hypotheses", {}).get("H1b", {}).get("norm_matched")}


def h1d_model(a):
    h = a.get("triviaqa", {}).get("H1d", {})
    d = h.get("selective_accuracy_diff_pp", {}).get("70", {})
    if not d or not d.get("defined"):
        v = "UNTESTABLE"
    elif d.get("tost_equivalent_3pp"):
        v = "SUPPORTED"
    elif _f(d.get("ci_low"), -99) > 3 or _f(d.get("ci_high"), 99) < -3:
        v = "REJECTED"
    else:
        v = "INCONCLUSIVE"
    aurc = h.get("boot", {})
    return {"verdict": v, "selective_accuracy_diff_pp_70": {k: d.get(k) for k in ("point", "ci_low", "ci_high", "ci90_low",
                                                                                  "ci90_high")},
            "aurc_diff": h.get("diff"), "aurc_ci": [aurc.get("ci_low"), aurc.get("ci_high")],
            "aurc_ci_includes_0": bool(aurc.get("defined") and _f(aurc.get("ci_low")) <= 0 <= _f(aurc.get("ci_high"))),
            "uniform_push_70": a.get("triviaqa", {}).get("H1d_uniform_push", {}).get("selective_accuracy_diff_pp", {}).get("70")}


def h1e_model(a):
    h = a.get("hypotheses", {}).get("H1e", {})
    k = h.get("knowledge_dir", {})
    if not k:
        return {"verdict": "UNTESTABLE"}
    rho, cv, mono = _f(k.get("spearman_m0_cflip")), _f(k.get("gain_cv")), _f(k.get("monotone_fraction"))
    if cv > 0.7 or mono < 0.7:
        v = "REJECTED"
    elif rho >= 0.7 and cv <= 0.35 and mono >= 0.8:
        v = "SUPPORTED"
    else:
        v = "MIXED"
    return {"verdict": v, "spearman": rho, "gain_cv": cv, "monotone_fraction": mono,
            "censored_fraction": k.get("censored_fraction"), "grid": h.get("grid"),
            "comparison_directions": {n: {kk: h[n].get(kk) for kk in ("spearman_m0_cflip", "gain_cv", "monotone_fraction")}
                                      for n in h if isinstance(h.get(n), dict) and "spearman_m0_cflip" in h[n] and n != "knowledge_dir"},
            "margin_ordering_spearman_vs_knowledge": h.get("margin_ordering_spearman_vs_knowledge"),
            "generation_flip_agreement": h.get("generation_flip_agreement")}


# ----------------------------------------------------------------------------- H2
def h2_model(a):
    h = a.get("hypotheses", {})
    pv = h.get("H2_primary_variant", "")
    if not pv or pv.startswith("none"):
        return {"verdict": "UNTESTABLE", "primary_variant": pv}
    tag = "" if pv == "refusal_dir_ablation" else f"[{pv}]"
    b, c, a2 = h.get(f"H2b{tag}", {}), h.get(f"H2c{tag}", {}), h.get("H2a", {})
    d = h.get("H2d", {}).get("knowledge_dir_steer", {})
    drop, frac = _f(b.get("relative_drop")), _f(c.get("primary_value"))
    spill = _f(d.get("vs_unsteered_pts"))
    ci = c.get("primary_ci") or [None, None]
    lost = b.get("lost_abstention_destination", {})
    rejects = bool(b.get("positive_control_passed") and drop < 0.15 and frac <= 0.20 and spill < 2)
    preds = {"H2a_cos_refusal_idk_ge_0.3": _f(a2.get("cos_refusal_idk_caa"), -1) >= 0.3,
             "H2a_cos_refusal_knowledge_ge_0.3": _f(a2.get("cos_refusal_knowledge"), -1) >= 0.3,
             "H2b_bottom_tercile_drop_ge_50pct": drop >= 0.5,
             "H2b_lost_abstentions_mostly_wrong": lost.get("W", 0) > lost.get("C", 0),
             "H2b_top_tercile_change_lt_3": abs(_f(b.get("top_tercile_accuracy_change_pts"), 99)) < 3,
             "H2c_ci_low_ge_0.5": _f(ci[0], -99) >= 0.5,
             "H2d_spillover_ge_5_above_random": spill >= 5 and bool(d.get("above_random_p95"))}
    v = "REJECTED" if rejects else ("SUPPORTED" if all(preds.values()) else "PARTIAL")
    return {"verdict": v, "primary_variant": pv, "predictions_met": preds, "rejection_conjunction": rejects,
            "bottom_tercile_relative_drop": drop, "lost_abstention_destination": lost,
            "H2c_primary_form": c.get("primary"), "H2c_fraction_removed": frac, "H2c_ci": ci,
            "H2c_registered": c.get("fraction_removed"), "H2c_conditional": c.get("fraction_removed_conditional", {}).get("point"),
            "H2c_logodds": c.get("fraction_removed_logodds", {}).get("point"),
            "benign_spillover_pts": spill, "benign_fisher_p": d.get("fisher_p_vs_unsteered"),
            "cos_refusal_idk_caa": a2.get("cos_refusal_idk_caa"), "cos_refusal_knowledge": a2.get("cos_refusal_knowledge"),
            "max_abs_cos_any_layer": {"idk_caa": a2.get("max_abs_cos_refusal_idk_caa_any_layer"),
                                      "knowledge": a2.get("max_abs_cos_refusal_knowledge_any_layer")}}


# ----------------------------------------------------------------------------- H3
def h3_from(f):
    if not f:
        return {"verdict": "UNTESTABLE"}
    d_reg = _f(f.get("vs_popularity_surprisal_logreg", {}).get("diff"))
    d_a6 = _f(f.get("vs_popularity+relation", {}).get("diff"))
    wd = _f(f.get("internal_within_decile_auroc"))
    rej_reg = wd >= 0.65 and d_reg >= 0.05
    rej_a6 = wd >= 0.65 and min(d_reg, d_a6) >= 0.05
    sup_reg = (d_reg <= 0.03 and _f(f.get("max_within_decile_auroc_any_internal"), 1) < 0.60
               and _f(f.get("increment_over_popularity"), 1) <= 0.03)
    sup_a6 = min(d_reg, d_a6) <= 0.03 if math.isfinite(d_a6) else False
    return {"verdict": "REJECTED" if rej_reg else ("SUPPORTED" if sup_reg else "PARTIAL"),
            "verdict_A6_comparator": "REJECTED" if rej_a6 else ("SUPPORTED" if sup_a6 else "PARTIAL"),
            "internal_signal": f.get("internal_signal_dev_selected"), "internal_within_decile_auroc": wd,
            "minus_popularity_surprisal": d_reg, "minus_popularity_relation": d_a6,
            "increment_over_popularity": f.get("increment_over_popularity"),
            "increment_over_popularity_relation": f.get("increment_over_popularity_relation")}


def h3_model(a):
    h = a.get("hypotheses", {})
    out = h3_from(h.get("H3_final"))
    if h.get("H3_final_judge_labels"):
        j = h3_from(h["H3_final_judge_labels"])
        out["labels"] = {"judge": {"verdict": j["verdict"], "verdict_A6_comparator": j["verdict_A6_comparator"]}}
    return out


# ----------------------------------------------------------------------------- H4
def h4_model(a):
    h = a.get("hypotheses", {})
    c, b, bo, h4a = h.get("H4c", {}), h.get("H4b", {}), h.get("H4b_own_dose", {}), h.get("H4a", {})
    rec_so, rec_mi = _f(c.get("recovery_stored_outcompeted")), _f(c.get("recovery_missing"))
    above = bool(b.get("above_both_nulls")) or bool(bo.get("above_both_nulls"))
    rejects = rec_so < 0.10 and not above
    knock = (rec_so >= 0.15 and rec_mi < 0.03 and _f(c.get("knockout_detection_auroc_A2")) >= 0.70
             and _f(c.get("A2_minus_A1_auroc")) >= 0.07)
    v = "REJECTED" if rejects else ("SUPPORTED" if (knock or above) else "PARTIAL")
    so = h4a.get("stored_outcompeted", {})
    h4a_ok = bool(_f(so.get("W_to_C"), 1) < 0.05 and _f(so.get("W_to_A")) >= 3 * _f(so.get("W_to_C"), 1))
    return {"verdict": v, "knockout_recovery_stored_outcompeted": rec_so,
            "knockout_recovery_wilson95": c.get("recovery_stored_outcompeted_wilson95"),
            "knockout_chance_stored_outcompeted": c.get("chance_stored_outcompeted"), "knockout_recovery_missing": rec_mi,
            "knockout_detection_auroc": c.get("knockout_detection_auroc_A2"), "knockout_detection_minus_A1": c.get("A2_minus_A1_auroc"),
            "dwq_shared_site": {k: b.get(k) for k in ("random_p95", "label_permuted_p95", "above_both_nulls")} | {
                "W_to_C": b.get("overall", {}).get("W_to_C")},
            "dwq_own_dose": {k: bo.get(k) for k in ("random_p95", "label_permuted_p95", "above_both_nulls")} | {
                "W_to_C": bo.get("overall", {}).get("W_to_C"), "c": (bo.get("dose") or {}).get("c")},
            "H4a_verdict": "SUPPORTED" if h4a_ok else "NOT_SUPPORTED",
            "H4a_so_W_to_C": so.get("W_to_C"), "H4a_so_W_to_C_wilson95": h4a.get("so_W_to_C_wilson95"),
            "H4a_so_W_to_A": so.get("W_to_A")}


# ----------------------------------------------------------------------------- combine
def combine(vs):
    vs = [v for v in vs if v is not None]
    if not vs:
        return "UNTESTABLE"
    if any(v == "UNTESTABLE" for v in vs):
        return "UNTESTABLE"
    if all(v == vs[0] for v in vs):
        return vs[0]
    return "MIXED"


DECISION_RULES = {
    ("H1", "SUPPORTED"): "H1 accepted at screening: steering does not beat its own detector. Under the amended kill rule this "
                         "is acted on now; later steering work must carry random_dir_placebo and confidence_threshold_margin, "
                         "and a 7-9B replication is optional.",
    ("H1", "REJECTED"): "H1 rejected in both models: gated steering (margin_gated_steer) becomes the main study.",
    ("H1", "MIXED"): "H1 not established (the two families disagree).",
    ("H3", "SUPPORTED"): "Frequency suffices: report the gate as API-compatible routing; restate mechanistic gate claims as "
                         "frequency tracking.",
    ("H4", "SUPPORTED"): "Advance within_question_dir_steer (if D_wq passed its nulls) / the knockout with their nulls.",
    ("H4", "REJECTED"): "Report the observed W->C and knockout rates as upper bounds; the recovery account is dropped.",
}


def compute_verdicts(runners, analyses, paired=None, judge_info=None):
    per = {r.name: {"H1": h1_model(a), "H1b": h1b_model(a), "H1d": h1d_model(a), "H1e": h1e_model(a),
                    "H2": h2_model(a), "H3": h3_model(a), "H4": h4_model(a)} for r, a in zip(runners, analyses)}
    models = list(per)
    overall = {}
    for hyp in ("H1", "H1b", "H1d", "H1e", "H2", "H3", "H4"):
        v = combine([per[m][hyp]["verdict"] for m in models])
        e = {"verdict": v, "per_model": {m: per[m][hyp]["verdict"] for m in models}}
        labs = set().union(*[set(per[m][hyp].get("labels", {})) for m in models])
        e["under_other_labels"] = {lab: combine([per[m][hyp].get("labels", {}).get(lab, {}).get("verdict") for m in models])
                                   for lab in sorted(labs)}
        if "judge" in e["under_other_labels"]:
            e["label_robust"] = e["under_other_labels"]["judge"] == v
        elif hyp in ("H1", "H3"):
            e["label_robust"] = None          # judge labels unavailable
        rule = DECISION_RULES.get((hyp, v))
        if rule:
            e["decision_rule"] = rule
        overall[hyp] = e
    overall["H1"]["forms"] = {m: per[m]["H1"]["form"] for m in models}
    overall["H1"]["seed_replication"] = {m: per[m]["H1"]["seed_replication"] for m in models}
    overall["H3"]["verdict_A6_comparator"] = combine([per[m]["H3"].get("verdict_A6_comparator") for m in models])
    overall["H4a"] = {"verdict": combine([per[m]["H4"]["H4a_verdict"] for m in models]),
                      "per_model": {m: per[m]["H4"]["H4a_verdict"] for m in models}}
    works = [h for h in ("H1", "H2", "H3", "H4") if overall[h]["verdict"] == "SUPPORTED"]
    fails = [h for h in ("H1", "H2", "H3", "H4") if overall[h]["verdict"] == "REJECTED"]
    other = [h for h in ("H1", "H2", "H3", "H4") if h not in works and h not in fails]
    return {"kill_rule": "amended 2026-11-07: both 3B families decide; no 7-9B confirmation gate",
            "judge": judge_info or {"status": "not run"}, "overall": overall, "per_model": per,
            "summary": {"supported": works, "rejected": fails, "mixed_or_untestable": other}}


def print_verdicts(vt):
    for hyp, e in vt.get("overall", {}).items():
        extra = ""
        if "label_robust" in e:
            extra += f" judge_labels={e['under_other_labels'].get('judge', 'unavailable')} label_robust={e['label_robust']}"
        if hyp == "H1":
            extra += f" forms={json.dumps(e.get('forms'))} seed_replication={json.dumps(e.get('seed_replication'))}"
        if hyp == "H3":
            extra += f" A6_comparator={e.get('verdict_A6_comparator')}"
        print(f"VERDICT: {hyp} {e['verdict']} per_model={json.dumps(e.get('per_model'))}{extra}", flush=True)
        if e.get("decision_rule"):
            print(f"DECISION_RULE: {hyp} {e['decision_rule']}", flush=True)
    s = vt.get("summary", {})
    print(f"VERDICT_SUMMARY: supported={s.get('supported')} rejected={s.get('rejected')} "
          f"mixed_or_untestable={s.get('mixed_or_untestable')}", flush=True)


def write_verdicts(vt):
    from report import jsonable
    VERDICTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VERDICTS_PATH, "w", encoding="utf-8") as f:
        json.dump(jsonable(vt), f, indent=1, ensure_ascii=False)

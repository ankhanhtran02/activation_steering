"""Metrics per condition and stratum, hypothesis analyses (H1-H4), knowledge-signal AUROCs,
paired statistics, harness-format printing, results.json, samples_<condition>.jsonl, audit.csv.

Console convention: bare `condition=<name>` lines pool both models (same PopQA test questions, so the
pooled score is the mean of the two per-model scores); per-model values use regime=<model>; the
paired bootstrap resamples question ids jointly for both models.
"""
import csv
import json
import math

import numpy as np
import torch

from baselines import PopularityGate, RandomDirectionPlacebo, cross_fitted_proba, fit_l2_logreg
from config import (AUDIT_PATH, AUDIT_PER_MODEL, C_GRID, COVERAGE_POINTS, LAMBDAS, N_COSINE_NULL,
                    N_SAMPLES_PER_CONDITION, N_SPLIT_RESAMPLES, OUT_DIR, RESULTS_PATH,
                    COSINE_NULL_SEED, STEERING_CONDITIONS, TOST_MARGIN)
from scoring import (A_, C_, W_, STATE_NAMES, abstention_score, aurc, auroc, bootstrap,
                     cochran_armitage, empirical_percentile, holm, match_coverage, matched_score,
                     mcnemar_exact, score_vec, selective_accuracy_at, summary_metrics, tost_equivalent,
                     transition_metrics)
from steering import cosine, random_cosine_null, unit, within_question_direction

KS_REGIME = {"known": "known", "stored_outcompeted": "outcompeted", "missing": "missing"}
SAME_SIGNAL = {"knowledge_dir_steer": "ku_projection", "within_question_dir_steer": "wq_answer_projection",
               "refusal_dir_steer": "refusal_projection"}


def fmt(v):
    return "nan" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.4f}"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, torch.Tensor):
        return jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else (None if math.isnan(f) else str(f))
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ----------------------------------------------------------------------------- strata
def test_strata(r):
    items = r.items["test"]
    terc = np.array([it.tercile for it in items])
    dec = np.array([it.decile for it in items])
    ks = r.kstatus["test"]
    lab = r.labels["test"]
    m = {f"pop_q{t + 1}": terc == t for t in range(3)}
    m.update({KS_REGIME[k]: ks == k for k in KS_REGIME})
    m.update({"robust_known": lab["robust_known"], "consistent_wrong": lab["consistent_wrong"],
              "unstable": lab["unstable"]})
    grid = {f"pop_q{t + 1}.{KS_REGIME[k]}": (terc == t) & (ks == k) for t in range(3) for k in KS_REGIME}
    deciles = {f"pop_d{d + 1}": dec == d for d in range(10)}
    return m, grid, deciles


def stratum_scores(states, masks, lam=1.0):
    return {k: {"n": int(v.sum()), "abstention_score": abstention_score(states[v], lam) if v.any() else float("nan"),
                "abstention_rate": float((states[v] == A_).mean()) if v.any() else float("nan"),
                "wrong_rate": float((states[v] == W_).mean()) if v.any() else float("nan"),
                "descriptive_only": bool(v.sum() < 30)}
            for k, v in masks.items()}


# ----------------------------------------------------------------------------- per-condition metrics
def condition_metrics(r, out, base, split="test"):
    tb = r.tiebreak[split]
    m = summary_metrics(out.states, out.degenerate, out.truncated, out.n_tokens)
    rk = r.labels[split]["robust_known"] if split in r.labels else None
    m.update(transition_metrics(base.states, out.states, rk))
    m["aurc"] = aurc(out.states, out.confidence, tb)
    m["confidence_kind"] = out.confidence_kind
    m["selective_accuracy_at"] = {f"{int(c * 100)}": selective_accuracy_at(out.states, out.confidence, tb, c)
                                  for c in COVERAGE_POINTS}
    m["S1_strict_exact"] = abstention_score(out.strict_states())
    m["answer_length_shift"] = m["mean_answer_tokens"] - float(np.mean(base.n_tokens[base.states != A_]))
    m["abstention_subtypes"] = {t: int(sum(1 for a in out.abst_type if a == t))
                                for t in ("idk_phrasing", "refusal_phrasing", "gate")}
    k = int((out.states != A_).sum())
    m["answered"] = k
    m["unmatched_coverage"] = bool(k > int((base.states != A_).sum()))
    margin = r.margin[split]
    m["surplus_vs_threshold"] = {f"lam{lam:g}": abstention_score(out.states, lam) - matched_score(base.states, margin, tb, k, lam)
                                 for lam in LAMBDAS}
    if out.name in SAME_SIGNAL:
        sig = r.signal(SAME_SIGNAL[out.name], split)
        m["surplus_vs_same_signal_threshold"] = abstention_score(out.states) - matched_score(base.states, sig, tb, k)
    if split == "test":
        pop = r.signal("popularity", split)
        m["surplus_vs_popularity_gate"] = abstention_score(out.states) - matched_score(base.states, pop, tb, k)
    m["flags"] = {"collateral_gt_3pts": bool(m.get("collateral", 0) * 100 > 3),
                  "length_shift_gt_2": bool(abs(m["answer_length_shift"]) > 2),
                  "degenerate_gt_5pct": bool(m["degenerate_rate"] > 0.05)}
    return m


def all_test_outputs(r):
    return {**r.test_out, **r.ablation_out}


# ----------------------------------------------------------------------------- knowledge signals (H3)
def knowledge_signals(r):
    L = r.frozen["L_star"]
    y = {s: (r.uout[s].states == C_).astype(int) for s in ("train", "dev", "test")}
    feats = {s: r.cache[s]["resid_last"] for s in ("train", "dev", "test")}
    ent = r.labels["train"]["semantic_entropy"]
    y_sep = (ent > np.median(ent)).astype(int)
    if y_sep.min() == y_sep.max():
        y_sep = (ent >= np.median(ent)).astype(int)

    def choose_layer(label_tr, sign):
        best, best_auc = None, -1
        for Lp in r.cache_layers:
            mdl = fit_l2_logreg(feats["train"][:, Lp].numpy(), label_tr, c_grid=[1.0])
            a = auroc(y["dev"], sign * mdl.predict_proba(feats["dev"][:, Lp].numpy())[:, 1])
            if a > best_auc:
                best, best_auc = Lp, a
        return best, best_auc

    probes = {}
    for pname, label_tr, sign in (("correctness_probe", y["train"], 1.0), ("sep_probe", y_sep, -1.0)):
        Lp, dev_auc = choose_layer(label_tr, sign)
        mdl = fit_l2_logreg(feats["train"][:, Lp].numpy(), label_tr)
        probes[pname] = {"layer": Lp, "dev_auroc": dev_auc, "C": mdl.C_,
                         "test": sign * mdl.predict_proba(feats["test"][:, Lp].numpy())[:, 1],
                         "train_oof": sign * cross_fitted_proba(feats["train"][:, Lp].numpy(), label_tr, mdl.C_)}
    sig_test = {"margin": r.margin["test"], "ku_projection": r.signal("ku_projection", "test"),
                "sep_probe": probes["sep_probe"]["test"], "correctness_probe": probes["correctness_probe"]["test"]}
    sig_train = {"margin": r.margin["train"], "ku_projection": r.signal("ku_projection", "train"),
                 "sep_probe": probes["sep_probe"]["train_oof"], "correctness_probe": probes["correctness_probe"]["train_oof"]}
    pop = {s: r.signal("popularity", s) for s in ("train", "test")}
    sur = {s: r.cache[s]["surprisal"] for s in ("train", "test")}
    psl = PopularityGate.fit_popularity_surprisal_logreg(pop["train"], sur["train"], y["train"])
    sig_test["popularity"] = pop["test"]
    sig_test["popularity_surprisal_logreg"] = psl.predict_proba(
        PopularityGate.features(pop["test"], sur["test"], psl.surprisal_fill_))[:, 1]
    for name in ("margin", "ku_projection", "sep_probe", "correctness_probe"):
        mdl = fit_l2_logreg(np.stack([pop["train"], sig_train[name]], 1), y["train"])
        sig_test[f"popularity+{name}"] = mdl.predict_proba(np.stack([pop["test"], sig_test[name]], 1))[:, 1]

    yt = y["test"].astype(bool)
    dec = np.array([it.decile for it in r.items["test"]])
    terc = np.array([it.tercile for it in r.items["test"]])
    pce = (terc == 2) & r.labels["test"]["consistent_wrong"]
    pop_correct = (terc == 2) & yt
    out = {}
    n = len(yt)
    for name, s in sig_test.items():
        boot = bootstrap(lambda idx, s=s: auroc(yt[idx], s[idx]), n)
        wd = [auroc(yt[dec == d], s[dec == d]) for d in range(10)
              if (yt[dec == d]).sum() >= 10 and (~yt[dec == d]).sum() >= 10]
        sel = pce | pop_correct
        h3d = bootstrap(lambda idx, s=s: auroc(~yt[sel][idx], -s[sel][idx]), int(sel.sum())) if sel.sum() > 2 else {}
        out[name] = {"auroc_known": auroc(yt, s), "ci": [boot["ci_low"], boot["ci_high"]],
                     "within_decile_auroc": float(np.mean(wd)) if wd else float("nan"), "n_deciles_used": len(wd),
                     "popular_consistent_error_auroc": auroc(~yt[sel], -s[sel]) if sel.sum() else float("nan"),
                     "popular_consistent_error_ci": [h3d.get("ci_low"), h3d.get("ci_high")]}
    for name in ("margin", "ku_projection", "sep_probe", "correctness_probe"):
        out[f"popularity+{name}"]["incremental_auroc"] = out[f"popularity+{name}"]["auroc_known"] - out["popularity"]["auroc_known"]
        a, b = sig_test[f"popularity+{name}"], sig_test["popularity"]
        inc = bootstrap(lambda idx, a=a, b=b: auroc(yt[idx], a[idx]) - auroc(yt[idx], b[idx]), n)
        out[f"popularity+{name}"]["incremental_ci"] = [inc["ci_low"], inc["ci_high"]]
    internal = ["margin", "ku_projection", "sep_probe", "correctness_probe"]
    best_internal = max(internal, key=lambda k: out[k]["auroc_known"])
    out["_H3a"] = {"best_internal": best_internal,
                   "best_minus_popularity_surprisal": out[best_internal]["auroc_known"] - out["popularity_surprisal_logreg"]["auroc_known"]}
    out["_probes"] = {k: {kk: vv for kk, vv in v.items() if kk in ("layer", "dev_auroc", "C")} for k, v in probes.items()}
    out["_n_popular_consistent_error"] = int(pce.sum())
    r.extras["_probe_objects"] = probes
    return out


# ----------------------------------------------------------------------------- split randomness
def split_randomness(r):
    L = r.frozen["L_star"]
    rng = np.random.default_rng(11)
    known, unknown = r.extras["known_unknown_masks"]
    R = r.cache["train"]["resid_last"][:, L]
    full_k = r.dirs["knowledge"][L]
    full_w = r.dirs["wq"][L]
    H = r.gray["H"][:, r.cand.index(L)]
    qidx, lab = r.gray["qidx"], r.gray["labels"]
    uq = np.unique(qidx)
    probe = r.extras["_probe_objects"]["correctness_probe"]
    Lp = probe["layer"]
    ytr = (r.uout["train"].states == C_).astype(int)
    yte = (r.uout["test"].states == C_)
    cos_k, cos_w, aucs = [], [], []
    for _ in range(N_SPLIT_RESAMPLES):
        b = rng.integers(0, len(R), len(R))
        kb, ub = known[b], unknown[b]
        if kb.any() and ub.any():
            cos_k.append(cosine(R[b][ub].mean(0) - R[b][kb].mean(0), full_k))
        qb = rng.choice(uq, len(uq), replace=True)
        rows = np.concatenate([np.flatnonzero(qidx == q) for q in qb])
        newq = np.concatenate([np.full((qidx == q).sum(), j) for j, q in enumerate(qb)])
        cos_w.append(cosine(within_question_direction(H[torch.as_tensor(rows)], newq, lab[rows]), full_w))
        if len(np.unique(ytr[b])) == 2:
            mdl = fit_l2_logreg(r.cache["train"]["resid_last"][b, Lp].numpy(), ytr[b], c_grid=[probe["C"]])
            aucs.append(auroc(yte, mdl.predict_proba(r.cache["test"]["resid_last"][:, Lp].numpy())[:, 1]))
    return {"knowledge_dir_cosine": cos_k, "wq_dir_cosine": cos_w, "correctness_probe_test_auroc": aucs,
            "correctness_probe_auroc_spread": float(np.ptp(aucs)) if aucs else float("nan")}


# ----------------------------------------------------------------------------- hypotheses per model
def random_draw_names(outs, prefix="random_dir_placebo_s"):
    return sorted([k for k in outs if k.startswith(prefix)], key=lambda k: int(k.rsplit("s", 1)[1]))


def analyze_model(r):
    base = r.uout["test"]
    tb = r.tiebreak["test"]
    margin = r.margin["test"]
    outs = all_test_outputs(r)
    strata, grid, deciles = test_strata(r)
    res = {"conditions": {}, "strata": {}, "grid": {}, "deciles": {}}
    for name, o in outs.items():
        res["conditions"][name] = condition_metrics(r, o, base)
        res["strata"][name] = stratum_scores(o.states, strata)
        res["grid"][name] = stratum_scores(o.states, grid)
        res["deciles"][name] = stratum_scores(o.states, deciles)
    hyp = {}
    draws = random_draw_names(outs)
    S = lambda name: res["conditions"][name]["abstention_score"] if name in res["conditions"] else float("nan")
    sur = lambda name: res["conditions"][name]["surplus_vs_threshold"]["lam1"] if name in res["conditions"] else float("nan")

    def surplus_boot(name):
        X = outs[name].states
        return bootstrap(lambda idx: abstention_score(X[idx]) - matched_score(
            base.states[idx], margin[idx], tb[idx], int((X[idx] != A_).sum())), len(X))

    if draws:
        dist = {"S1": [S(d) for d in draws], "surplus": [sur(d) for d in draws],
                "selectivity_lift": [res["conditions"][d]["selectivity_lift"] for d in draws],
                "recovered_rate": [res["conditions"][d]["recovered_rate"] for d in draws]}
        res["random_distribution"] = {}
        for metric, vals in dist.items():
            res["random_distribution"][metric] = RandomDirectionPlacebo.summarize_distribution(vals)
            for name in STEERING_CONDITIONS:
                if name in res["conditions"]:
                    learned = {"S1": S(name), "surplus": sur(name),
                               "selectivity_lift": res["conditions"][name]["selectivity_lift"],
                               "recovered_rate": res["conditions"][name]["recovered_rate"]}[metric]
                    res["random_distribution"][f"{metric}|{name}"] = RandomDirectionPlacebo.summarize_distribution(vals, learned)
                    res["random_distribution"][f"{metric}|{name}"]["percentile"] = empirical_percentile(learned, vals)
        D = np.stack([outs[d].states for d in draws])
        for name in STEERING_CONDITIONS:
            if name in outs:
                X = outs[name].states
                res["random_distribution"][f"learned_minus_draw_mean|{name}"] = bootstrap(
                    lambda idx, X=X: abstention_score(X[idx]) - np.mean([abstention_score(dd[idx]) for dd in D]), len(X))

    # H1
    if "knowledge_dir_steer" in outs:
        b = surplus_boot("knowledge_dir_steer")
        hyp["H1a"] = {"surplus": sur("knowledge_dir_steer"), **b,
                      "tost_equivalent_3pts": tost_equivalent(b["ci90_low"], b["ci90_high"], TOST_MARGIN),
                      "pragmatist_surplus_ge_5": bool(sur("knowledge_dir_steer") >= 5 and b["ci_low"] > 0),
                      "underpowered_se_gt_1.6": bool(b["sd"] > 1.6)}
        cm = random_draw_names(outs, "random_dir_placebo_coverage_matched_s")
        h1b = {}
        if draws:
            h1b["learned_surplus_above_p95_of_draws"] = bool(sur("knowledge_dir_steer") > np.percentile([sur(d) for d in draws], 95))
            h1b["lift_learned"] = res["conditions"]["knowledge_dir_steer"]["selectivity_lift"]
            h1b["lift_random_median"] = float(np.median([res["conditions"][d]["selectivity_lift"] for d in draws]))
        if cm:
            best = max(cm, key=sur)
            Xk, Xb = outs["knowledge_dir_steer"].states, outs[best].states

            def diff(idx):
                s1 = abstention_score(Xk[idx]) - matched_score(base.states[idx], margin[idx], tb[idx], int((Xk[idx] != A_).sum()))
                s2 = abstention_score(Xb[idx]) - matched_score(base.states[idx], margin[idx], tb[idx], int((Xb[idx] != A_).sum()))
                return s1 - s2
            h1b.update({"best_coverage_matched_draw": best, "learned_minus_best_cm_surplus": sur("knowledge_dir_steer") - sur(best),
                        **{f"cm_{k}": v for k, v in bootstrap(diff, len(Xk)).items()},
                        "cm_draw_surplus": {d: sur(d) for d in cm},
                        "cm_draw_lift": {d: res["conditions"][d]["selectivity_lift"] for d in cm},
                        "cm_matched_on_dev": {d: outs[d].extra.get("matched_on_dev") for d in cm}})
        hyp["H1b"] = h1b
        ku = outs["knowledge_dir_steer"]
        h1d = {"steered": res["conditions"]["knowledge_dir_steer"]["selective_accuracy_at"],
               "unsteered": res["conditions"]["unsteered"]["selective_accuracy_at"] if "unsteered" in res["conditions"] else None}
        h1d["aurc_diff"] = bootstrap(lambda idx: aurc(ku.states[idx], ku.confidence[idx], tb[idx])
                                     - aurc(base.states[idx], margin[idx], tb[idx]), len(base))
        h1d["aurc_diff_point"] = res["conditions"]["knowledge_dir_steer"]["aurc"] - aurc(base.states, margin, tb)
        hyp["H1d"] = h1d
        if "confidence_prompt_t050" in outs:
            hyp["H1f"] = {"confidence_prompt_t050_surplus": sur("confidence_prompt_t050"),
                          "knowledge_surplus_ci": [b["ci_low"], b["ci_high"]],
                          "inside_ci": bool(b["ci_low"] <= sur("confidence_prompt_t050") <= b["ci_high"])}
    op = outs.get("confidence_threshold_margin")
    for inst, hid in (("margin_gated_steer_knowledge", "H1c"), ("margin_gated_steer_wq", "H4d_gated")):
        if inst in outs and op is not None:
            g = outs[inst].extra["gate"]
            X = outs[inst].states
            from methods import MarginGatedSteer
            delta = MarginGatedSteer.compare_with_gated_refusal(X, op.states, g)
            p, bb, cc = mcnemar_exact((X == C_)[g], (op.states == C_)[g])
            hyp[hid] = {"delta_vs_gated_refusal": delta, "n_gated": int(g.sum()),
                        **bootstrap(lambda idx: abstention_score(X[idx]) - abstention_score(op.states[idx]), len(X)),
                        "mcnemar_p_gated": p, "discordant": [bb, cc]}
    dose = r.extras.get("teacher_forced_dose_response")
    if dose:
        hyp["H1e"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("margins", "c_flip", "m0")}
                      for k, v in dose.items() if isinstance(v, dict) and "spearman_m0_cflip" in v}
        hyp["H1e"]["generation_flip_agreement"] = dose.get("generation_flip_agreement")
        if "random_s0" in dose:
            hyp["H1e"]["random_margin_ordering_spearman"] = {
                k: (float(np.corrcoef(dose["knowledge_dir"]["c_flip"], dose[k]["c_flip"])[0, 1])
                    if np.std(dose[k]["c_flip"]) > 0 else float("nan")) for k in ("random_s0", "random_s1")}

    # H2
    L = r.frozen["L_star"]
    h2a = {}
    null = random_cosine_null(r.dirs["refusal"][L], N_COSINE_NULL, COSINE_NULL_SEED)
    for k in ("knowledge", "wq", "idk_caa"):
        if L in r.dirs.get(k, {}):
            h2a[f"cos_refusal_{k}"] = cosine(r.dirs["refusal"][L], r.dirs[k][L])
    h2a["null_abs_mean"] = float(np.abs(null).mean())
    h2a["null_abs_p95"] = float(np.percentile(np.abs(null), 95))
    hyp["H2a"] = h2a
    terc = np.array([it.tercile for it in r.items["test"]])
    if "refusal_dir_ablation" in outs:
        X = outs["refusal_dir_ablation"].states
        bot, top = terc == 0, terc == 2
        a0, a1 = float((base.states[bot] == A_).mean()), float((X[bot] == A_).mean())
        lost = (base.states == A_) & (X != A_)
        hyp["H2b"] = {"bottom_tercile_abstention_unsteered": a0, "bottom_tercile_abstention_ablated": a1,
                      "relative_drop": (a0 - a1) / a0 if a0 > 0 else float("nan"),
                      "lost_abstention_destination": {"C": int((lost & (X == C_)).sum()), "W": int((lost & (X == W_)).sum())},
                      "A_to_A": int(((base.states == A_) & (X == A_)).sum()),
                      "top_tercile_accuracy_change_pts": 100 * (float((X[top] == C_).mean()) - float((base.states[top] == C_).mean())),
                      "positive_control_passed": r.side["refusal_control"]["passed"]}
    if all(k in outs for k in ("refusal_dir_ablation", "refusal_dir_ablation_plus_knowledge_steer", "knowledge_dir_steer")):
        A = lambda n: float((outs[n].states == A_).mean())
        A_u = float((base.states == A_).mean())
        denom = A("knowledge_dir_steer") - A_u
        hyp["H2c"] = {"fraction_removed": 1 - (A("refusal_dir_ablation_plus_knowledge_steer") - A("refusal_dir_ablation")) / denom
                      if denom != 0 else float("nan"), "steer_abstention_increase": denom}
    br = r.side.get("benign_refusal_rate")
    if br:
        rnd = [v for k, v in br.items() if k.startswith("random_dir_placebo_s")]
        hyp["H2d"] = {name: {"benign_refusal": br[name], "vs_unsteered_pts": 100 * (br[name] - br["unsteered"]),
                             "above_random_p95": bool(rnd and br[name] > np.percentile(rnd, 95))}
                      for name in STEERING_CONDITIONS if name in br}
    if "refusal_dir_steer" in outs and "knowledge_dir_steer" in outs:
        Xr, Xk = outs["refusal_dir_steer"].states, outs["knowledge_dir_steer"].states

        def d_sur(idx):
            f = lambda X: abstention_score(X[idx]) - matched_score(base.states[idx], margin[idx], tb[idx], int((X[idx] != A_).sum()))
            return f(Xr) - f(Xk)
        b = bootstrap(d_sur, len(Xr))
        hyp["H2e"] = {"refusal_minus_knowledge_surplus": sur("refusal_dir_steer") - sur("knowledge_dir_steer"), **b,
                      "tost_equivalent_3pts": tost_equivalent(b["ci90_low"], b["ci90_high"], TOST_MARGIN),
                      "lift_refusal": res["conditions"]["refusal_dir_steer"]["selectivity_lift"],
                      "lift_knowledge": res["conditions"]["knowledge_dir_steer"]["selectivity_lift"],
                      "refusal_abstention_subtypes": res["conditions"]["refusal_dir_steer"]["abstention_subtypes"]}

    # H3
    try:
        sig = knowledge_signals(r)
        res["knowledge_signals"] = sig
        hyp["H3a"] = sig["_H3a"]
        hyp["H3b"] = {k: sig[k]["within_decile_auroc"] for k in ("margin", "ku_projection", "sep_probe", "correctness_probe")}
        hyp["H3c"] = {k: sig[f"popularity+{k}"]["incremental_auroc"] for k in ("margin", "ku_projection", "sep_probe", "correctness_probe")}
        hyp["H3d"] = {k: {"auroc": sig[k]["popular_consistent_error_auroc"], "ci": sig[k]["popular_consistent_error_ci"]}
                      for k in ("margin", "ku_projection", "sep_probe", "correctness_probe")}
        res["split_randomness"] = split_randomness(r)
    except Exception as e:  # analysis failure must not hide the condition results
        print(f"ANALYSIS_FAILED: knowledge_signals model={r.name} {type(e).__name__}: {e}", flush=True)
    if op is not None and "popularity_gate" in outs:
        k = int((op.states != A_).sum())
        pop = r.signal("popularity", "test")
        pm = match_coverage(base.states, pop, tb, k)
        if pm is not None:
            hyp["H3e"] = {"threshold_minus_popularity_gate": abstention_score(op.states) - abstention_score(pm),
                          **bootstrap(lambda idx: abstention_score(op.states[idx]) - matched_score(
                              base.states[idx], pop[idx], tb[idx], int((op.states[idx] != A_).sum())), len(base))}

    # H4
    ks = r.kstatus["test"]

    def wc_by_status(X):
        d = {}
        for s in ("known", "stored_outcompeted", "missing"):
            w = (base.states == W_) & (ks == s)
            d[s] = {"n_wrong": int(w.sum()), "W_to_C": float((X[w] == C_).mean()) if w.any() else float("nan"),
                    "W_to_A": float((X[w] == A_).mean()) if w.any() else float("nan")}
            na = (base.states == A_) & (ks == s)
            d[s]["A_to_C"] = float((X[na] == C_).mean()) if na.any() else float("nan")
        w = base.states == W_
        d["overall"] = {"W_to_C": float((X[w] == C_).mean()) if w.any() else float("nan")}
        return d
    if "knowledge_dir_steer" in outs:
        d = wc_by_status(outs["knowledge_dir_steer"].states)
        so = d["stored_outcompeted"]
        hyp["H4a"] = {**d, "so_W_to_C_lt_5pct": bool(so["W_to_C"] < 0.05),
                      "so_W_to_A_ge_3x_W_to_C": bool(so["W_to_A"] >= 3 * so["W_to_C"])}
    if "within_question_dir_steer" in outs:
        d = wc_by_status(outs["within_question_dir_steer"].states)
        perm = random_draw_names(outs, "label_permuted_wq_dir_steer_s")
        wc_rand = [wc_by_status(outs[n].states)["overall"]["W_to_C"] for n in draws]
        wc_perm = [wc_by_status(outs[n].states)["overall"]["W_to_C"] for n in perm]
        hyp["H4b"] = {**d, "random_p95": float(np.percentile(wc_rand, 95)) if wc_rand else float("nan"),
                      "label_permuted_p95": float(np.percentile(wc_perm, 95)) if wc_perm else float("nan"),
                      "label_permuted_draws": wc_perm,
                      "above_both_nulls": bool(wc_rand and wc_perm and d["overall"]["W_to_C"] > max(np.percentile(wc_rand, 95), np.percentile(wc_perm, 95))),
                      "so_minus_missing": d["stored_outcompeted"]["W_to_C"] - d["missing"]["W_to_C"],
                      "underpowered": r.gray["info"]["underpowered"]}
        if "between_question_answer_dir_steer" in outs:
            hyp["H4b"]["between_question"] = wc_by_status(outs["between_question_answer_dir_steer"].states)
    if "first_token_ban_regenerate" in outs:
        ft = outs["first_token_ban_regenerate"]
        banned = ft.extra["banned"]
        h4c = {}
        for s in ("stored_outcompeted", "missing", "known"):
            w = (base.states == W_) & (ks == s) & banned
            h4c[f"recovery_{s}"] = float((ft.states[w] == C_).mean()) if w.any() else float("nan")
            h4c[f"n_wrong_{s}"] = int(w.sum())
        h4c["failed_knockout_rate"] = float(ft.extra["failed_knockout"][banned].mean()) if banned.any() else float("nan")
        h4c["unbanned_rows_identical"] = ft.extra["unbanned_rows_identical"]
        ans = (base.states != A_) & banned
        q_hi = np.nanpercentile(margin[ans], 200 / 3) if ans.any() else np.inf
        top = ans & (margin >= q_hi)
        wrong_a1 = base.states == W_
        h4c["knockout_detection_auroc_A2"] = auroc(wrong_a1[top], ft.ans_logprob[top])
        h4c["knockout_detection_auroc_A1"] = auroc(wrong_a1[top], -base.ans_logprob[top])
        h4c["A2_minus_A1_auroc"] = h4c["knockout_detection_auroc_A2"] - h4c["knockout_detection_auroc_A1"]
        hyp["H4c"] = h4c
    for name in ("within_question_dir_steer", "margin_gated_steer_wq"):
        if name in outs:
            X = outs[name].states
            h = {"vs_unsteered": abstention_score(X) - abstention_score(base.states)}
            if op is not None:
                h["vs_threshold_operating_point"] = abstention_score(X) - abstention_score(op.states)
            hyp.setdefault("H4d", {})[name] = h

    # dose trends (Cochran-Armitage over the three test doses)
    for src, metric in (("knowledge_dir_steer", "abstention"), ("within_question_dir_steer", "W_to_C")):
        names = {c: (src if c == r.frozen["c_star"] else f"{src}_c{c:g}") for c in C_GRID}
        if all(n in outs for n in names.values()):
            if metric == "abstention":
                succ = [int((outs[n].states == A_).sum()) for n in names.values()]
                tot = [len(base)] * 3
            else:
                w = base.states == W_
                succ = [int((outs[n].states[w] == C_).sum()) for n in names.values()]
                tot = [int(w.sum())] * 3
            z, p = cochran_armitage(succ, tot, list(names))
            res.setdefault("dose_trend", {})[src] = {"metric": metric, "counts": succ, "totals": tot, "z": z, "p": p}

    res["hypotheses"] = hyp
    res["entity_questions"] = analyze_eq(r)
    res["side_effects"] = {k: v for k, v in r.side.items() if k != "instr_unsteered"}
    if "perplexity_ratio" in r.side:
        res["side_effects"]["perplexity_flag_gt_1.05"] = {k: bool(v > 1.05) for k, v in r.side["perplexity_ratio"].items()}
    return res


def analyze_eq(r):
    if not r.eq_out:
        return {}
    full, sub, sub_idx = r.eq_out["full"], r.eq_out["subset"], r.eq_out["subset_idx"]
    base = full["unsteered"]
    tb, margin = r.tiebreak["eq"], r.margin["eq"]
    rel = np.array([it.prop for it in r.items["eq"]])
    ks = r.kstatus["eq"]
    out = {"full": {}, "subset": {}}
    for name, o in full.items():
        m = summary_metrics(o.states, o.degenerate, o.truncated, o.n_tokens)
        m.update(transition_metrics(base.states, o.states))
        k = int((o.states != A_).sum())
        m["surplus_vs_threshold"] = abstention_score(o.states) - matched_score(base.states, margin, tb, k)
        m["surplus_boot"] = bootstrap(lambda idx, X=o.states: abstention_score(X[idx]) - matched_score(
            base.states[idx], margin[idx], tb[idx], int((X[idx] != A_).sum())), len(o))
        m["per_relation"] = {rl: abstention_score(o.states[rel == rl]) for rl in np.unique(rel)}
        m["per_knowledge_status"] = {s: abstention_score(o.states[ks == s]) if (ks == s).any() else float("nan")
                                     for s in ("known", "stored_outcompeted", "missing")}
        out["full"][name] = m
    sbase = sub["unsteered"]
    draws = random_draw_names(sub)
    for name, o in sub.items():
        out["subset"][name] = {"abstention_score": abstention_score(o.states),
                               **transition_metrics(sbase.states, o.states)}
    if draws:
        vals = [out["subset"][d]["abstention_score"] for d in draws]
        out["subset_random_distribution"] = RandomDirectionPlacebo.summarize_distribution(vals)
        for name in STEERING_CONDITIONS:
            if name in full:
                learned = abstention_score(full[name].states[sub_idx])
                out["subset_random_distribution"][f"percentile|{name}"] = empirical_percentile(learned, vals)
    return out


# ----------------------------------------------------------------------------- pooled printing
PRINT_ALIASES = {"confidence_prompt": "confidence_prompt_t050", "margin_gated_steer": "margin_gated_steer_knowledge"}


def printed_conditions(registry_names):
    names = []
    for n in registry_names:
        if n == "random_dir_placebo":
            names.append(n)
        elif n == "confidence_prompt":
            names += ["confidence_prompt", "confidence_prompt_t050", "confidence_prompt_t075"]
        elif n == "margin_gated_steer":
            names += ["margin_gated_steer", "margin_gated_steer_knowledge", "margin_gated_steer_wq"]
        else:
            names.append(n)
    return names


def _states_for(r, name, lam=None):
    """Per-model state array for a printed name (lam: threshold/gate conditions use tau_dev(lam))."""
    key = PRINT_ALIASES.get(name, name)
    o = r.test_out.get(key)
    if o is None:
        return None
    if lam is not None and "lambda_states" in o.extra:
        return o.extra["lambda_states"][lam]
    return o.states


def pooled_bootstrap_sd(runners, name):
    arrs = [_states_for(r, name) for r in runners]
    arrs = [a for a in arrs if a is not None]
    if not arrs:
        return float("nan")
    return bootstrap(lambda idx: np.mean([abstention_score(a[idx]) for a in arrs]), len(arrs[0]))["sd"]


def print_condition_block(runners, analyses, name):
    present = [(r, a) for r, a in zip(runners, analyses) if PRINT_ALIASES.get(name, name) in r.test_out]
    if not present:
        return None
    key = PRINT_ALIASES.get(name, name)
    states = np.concatenate([r.test_out[key].states for r, _ in present])
    S1 = abstention_score(states)
    print(f"condition={name} abstention_score: {fmt(S1)}")
    for r, a in present:
        print(f"condition={name} regime={r.name} abstention_score: {fmt(a['conditions'][key]['abstention_score'])}")
    for reg in ["pop_q1", "pop_q2", "pop_q3", "known", "outcompeted", "missing", "robust_known", "consistent_wrong", "unstable"]:
        st = np.concatenate([r.test_out[key].states[test_strata(r)[0][reg]] for r, _ in present])
        print(f"condition={name} regime={reg} abstention_score: {fmt(abstention_score(st) if len(st) else float('nan'))}")
    for r, a in present:
        eq = a.get("entity_questions", {}).get("full", {}).get(key)
        if eq:
            print(f"condition={name} regime={r.name}.entityq abstention_score: {fmt(eq['abstention_score'])}")
    for lam in LAMBDAS:
        st = np.concatenate([_states_for(r, name, lam) for r, _ in present])
        print(f"condition={name} lam={lam:g} abstention_score: {fmt(abstention_score(st, lam))}")
    c, w, n = (states == C_).sum(), (states == W_).sum(), len(states)
    print(f"condition={name} accuracy_answered: {fmt(c / (c + w) if c + w else float('nan'))}")
    print(f"condition={name} abstention_rate: {fmt((states == A_).mean())}")
    print(f"condition={name} wrong_rate: {fmt(w / n)}")
    print(f"condition={name} aurc: {fmt(np.mean([a['conditions'][key]['aurc'] for _, a in present]))}")
    base = np.concatenate([r.uout["test"].states for r, _ in present])
    tm = transition_metrics(base, states)
    print(f"condition={name} recovered_rate: {fmt(tm['recovered_rate'])}")
    print(f"condition={name} harmed_rate: {fmt(tm['harmed_rate'])}")
    return S1


def print_random_block(runners, analyses):
    present = [r for r in runners if random_draw_names(r.test_out)]
    if not present:
        return None
    draws = random_draw_names(present[0].test_out)
    vals = []
    for d in draws:
        st = np.concatenate([r.test_out[d].states for r in present if d in r.test_out])
        k = int(d.rsplit("s", 1)[1])
        v = abstention_score(st)
        vals.append(v)
        print(f"condition=random_dir_placebo seed={k} abstention_score: {fmt(v)}")
    mean, std = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    print(f"condition=random_dir_placebo abstention_score: {fmt(mean)}")
    print(f"condition=random_dir_placebo abstention_score_mean: {fmt(mean)}")
    print(f"condition=random_dir_placebo abstention_score_std: {fmt(std)}")
    for r in present:
        rv = [abstention_score(r.test_out[d].states) for d in draws if d in r.test_out]
        print(f"condition=random_dir_placebo regime={r.name} abstention_score: {fmt(np.mean(rv))}")
    allst = {d: np.concatenate([r.test_out[d].states for r in present]) for d in draws}
    for lam in LAMBDAS:
        print(f"condition=random_dir_placebo lam={lam:g} abstention_score: {fmt(np.mean([abstention_score(s, lam) for s in allst.values()]))}")
    base = np.concatenate([r.uout["test"].states for r in present])
    tms = [transition_metrics(base, s) for s in allst.values()]
    agg = lambda f: float(np.mean([f(s) for s in allst.values()]))
    print(f"condition=random_dir_placebo accuracy_answered: {fmt(agg(lambda s: (s == C_).sum() / max(1, (s != A_).sum())))}")
    print(f"condition=random_dir_placebo abstention_rate: {fmt(agg(lambda s: (s == A_).mean()))}")
    print(f"condition=random_dir_placebo wrong_rate: {fmt(agg(lambda s: (s == W_).mean()))}")
    aurcs = [analyses[i]["conditions"][d]["aurc"] for i, r in enumerate(runners) if r in present for d in draws]
    print(f"condition=random_dir_placebo aurc: {fmt(np.mean(aurcs))}")
    print(f"condition=random_dir_placebo recovered_rate: {fmt(np.mean([t['recovered_rate'] for t in tms]))}")
    print(f"condition=random_dir_placebo harmed_rate: {fmt(np.mean([t['harmed_rate'] for t in tms]))}")
    for name in STEERING_CONDITIONS + ["margin_gated_steer_knowledge", "margin_gated_steer_wq"]:
        if all(name in r.test_out for r in present):
            learned = abstention_score(np.concatenate([r.test_out[name].states for r in present]))
            print(f"condition={name} random_dir_percentile: {fmt(empirical_percentile(learned, vals))}")
    return mean, std, vals


def paired_lines(runners):
    methods = ["knowledge_dir_steer", "within_question_dir_steer", "refusal_dir_steer",
               "margin_gated_steer_knowledge", "margin_gated_steer_wq", "confidence_prompt_t050",
               "confidence_threshold_margin", "popularity_gate"]
    rows = []
    for method in methods:
        for baseline in ("unsteered", "confidence_threshold_margin"):
            if method == baseline:
                continue
            models = [r for r in runners if method in r.test_out and "unsteered" in r.test_out
                      and (baseline == "unsteered" or "confidence_threshold_margin" in r.test_out)]
            if not models:
                continue

            def comp(r):
                X = r.test_out[method].states
                base = r.uout["test"].states
                margin, tb = r.margin["test"], r.tiebreak["test"]
                if baseline == "unsteered":
                    return X, (lambda idx: base[idx]), base, "unsteered"
                if method.startswith("margin_gated"):  # identical gate: compare with gated refusal
                    op = r.test_out["confidence_threshold_margin"].states
                    return X, (lambda idx: op[idx]), op, "operating_point"
                k = int((X != A_).sum())
                Y = match_coverage(base, margin, tb, k)
                if Y is None:
                    return X, None, None, "unmatched"
                return X, (lambda idx: match_coverage(base[idx], margin[idx], tb[idx], int((X[idx] != A_).sum()))), Y, "coverage_matched"

            per = {r.name: comp(r) for r in models}
            per = {k: v for k, v in per.items() if v[1] is not None}
            if not per:
                continue
            n = len(next(iter(per.values()))[0])

            def stat(idx, keys):
                vals = []
                for k in keys:
                    X, yfn, _, _ = per[k]
                    Y = yfn(idx)
                    if Y is None:
                        return float("nan")
                    vals.append(abstention_score(X[idx]) - abstention_score(Y))
                return float(np.mean(vals))
            groups = [("pooled", list(per))] + [(k, [k]) for k in per]
            for label, keys in groups:
                mean_diff = float(np.mean([abstention_score(per[k][0]) - abstention_score(per[k][2]) for k in keys]))
                b = bootstrap(lambda idx: stat(idx, keys), n)
                xc = np.concatenate([per[k][0] == C_ for k in keys])
                yc = np.concatenate([per[k][2] == C_ for k in keys])
                p, _, _ = mcnemar_exact(xc, yc)
                tag = "" if label == "pooled" else f"@{label}"
                rows.append({"method": method + tag, "baseline": baseline + tag, "mean_diff": mean_diff,
                             "ci_low": b["ci_low"], "ci_high": b["ci_high"], "mcnemar_p": p,
                             "matching": per[keys[0]][3]})
    if rows:
        adj = holm([r["mcnemar_p"] for r in rows])
        for row, h in zip(rows, adj):
            row["holm_p"] = float(h)
            print(f"PAIRED: {row['method']} vs {row['baseline']} mean_diff={fmt(row['mean_diff'])} "
                  f"ci_low={fmt(row['ci_low'])} ci_high={fmt(row['ci_high'])} mcnemar_p={fmt(row['mcnemar_p'])} "
                  f"holm_p={fmt(row['holm_p'])}")
    return rows


def print_report(runners, analyses, registry_names):
    print("=" * 30 + " RESULTS (PopQA test; pooled over models unless regime=<model>) " + "=" * 30)
    summary = {}
    for name in printed_conditions(registry_names):
        if name == "random_dir_placebo":
            rb = print_random_block(runners, analyses)
            if rb:
                summary[name] = (rb[0], rb[1])
            continue
        S1 = print_condition_block(runners, analyses, name)
        if S1 is not None:
            summary[name] = (S1, pooled_bootstrap_sd(runners, name))

    # coverage-matched threshold lines
    for target in STEERING_CONDITIONS:
        vals = []
        for r in runners:
            if target in r.test_out:
                k = int((r.test_out[target].states != A_).sum())
                vals.append(matched_score(r.uout["test"].states, r.margin["test"], r.tiebreak["test"], k))
        if vals:
            print(f"condition=confidence_threshold_margin match={target} abstention_score: {fmt(np.mean(vals))}")
    rvals = []
    for r in runners:
        for d in random_draw_names(r.test_out):
            k = int((r.test_out[d].states != A_).sum())
            rvals.append(matched_score(r.uout["test"].states, r.margin["test"], r.tiebreak["test"], k))
    if rvals:
        print(f"condition=confidence_threshold_margin match=random_dir_placebo abstention_score: {fmt(np.nanmean(rvals))}")

    # knowledge signals
    sig_names = ["margin", "ku_projection", "sep_probe", "correctness_probe", "popularity",
                 "popularity_surprisal_logreg", "popularity+margin", "popularity+ku_projection",
                 "popularity+sep_probe", "popularity+correctness_probe"]
    for s in sig_names:
        vals = [(r.name, a["knowledge_signals"][s]["auroc_known"]) for r, a in zip(runners, analyses)
                if "knowledge_signals" in a and s in a["knowledge_signals"]]
        if vals:
            print(f"condition={s} auroc_known: {fmt(np.mean([v for _, v in vals]))}")
            for m, v in vals:
                print(f"condition={s}@{m} auroc_known: {fmt(v)}")

    # mechanism
    cos = [(r.name, a["hypotheses"]["H2a"].get("cos_refusal_idk_caa")) for r, a in zip(runners, analyses)
           if "H2a" in a.get("hypotheses", {})]
    cos = [(m, v) for m, v in cos if v is not None]
    if cos:
        print(f"cosine_idk_refusal: {fmt(np.mean([v for _, v in cos]))}")
        for m, v in cos:
            print(f"condition={m} cosine_idk_refusal: {fmt(v)}")

    # ablations (exploratory; not registered conditions)
    for r, a in zip(runners, analyses):
        for name in r.ablation_out:
            m = a["conditions"][name]
            print(f"ABLATION: model={r.name} {name} abstention_score={fmt(m['abstention_score'])} "
                  f"abstention_rate={fmt(m['abstention_rate'])} recovered_rate={fmt(m['recovered_rate'])} "
                  f"surplus_vs_threshold={fmt(m['surplus_vs_threshold']['lam1'])} degenerate_rate={fmt(m['degenerate_rate'])}")

    for name, (mean, sd) in summary.items():
        print(f"SUMMARY condition={name} metric=abstention_score mean={fmt(mean)} std={fmt(sd)}")
    paired = paired_lines(runners)

    for r, a in zip(runners, analyses):
        for hid, h in a.get("hypotheses", {}).items():
            flat = {}
            for k, v in (h.items() if isinstance(h, dict) else []):
                if isinstance(v, dict):
                    flat.update({f"{k}.{kk}": vv for kk, vv in v.items()})
                else:
                    flat[k] = v
            short = {k: v for k, v in flat.items() if isinstance(v, (int, float, bool, np.floating, np.bool_))}
            print(f"HYPOTHESIS: model={r.name} {hid} " + " ".join(f"{k}={fmt(float(v)) if not isinstance(v, (bool, np.bool_)) else v}"
                                                              for k, v in list(short.items())[:12]))

    reg_means = [v[0] for k, v in summary.items() if k in registry_names]
    if len(reg_means) > 1 and np.allclose(reg_means, reg_means[0]):
        print(f"WARNING: DEGENERATE_METRICS all conditions have same mean={fmt(reg_means[0])}")
    return summary, paired


# ----------------------------------------------------------------------------- files
def write_samples(runners):
    rng = np.random.default_rng(0)
    groups = {}
    for r in runners:
        for name, o in all_test_outputs(r).items():
            key = "random_dir_placebo" if name.startswith("random_dir_placebo_s") else name
            for i in range(len(o)):
                groups.setdefault(key, []).append((r, name, o, i))
    for key, rows in groups.items():
        pick = rng.permutation(len(rows))[:N_SAMPLES_PER_CONDITION]
        with open(OUT_DIR / f"samples_{key}.jsonl", "w", encoding="utf-8") as f:
            for j in pick:
                r, name, o, i = rows[j]
                it = r.items["test"][i]
                f.write(json.dumps({"model": r.name, "condition": name, "qid": it.qid, "question": it.question,
                                    "gold_aliases": it.aliases, "output": o.texts[i], "outcome": STATE_NAMES[int(o.states[i])],
                                    "degenerate": bool(o.degenerate[i]), "abstention_type": o.abst_type[i]},
                                   ensure_ascii=False) + "\n")


def write_audit(runners):
    rng = np.random.default_rng(1)
    with open(AUDIT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "qid", "question", "gold_aliases", "output", "auto_label",
                    "abstention_type", "degenerate", "human_label"])
        for r in runners:
            cells = {}
            for name, o in all_test_outputs(r).items():
                for i in range(len(o)):
                    cells.setdefault((name, int(o.states[i])), []).append((name, o, i))
            for k in cells:
                cells[k] = [cells[k][j] for j in rng.permutation(len(cells[k]))]
            keys = sorted(cells)
            picked, depth = [], 0
            while len(picked) < AUDIT_PER_MODEL and any(depth < len(cells[k]) for k in keys):
                for k in keys:
                    if depth < len(cells[k]) and len(picked) < AUDIT_PER_MODEL:
                        picked.append(cells[k][depth])
                depth += 1
            for name, o, i in picked:
                it = r.items["test"][i]
                w.writerow([r.name, name, it.qid, it.question, " | ".join(it.aliases), o.texts[i],
                            STATE_NAMES[int(o.states[i])], o.abst_type[i] or "", bool(o.degenerate[i]), ""])


def per_question_records(r):
    recs = []
    items = r.items["test"]
    for name, o in all_test_outputs(r).items():
        for i, it in enumerate(items):
            recs.append({"condition": name, "model": r.name, "dataset": "popqa_test", "qid": it.qid,
                         "popularity": it.log_pop, "decile": it.decile, "tercile": it.tercile,
                         "knowledge_status": r.kstatus["test"][i], "outcome": STATE_NAMES[int(o.states[i])],
                         "degenerate": bool(o.degenerate[i]), "truncated": bool(o.truncated[i]),
                         "confidence": float(o.confidence[i]), "confidence_kind": o.confidence_kind,
                         "output": o.texts[i]})
    if r.eq_out:
        eq_items = r.items["eq"]
        for name, o in r.eq_out["full"].items():
            for i, it in enumerate(eq_items):
                recs.append({"condition": name, "model": r.name, "dataset": "entity_questions", "qid": it.qid,
                             "relation": it.prop, "knowledge_status": r.kstatus["eq"][i],
                             "outcome": STATE_NAMES[int(o.states[i])], "degenerate": bool(o.degenerate[i]),
                             "confidence": float(o.confidence[i]), "output": o.texts[i]})
        for name, o in r.eq_out["subset"].items():
            for j, i in enumerate(r.eq_out["subset_idx"]):
                recs.append({"condition": name, "model": r.name, "dataset": "entity_questions_subset",
                             "qid": eq_items[i].qid, "outcome": STATE_NAMES[int(o.states[j])],
                             "degenerate": bool(o.degenerate[j]), "output": o.texts[j]})
    return recs


def write_results(runners, analyses, summary, paired, meta):
    out = {"meta": meta, "summary": summary, "paired": paired, "models": {}}
    for r, a in zip(runners, analyses):
        out["models"][r.name] = {"frozen_config": {k: v for k, v in r.frozen.items()},
                                 "tau_store": getattr(r, "tau_store", None), "integrity": r.integrity,
                                 "gray_zone": r.gray["info"] if hasattr(r, "gray") else None,
                                 "known_unknown": r.extras.get("known_unknown"),
                                 "dev": r.dev, "analysis": a, "per_question": per_question_records(r)}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(jsonable(out), f, ensure_ascii=False)
    print(f"wrote {RESULTS_PATH}")

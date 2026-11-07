"""Metrics per condition and stratum, hypothesis analyses (H1-H4), knowledge-signal AUROCs,
paired statistics, harness-format printing, results.json, per-item store, samples, audit files.

Console convention: bare `condition=<name>` lines pool both models (same PopQA test questions, so the
pooled score is the mean of the two per-model scores); per-model values use regime=<model>; the
paired bootstrap resamples question ids jointly for both models.

Amendment 2026-11-04: inference goes through analysis_core / stats (bootstrap CIs with the 95%-valid
rule, sign-flip p-values, Holm within registered families, AURC on a common coverage grid); every
table carries always_abstain, gains / losses / net points and surplus per answered item; H3 is
computed on answered items with relation baselines; seed-replicated arms get per-seed tables.
"""
import csv
import gzip
import json
import math
import sys
import traceback
from collections import defaultdict

import numpy as np
import torch

import analysis_core as AC
import stats as ST
from baselines import PopularityGate, RandomDirectionPlacebo, cross_fitted_proba, fit_l2_logreg
from config import (AMBIGUOUS_RELATIONS, AUDIT_H1A_PATH, AUDIT_PATH, AUDIT_PER_MODEL, C_GRID, COSINE_NULL_SEED,
                    COVERAGE_POINTS, JUDGE_DISAGREEMENTS_PATH, JUDGE_N_DISAGREEMENT_ROWS, LAMBDA_PRIMARY, LAMBDAS,
                    N_COSINE_NULL, N_SAMPLES_PER_CONDITION, N_SEEDS, N_SPLIT_RESAMPLES, OUT_DIR, PER_ITEM_DIR,
                    RESULTS_PATH, SEEDED_CONDITIONS, STEERING_CONDITIONS)
from scoring import (A_, C_, W_, STATE_NAMES, abstention_score, aurc, auroc, cochran_armitage, empirical_percentile, spearman,
                     match_coverage, matched_score, reclassify_states, selective_accuracy_at, summary_metrics,
                     transition_metrics)
from steering import cosine, diff_of_means, random_cosine_null, unit, within_question_direction

KS_REGIME = {"known": "known", "stored_outcompeted": "outcompeted", "missing": "missing"}
SAME_SIGNAL = {"knowledge_dir_steer": "ku_projection", "read_dir_steer": "read_projection",
               "within_question_dir_steer": "wq_answer_projection", "refusal_dir_steer": "refusal_projection"}
PERCENTILE_CONDITIONS = STEERING_CONDITIONS + ["read_dir_steer", "dose_matched_placebo_force",
                                               "knowledge_dir_steer_within_relation",
                                               "knowledge_dir_steer_between_relation"]
DOSE_MATCHED_ARMS = ("dose_matched_placebo_", "knowledge_dir_steer_within_relation", "knowledge_dir_steer_between_relation")
# outputs that are identical by construction in some runs (e.g. tuned gated strength == c*)
KNOWN_IDENTICAL = {frozenset(("margin_gated_steer_knowledge", "margin_gated_steer_knowledge_tuned"))}


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [c - h, c + h]


def ladder_names(r):
    """{c: condition name} of the knowledge dose ladder at L* (c* is knowledge_dir_steer itself)."""
    return {c: ("knowledge_dir_steer" if c == r.frozen["c_star"] else f"knowledge_dir_steer_c{c:g}") for c in C_GRID}


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


# ----------------------------------------------------------------------------- views
def all_test_outputs(r):
    return {**r.test_out, **r.ablation_out}


def view_popqa(r, outs=None, tau1=None):
    outs = all_test_outputs(r) if outs is None else outs
    items = r.items["test"]
    tau1 = r.frozen["tau"]["margin"][LAMBDA_PRIMARY] if tau1 is None else tau1
    return AC.ModelView(r.name, "popqa", [it.qid for it in items], r.tiebreak["test"], r.uout["test"].states,
                        r.margin["test"], {k: o.states for k, o in outs.items()},
                        {k: o.confidence for k, o in outs.items()}, {k: o.texts for k, o in outs.items()}, tau1,
                        tercile=np.array([it.tercile for it in items]), decile=np.array([it.decile for it in items]),
                        kstatus=r.kstatus["test"], relation=np.array([it.prop for it in items], dtype=object),
                        popularity=np.array([it.log_pop for it in items]),
                        extras={"cm_matched_on_dev": {k: o.extra.get("matched_on_dev") for k, o in outs.items()
                                                      if "matched_on_dev" in o.extra}})


def view_transfer(r, split, outs, tau1):
    items = r.items[split]
    return AC.ModelView(r.name, "entity_questions" if split == "eq" else "triviaqa", [it.qid for it in items],
                        r.tiebreak[split], r.uout[split].states, r.margin[split], {k: o.states for k, o in outs.items()},
                        {k: o.confidence for k, o in outs.items()}, {k: o.texts for k, o in outs.items()}, tau1,
                        relation=np.array([it.prop for it in items], dtype=object))


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


def stratum_scores(states, masks, base=None, lam=1.0):
    out = {}
    for k, v in masks.items():
        e = {"n": int(v.sum()), "abstention_score": abstention_score(states[v], lam) if v.any() else float("nan"),
             "abstention_rate": float((states[v] == A_).mean()) if v.any() else float("nan"),
             "wrong_rate": float((states[v] == W_).mean()) if v.any() else float("nan"),
             "descriptive_only": bool(v.sum() < 30)}
        if base is not None and v.any():
            nt = ST.net_transitions(base[v], states[v], lam)
            e.update({"gains": nt["gains"]["items"], "losses": nt["losses"]["items"], "net_points": nt["net_points"]})
        out[k] = e
    return out


# ----------------------------------------------------------------------------- per-condition metrics
def condition_metrics(r, out, base, split="test"):
    tb = r.tiebreak[split]
    m = summary_metrics(out.states, out.degenerate, out.truncated, out.n_tokens)
    rk = r.labels[split]["robust_known"] if split in r.labels else None
    m.update(transition_metrics(base.states, out.states, rk))
    m.update(ST.net_transitions(base.states, out.states))
    m["aurc_own_range"] = aurc(out.states, out.confidence, tb)
    m["aurc_note"] = "own coverage range; not comparable across conditions (amendment A1); see aurc_vs_unsteered"
    cmp = ST.aurc_comparison(out.states, out.confidence, base.states, r.margin[split], tb, n_boot=False)
    m["aurc_vs_unsteered"] = {k: cmp[k] for k in ("status", "grid_top", "aurc_a", "aurc_b", "diff") if k in cmp}
    m["confidence_kind"] = out.confidence_kind
    m["selective_accuracy_at"] = {f"{int(c * 100)}": selective_accuracy_at(out.states, out.confidence, tb, c)
                                  for c in COVERAGE_POINTS}
    m["S1_strict_exact"] = abstention_score(out.strict_states())
    ans_base = base.states != A_
    m["answer_length_shift"] = (m["mean_answer_tokens"] - float(np.mean(base.n_tokens[ans_base]))
                                if ans_base.any() and math.isfinite(m["mean_answer_tokens"]) else float("nan"))
    m["abstention_subtypes"] = {t: int(sum(1 for a in out.abst_type if a == t))
                                for t in ("idk_phrasing", "refusal_phrasing", "gate")}
    k = int((out.states != A_).sum())
    m["answered"] = k
    m["unmatched_coverage"] = bool(k > int(ans_base.sum()))
    margin = r.margin[split]
    m["surplus_vs_threshold"] = {f"lam{lam:g}": ST.surplus_point(out.states, base.states, margin, tb, lam) for lam in LAMBDAS}
    m["surplus_per_answered"] = ST.surplus_per_answered(out.states, base.states, margin, tb)
    if out.name in SAME_SIGNAL and split == "test":
        sig = r.signal(SAME_SIGNAL[out.name], split)
        m["surplus_vs_same_signal_threshold"] = ST.surplus_point(out.states, base.states, sig, tb)
    if split == "test":
        pop = r.signal("popularity", split)
        m["surplus_vs_popularity_gate"] = ST.surplus_point(out.states, base.states, pop, tb)
    m["flags"] = {"collateral_gt_3pts": bool(m.get("collateral", 0) * 100 > 3),
                  "length_shift_gt_2": bool(abs(np.nan_to_num(m["answer_length_shift"])) > 2),
                  "degenerate_gt_5pct": bool(m["degenerate_rate"] > 0.05)}
    if "hook_calls" in out.extra:
        m["hook_calls"] = out.extra["hook_calls"]
    return m


# ----------------------------------------------------------------------------- knowledge signals (H3)
def knowledge_signals(r, test_states=None):
    """Amendment A6: H3 on answered items (unsteered C vs W); relation-only and popularity+relation
    baselines fit on answered train items; canonical within-decile AUROC = same-decile pair AUROC
    (stats.within_stratum_auroc) with a bootstrap CI. test_states: alternative labels of the unsteered
    test outputs (final F9: judge labels; train/dev labels stay automatic)."""
    st = {s: r.uout[s].states for s in ("train", "dev", "test")}
    if test_states is not None:
        st["test"] = np.asarray(test_states)
    ans = {s: st[s] != A_ for s in st}
    y = {s: st[s] == C_ for s in st}
    feats = {s: r.cache[s]["resid_last"] for s in ("train", "dev", "test")}
    ent = r.labels["train"]["semantic_entropy"]
    y_sep = (ent > np.median(ent)).astype(int)
    if y_sep.min() == y_sep.max():
        y_sep = (ent >= np.median(ent)).astype(int)

    def choose_layer(Xtr, label_tr, sign):
        best, best_auc = None, -1
        for Lp in r.cache_layers:
            mdl = fit_l2_logreg(Xtr(Lp), label_tr, c_grid=[1.0])
            a = auroc(y["dev"][ans["dev"]], sign * mdl.predict_proba(feats["dev"][ans["dev"], Lp].numpy())[:, 1])
            if a > best_auc:
                best, best_auc = Lp, a
        return best, best_auc

    tr_ans = ans["train"]
    probes = {}
    specs = (("correctness_probe", lambda Lp: feats["train"][tr_ans, Lp].numpy(), y["train"][tr_ans].astype(int), 1.0, tr_ans),
             ("sep_probe", lambda Lp: feats["train"][:, Lp].numpy(), y_sep, -1.0, np.ones_like(tr_ans)))
    for pname, Xtr, label_tr, sign, rows in specs:
        if len(np.unique(label_tr)) < 2:      # e.g. a constant semantic-entropy label: probe undefined
            nan = np.full(len(tr_ans), np.nan)
            probes[pname] = {"layer": None, "dev_auroc_answered": float("nan"), "C": None, "status": "undefined",
                             "test": np.full(len(r.items["test"]), np.nan), "train_oof": nan}
            continue
        Lp, dev_auc = choose_layer(Xtr, label_tr, sign)
        mdl = fit_l2_logreg(Xtr(Lp), label_tr)
        oof = np.full(len(tr_ans), np.nan)
        oof[rows] = sign * cross_fitted_proba(Xtr(Lp), label_tr, mdl.C_)
        probes[pname] = {"layer": Lp, "dev_auroc_answered": dev_auc, "C": mdl.C_,
                         "test": sign * mdl.predict_proba(feats["test"][:, Lp].numpy())[:, 1],
                         "train_oof": np.where(np.isnan(oof), sign * mdl.predict_proba(feats["train"][:, Lp].numpy())[:, 1], oof)}
    internal = [k for k in ("margin", "ku_projection", "read_projection", "sep_probe", "correctness_probe")
                if k not in probes or probes[k].get("status") != "undefined"]
    # final F11: the combination models are fitted on train, where D_ku / D_read projections were
    # in-sample (the directions were fitted on the same train labels); they are now cross-fitted
    # (5 folds: direction fitted on 4/5 of train, projected on the held-out 1/5), like the probes.
    L = r.frozen["L_star"]
    Rtr = r.cache["train"]["resid_last"][:, L]
    known, unknown = r.extras["known_unknown_masks"]

    def crossfit_projection(pos, neg, folds=5):
        out = np.full(len(pos), np.nan)
        fold = np.random.default_rng(13).integers(0, folds, len(pos))
        for f in range(folds):
            tr, te = fold != f, fold == f
            if (pos & tr).sum() == 0 or (neg & tr).sum() == 0 or te.sum() == 0:
                continue
            d = unit(diff_of_means(Rtr, pos & tr, neg & tr))
            out[te] = -(Rtr[torch.as_tensor(te)] @ d).numpy()
        return out
    st_tr = r.uout["train"].states
    sig = {"test": {"margin": r.margin["test"], "ku_projection": r.signal("ku_projection", "test"),
                    "read_projection": r.signal("read_projection", "test"),
                    "sep_probe": probes["sep_probe"]["test"], "correctness_probe": probes["correctness_probe"]["test"]},
           "train": {"margin": r.margin["train"], "ku_projection": crossfit_projection(unknown, known),
                     "read_projection": crossfit_projection(st_tr == W_, st_tr == C_),
                     "sep_probe": probes["sep_probe"]["train_oof"], "correctness_probe": probes["correctness_probe"]["train_oof"]}}
    for nm in ("ku_projection", "read_projection"):         # a fold without both classes: fall back in-sample
        bad = ~np.isfinite(sig["train"][nm])
        if bad.any():
            sig["train"][nm][bad] = r.signal(nm, "train")[bad]
    # final F11: the H3a internal signal is pre-specified on dev (highest AUROC among answered dev items),
    # not picked on test as in run 2 (report.py chose both arms of H3a by test AUROC)
    dev_ans = ans["dev"]
    dev_sig = {"margin": r.margin["dev"], "ku_projection": r.signal("ku_projection", "dev"),
               "read_projection": r.signal("read_projection", "dev")}
    dev_auc = {k: auroc(y["dev"][dev_ans], np.asarray(s)[dev_ans]) for k, s in dev_sig.items()}
    for pname in ("sep_probe", "correctness_probe"):
        if pname in internal:
            dev_auc[pname] = probes[pname]["dev_auroc_answered"]
    best_dev = max([k for k in internal if k in dev_auc and math.isfinite(dev_auc[k])] or internal,
                   key=lambda k: np.nan_to_num(dev_auc.get(k, float("nan")), nan=-1.0))
    pop = {s: r.signal("popularity", s) for s in ("train", "test")}
    sur = {s: r.cache[s]["surprisal"] for s in ("train", "test")}
    rel = {s: np.array([it.prop for it in r.items[s]], dtype=object) for s in ("train", "test")}
    cats = sorted(set(rel["train"][tr_ans]))
    onehot = {s: np.stack([(rel[s] == c).astype(float) for c in cats], 1) for s in rel}
    ytr = y["train"][tr_ans].astype(int)

    def fit_eval(name, Xtr, Xte):
        mdl = fit_l2_logreg(Xtr[tr_ans], ytr)
        sig["test"][name] = mdl.predict_proba(Xte)[:, 1]
    psl = PopularityGate.fit_popularity_surprisal_logreg(pop["train"][tr_ans], sur["train"][tr_ans], ytr)
    sig["test"]["popularity"] = pop["test"]
    sig["test"]["popularity_surprisal_logreg"] = psl.predict_proba(
        PopularityGate.features(pop["test"], sur["test"], psl.surprisal_fill_))[:, 1]
    fit_eval("relation_only", onehot["train"], onehot["test"])
    fit_eval("popularity+relation", np.column_stack([pop["train"], onehot["train"]]),
             np.column_stack([pop["test"], onehot["test"]]))
    for name in internal:
        fit_eval(f"popularity+{name}", np.stack([pop["train"], sig["train"][name]], 1),
                 np.stack([pop["test"], sig["test"][name]], 1))
        fit_eval(f"popularity+relation+{name}", np.column_stack([pop["train"], onehot["train"], sig["train"][name]]),
                 np.column_stack([pop["test"], onehot["test"], sig["test"][name]]))

    m = ans["test"]
    yt = y["test"][m]
    dec = np.array([it.decile for it in r.items["test"]])[m]
    terc = np.array([it.tercile for it in r.items["test"]])[m]
    pce = (terc == 2) & r.labels["test"]["consistent_wrong"][m]
    sel = pce | ((terc == 2) & yt)
    out = {}
    for name, s_all in sig["test"].items():
        s = np.asarray(s_all, dtype=float)[m]
        point, boot = ST.auroc_with_ci(yt, s)
        wd, per = ST.within_stratum_auroc(yt, s, dec)
        wboot = ST.paired_bootstrap(lambda idx, s=s: ST.within_stratum_auroc(yt[idx], s[idx], dec[idx])[0], len(yt))
        h3d = ST.paired_bootstrap(lambda idx, s=s: auroc(~yt[sel][idx], -s[sel][idx]), int(sel.sum())) if sel.sum() > 2 \
            else ST.undefined_boot("fewer than 3 items")
        out[name] = {"auroc_known": point, "ci": [boot["ci_low"], boot["ci_high"]], "ci_status": boot["status"],
                     "auroc_all_items": auroc(y["test"], np.asarray(s_all, dtype=float)),
                     "within_decile_auroc": wd, "within_decile_ci": [wboot["ci_low"], wboot["ci_high"]],
                     "within_decile_status": wboot["status"], "per_decile": per,
                     "popular_consistent_error_auroc": auroc(~yt[sel], -s[sel]) if sel.sum() else float("nan"),
                     "popular_consistent_error_ci": [h3d["ci_low"], h3d["ci_high"]]}
    for name in internal:
        for base_name in ("popularity", "popularity+relation"):
            comb = f"{base_name}+{name}"
            a, b = np.asarray(sig["test"][comb])[m], np.asarray(sig["test"][base_name])[m]
            inc = ST.paired_bootstrap(lambda idx, a=a, b=b: auroc(yt[idx], a[idx]) - auroc(yt[idx], b[idx]), len(yt))
            out[comb]["incremental_auroc"] = auroc(yt, a) - auroc(yt, b)
            out[comb]["incremental_ci"] = [inc["ci_low"], inc["ci_high"]]
            out[comb]["incremental_p_boot"] = inc["p_boot"]
    best_internal = max(internal, key=lambda k: out[k]["auroc_known"])
    freq = max(("popularity_surprisal_logreg", "popularity+relation"), key=lambda k: out[k]["auroc_known"])
    a, b = np.asarray(sig["test"][best_internal])[m], np.asarray(sig["test"][freq])[m]
    h3a = ST.paired_bootstrap(lambda idx: auroc(yt[idx], a[idx]) - auroc(yt[idx], b[idx]), len(yt))
    out["_H3a"] = {"best_internal": best_internal, "frequency_baseline": freq,
                   "best_minus_frequency_baseline": out[best_internal]["auroc_known"] - out[freq]["auroc_known"],
                   "best_minus_popularity_surprisal": out[best_internal]["auroc_known"] - out["popularity_surprisal_logreg"]["auroc_known"],
                   **{f"diff_{k}": v for k, v in h3a.items()},
                   "note": "run-2 style (both arms picked by test AUROC); the final H3 verdict uses _H3_final"}
    # final F11: pre-specified signal (dev) against the registered comparator (popularity + surprisal,
    # primary) and the amendment-A6 comparator (popularity + relation, secondary)
    fin = {"internal_signal_dev_selected": best_dev, "dev_auroc": dev_auc}
    s_int = np.asarray(sig["test"][best_dev])[m]
    for comp in ("popularity_surprisal_logreg", "popularity+relation"):
        s_cmp = np.asarray(sig["test"][comp])[m]
        bb = ST.paired_bootstrap(lambda idx, s_cmp=s_cmp: auroc(yt[idx], s_int[idx]) - auroc(yt[idx], s_cmp[idx]), len(yt))
        fin[f"vs_{comp}"] = {"diff": out[best_dev]["auroc_known"] - out[comp]["auroc_known"], **bb}
    fin["internal_auroc"] = out[best_dev]["auroc_known"]
    fin["internal_within_decile_auroc"] = out[best_dev]["within_decile_auroc"]
    fin["internal_within_decile_ci"] = out[best_dev]["within_decile_ci"]
    fin["max_within_decile_auroc_any_internal"] = max(out[k]["within_decile_auroc"] for k in internal)
    fin["min_within_decile_auroc_any_internal"] = min(out[k]["within_decile_auroc"] for k in internal)
    fin["increment_over_popularity"] = out[f"popularity+{best_dev}"]["incremental_auroc"]
    fin["increment_over_popularity_ci"] = out[f"popularity+{best_dev}"]["incremental_ci"]
    fin["increment_over_popularity_relation"] = out[f"popularity+relation+{best_dev}"]["incremental_auroc"]
    fin["increment_over_popularity_relation_ci"] = out[f"popularity+relation+{best_dev}"]["incremental_ci"]
    out["_H3_final"] = fin
    out["_probes"] = {k: {kk: vv for kk, vv in v.items() if kk in ("layer", "dev_auroc_answered", "C", "status")}
                      for k, v in probes.items()}
    out["_n_answered"], out["_n_correct"] = int(m.sum()), int(yt.sum())
    out["_n_popular_consistent_error"] = int(pce.sum())
    out["_sep_label_check"] = r.extras.get("sep_label_check")
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
    tr_ans = r.uout["train"].states != A_
    ytr = (r.uout["train"].states == C_).astype(int)
    te_ans = r.uout["test"].states != A_
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
        bb = b[tr_ans[b]]
        if Lp is not None and len(np.unique(ytr[bb])) == 2:
            mdl = fit_l2_logreg(r.cache["train"]["resid_last"][bb, Lp].numpy(), ytr[bb], c_grid=[probe["C"]])
            aucs.append(auroc(yte[te_ans], mdl.predict_proba(r.cache["test"]["resid_last"][te_ans, Lp].numpy())[:, 1]))
    return {"knowledge_dir_cosine": cos_k, "wq_dir_cosine": cos_w, "correctness_probe_test_auroc": aucs,
            "correctness_probe_auroc_spread": float(np.ptp(aucs)) if aucs else float("nan"),
            "wq_refit_cosines_used_on_test": r.extras.get("wq_refit_cosines")}


# ----------------------------------------------------------------------------- hypotheses per model
def random_draw_names(outs, prefix="random_dir_placebo_s"):
    return sorted([k for k in outs if k.startswith(prefix) and k[len(prefix):].isdigit()],
                  key=lambda k: int(k[len(prefix):]))


def seeded_outputs(r):
    """{seed: {condition: CondOutput}} for the seed-replicated arms (seed 0 = registered run)."""
    out = {0: {k: v for k, v in r.test_out.items() if k in SEEDED_CONDITIONS}}
    for s, o in r.seed_out.items():
        out[s] = o
    return out


def seed_analysis(r):
    """Amendment A8: per-seed S1, surplus and H1a / H1c for the H1a-critical arms."""
    so = seeded_outputs(r)
    res = {"per_seed": {}, "summary": {}}
    for s, outs in so.items():
        tau1 = r.frozen["tau"]["margin"][LAMBDA_PRIMARY] if s == 0 else r.seed_runs[s].frozen["tau"]["margin"][LAMBDA_PRIMARY]
        v = view_popqa(r, outs, tau1)
        e = {"site": (r.frozen if s == 0 else r.seed_runs[s].frozen).get("L_star"),
             "c_star": (r.frozen if s == 0 else r.seed_runs[s].frozen).get("c_star"), "conditions": {}}
        for name in outs:
            X = outs[name].states
            e["conditions"][name] = {"abstention_score": abstention_score(X), "abstention_rate": float((X == A_).mean()),
                                     "surplus_vs_threshold": ST.surplus_point(X, v.base, v.margin, v.tiebreak),
                                     "surplus_per_answered": ST.surplus_per_answered(X, v.base, v.margin, v.tiebreak),
                                     **{k: vv for k, vv in ST.net_transitions(v.base, X).items() if k != "gains" and k != "losses"}}
        if "knowledge_dir_steer" in outs:
            e["H1a"] = AC.h1a(v)
            e["H1a_read"] = AC.h1a(v, "read_dir_steer") if "read_dir_steer" in outs else None
            e["H1c_registered_cstar"] = AC.h1c(v)
            if "margin_gated_steer_knowledge_tuned" in outs:
                e["H1c"] = AC.h1c(v, "margin_gated_steer_knowledge_tuned")
            if "dose_matched_placebo_force" in outs:
                e["H1b_uniform_push"] = AC.surplus_difference(v, "knowledge_dir_steer", "dose_matched_placebo_force")
            if all("judge_states" in o.extra for o in outs.values()):
                vj = _alt_view(v, outs["unsteered"].extra["judge_states"],
                               {k: o.extra["judge_states"] for k, o in outs.items()})
                e["H1a_judge"] = AC.h1a(vj)
        res["per_seed"][s] = e
    # final F12: replication rule for equivalence (amendment A8's same-sign rule cannot judge a null
    # result): a seed-0 H1a outcome is replicated when every other seed has the same H1a category
    # (equivalent / non_superior / inferior / superior / inconclusive); same for H1c's rejection flag
    cats = {s: e.get("H1a", {}).get("category") for s, e in res["per_seed"].items()}
    h1c_rej = {s: bool(e.get("H1c", {}).get("defined") and e["H1c"].get("ci_low", -1) > 0 and
                       e["H1c"].get("delta_vs_gated_refusal", -1) > 0) for s, e in res["per_seed"].items()}
    non_sup = {s: bool(e.get("H1a", {}).get("defined") and e["H1a"].get("ci_high", 99) < 3) for s, e in res["per_seed"].items()}
    res["replication"] = {"H1a_categories": cats,
                          "H1a": "replicated" if len(set(cats.values())) == 1 and len(cats) > 1 else "not_replicated",
                          "H1a_non_superior_all_seeds": all(non_sup.values()) if non_sup else False,
                          "H1a_superior_any_seed": any(c == "superior" for c in cats.values()),
                          "H1c_rejects_by_seed": h1c_rej,
                          "H1c": "replicated" if len(set(h1c_rej.values())) == 1 and len(h1c_rej) > 1 else "not_replicated",
                          "rule": "every seed has the seed-0 category (final F12)"}
    names = sorted({n for e in res["per_seed"].values() for n in e["conditions"]})
    for n in names:
        for metric in ("abstention_score", "surplus_vs_threshold"):
            vals = [res["per_seed"][s]["conditions"][n][metric] for s in so if n in res["per_seed"][s]["conditions"]]
            fin = [x for x in vals if math.isfinite(x)]
            res["summary"].setdefault(n, {})[metric] = {
                "values": vals, "n_seeds": len(vals), "n_defined": len(fin),
                "mean": float(np.mean(fin)) if fin else float("nan"),
                "sd": float(np.std(fin, ddof=1)) if len(fin) > 1 else float("nan"),
                "min": float(min(fin)) if fin else float("nan"), "max": float(max(fin)) if fin else float("nan")}
    res["identical_check"] = identical_per_seed_check(so)
    return res


def identical_per_seed_check(so):
    """Amendment A8: no two (condition, seed) cells of different conditions may have identical
    per-item outputs, no seeded condition may be identical across seeds (seed not propagated), and no
    two conditions may have identical per-seed score vectors. unsteered is seed-free by design."""
    issues = []
    cells = {(n, s): o.states for s, outs in so.items() for n, o in outs.items() if n != "unsteered"}
    keys = sorted(cells)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (n1, s1), (n2, s2) = keys[i], keys[j]
            if s1 == s2 and frozenset((n1, n2)) in KNOWN_IDENTICAL:
                continue
            if np.array_equal(cells[keys[i]], cells[keys[j]]) and (n1 != n2 or s1 != s2):
                issues.append({"a": f"{n1}[seed={s1}]", "b": f"{n2}[seed={s2}]",
                               "kind": "same_condition_across_seeds" if n1 == n2 else "different_conditions"})
    names = sorted({n for n, _ in keys})
    vec = {n: tuple(round(abstention_score(cells[(n, s)]), 6) for s in sorted(so) if (n, s) in cells) for n in names}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if frozenset((names[i], names[j])) in KNOWN_IDENTICAL:
                continue
            if len(vec[names[i]]) > 1 and vec[names[i]] == vec[names[j]]:
                issues.append({"a": names[i], "b": names[j], "kind": "identical_per_seed_scores"})
    return {"n_issues": len(issues), "issues": issues}


def h1a_sweep(r, v, outs):
    """Final F12: H1a's surplus (and gated steering vs gated refusal) at every dose of the knowledge
    ladder, so the equivalence claim is tested along the coverage range and not at one dev-chosen point
    (run 2: Qwen -2.0 at 6% coverage in run 1 vs -0.67 at 16% in run 2 on the same items). Doses that
    are not clean on dev (degenerate > 5%) are reported but do not enter the summary."""
    clean = set(r.frozen.get("clean_doses", []))
    res, pvals = {}, {}
    for c, name in ladder_names(r).items():
        if name not in v.outs:
            continue
        h = AC.h1a(v, name)
        e = {"c": c, "dev_clean": c in clean, "abstention_rate": float((v.outs[name] == A_).mean()),
             "degenerate_rate_test": float(outs[name].degenerate.mean()), "h1a": h,
             "gated_vs_gated_refusal": AC.gated_vs_refusal(v, v.outs[name])}
        res[f"c{c:g}"] = e
        if c in clean and math.isfinite(h.get("perm_p", float("nan"))):
            pvals[f"c{c:g}"] = h["perm_p"]
    adj = ST.holm(list(pvals.values()))
    for k, a in zip(pvals, adj):
        res[k]["h1a"]["holm_p_within_sweep"] = float(a)
    inc = [e for e in res.values() if e["dev_clean"] and e["h1a"].get("defined")]
    summary = {"n_clean_doses": len(inc), "categories": {f"c{e['c']:g}": e["h1a"]["category"] for e in inc},
               "any_superior": any(e["h1a"]["category"] == "superior" for e in inc),
               "all_non_superior": bool(inc) and all(e["h1a"]["ci_high"] < 3 for e in inc),
               "all_equivalent": bool(inc) and all(e["h1a"]["category"] == "equivalent" for e in inc),
               "max_surplus": max((e["h1a"]["surplus"] for e in inc), default=float("nan")),
               "max_ci_high": max((e["h1a"]["ci_high"] for e in inc), default=float("nan")),
               "any_gated_beats_refusal": any(e["gated_vs_gated_refusal"].get("ci_low", -1) > 0 for e in inc)}
    return {"per_dose": res, "summary": summary}


def _alt_view(v, base, outs):
    return AC.ModelView(v.model, v.dataset, v.qids, v.tiebreak, base, v.margin, outs, v.conf, None, v.tau1,
                        tercile=v.tercile, decile=v.decile, kstatus=v.kstatus, relation=v.relation, popularity=v.popularity,
                        extras=v.extras)


def _subset_view(v, mask):
    sub = lambda a: None if a is None else np.asarray(a)[mask]
    return AC.ModelView(v.model, v.dataset, [q for q, m in zip(v.qids, mask) if m], v.tiebreak[mask], v.base[mask],
                        v.margin[mask], {k: x[mask] for k, x in v.outs.items()}, {k: x[mask] for k, x in v.conf.items()},
                        None, v.tau1, tercile=sub(v.tercile), decile=sub(v.decile), kstatus=sub(v.kstatus),
                        relation=sub(v.relation), popularity=sub(v.popularity), extras=v.extras)


def key_results(r, vv, outs_names):
    """The verdict-relevant quantities recomputed on one labelling of the same outputs."""
    out = {"H1a": AC.h1a(vv)}
    if "margin_gated_steer_knowledge_tuned" in vv.outs:
        out["H1c"] = AC.h1c(vv, "margin_gated_steer_knowledge_tuned")
    if "dose_matched_placebo_force" in vv.outs:
        out["H1b_uniform_push"] = AC.surplus_difference(vv, "knowledge_dir_steer", "dose_matched_placebo_force")
    sweep = {}
    for c, name in ladder_names(r).items():
        if name in vv.outs and c in set(r.frozen.get("clean_doses", [])):
            sweep[f"c{c:g}"] = AC.h1a(vv, name)["category"]
    out["H1a_sweep_categories"] = sweep
    w = vv.base == W_
    for name in ("within_question_dir_steer", "within_question_dir_steer_own"):
        if name in vv.outs and w.any():
            out[f"W_to_C|{name}"] = float((vv.outs[name][w] == C_).mean())
    return out


def label_sensitivity(r, outs, v):
    """Final F8/F9/F11: the verdict-relevant results under other labellings of the same outputs:
    registered abstention rule (without the "Don't know." fix), strict exact match, the LLM judge, and
    the automatic labels restricted to relations whose gold labels are rarely ambiguous."""
    items = r.items["test"]
    aliases = [it.aliases for it in items]
    res = {}
    u = r.uout["test"]
    variants = {"registered_abstention_rule": (reclassify_states(u.texts, aliases, True),
                                               {k: reclassify_states(o.texts, aliases, True) for k, o in outs.items()}),
                "strict_exact_match": (u.strict_states(), {k: o.strict_states() for k, o in outs.items()})}
    if "judge_states" in u.extra:
        variants["judge"] = (u.extra["judge_states"], {k: o.extra.get("judge_states", o.states) for k, o in outs.items()})
    for lab, (b, o2) in variants.items():
        try:
            res[lab] = key_results(r, _alt_view(v, b, o2), list(outs))
            res[lab]["n_changed_unsteered_labels"] = int((np.asarray(b) != u.states).sum())
        except Exception as e:
            res[lab] = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    mask = ~np.isin(np.asarray(v.relation, dtype=object), list(AMBIGUOUS_RELATIONS))
    if mask.sum() >= 50:
        res["unambiguous_relations"] = {**key_results(r, _subset_view(v, mask), list(outs)), "n_items": int(mask.sum()),
                                        "excluded_relations": list(AMBIGUOUS_RELATIONS)}
    else:
        res["unambiguous_relations"] = {"status": "undefined", "reason": f"only {int(mask.sum())} items"}
    return res


def analyze_model(r):
    base = r.uout["test"]
    tb = r.tiebreak["test"]
    margin = r.margin["test"]
    outs = all_test_outputs(r)
    strata, grid, deciles = test_strata(r)
    res = {"conditions": {}, "strata": {}, "grid": {}, "deciles": {}}
    for name, o in outs.items():
        res["conditions"][name] = condition_metrics(r, o, base)
        res["strata"][name] = stratum_scores(o.states, strata, base.states)
        res["grid"][name] = stratum_scores(o.states, grid, base.states)
        res["deciles"][name] = stratum_scores(o.states, deciles)
    v = view_popqa(r)
    hyp = {}
    draws = random_draw_names(outs)
    S = lambda name: res["conditions"][name]["abstention_score"] if name in res["conditions"] else float("nan")
    sur = lambda name: res["conditions"][name]["surplus_vs_threshold"]["lam1"] if name in res["conditions"] else float("nan")

    if draws:
        dist = {"S1": [S(d) for d in draws], "surplus": [sur(d) for d in draws],
                "selectivity_lift": [res["conditions"][d]["selectivity_lift"] for d in draws],
                "recovered_rate": [res["conditions"][d]["recovered_rate"] for d in draws]}
        res["random_distribution"] = {"n_draws_surplus_defined": int(np.isfinite(dist["surplus"]).sum())}
        for metric, vals in dist.items():
            res["random_distribution"][metric] = RandomDirectionPlacebo.summarize_distribution(vals)
            for name in PERCENTILE_CONDITIONS:
                if name in res["conditions"]:
                    learned = {"S1": S(name), "surplus": sur(name),
                               "selectivity_lift": res["conditions"][name]["selectivity_lift"],
                               "recovered_rate": res["conditions"][name]["recovered_rate"]}[metric]
                    res["random_distribution"][f"{metric}|{name}"] = RandomDirectionPlacebo.summarize_distribution(vals, learned)
                    fin = [x for x in vals if math.isfinite(x)]
                    res["random_distribution"][f"{metric}|{name}"]["percentile"] = (
                        empirical_percentile(learned, fin) if fin and math.isfinite(learned) else float("nan"))
        D = np.stack([outs[d].states for d in draws])
        for name in PERCENTILE_CONDITIONS:
            if name in outs:
                X = outs[name].states
                res["random_distribution"][f"learned_minus_draw_mean|{name}"] = ST.paired_bootstrap(
                    lambda idx, X=X: abstention_score(X[idx]) - np.mean([abstention_score(dd[idx]) for dd in D]), len(X))

    # H1 (analysis_core: valid-resample rule, common-grid AURC)
    if "knowledge_dir_steer" in outs:
        hyp["H1a"] = AC.h1a(v)
        hyp["H1a_read"] = AC.h1a(v, "read_dir_steer") if "read_dir_steer" in outs else None
        hyp["H1b"] = AC.h1b(v)
        # final F3-F5: dose-matched arms (uniform push, D_ku decomposition, sign-aligned random draws)
        arms = {n: bool(o.extra.get("matched_on_dev")) for n, o in outs.items() if n.startswith(DOSE_MATCHED_ARMS)}
        hyp["H1b_final"] = AC.h1b_dose_matched(v, arms)
        hyp["H1b_final"]["dev_matching"] = r.frozen.get("placebos")
        hyp["H1b_final"]["ku_decomposition"] = r.extras.get("ku_decomposition", {}).get(r.frozen["L_star"])
        hyp["H1a_sweep"] = h1a_sweep(r, v, outs)
        hyp["H1d"] = AC.h1d(v)
        hyp["H1f"] = AC.h1f(v, hyp["H1a"])
        hyp["always_abstain_reference"] = {n: S(n) for n in ("always_abstain", "unsteered", "confidence_threshold_margin",
                                                               "knowledge_dir_steer") if n in res["conditions"]}
        if "always_abstain" in outs:
            X, Z = outs["knowledge_dir_steer"].states, outs["always_abstain"].states
            hyp["always_abstain_reference"]["knowledge_minus_always_abstain"] = {
                "diff": S("knowledge_dir_steer") - S("always_abstain"),
                **ST.paired_bootstrap(lambda idx: abstention_score(X[idx]) - abstention_score(Z[idx]), len(X))}
    if "margin_gated_steer_knowledge" in outs:
        hyp["H1c_registered_cstar"] = AC.h1c(v)
    if "margin_gated_steer_knowledge_tuned" in outs:
        hyp["H1c"] = {**AC.h1c(v, "margin_gated_steer_knowledge_tuned"), "c_gate": r.frozen.get("c_gate"),
                      "c_star": r.frozen.get("c_star"), "dev_tuning": r.frozen.get("gated_tuning"),
                      "definition": "final F2: gated steering at its dev-tuned strength"}
    if "margin_gated_steer_wq" in outs:
        hyp["H4d_gated"] = AC.h1c(v, "margin_gated_steer_wq")
    for name in ("majority_vote_k8", "best_of_k8"):
        if name in outs:
            hyp[f"baseline_{name}"] = {"surplus": sur(name), "S1": S(name),
                                       "vs_threshold_operating_point": S(name) - S("confidence_threshold_margin")}
    dose = r.extras.get("teacher_forced_dose_response")
    if dose:
        # final F6: statistics on the dev-clean grid; comparison directions on the same grid
        hyp["H1e"] = {k: {kk: vv for kk, vv in val.items() if kk not in ("margins", "c_flip", "m0")}
                      for k, val in dose.items() if isinstance(val, dict) and "spearman_m0_cflip" in val}
        hyp["H1e"]["grid"] = dose.get("grid")
        hyp["H1e"]["clean_doses_dev"] = dose.get("clean_doses_dev")
        hyp["H1e"]["generation_flip_agreement"] = dose.get("generation_flip_agreement")
        kc = dose["knowledge_dir"]["c_flip"]
        hyp["H1e"]["margin_ordering_spearman_vs_knowledge"] = {
            k: (spearman(kc, dose[k]["c_flip"]) if np.nanstd(dose[k]["c_flip"]) > 0 else float("nan"))
            for k in dose if isinstance(dose[k], dict) and "c_flip" in dose[k] and k != "knowledge_dir"}
        hyp["H1e"]["margins"] = {k: dose[k]["margins"] for k in dose if isinstance(dose[k], dict) and "margins" in dose[k]}
        hyp["H1e"]["item_indices"] = dose.get("item_indices")

    # H2 (amendment A12: every ablation variant; H2c validity rule; final F7: one L_ref, two-sided rule)
    L = r.frozen["L_star"]
    h2a = {}
    null = random_cosine_null(r.dirs["refusal"][L], N_COSINE_NULL, COSINE_NULL_SEED)
    for k in ("knowledge", "read", "wq", "idk_caa", "force"):
        if L in r.dirs.get(k, {}):
            h2a[f"cos_refusal_{k}"] = cosine(r.dirs["refusal"][L], r.dirs[k][L])
    # final F7: the same cosines at every candidate layer (run 2 reported L* only)
    h2a["per_layer"] = {str(Lc): {f"cos_refusal_{k}": cosine(r.dirs["refusal"][Lc], r.dirs[k][Lc])
                                  for k in ("knowledge", "idk_caa", "force") if Lc in r.dirs.get(k, {})}
                        for Lc in r.cand if Lc in r.dirs["refusal"]}
    h2a["null_abs_mean"] = float(np.abs(null).mean())
    h2a["null_abs_p95"] = float(np.percentile(np.abs(null), 95))
    h2a["max_abs_cos_refusal_idk_caa_any_layer"] = max((abs(x.get("cos_refusal_idk_caa", 0.0)) for x in h2a["per_layer"].values()),
                                                       default=float("nan"))
    h2a["max_abs_cos_refusal_knowledge_any_layer"] = max((abs(x.get("cos_refusal_knowledge", 0.0)) for x in h2a["per_layer"].values()),
                                                         default=float("nan"))
    hyp["H2a"] = h2a
    variants = r.side.get("refusal_control", {}).get("variants", {})
    for vname, info in variants.items():
        tag = "" if vname == "refusal_dir_ablation" else f"[{vname}]"
        hyp[f"H2b{tag}"] = AC.h2b(v, vname, info.get("passed"))
        hyp[f"H2b{tag}"]["L_ref"] = info.get("L_ref")
        hyp[f"H2c{tag}"] = AC.h2c_final(v, ablation=vname, combo=f"{vname}_plus_knowledge_steer",
                                        positive_control_passed=info.get("passed"))
        prefix = "random_dir_ablation_s" if vname == "refusal_dir_ablation" else vname.replace("refusal_dir", "random_dir") + "_s"
        rnd = random_draw_names(outs, prefix)
        if rnd:
            hyp[f"H2b{tag}"]["random_ablation_abstention_change_pts"] = [
                100 * (float((outs[n].states == A_).mean()) - float((base.states == A_).mean())) for n in rnd]
    # amendment A12 primary arm: registered directional if it passes the positive control without damaging
    # the model; otherwise mean-ablation (all layers); otherwise mean-ablation at L_ref
    primary = None
    for vname in ("refusal_dir_ablation", "refusal_dir_mean_ablation", "refusal_dir_mean_ablation_single_layer"):
        tag = "" if vname == "refusal_dir_ablation" else f"[{vname}]"
        h = hyp.get(f"H2b{tag}")
        if h and h.get("positive_control_passed") and not h.get("ablation_damages_model"):
            primary = vname
            break
    hyp["H2_primary_variant"] = primary or "none (every variant fails its positive control or damages the model)"
    br = r.side.get("benign_refusal_rate")
    if br:
        from scipy.stats import fisher_exact
        nb = int(r.side.get("benign_n_prompts", 100))
        rnd = [val for k, val in br.items() if k.startswith("random_dir_placebo_s")]
        h2d = {}
        for name in STEERING_CONDITIONS + ["read_dir_steer"]:
            if name in br:
                a1, a0 = int(round(br[name] * nb)), int(round(br["unsteered"] * nb))
                h2d[name] = {"benign_refusal": br[name], "vs_unsteered_pts": 100 * (br[name] - br["unsteered"]),
                             "n_prompts": nb, "fisher_p_vs_unsteered": float(fisher_exact([[a1, nb - a1], [a0, nb - a0]])[1]),
                             "above_random_p95": bool(rnd and br[name] > np.percentile(rnd, 95))}
        hyp["H2d"] = h2d
    if "refusal_dir_steer" in outs and "knowledge_dir_steer" in outs:
        Xr, Xk = outs["refusal_dir_steer"].states, outs["knowledge_dir_steer"].states
        fr, fk = ST.surplus_stat(Xr, base.states, margin, tb), ST.surplus_stat(Xk, base.states, margin, tb)
        b = ST.paired_bootstrap(lambda idx: fr(idx) - fk(idx), len(Xr))
        hyp["H2e"] = {"refusal_minus_knowledge_surplus": sur("refusal_dir_steer") - sur("knowledge_dir_steer"), **b,
                      "tost_equivalent_3pts": bool(b["defined"] and b["ci90_low"] > -3 and b["ci90_high"] < 3)
                      if b["defined"] else None,
                      "lift_refusal": res["conditions"]["refusal_dir_steer"]["selectivity_lift"],
                      "lift_knowledge": res["conditions"]["knowledge_dir_steer"]["selectivity_lift"],
                      "refusal_abstention_subtypes": res["conditions"]["refusal_dir_steer"]["abstention_subtypes"]}

    # H3 (answered items)
    try:
        sig = knowledge_signals(r)
        res["knowledge_signals"] = sig
        internal = [k for k in ("margin", "ku_projection", "read_projection", "sep_probe", "correctness_probe")
                    if f"popularity+{k}" in sig]
        hyp["H3a"] = sig["_H3a"]
        hyp["H3b"] = {k: {"within_decile_auroc": sig[k]["within_decile_auroc"], "ci": sig[k]["within_decile_ci"],
                          "status": sig[k]["within_decile_status"]} for k in internal}
        hyp["H3c"] = {k: {"over_popularity": sig[f"popularity+{k}"]["incremental_auroc"],
                          "over_popularity_ci": sig[f"popularity+{k}"]["incremental_ci"],
                          "over_popularity_relation": sig[f"popularity+relation+{k}"]["incremental_auroc"],
                          "over_popularity_relation_ci": sig[f"popularity+relation+{k}"]["incremental_ci"]} for k in internal}
        hyp["H3d"] = {k: {"auroc": sig[k]["popular_consistent_error_auroc"], "ci": sig[k]["popular_consistent_error_ci"]}
                      for k in internal}
        hyp["H3_final"] = sig["_H3_final"]
        js = r.uout["test"].extra.get("judge_states")
        if js is not None:            # final F9: H3 with judge-labelled test correctness
            sj = knowledge_signals(r, test_states=js)
            hyp["H3_final_judge_labels"] = sj["_H3_final"]
            res["knowledge_signals_judge_labels"] = {k: {kk: vv for kk, vv in val.items() if kk != "per_decile"}
                                                     for k, val in sj.items() if isinstance(val, dict) and not k.startswith("_")}
        res["split_randomness"] = split_randomness(r)
    except Exception as e:  # analysis failure must not hide the condition results
        print(f"ANALYSIS_FAILED: knowledge_signals model={r.name} {type(e).__name__}: {e}", flush=True)
        traceback.print_exc(file=sys.stderr)
    op = outs.get("confidence_threshold_margin")
    if op is not None and "popularity_gate" in outs:
        pop = r.signal("popularity", "test")
        pm = match_coverage(base.states, pop, tb, int((op.states != A_).sum()))
        if pm is not None:
            b = ST.paired_bootstrap(lambda idx: abstention_score(op.states[idx]) - matched_score(
                base.states[idx], pop[idx], tb[idx], int((op.states[idx] != A_).sum())), len(base))
            hyp["H3e"] = {"threshold_minus_popularity_gate": abstention_score(op.states) - abstention_score(pm), **b}

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
    wc = lambda n: wc_by_status(outs[n].states)["overall"]["W_to_C"]
    if "knowledge_dir_steer" in outs:
        d = wc_by_status(outs["knowledge_dir_steer"].states)
        so = d["stored_outcompeted"]
        k_so = int(round(so["W_to_C"] * so["n_wrong"])) if so["n_wrong"] else 0
        hyp["H4a"] = {**d, "so_W_to_C_lt_5pct": bool(so["W_to_C"] < 0.05),
                      "so_W_to_A_ge_3x_W_to_C": bool(so["W_to_A"] >= 3 * so["W_to_C"]),
                      "so_W_to_C_wilson95": wilson(k_so, so["n_wrong"]),
                      "so_W_to_C_upper_lt_5pct": bool(so["n_wrong"] and wilson(k_so, so["n_wrong"])[1] < 0.05)}
    perm = random_draw_names(outs, "label_permuted_wq_dir_steer_s")
    wc_perm = [wc(n) for n in perm]
    # final F10: H4b at the registered shared site (norm-matched random draws, label-permuted D_wq) and
    # at D_wq's own dev-selected dose (nulls at that same norm)
    for name, rnd_names, perm_names, key in (
            ("within_question_dir_steer", draws, perm, "H4b"),
            ("within_question_dir_steer_own", random_draw_names(outs, "wq_own_random_s"),
             random_draw_names(outs, "label_permuted_wq_own_s"), "H4b_own_dose")):
        if name not in outs:
            continue
        d = wc_by_status(outs[name].states)
        wr, wp = [wc(n) for n in rnd_names], [wc(n) for n in perm_names]
        hyp[key] = {**d, "random_p95": float(np.percentile(wr, 95)) if wr else float("nan"), "random_draws": wr,
                    "label_permuted_p95": float(np.percentile(wp, 95)) if wp else float("nan"),
                    "label_permuted_draws": wp,
                    "above_both_nulls": bool(wr and wp and d["overall"]["W_to_C"] > max(np.percentile(wr, 95), np.percentile(wp, 95))),
                    "so_minus_missing": d["stored_outcompeted"]["W_to_C"] - d["missing"]["W_to_C"],
                    "S1": S(name), "underpowered": r.gray["info"]["underpowered"]}
        if key == "H4b_own_dose":
            hyp[key]["dose"] = r.frozen.get("wq_own")
            hyp[key]["dev_sweep"] = r.dev.get("wq_own_sweep")
    if "first_token_ban_regenerate" in outs:
        ft = outs["first_token_ban_regenerate"]
        banned = ft.extra["banned"]
        h4c = {}
        prior_rel = defaultdict(list)
        for pool in r.data["pools"].values():
            for it in pool:
                if it.aliases:
                    prior_rel[it.prop].append(it.aliases[0])
        prior = AC.relation_answer_prior(prior_rel)
        titems = r.items["test"]
        for s in ("stored_outcompeted", "missing", "known"):
            w = (base.states == W_) & (ks == s) & banned
            h4c[f"recovery_{s}"] = float((ft.states[w] == C_).mean()) if w.any() else float("nan")
            h4c[f"n_wrong_{s}"] = int(w.sum())
            h4c[f"recovery_{s}_wilson95"] = wilson(int((ft.states[w] == C_).sum()), int(w.sum()))
            # final F10: chance level of a second guess drawn from the relation's answer prior
            ch = [AC.chance_second_guess(titems[i].prop, titems[i].aliases, titems[i].aliases[0] if titems[i].aliases else "",
                                         base.texts[i], prior) for i in np.flatnonzero(w)]
            ch = [x for x in ch if math.isfinite(x)]
            h4c[f"chance_{s}"] = float(np.mean(ch)) if ch else float("nan")
        h4c["failed_knockout_rate"] = float(ft.extra["failed_knockout"][banned].mean()) if banned.any() else float("nan")
        h4c["unbanned_rows_identical"] = ft.extra["unbanned_rows_identical"]
        ans_b = (base.states != A_) & banned
        q_hi = np.nanpercentile(margin[ans_b], 200 / 3) if ans_b.any() else np.inf
        top = ans_b & (margin >= q_hi)
        wrong_a1 = base.states == W_
        h4c["knockout_detection_auroc_A2"] = auroc(wrong_a1[top], ft.ans_logprob[top])
        h4c["knockout_detection_auroc_A1"] = auroc(wrong_a1[top], -base.ans_logprob[top])
        h4c["A2_minus_A1_auroc"] = h4c["knockout_detection_auroc_A2"] - h4c["knockout_detection_auroc_A1"]
        hyp["H4c"] = h4c
        items = r.items["test"]
        by_rel = defaultdict(list)
        for pool in r.data["pools"].values():
            for it in pool:
                if it.aliases:
                    by_rel[it.prop].append(it.aliases[0])
        hyp["gated_knockout_exploratory"] = AC.gated_knockout(
            v, aliases=[it.aliases for it in items], relations=[it.prop for it in items],
            first_aliases=[it.aliases[0] if it.aliases else "" for it in items], prior=AC.relation_answer_prior(by_rel))
    for name in ("within_question_dir_steer", "margin_gated_steer_wq", "within_question_dir_steer_own"):
        if name in outs:
            h = {"vs_unsteered": S(name) - abstention_score(base.states)}
            if op is not None:
                h["vs_threshold_operating_point"] = S(name) - abstention_score(op.states)
            hyp.setdefault("H4d", {})[name] = h

    # dose trend of abstention (Cochran-Armitage) over the dev-clean doses only (final F6: run 2's z = -53
    # was driven by the degenerate doses)
    names = {c: n for c, n in ladder_names(r).items() if c in set(r.frozen.get("clean_doses", [])) and n in outs}
    if len(names) >= 2:
        succ = [int((outs[n].states == A_).sum()) for n in names.values()]
        z, p = cochran_armitage(succ, [len(base)] * len(names), list(names))
        res["dose_trend"] = {"knowledge_dir_steer": {"metric": "abstention", "doses": list(names), "counts": succ,
                                                     "z": z, "p": p, "clean_doses_only": True}}

    res["label_sensitivity"] = label_sensitivity(r, outs, v)
    res["hypotheses"] = hyp
    res["seeds"] = seed_analysis(r)
    res["entity_questions"] = analyze_transfer(r, "eq")
    res["triviaqa"] = analyze_transfer(r, "tqa")
    res["side_effects"] = {k: val for k, val in r.side.items() if k != "instr_unsteered"}
    if "perplexity_ratio" in r.side:
        res["side_effects"]["perplexity_flag_gt_1.05"] = {k: bool(val > 1.05) for k, val in r.side["perplexity_ratio"].items()}
    res["diagnostics"] = {"directions": r.diag["directions"], "integrity": r.integrity,
                          "sep_label_check": r.extras.get("sep_label_check")}
    return res


def analyze_transfer(r, split):
    store = r.eq_out if split == "eq" else r.tqa_out
    if not store:
        return {}
    full, sub, sub_idx = store["full"], store["subset"], store["subset_idx"]
    tau1 = r.frozen["tau"]["margin" if split == "eq" else "tqa_margin"][LAMBDA_PRIMARY]
    v = view_transfer(r, split, full, tau1)
    base = full["unsteered"]
    rel = np.array([it.prop for it in r.items[split]])
    out = {"full": {}, "subset": {}}
    for name, o in full.items():
        m = summary_metrics(o.states, o.degenerate, o.truncated, o.n_tokens)
        m.update(transition_metrics(base.states, o.states))
        m.update(ST.net_transitions(base.states, o.states))
        m["surplus_vs_threshold"] = ST.surplus_point(o.states, base.states, v.margin, v.tiebreak)
        m["surplus_per_answered"] = ST.surplus_per_answered(o.states, base.states, v.margin, v.tiebreak)
        m["selective_accuracy_at"] = {f"{int(c * 100)}": selective_accuracy_at(o.states, o.confidence, v.tiebreak, c)
                                      for c in COVERAGE_POINTS}
        if split == "eq":
            ks = r.kstatus["eq"]
            m["per_relation"] = {rl: abstention_score(o.states[rel == rl]) for rl in np.unique(rel)}
            m["per_knowledge_status"] = {s: abstention_score(o.states[ks == s]) if (ks == s).any() else float("nan")
                                         for s in ("known", "stored_outcompeted", "missing")}
        out["full"][name] = m
    out["H1a"] = AC.h1a(v)
    if "margin_gated_steer_knowledge" in full and "confidence_threshold_margin" in full:
        out["H1c"] = AC.h1c(v)
    if split == "tqa":
        out["H1d"] = AC.h1d(v)
        if "dose_matched_placebo_force" in full:           # final F3: placebo at the high-coverage point
            out["H1d_uniform_push"] = AC.h1d(v, "dose_matched_placebo_force")
        out["coverage_regime"] = {"unsteered_coverage": 1 - float((base.states == A_).mean()),
                                  "selective_accuracy_unsteered_at": {f"{int(c * 100)}": selective_accuracy_at(
                                      base.states, v.margin, v.tiebreak, c) for c in COVERAGE_POINTS}}
        if "judge_states" in base.extra:
            vj = _alt_view(v, base.extra["judge_states"], {k: o.extra.get("judge_states", o.states) for k, o in full.items()})
            out["H1d_judge"] = AC.h1d(vj)
    if not sub:
        out["_view"] = v
        return out
    sbase = sub["unsteered"]
    draws = random_draw_names(sub)
    for name, o in sub.items():
        out["subset"][name] = {"abstention_score": abstention_score(o.states), **transition_metrics(sbase.states, o.states)}
    if draws:
        vals = [out["subset"][d]["abstention_score"] for d in draws]
        out["subset_random_distribution"] = RandomDirectionPlacebo.summarize_distribution(vals)
        for name in PERCENTILE_CONDITIONS:
            if name in full:
                out["subset_random_distribution"][f"percentile|{name}"] = empirical_percentile(
                    abstention_score(full[name].states[sub_idx]), vals)
    out["_view"] = v
    return out


# ----------------------------------------------------------------------------- pooled printing
PRINT_ALIASES = {"confidence_prompt": "confidence_prompt_t050", "margin_gated_steer": "margin_gated_steer_knowledge_tuned",
                 "dose_matched_placebo": "dose_matched_placebo_force", "sample_vote": "majority_vote_k8",
                 "within_question_dir_steer_own": "within_question_dir_steer_own"}
EXPANDED = {"confidence_prompt": ["confidence_prompt", "confidence_prompt_t050", "confidence_prompt_t075"],
            "margin_gated_steer": ["margin_gated_steer", "margin_gated_steer_knowledge", "margin_gated_steer_knowledge_tuned",
                                   "margin_gated_steer_wq"],
            "knowledge_dir_steer_dose_ladder": [f"knowledge_dir_steer_c{c:g}" for c in C_GRID],
            "dose_matched_placebo": ["dose_matched_placebo_force", "knowledge_dir_steer_within_relation",
                                     "knowledge_dir_steer_between_relation"],
            "sample_vote": ["majority_vote_k8", "best_of_k8"]}


def printed_conditions(registry_names):
    names = []
    for n in registry_names:
        names += EXPANDED.get(n, [n])
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
    return ST.paired_bootstrap(lambda idx: np.mean([abstention_score(a[idx]) for a in arrs]), len(arrs[0]))["sd"]


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
        for tag, block in (("entityq", "entity_questions"), ("triviaqa", "triviaqa")):
            e = a.get(block, {}).get("full", {}).get(key)
            if e:
                print(f"condition={name} regime={r.name}.{tag} abstention_score: {fmt(e['abstention_score'])}")
    for lam in LAMBDAS:
        st = np.concatenate([_states_for(r, name, lam) for r, _ in present])
        print(f"condition={name} lam={lam:g} abstention_score: {fmt(abstention_score(st, lam))}")
    c, w, n = (states == C_).sum(), (states == W_).sum(), len(states)
    print(f"condition={name} accuracy_answered: {fmt(c / (c + w) if c + w else float('nan'))}")
    print(f"condition={name} abstention_rate: {fmt((states == A_).mean())}")
    print(f"condition={name} wrong_rate: {fmt(w / n)}")
    print(f"condition={name} aurc: {fmt(np.mean([a['conditions'][key]['aurc_own_range'] for _, a in present]))}")
    base = np.concatenate([r.uout["test"].states for r, _ in present])
    tm = transition_metrics(base, states)
    print(f"condition={name} recovered_rate: {fmt(tm['recovered_rate'])}")
    print(f"condition={name} harmed_rate: {fmt(tm['harmed_rate'])}")
    # amendment A3: gains / losses / net, surplus and surplus per answered item, common-grid AURC
    nt = ST.net_transitions(base, states)
    print(f"condition={name} gains_items: {nt['gains']['items']}")
    print(f"condition={name} losses_items: {nt['losses']['items']}")
    print(f"condition={name} gains_points: {fmt(nt['gains']['points'])}")
    print(f"condition={name} losses_points: {fmt(nt['losses']['points'])}")
    print(f"condition={name} net_points: {fmt(nt['net_points'])}")
    for r, a in present:
        cm = a["conditions"][key]
        print(f"condition={name} regime={r.name} surplus_vs_threshold: {fmt(cm['surplus_vs_threshold']['lam1'])}")
        print(f"condition={name} regime={r.name} surplus_per_answered: {fmt(cm['surplus_per_answered'])}")
        av = cm["aurc_vs_unsteered"]
        print(f"condition={name} regime={r.name} aurc_common_grid_minus_unsteered: "
              f"{fmt(av.get('diff')) if av.get('status') == 'defined' else 'undefined'}")
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
        val = abstention_score(st)
        vals.append(val)
        print(f"condition=random_dir_placebo seed={k} abstention_score: {fmt(val)}")
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
    aurcs = [analyses[i]["conditions"][d]["aurc_own_range"] for i, r in enumerate(runners) if r in present for d in draws]
    print(f"condition=random_dir_placebo aurc: {fmt(np.mean(aurcs))}")
    print(f"condition=random_dir_placebo recovered_rate: {fmt(np.mean([t['recovered_rate'] for t in tms]))}")
    print(f"condition=random_dir_placebo harmed_rate: {fmt(np.mean([t['harmed_rate'] for t in tms]))}")
    print(f"condition=random_dir_placebo net_points: {fmt(agg(lambda s: ST.net_transitions(base, s)['net_points']))}")
    for r, a in zip(runners, analyses):
        if r in present:
            print(f"condition=random_dir_placebo regime={r.name} n_draws_surplus_defined: "
                  f"{a.get('random_distribution', {}).get('n_draws_surplus_defined')}")
    for name in PERCENTILE_CONDITIONS + ["margin_gated_steer_knowledge", "margin_gated_steer_wq"]:
        if all(name in r.test_out for r in present):
            learned = abstention_score(np.concatenate([r.test_out[name].states for r in present]))
            print(f"condition={name} random_dir_percentile: {fmt(empirical_percentile(learned, vals))}")
    return mean, std, vals


def print_seed_block(runners, analyses):
    """Amendment A8: pooled per-seed scores of the seed-replicated arms, then across-seed mean/SD."""
    summary = {}
    for name in SEEDED_CONDITIONS:
        if name == "unsteered":
            continue
        per = []
        for s in range(N_SEEDS):
            arrs = []
            for r in runners:
                so = seeded_outputs(r)
                if s in so and name in so[s]:
                    arrs.append(so[s][name].states)
            if len(arrs) == len(runners) and arrs:
                v = abstention_score(np.concatenate(arrs))
                per.append(v)
                print(f"condition={name} seed={s} abstention_score: {fmt(v)}")
        if per:
            m, sd = float(np.mean(per)), float(np.std(per, ddof=1)) if len(per) > 1 else float("nan")
            print(f"condition={name} abstention_score_seed_mean: {fmt(m)}")
            print(f"condition={name} abstention_score_seed_std: {fmt(sd)}")
            summary[name] = (m, sd, len(per))
    for r, a in zip(runners, analyses):
        chk = a.get("seeds", {}).get("identical_check", {})
        print(f"SEED_CHECK: model={r.name} identical_pairs={chk.get('n_issues')}")
        for iss in chk.get("issues", []):
            print(f"WARNING: IDENTICAL_PER_SEED model={r.name} {iss['a']} == {iss['b']} kind={iss['kind']}")
    return summary


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
    seed_summary = print_seed_block(runners, analyses)

    # coverage-matched threshold lines (per model; pooled only if every model is matched)
    for target in STEERING_CONDITIONS + ["read_dir_steer"]:
        vals = []
        for r in runners:
            if target in r.test_out:
                k = int((r.test_out[target].states != A_).sum())
                val = matched_score(r.uout["test"].states, r.margin["test"], r.tiebreak["test"], k)
                vals.append(val)
                print(f"condition=confidence_threshold_margin match={target} regime={r.name} abstention_score: "
                      f"{fmt(val) if math.isfinite(val) else 'undefined'}")
        if vals:
            print(f"condition=confidence_threshold_margin match={target} abstention_score: "
                  f"{fmt(np.mean(vals)) if all(math.isfinite(x) for x in vals) else 'undefined'}")
    rvals = []
    for r in runners:
        for d in random_draw_names(r.test_out):
            k = int((r.test_out[d].states != A_).sum())
            rvals.append(matched_score(r.uout["test"].states, r.margin["test"], r.tiebreak["test"], k))
    if rvals:
        n_def = int(np.isfinite(rvals).sum())
        print(f"condition=confidence_threshold_margin match=random_dir_placebo abstention_score: "
              f"{fmt(np.nanmean(rvals)) if n_def >= 0.95 * len(rvals) else 'undefined'} n_defined={n_def}/{len(rvals)}")

    # knowledge signals (answered items, amendment A6)
    sig_names = ["margin", "ku_projection", "read_projection", "sep_probe", "correctness_probe", "popularity",
                 "popularity_surprisal_logreg", "relation_only", "popularity+relation", "popularity+margin",
                 "popularity+ku_projection", "popularity+sep_probe", "popularity+correctness_probe",
                 "popularity+relation+margin", "popularity+relation+correctness_probe"]
    for s in sig_names:
        vals = [(r.name, a["knowledge_signals"][s]) for r, a in zip(runners, analyses)
                if "knowledge_signals" in a and s in a["knowledge_signals"]]
        if vals:
            print(f"condition={s} auroc_known: {fmt(np.mean([v['auroc_known'] for _, v in vals]))}")
            for m, v in vals:
                print(f"condition={s}@{m} auroc_known: {fmt(v['auroc_known'])}")
                print(f"condition={s}@{m} within_decile_auroc: {fmt(v['within_decile_auroc'])}")

    cos = [(r.name, a["hypotheses"]["H2a"].get("cos_refusal_idk_caa")) for r, a in zip(runners, analyses)
           if "H2a" in a.get("hypotheses", {})]
    cos = [(m, v) for m, v in cos if v is not None]
    if cos:
        print(f"cosine_idk_refusal: {fmt(np.mean([v for _, v in cos]))}")
        for m, v in cos:
            print(f"condition={m} cosine_idk_refusal: {fmt(v)}")

    for r, a in zip(runners, analyses):
        for name in r.ablation_out:
            m = a["conditions"][name]
            print(f"ABLATION: model={r.name} {name} abstention_score={fmt(m['abstention_score'])} "
                  f"abstention_rate={fmt(m['abstention_rate'])} recovered_rate={fmt(m['recovered_rate'])} "
                  f"surplus_vs_threshold={fmt(m['surplus_vs_threshold']['lam1'])} degenerate_rate={fmt(m['degenerate_rate'])} "
                  f"net_points={fmt(m['net_points'])} surplus_per_answered={fmt(m['surplus_per_answered'])}")

    for name, (mean, sd) in summary.items():
        print(f"SUMMARY condition={name} metric=abstention_score mean={fmt(mean)} std={fmt(sd)}")
    for name, (mean, sd, n) in seed_summary.items():
        print(f"SUMMARY condition={name} metric=abstention_score_across_seeds mean={fmt(mean)} std={fmt(sd)} n_seeds={n}")
    paired = paired_lines(runners, analyses)

    for r, a in zip(runners, analyses):
        for hid, h in a.get("hypotheses", {}).items():
            if not isinstance(h, dict):
                continue
            flat = ST_flat(h)
            short = [(k, val) for k, val in flat.items() if isinstance(val, (int, float, bool, np.floating, np.bool_, str))
                     and not any(x in k for x in ("per_decile", "per_relation", "draw_surplus", "confusion"))]
            print(f"HYPOTHESIS: model={r.name} {hid} " + " ".join(
                f"{k}={(val.replace(' ', '_') if isinstance(val, str) else ST.fmt_stat(val))}" for k, val in short[:16]))
        for s, e in a.get("seeds", {}).get("per_seed", {}).items():
            h = e.get("H1a") or {}
            print(f"HYPOTHESIS: model={r.name} H1a_seed seed={s} L_star={e['site']} c_star={e['c_star']} "
                  f"surplus={ST.fmt_stat(h.get('surplus'))} ci_low={ST.fmt_stat(h.get('ci_low'), h.get('status'))} "
                  f"ci_high={ST.fmt_stat(h.get('ci_high'), h.get('status'))} perm_p={ST.fmt_stat(h.get('perm_p'))}")
        for block in ("entity_questions", "triviaqa"):
            for hid in ("H1a", "H1c", "H1d"):
                h = a.get(block, {}).get(hid)
                if h:
                    flat = ST_flat(h)
                    print(f"HYPOTHESIS: model={r.name} {hid}[{block}] " + " ".join(
                        f"{k}={(val.replace(' ', '_') if isinstance(val, str) else ST.fmt_stat(val))}"
                        for k, val in list(flat.items())[:12]))

    reg_means = [val[0] for k, val in summary.items() if k in registry_names]
    if len(reg_means) > 1 and np.allclose(reg_means, reg_means[0]):
        print(f"WARNING: DEGENERATE_METRICS all conditions have same mean={fmt(reg_means[0])}")
    return summary, paired


def ST_flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(ST_flat(v, f"{prefix}{k}."))
        elif isinstance(v, (list, tuple)):
            continue
        else:
            out[f"{prefix}{k}"] = v
    return out


def paired_lines(runners, analyses):
    """Amendment A1: bootstrap CIs (95%-valid rule), sign-flip p, Holm within registered families."""
    rows = AC.paired_rows([view_popqa(r) for r in runners], "popqa")
    for block, ds in (("entity_questions", "entity_questions"), ("triviaqa", "triviaqa")):
        vs = [a[block]["_view"] for a in analyses if a.get(block, {}).get("_view") is not None]
        if len(vs) == len(runners) and vs:
            rows += AC.paired_rows(vs, ds)
    AC.finalize_paired(rows)
    for row in rows:
        print(AC.format_paired(row))
    return rows


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
    """Registered audit file (300 outputs per model, stratified by condition and label)."""
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


def write_audit_h1a(runners):
    """Amendment A4: up to AUDIT_ROWS_PER_CELL rows per (model, H1a arm, transition vs unsteered)."""
    import audit
    rows = []
    for r in runners:
        items = r.items["test"]
        arms = audit.h1a_arms(r.uout["test"].states, r.margin["test"], r.tiebreak["test"],
                              r.test_out["knowledge_dir_steer"].states)
        texts = {"knowledge_dir_steer": r.test_out["knowledge_dir_steer"].texts,
                 "confidence_threshold_margin_matched": r.uout["test"].texts}
        strict = {"knowledge_dir_steer": r.test_out["knowledge_dir_steer"].strict_states(),
                  "confidence_threshold_margin_matched": r.uout["test"].strict_states()}
        rows += audit.stratified_rows(r.name, items, r.uout["test"].states, arms, texts, strict, seed=1)
    audit.write_rows(AUDIT_H1A_PATH, rows)
    return len(rows)


def per_item_records(name, split, seed, o, items, extra=None):
    for i, it in enumerate(items):
        rec = {"seed": seed, "split": split, "condition": name, "qid": it.qid, "state": int(o.states[i]),
               "outcome": STATE_NAMES[int(o.states[i])], "degenerate": bool(o.degenerate[i]),
               "truncated": bool(o.truncated[i]), "abstention_type": o.abst_type[i],
               "strict_correct": bool(o.strict_correct[i]), "confidence": float(o.confidence[i]),
               "confidence_kind": o.confidence_kind, "ans_logprob": float(o.ans_logprob[i]),
               "n_tokens": int(o.n_tokens[i]), "output": o.texts[i]}
        if "judge_states" in o.extra:
            rec["judge_state"] = int(o.extra["judge_states"][i])
        if extra:
            rec.update({k: (v[i] if hasattr(v, "__len__") and len(v) == len(items) else v) for k, v in extra.items()})
        yield rec


def write_per_item(runners):
    """Amendment A14: every per-item output of every condition, split and seed, plus the unsteered
    labels and the 8 samples, as gzip JSONL under outputs/per_item/<model>/."""
    PER_ITEM_DIR.mkdir(parents=True, exist_ok=True)
    written = {}
    for r in runners:
        d = PER_ITEM_DIR / r.name
        d.mkdir(parents=True, exist_ok=True)
        groups = {"test": [], "seeds": [], "transfer": [], "dev": [], "unsteered_labels": []}
        items = r.items["test"]
        meta = {"decile": [it.decile for it in items], "tercile": [it.tercile for it in items],
                "popularity": [it.log_pop for it in items], "relation": [it.prop for it in items],
                "knowledge_status": list(r.kstatus["test"]), "margin": r.margin["test"]}
        for name, o in all_test_outputs(r).items():
            groups["test"] += per_item_records(name, "test", 0, o, items, meta)
        for s, outs in r.seed_out.items():
            for name, o in outs.items():
                groups["seeds"] += per_item_records(name, "test", s, o, items)
        for split, store in (("eq", r.eq_out), ("tqa", r.tqa_out)):
            if store:
                for name, o in store["full"].items():
                    groups["transfer"] += per_item_records(name, split, 0, o, r.items[split],
                                                           {"relation": [it.prop for it in r.items[split]],
                                                            "margin": r.margin[split]})
                sub_items = [r.items[split][i] for i in store["subset_idx"]]
                for name, o in store["subset"].items():
                    groups["transfer"] += per_item_records(name, f"{split}_subset", 0, o, sub_items)
        for runner, s in [(r, 0)] + [(f, s) for s, f in r.seed_runs.items()]:
            for name, o in runner.dev_outputs.items():
                its = runner.items["dev"] if len(o) == len(runner.items["dev"]) else [runner.items["dev"][i] for i in runner.dev_subset]
                groups["dev"] += per_item_records(name, "dev", s, o, its)
            for split in ("train", "train_reserve", "dev"):
                if split in runner.uout:
                    lab = runner.labels.get(split, {})
                    samp = runner.samples.get(split)
                    ex = {"margin": runner.margin[split]} if split in runner.margin else {}
                    if samp is not None:
                        ex["samples"] = [[t for t, _, _ in ss] for ss in samp]
                        ex["semantic_entropy"] = lab.get("semantic_entropy")
                    groups["unsteered_labels"] += per_item_records("unsteered", split, s, runner.uout[split],
                                                                   runner.items[split], ex)
        lab = r.labels["test"]
        groups["unsteered_labels"] += per_item_records("unsteered_samples", "test", 0, r.uout["test"], items,
                                                       {"samples": [[t for t, _, _ in ss] for ss in r.samples["test"]],
                                                        "semantic_entropy": lab["semantic_entropy"]})
        for g, recs in groups.items():
            path = d / f"{g}.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as f:
                n = 0
                for rec in recs:
                    f.write(json.dumps(jsonable(rec), ensure_ascii=False) + "\n")
                    n += 1
            written[f"{r.name}/{g}"] = n
    return written


def write_judge_disagreements(runners):
    """Final F9: up to JUDGE_N_DISAGREEMENT_ROWS rows per model where the judge and the automatic label
    disagree, stratified by condition (round robin), with an empty human_label column. Rows of the H1-
    critical arms come first. A human label on these rows settles which labelling a verdict should
    follow when automatic and judge labels disagree."""
    rng = np.random.default_rng(3)
    first = ("unsteered", "knowledge_dir_steer", "margin_gated_steer_knowledge_tuned", "dose_matched_placebo_force")
    rows = []
    for r in runners:
        items = r.items["test"]
        cells = {}
        seen = set()
        for name, o in {"unsteered": r.uout["test"], **all_test_outputs(r)}.items():
            js = o.extra.get("judge_states")
            if js is None:
                continue
            for i in np.flatnonzero(js != o.states):
                key = (items[i].qid, o.texts[i])
                if key in seen:
                    continue
                seen.add(key)
                cells.setdefault(name, []).append(i)
        order = [n for n in first if n in cells] + sorted(n for n in cells if n not in first)
        for n in order:
            cells[n] = [cells[n][j] for j in rng.permutation(len(cells[n]))]
        picked, depth = [], 0
        while len(picked) < JUDGE_N_DISAGREEMENT_ROWS and any(depth < len(cells[n]) for n in order):
            for n in order:
                if depth < len(cells[n]) and len(picked) < JUDGE_N_DISAGREEMENT_ROWS:
                    picked.append((n, cells[n][depth]))
            depth += 1
        outs = {"unsteered": r.uout["test"], **all_test_outputs(r)}
        for n, i in picked:
            o, it = outs[n], items[i]
            rows.append([r.name, n, it.qid, it.prop, it.question, " | ".join(it.aliases), o.texts[i],
                         STATE_NAMES[int(o.states[i])], STATE_NAMES[int(o.extra["judge_states"][i])], ""])
    if not rows:
        return 0
    with open(JUDGE_DISAGREEMENTS_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "qid", "relation", "question", "gold_aliases", "output", "auto_label",
                    "judge_label", "human_label"])
        w.writerows(rows)
    return len(rows)


def write_results(runners, analyses, summary, paired, meta, verdict_table=None):
    out = {"meta": meta, "summary": summary, "paired": paired, "verdicts": verdict_table or {}, "models": {}}
    for r, a in zip(runners, analyses):
        a = {k: ({kk: vv for kk, vv in v.items() if kk != "_view"} if k in ("entity_questions", "triviaqa") and isinstance(v, dict)
                 else v) for k, v in a.items()}
        out["models"][r.name] = {"frozen_config": dict(r.frozen),
                                 "seed_frozen_configs": {s: dict(f.frozen) for s, f in r.seed_runs.items()},
                                 "tau_store": getattr(r, "tau_store", None), "integrity": r.integrity,
                                 "gray_zone": r.gray["info"] if hasattr(r, "gray") else None,
                                 "known_unknown": r.extras.get("known_unknown"),
                                 "dev": r.dev, "seed_dev": {s: f.dev for s, f in r.seed_runs.items()},
                                 "analysis": a, "per_item_dir": str(PER_ITEM_DIR / r.name)}
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(jsonable(out), f, ensure_ascii=False)
    print(f"wrote {RESULTS_PATH}")

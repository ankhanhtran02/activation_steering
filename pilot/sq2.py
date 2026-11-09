"""SQ2: does the inside of the model know about its mistakes when its outputs don't?

Disagreement questions: one model right in >= 8 of its 10 tries, the other in <= 2. They come from the
main pool (3,000 questions) and from a screening extension (~3,000 more questions that get only the 10
tries; the disagreement questions found there then get the full record). On these, for each model, two
predictors of whether the main answer is right are compared:
  outputs only      : confidence (margin, mean token log-prob) + answer consistency from the separate
                      8 tries (agreement, distinct answers, decline share);
  outputs + detector: the same plus the internal-state probe score.
Both are logistic regressions trained on answered main-pool questions: out-of-fold for main-pool
questions; fitted on the whole main pool for extension questions (which are never used for training).
The pooled effect is the mean of the two models' AUROC gains; its CI resamples disagreement questions
(both models' rows move together).
"""
import numpy as np

from pilot import config as PC
from pilot.metrics import bootstrap_ci, safe_auroc
from pilot.probes import OUTPUT_FEATURE_NAMES, crossfit_lr, fit_lr_score, full_probe, output_features
from scoring import A_, C_


def _disagree(ca, cb):
    return ((ca >= PC.SQ2_HI) & (cb <= PC.SQ2_LO)) | ((cb >= PC.SQ2_HI) & (ca <= PC.SQ2_LO))


def analyze(pool, shared, probes, ext=None):
    """ext (optional): {"items": extension disagreement items, "n_screened": int,
    "followup": {model: run_shared output on those items}}."""
    names = list(shared)
    folds = np.array([it.extra["fold"] for it in pool])
    out = {"models": names}
    if len(names) < 2:
        out["error"] = "SQ2 needs two models"
        return out
    a, b = names[:2]
    ca, cb = shared[a]["dis_correct"], shared[b]["dis_correct"]
    D = _disagree(ca, cb)
    dmain = np.flatnonzero(D)
    ext_items = (ext or {}).get("items", [])
    has_ext = bool(ext_items) and all(m in ext.get("followup", {}) for m in names[:2])
    n_ext = len(ext_items) if has_ext else 0
    q_items = [pool[i] for i in dmain] + (list(ext_items) if has_ext else [])
    out.update({"n_disagreement": int(len(q_items)), "n_disagreement_main": int(len(dmain)),
                "n_disagreement_extension": int(n_ext), "n_screened_extension": int((ext or {}).get("n_screened", 0)),
                "n_disagreement_by_dataset": {d: sum(it.dataset == d for it in q_items) for d in sorted({it.dataset for it in pool})},
                "n_a_knows_main": int(((ca >= PC.SQ2_HI) & (cb <= PC.SQ2_LO)).sum()),
                "n_b_knows_main": int(((cb >= PC.SQ2_HI) & (ca <= PC.SQ2_LO)).sum())})
    if (ext or {}).get("items") and not has_ext:
        out["extension_note"] = "extension follow-up missing for a model; extension not used"

    per, rows = {}, {}
    for m in names[:2]:
        sh = shared[m]
        y = sh["states"] == C_
        probe = probes[m]["oof"]
        pop = (sh["states"] != A_) & np.isfinite(probe)
        F_out = output_features(sh)
        F_full = np.column_stack([F_out, probe])
        s_out = crossfit_lr(F_out, y, pop, folds)
        s_full = crossfit_lr(F_full, y, pop, folds)
        yq, soq, sfq, prq, Fq = y[dmain], s_out[dmain], s_full[dmain], probe[dmain], F_out[dmain]
        okq = pop[dmain] & np.isfinite(soq) & np.isfinite(sfq)
        ext_info = None
        if has_ext:
            fu = ext["followup"][m]
            pr_e, ext_info = full_probe(sh["res_prompt"], sh["res_ans"], y, pop, fu["res_prompt"], fu["res_ans"])
            Fe = output_features(fu)
            so_e = fit_lr_score(F_out, y, pop, Fe)
            sf_e = fit_lr_score(F_full, y, pop, np.column_stack([Fe, pr_e]))
            ye = fu["states"] == C_
            ok_e = (fu["states"] != A_) & np.isfinite(so_e) & np.isfinite(sf_e)
            yq, soq, sfq = np.concatenate([yq, ye]), np.concatenate([soq, so_e]), np.concatenate([sfq, sf_e])
            prq, Fq, okq = np.concatenate([prq, pr_e]), np.concatenate([Fq, Fe]), np.concatenate([okq, ok_e])
        rows[m] = (yq, soq, sfq, okq)
        r = okq
        ev = {"n_rows": int(r.sum()), "n_rows_extension": int(r[len(dmain):].sum()), "n_right": int((yq & r).sum()),
              "n_wrong": int((~yq & r).sum()), "auroc_outputs": safe_auroc(yq[r], soq[r]),
              "auroc_outputs_plus_detector": safe_auroc(yq[r], sfq[r]), "auroc_detector_alone": safe_auroc(yq[r], prq[r]),
              "extension_probe": ext_info}
        ev["delta"] = ev["auroc_outputs_plus_detector"] - ev["auroc_outputs"]
        for j, nm in enumerate(OUTPUT_FEATURE_NAMES):
            ev[f"auroc_{nm}_alone"] = safe_auroc(yq[r], Fq[r, j])
        allr = pop & np.isfinite(s_out) & np.isfinite(s_full)
        ev["all_answered_main"] = {"n": int(allr.sum()), "auroc_outputs": safe_auroc(y[allr], s_out[allr]),
                                   "auroc_outputs_plus_detector": safe_auroc(y[allr], s_full[allr]),
                                   "auroc_detector_alone": safe_auroc(y[allr], probe[allr])}
        per[m] = ev
    out["per_model"] = per
    deltas = [per[m]["delta"] for m in names[:2]]
    out["delta_pooled"] = float(np.mean(deltas)) if all(np.isfinite(deltas)) else float("nan")

    nq = len(q_items)

    def delta_m(m, q):
        yq, so, sf, ok = rows[m]
        qq = q[ok[q]]
        return safe_auroc(yq[qq], sf[qq]) - safe_auroc(yq[qq], so[qq])

    na = {"ci_low": float("nan"), "ci_high": float("nan")}
    out["delta_pooled_ci"] = bootstrap_ci(lambda q: float(np.mean([delta_m(m, q) for m in names[:2]])), nq) if nq >= 10 else na
    for m in names[:2]:
        out["per_model"][m]["delta_ci"] = bootstrap_ci(lambda q: delta_m(m, q), nq) if nq >= 10 else na
    return out


def extension_disagreements(screens, model_names):
    """Indices (into the screened extension list) of disagreement questions, using only questions every
    model screened (a budget step may have halved the extension for a later model)."""
    a, b = model_names[:2]
    n = min(len(screens[a]["dis_correct"]), len(screens[b]["dis_correct"]))
    return np.flatnonzero(_disagree(screens[a]["dis_correct"][:n], screens[b]["dis_correct"][:n])), n

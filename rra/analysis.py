"""Phase F: fresh-test evaluation with the frozen configuration, paired bootstrap (2,000 resamples over fresh-test
questions, both models' rows together, percentile intervals; pooled = mean of the two models), the verdict,
probe diagnostics, strata, the replay set (diagnostic), per-question records, the audit sheet and the report.

    python -m rra.analysis --run-dir outputs_rra/run_<id>      (re-run the analysis only; CPU)
"""
import argparse
import csv
import traceback

import numpy as np

from pilot.metrics import safe_auroc, same_answer
from rra import arms as AR
from rra import probes as PR
from rra.verdicts import rra_verdict
from scoring import A_, C_, W_
from vesecs import stats as S
from vesecs.io import read_json, write_json, write_jsonl

LOW = -1e9          # score given to a null A2 when D(A2) is used as a predictor (it would be rejected by any check)


def _num(v):
    return v is not None and isinstance(v, (int, float)) and np.isfinite(v)


def main_table(outs, models, arms):
    table = {}
    for a in arms:
        rows = {}
        for m in models:
            o1, o3 = outs[1][m].get(a), outs[3][m].get(a)
            if o1 is not None:
                rows[m] = AR.arm_metrics(o1, o3=o3)
        if not rows:
            continue
        keys = next(iter(rows.values())).keys()
        pooled = {}
        for k in keys:
            vals = [r[k] for r in rows.values()]
            pooled[k] = float(np.mean(vals)) if len(rows) == len(models) and all(_num(v) for v in vals) else None
        table[a] = {"per_model": rows, "pooled": pooled}
    return table


def contrast_fn(outs, models, boot):
    def contrast(a, b, lam=1, masks=None):
        if not all(a in outs[lam][m] and b in outs[lam][m] for m in models):
            return None
        vec = {m: 100.0 * (AR.uvec(outs[lam][m][a].label, lam) - AR.uvec(outs[lam][m][b].label, lam)) for m in models}
        return boot.mean_contrast(vec, masks)
    return contrast


def contrasts(outs, models, boot, comp):
    c = contrast_fn(outs, models, boot)
    C1, C3 = comp.get("lam1", {}).get("arm", "R-check"), comp.get("lam3", {}).get("arm", "R-check")
    out = {"H": c("Oracle", "Oracle-noswitch"), "d_main": c("RRA", "B3"), "d_mech": c("RRA", C1),
           "d_main_lam3": c("RRA", "B3", 3), "d_mech_lam3": c("RRA", C3, 3), "H_lam3": c("Oracle", "Oracle-noswitch", 3)}
    for a in ("RRA-EU", "RRA+check", "R-all", "R-band", "R-check", "RRA-peer"):
        out[f"{a} - B3"] = c(a, "B3")
    for a in ("R-band", "R-check", "R-all", "RRA-EU"):
        out[f"RRA - {a}"] = c("RRA", a)
    out["RRA+check - RRA"] = c("RRA+check", "RRA")
    return {k: v for k, v in out.items() if v is not None}


def n_recoverable(r):
    return int(np.sum(r.answered & (r.a1_lab == W_) & ~r.a2_null & (r.a2_lab == C_)))


def probe_diagnostics(R, models, fz):
    out = {}
    for m in models:
        r = R[m]
        t1 = fz["models"][m]["arms"]["RRA"]["lam1"]["t1"]
        ans = r.answered
        flag = ans & ~AR.ge(r.s1, t1)
        y2 = ~r.a2_null & (r.a2_lab == C_)
        dA2 = np.where(r.a2_null, LOW, r.dA2)
        d = {"tau1_RRA": t1, "n_flagged": int(flag.sum()), "n_flagged_success": int((flag & y2).sum())}
        for nm, mk in (("flagged", flag), ("answered", ans)):
            nn = mk & ~r.a2_null
            d[nm] = {"auroc_s2": safe_auroc(y2[mk], r.s2[mk]), "auroc_D_A2": safe_auroc(y2[mk], dA2[mk]),
                     "auroc_D_A2_non_null": safe_auroc(y2[nn], r.dA2[nn]), "auroc_minus_s1": safe_auroc(y2[mk], -r.s1[mk]),
                     "n": int(mk.sum()), "n_success": int((mk & y2).sum())}
        d["calibration_test"] = {"p2": PR.calibration_curve(r.p2[ans], y2[ans]),
                                 "p1": PR.calibration_curve(r.p1[ans], (r.a1_lab == C_)[ans])}
        reasons = {}
        for x, a in zip(r.a2_null_reason, ans):
            if a:
                k = x or "usable"
                reasons[k] = reasons.get(k, 0) + 1
        d["A2"] = {"answered": int(ans.sum()), "reasons": reasons, "right": int((ans & y2).sum()),
                   "recoverable_A1_wrong_A2_right": n_recoverable(r),
                   "A1_right_A2_right": int((ans & (r.a1_lab == C_) & y2).sum())}
        out[m] = d
    return out


def strata_masks(pl, R, models):
    masks = {"all": {m: np.ones(R[m].N, bool) for m in models}}
    src = {m: np.array([it["source"] for it in R[m].items]) for m in models}
    masks["triviaqa"] = {m: src[m] == "tqa" for m in models}
    masks["popqa"] = {m: src[m] == "popqa" for m in models}
    if len(models) >= 2:
        a, b = models[:2]
        dis = (R[a].a1_lab == C_) != (R[b].a1_lab == C_)
        masks["disagreement"] = {m: dis for m in models}
    tries = {m: pl.pdir(m, "test_fresh") / "tries.json" for m in models}
    if all(p.exists() for p in tries.values()):
        kmin = int(pl.cfg["strata"]["confident_error_min_match"])
        ce = {}
        for m in models:
            tr = read_json(tries[m])
            r = R[m]
            ce[m] = np.array([bool(r.answered[i] and r.a1_lab[i] == W_ and
                                   sum(same_answer(t["text"], r.a1_text[i]) for t in tr[i]) >= kmin) for i in range(r.N)])
        masks["confident_errors"] = ce
    return masks


def strata(outs, models, boot, masks, comp):
    c = contrast_fn(outs, models, boot)
    C1 = comp.get("lam1", {}).get("arm", "R-check")
    out = {}
    for name, mk in masks.items():
        row = {"n": {m: int(mk[m].sum()) for m in models}}
        if any(mk[m].sum() == 0 for m in models):
            out[name] = row
            continue
        for a in ("B0", "B3", "RRA", C1, "Oracle"):
            if all(a in outs[1][m] for m in models):
                row[f"U1_{a}"] = boot.mean_contrast({m: 100.0 * AR.uvec(outs[1][m][a].label, 1) for m in models}, mk)
        row["d_main"] = c("RRA", "B3", 1, mk)
        row["d_mech"] = c("RRA", C1, 1, mk)
        row["H"] = c("Oracle", "Oracle-noswitch", 1, mk)
        out[name] = row
    return out


def evaluate(pl, split, models, fz, n_boot=None):
    """Outputs, main table and contrasts of every arm on one question set."""
    cfg = pl.cfg
    R = {m: pl.recs(m, split) for m in models}
    common = [q for q in R[models[0]].qids if all(q in set(R[x].qids) for x in models)]
    for m in models:
        if R[m].qids != common:
            pos = {q: i for i, q in enumerate(R[m].qids)}
            R[m] = R[m].take([pos[q] for q in common])
    fpe = cfg["fpe"]
    outs = {lam: {m: AR.all_outputs(R[m], fz["models"][m]["arms"], lam, fpe) for m in models} for lam in (1, 3)}
    boot = S.Boot(len(common), int(n_boot or cfg["stats"]["n_boot"]), int(cfg["stats"]["boot_seed"]),
                  float(cfg["stats"]["min_valid_frac"]))
    return R, outs, boot


def write_records(pl, m, split, r, outs1=None):
    rows = []
    for i, it in enumerate(r.items):
        row = {"qid": it["qid"], "source": it["source"], "relation": it["relation"], "subj_id": it.get("subj_id"),
               "obj_id": it.get("obj_id"), "set": split, "model": m,
               "A1": {"text": r.a1_text[i], "label_strict": int(r.a1_lab[i]), "label_sq2": int(r.a1_lab2[i]),
                      "is_idk": bool(r.a1_idk[i]), "s1": float(r.s1[i]), "first_token": r.a1_first[i]},
               "A2": {"text": r.a2_text[i], "label_strict": None if r.a2_text[i] is None else int(r.a2_lab[i]),
                      "null_reason": r.a2_null_reason[i], "D_score": None if not np.isfinite(r.dA2[i]) else float(r.dA2[i])},
               "s2": float(r.s2[i]), "p1": float(r.p1[i]), "p2": float(r.p2[i])}
        if r.pr_text is not None:
            row["peer"] = {"A1_text": r.pr_text[i], "A1_label": int(r.pr_lab[i]), "s1_peer": float(r.pr_s1[i]),
                           "null": bool(r.pr_null[i])}
        if outs1:
            row["outputs_lam1"] = {a: {"label": int(o.label[i]), "kind": str(o.kind[i])} for a, o in outs1.items()}
        rows.append(row)
    write_jsonl(pl.pdir(m, split) / "records.jsonl", rows)


def audit_sheet(pl, R, models, outs1):
    n = int(pl.cfg["audit_n"])
    names = {C_: "correct", A_: "abstained", W_: "wrong"}
    for m in models:
        r, o = R[m], outs1[m]["RRA"]
        cand = [i for i in range(r.N) if o.kind[i] == "A2"]
        rng = np.random.default_rng([int(pl.cfg["seed"]), 11])
        pick = sorted(cand[k] for k in rng.permutation(len(cand))[:n])
        with open(pl.run.audit / f"human_check_{m}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "qid", "source", "question", "gold_aliases", "A1", "A1_auto_label", "A2_output",
                        "auto_label", "auto_label_sq2", "s1", "s2", "human_label"])
            for i in pick:
                it = r.items[i]
                w.writerow([m, it["qid"], it["source"], it["question"], " | ".join(it["aliases"]), r.a1_text[i],
                            names[int(r.a1_lab[i])], r.a2_text[i], names[int(r.a2_lab[i])], names[int(r.a2_lab2[i])],
                            f"{r.s1[i]:.4f}", f"{r.s2[i]:.4f}", ""])


def analyze(pl):
    cfg, run = pl.cfg, pl.run
    summary = {"run": str(run.root), "smoke": cfg["smoke"], "missing": list(dict.fromkeys(pl.missing)),
               "models": [], "ves_run": str(pl.ves)}
    fz = read_json(run.frozen_path) if run.frozen_path.exists() else None
    summary["frozen"] = fz
    summary["integrity"] = pl.integ.d
    summary["integrity_status"] = dict(zip(("all_passed", "problems"), pl.integ.status(pl.names)))
    summary["budget"] = {"cuts": pl.budget.cuts, "elapsed_h": pl.budget.elapsed() / 3600, "probe": pl.budget.probe,
                         "measured": {m: {u: {"units": t[0], "seconds": t[1], "rate_per_s": t[0] / t[1] if t[1] else None}
                                          for u, t in d.items()} for m, d in pl.budget.totals.items()}}
    qp = run.root / "runs" / "questions.json"
    summary["questions"] = read_json(qp)["info"] if qp.exists() else None
    summary["pool"] = {}
    for m in pl.names:
        p = pl.pdir(m)
        d = {}
        if (p / "R.json").exists():
            d["R"] = read_json(p / "R.json")
        if (p / "labels.json").exists():
            lab = read_json(p / "labels.json")
            ls = np.array(lab["label_strict"])
            d["A1"] = {"n": int(len(ls)), "right": int((ls == C_).sum()), "idk": int((ls == A_).sum()),
                       "matches_ves_labels": lab.get("matches_ves_labels")}
        if (p / "a2.json").exists():
            rows = read_json(p / "a2.json")["rows"]
            d["A2"] = {"n": sum(r is not None for r in rows),
                       "null": sum(r is not None and r["null_reason"] is not None for r in rows),
                       "right": sum(r is not None and r["null_reason"] is None and r["label_strict"] == C_ for r in rows)}
        summary["pool"][m] = d
    gp = run.results / "generality.json"
    if gp.exists():
        summary["generality"] = read_json(gp)
    verd = {"RRA": {"verdict": "NOT_RUN", "reason": "no fresh-test results"}}
    models = []
    if fz is not None:
        models = [m for m in pl.names if m in fz["models"] and all((pl.pdir(m, "test_fresh") / f).exists()
                                                                     for f in ("a1.json", "states.npz", "a2.json"))]
    if fz is not None:
        summary["tuning"] = {m: {a: v for a, v in fz["models"][m]["arms"].items()} for m in fz["models"]}
        summary["comparator"] = fz.get("comparator")
        try:
            vm = [m for m in pl.names if m in fz["models"]]
            Rr, outr, bootr = evaluate(pl, "test_ves", vm, fz)
            summary["replay"] = {"n": Rr[vm[0]].N, "main_table": main_table(outr, vm, AR.ARMS_MAIN),
                                 "contrasts": contrasts(outr, vm, bootr, fz.get("comparator", {})),
                                 "n_recoverable": {m: n_recoverable(Rr[m]) for m in vm}, "diagnostic_only": True}
            Rd, outd, _ = evaluate(pl, "dev_fresh", vm, fz, n_boot=1)
            summary["dev_fresh"] = {"n": Rd[vm[0]].N, "main_table": main_table(outd, vm, AR.ARMS_MAIN)}
            for m in vm:
                write_records(pl, m, "dev_fresh", Rd[m])
                write_records(pl, m, "test_ves", Rr[m])
        except Exception:
            traceback.print_exc()
            summary["missing"].append("replay / dev evaluation failed (see the log)")
    if len(models) < len(pl.names):
        if fz is not None:
            verd["RRA"]["reason"] = f"fresh-test results incomplete (have {models}, need {pl.names})"
        summary["models"] = models
        _finish(pl, summary, verd)
        return summary

    summary["models"] = models
    R, outs, boot = evaluate(pl, "test_fresh", models, fz)
    summary["n_test"] = R[models[0]].N
    summary["n_boot"] = boot.B
    summary["main_table"] = main_table(outs, models, AR.ARMS_MAIN)
    con = contrasts(outs, models, boot, fz.get("comparator", {}))
    summary["contrasts"] = con
    nrec = {m: n_recoverable(R[m]) for m in models}
    try:
        inp = {"n_recoverable": nrec, "H": con["H"]["pooled"], "d_main": con["d_main"]["pooled"],
               "d_mech": con["d_mech"]["pooled"], "comparator": fz["comparator"]["lam1"]["arm"]}
        verd["RRA"] = {**rra_verdict(inp, cfg["rules"]), "inputs": inp}
    except Exception as e:
        traceback.print_exc()
        verd["RRA"] = {"verdict": "UNTESTABLE", "reason": f"{type(e).__name__}: {e}", "inputs": {"n_recoverable": nrec}}
    if summary["integrity_status"]["all_passed"] is False:
        verd["RRA"]["integrity_note"] = "some integrity checks did not pass: " + "; ".join(
            summary["integrity_status"]["problems"])
    summary["diagnostics"] = probe_diagnostics(R, models, fz)
    masks = strata_masks(pl, R, models)
    summary["strata"] = strata(outs, models, boot, masks, fz.get("comparator", {}))
    for m in models:
        try:
            write_records(pl, m, "test_fresh", R[m], outs[1][m])
        except Exception:
            traceback.print_exc()
    try:
        audit_sheet(pl, R, models, outs[1])
    except Exception:
        traceback.print_exc()
    _finish(pl, summary, verd)
    return summary


def generality(pl, g):
    """Stage G: Gemma tuned on its fresh dev set (1,000), evaluated on the fresh test; reported separately."""
    cfg = pl.cfg
    lams, step = cfg["lambdas"], float(cfg["thresholds"]["quantile_step"])
    rd = pl.recs(g, "dev_fresh", peer=False)
    rt = pl.recs(g, "test_fresh", peer=False)
    T = AR.tune(rd, lams, step)
    fzg = {"models": {g: {"arms": T}}}
    fz = read_json(pl.run.frozen_path)
    comp = fz.get("comparator", {})
    outs = {lam: {g: AR.all_outputs(rt, T, lam, cfg["fpe"])} for lam in (1, 3)}
    boot = S.Boot(rt.N, int(cfg["stats"]["n_boot"]), int(cfg["stats"]["boot_seed"]), float(cfg["stats"]["min_valid_frac"]))
    pdir = pl.pdir(g)
    return {"model": g, "spec": {k: pl.gspec[k] for k in ("hf_id", "revision", "attn_implementation") if k in pl.gspec},
            "n_dev": rd.N, "n_test": rt.N, "tuning": T, "comparator_from_main": comp,
            "main_table": main_table(outs, [g], AR.ARMS_MAIN), "contrasts": contrasts(outs, [g], boot, comp),
            "n_recoverable": n_recoverable(rt), "diagnostics": probe_diagnostics({g: rt}, [g], fzg)[g],
            "D": read_json(pdir / "D.json"), "R": read_json(pdir / "R.json")}


def _finish(pl, summary, verd):
    summary["verdicts"] = verd
    write_json(pl.run.results / "summary.json", summary, indent=1)
    write_json(pl.run.results / "verdicts.json", verd, indent=1)
    for k, v in verd.items():
        print(f"VERDICT: {k} {v['verdict']} - {v.get('reason', '')}" + (f" [{v['note']}]" if v.get("note") else ""))
    try:
        from rra.report import write_report
        write_report(summary, pl.run.results / "results_report.tex")
        print(f"REPORT: {pl.run.results / 'results_report.tex'}")
    except Exception:
        traceback.print_exc()
        print("REPORT_FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    from rra.pipeline import Pipeline
    pl = Pipeline(out_dir=args.run_dir)
    q = read_json(pl.run.root / "runs" / "questions.json")
    pl.fresh = q["splits"]
    pl.ves_frozen = read_json(pl.ves / "configs" / "frozen_config.json")
    analyze(pl)


if __name__ == "__main__":
    main()

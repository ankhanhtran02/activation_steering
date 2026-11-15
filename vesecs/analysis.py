"""Phase F: test evaluation with the frozen configuration, bootstrap CIs, G, strata, diagnostics, verdicts,
summary.json, verdicts.json, the audit sheet, per-question records (records.jsonl) and the LaTeX report.

    python -m vesecs.analysis --run-dir outputs_ves/run_<id>     (re-run the analysis only; CPU)
"""
import argparse
import csv
import traceback

import numpy as np

from pilot.metrics import same_answer
from scoring import A_, C_, W_
from vesecs import methods as M
from vesecs import stats as S
from vesecs.io import read_json, write_json, write_jsonl
from vesecs.verdicts import ecs_verdict, ves_verdict

MAIN_ORDER = ["B0", "B1", "B3", "B3b", "B5@1", "B5@2", "B5@4", "B5@8", "B5@Nm", "B7", "VES", "VES_free", "VES_placebo",
              "VES_unclean", "ECS_1p", "ECS_exact", "ECS_placebo", "R1", "R2", "R3"]


def method_outputs(pl, sr, mt, lam):
    """{name: Out} for every method with its lambda-tuned parameters (lam in (1, 3))."""
    k = f"lam{lam}"
    fx = pl.cfg["fpe_fixed"]
    out = {"B0": M._a1(sr, fx["B0"])}
    if "B1" in mt:
        out["B1"] = M.gate(sr, sr.m, mt["B1"][k]["tau"], fx["B1"])
    out["B3"] = M.gate(sr, sr.sD, mt["B3"][k]["tau"], fx["B3"])
    if "B3b" in mt:
        out["B3b"] = M.gate(sr, sr.sD_before, mt["B3b"][k]["tau"], fx["B3b"])
    if "B5" in mt and sr.b5 is not None:
        for N in pl.cfg["b5"]["N_report"]:
            out[f"B5@{N}"] = M.search_out(sr, sr.b5, mt["B5"][f"N{N}"][k]["tau"], "B5", max_rounds=N)
        if "N_matched" in mt["B5"]:
            N = mt["B5"]["N_matched"]["N"]
            out["B5@Nm"] = M.search_out(sr, sr.b5, mt["B5"][f"N{N}"][k]["tau"], "B5", max_rounds=N)
    if "B7" in mt and "B7" in sr.searches:
        out["B7"] = M.search_out(sr, sr.searches["B7"], mt["B7"][k]["tau"], "B7")
    if "VES" in mt:
        v = mt["VES"][k]
        tag = pl.ves_tag("VES", v["depth"], v["alpha"])
        if tag in sr.searches:
            out["VES"] = M.search_out(sr, sr.searches[tag], v["tau"], "VES")
        v1 = mt["VES"]["lam1"]
        if "VES_free" in mt:
            f = mt["VES_free"][k]
            tag = pl.ves_tag("VES_free", v1["depth"], f["alpha"])
            if tag in sr.searches:
                out["VES_free"] = M.search_out(sr, sr.searches[tag], f["tau"], "VES_free")
        if "VES_placebo" in mt:
            tag = pl.ves_tag("VES_placebo", v1["depth"], v1["alpha"])
            if tag in sr.searches:
                out["VES_placebo"] = M.search_out(sr, sr.searches[tag], mt["VES_placebo"][k]["tau"], "VES_placebo")
        if "VES_unclean" in mt:
            tag = pl.ves_tag("VES", v1["depth"], v1["alpha"])
            if tag in sr.searches:
                out["VES_unclean"] = M.search_out(sr, sr.searches[tag], mt["VES_unclean"][k]["tau"], "VES",
                                                  key="sD_steered")
    if sr.ecs is not None:
        for name in ("1p", "exact", "placebo"):
            key = f"ECS_{name}"
            if key in mt and name in sr.ecs_variants:
                p = mt[key][k]
                out[key] = M.ecs_out(sr, M.ecs_arrays(sr, name), p["tau_keep"], p["tau_sw"], p["tau_D"], fx["ECS"])[0]
    if sr.judges and "R1" in mt and "R1" in sr.judges:
        out["R1"] = M.gate(sr, sr.judges["R1"], mt["R1"][k]["tau"], fx["R1"])
        out["R2"] = M.gate(sr, sr.judges["R2"], mt["R2"][k]["tau"], fx["R2"])
        out["R3"] = M.r3_out(sr, fx["R3"])
    return out


def strata_masks(R, models, n_sel, cfg):
    st = cfg["strata"]
    masks = {"all": {m: np.ones(R[m].N, bool) for m in models}}
    ce = {}
    for m in models:
        sr = R[m]
        ce[m] = np.array([sr.a1_label[i] == W_ and not sr.a1_idk[i] and
                          sum(same_answer(t["text"], sr.a1_text[i]) for t in sr.tries[i][:n_sel]) >= st["confident_error_min_match"]
                          for i in range(sr.N)]) if sr.tries else np.zeros(sr.N, bool)
    masks["confident_errors"] = ce
    if len(models) >= 2 and all(R[m].tries for m in models[:2]):
        a, b = models[:2]
        ca = np.array([sum(t["label_strict"] == C_ for t in row[:n_sel]) for row in R[a].tries])
        cb = np.array([sum(t["label_strict"] == C_ for t in row[:n_sel]) for row in R[b].tries])
        dis = ((ca >= st["disagree_hi"]) & (cb <= st["disagree_lo"])) | ((cb >= st["disagree_hi"]) & (ca <= st["disagree_lo"]))
        masks["disagreement"] = {m: dis for m in models}
    src = {m: np.array([it["source"] for it in R[m].items]) for m in models}
    masks["triviaqa"] = {m: src[m] == "tqa" for m in models}
    masks["popqa"] = {m: src[m] == "popqa" for m in models}
    return masks


def ecs_slot_scores(sr, variant):
    return np.array([r["E"][variant][r["slot_A1"]] if (r["slot_A1"] is not None and variant in r["E"]) else np.nan
                     for r in sr.ecs])


def diagnostics(pl, R, models, outs1, mt_all, masks):
    K = int(pl.cfg["ves"]["K"])
    d = {"hit_rate": {}, "distinct_rate": {}, "fooling": {}, "ecs": {}, "net_fixes_vs_B3": {}}
    for m in models:
        sr, mt = R[m], mt_all[m]
        wrong = sr.answered & (sr.a1_label == W_)
        ce = masks["confident_errors"][m]
        lists = {}
        if "VES" in mt:
            v1 = mt["VES"]["lam1"]
            for meth, tag in (("VES", pl.ves_tag("VES", v1["depth"], v1["alpha"])),
                              ("VES_free", pl.ves_tag("VES_free", v1["depth"], mt.get("VES_free", {}).get("lam1", {}).get("alpha", -1))),
                              ("VES_placebo", pl.ves_tag("VES_placebo", v1["depth"], v1["alpha"]))):
                if tag in sr.searches:
                    lists[meth] = sr.searches[tag]
        if "B7" in sr.searches:
            lists["B7"] = sr.searches["B7"]
        if sr.b5 is not None:
            lists["B5 (first 3 tries)"] = [None if r is None else r[:3] for r in sr.b5]
            lists["B5 (8 tries)"] = sr.b5
        hr, dr = {}, {}
        for meth, L in lists.items():
            hit = np.array([bool(L[i]) and any(r["label_strict"] == C_ for r in L[i][:K if "B5" not in meth else None])
                            for i in range(sr.N)])
            hr[meth] = {"all": float(hit[wrong].mean()) if wrong.any() else None, "n_wrong": int(wrong.sum()),
                        "confident_errors": float(hit[wrong & ce].mean()) if (wrong & ce).any() else None,
                        "n_confident_errors": int((wrong & ce).sum())}
            rounds = [r for i in range(sr.N) if L[i] for r in L[i]]
            dr[meth] = float(np.mean([r["status"] in ("valid", "hedged") for r in rounds])) if rounds else None
        d["hit_rate"][m], d["distinct_rate"][m] = hr, dr
        if "VES" in lists:
            diffs = [r["sD_steered"] - r["sD_clean"] for i in range(sr.N) if lists["VES"][i] for r in lists["VES"][i]
                     if r["sD_steered"] is not None and r["sD_clean"] is not None]
            d["fooling"][m] = {"mean_steered_minus_clean": float(np.mean(diffs)) if diffs else None, "n": len(diffs),
                               "VES_unclean_U1": M.utility(outs1[m]["VES_unclean"].label, 1) if "VES_unclean" in outs1[m] else None,
                               "VES_U1": M.utility(outs1[m]["VES"].label, 1) if "VES" in outs1[m] else None}
        if sr.ecs is not None and "ECS_1p" in mt:
            p = mt["ECS_1p"]["lam1"]
            arr = M.ecs_arrays(sr, "1p")
            ch = M.ecs_choice(arr, p["tau_keep"], p["tau_sw"])
            sw = ch >= 0
            jj = np.maximum(ch, 0)
            new = np.where(sw, arr["lab"][np.arange(sr.N), jj], sr.a1_label)
            ft = np.array([f if f is not None else -1 for f in sr.first_token])
            topa = np.array([r["top_a"] for r in sr.ecs])
            real_switch = sw & np.array([sr.ecs[i]["completion"][str(jj[i])]["is_A1"] is False for i in range(sr.N)])
            d["ecs"][m] = {"agreement_top_a_eq_first_token": float(np.mean(topa[ft >= 0] == ft[ft >= 0])),
                           "n_switches": int(real_switch.sum()),
                           "switch_wrong_to_right": int((real_switch & (sr.a1_label == W_) & (new == C_)).sum()),
                           "switch_right_to_wrong": int((real_switch & (sr.a1_label == C_) & (new == W_)).sum()),
                           "switch_idk_to_right": int((real_switch & (sr.a1_label == A_) & (new == C_)).sum()),
                           "switch_right_to_idk": int((real_switch & (sr.a1_label == C_) & (new == A_)).sum())}
            e = d["ecs"][m]
            e["net"] = e["switch_wrong_to_right"] + e["switch_idk_to_right"] - e["switch_right_to_wrong"] - e["switch_right_to_idk"]
        if "B3" in outs1[m]:
            d["net_fixes_vs_B3"][m] = {k: S.net_fixes(o.label, outs1[m]["B3"].label) for k, o in outs1[m].items()}
    return d


def risk_coverage_curves(pl, R, models, mt_all):
    out = {}
    for m in models:
        sr, mt = R[m], mt_all[m]
        grid = M.tau_grid(sr.sD[sr.answered], pl.cfg["thresholds"]["quantile_step"])
        c = {"B3": S.risk_coverage([(t, M.gate(sr, sr.sD, t, 1).label) for t in grid])}
        if "VES" in mt:
            v = mt["VES"]["lam1"]
            tag = pl.ves_tag("VES", v["depth"], v["alpha"])
            if tag in sr.searches:
                c["VES"] = S.risk_coverage([(t, M.search_out(sr, sr.searches[tag], t, "VES").label) for t in grid])
        if "B7" in sr.searches:
            c["B7"] = S.risk_coverage([(t, M.search_out(sr, sr.searches["B7"], t, "B7").label) for t in grid])
        if sr.b5 is not None:
            c["B5@8"] = S.risk_coverage([(t, M.search_out(sr, sr.b5, t, "B5", max_rounds=8).label) for t in grid])
        if sr.ecs is not None and "ECS_1p" in mt:
            p = mt["ECS_1p"]["lam1"]
            arr = M.ecs_arrays(sr, "1p")
            c["ECS_1p (tau_D swept)"] = S.risk_coverage([(t, M.ecs_out(sr, arr, p["tau_keep"], p["tau_sw"], t)[0].label)
                                                         for t in grid])
        out[m] = c
    return out


def write_records(pl, m, split, sr, masks=None):
    rows = []
    n_sel = sr.n_sel
    for i, it in enumerate(sr.items):
        r = {"qid": it["qid"], "source": it["source"], "relation": it["relation"], "subj_id": it.get("subj_id"),
             "obj_id": it.get("obj_id"), "split": split, "model": m,
             "A1": {"text": sr.a1_text[i], "label_strict": int(sr.a1_label[i]), "label_sq2": int(sr.a1_label2[i]),
                    "first_token": sr.first_token[i], "is_idk": bool(sr.a1_idk[i]), "m": float(sr.m[i]),
                    "sD": float(sr.sD[i]), "sD_before": float(sr.sD_before[i]), "sD_after": float(sr.sD_after[i])}}
        if sr.tries:
            sdl = {}
            if sr.b5 and sr.b5[i]:
                sdl = {x["try_idx"]: x["sD_clean"] for x in sr.b5[i]}
            r["tries"] = [{"idx": j + 1, "text": t["text"], "label_strict": t["label_strict"],
                           **({"sD": sdl.get(j + 1)} if j >= n_sel else {})} for j, t in enumerate(sr.tries[i])]
        if masks:
            r["strata"] = {"confident_error": bool(masks["confident_errors"][m][i]),
                           "disagreement": bool(masks.get("disagreement", {}).get(m, np.zeros(sr.N, bool))[i])}
        r["search"] = {k: v[i] for k, v in sr.searches.items()}
        if sr.b5 is not None:
            r["search"]["B5"] = sr.b5[i]
        if sr.ecs is not None:
            e = sr.ecs[i]
            r["ecs"] = {"top_a": e["top_a"], "C": e["C"], "slots": e["slots"], "first_token_A1": e["first_token"],
                        **{f"E_{k}": {str(c): e["E"][k][j] for j, c in enumerate(e["slots"])} for k in e["E"]},
                        "completion": {str(c): {kk: e["completion"][str(j)][kk] for kk in ("text", "label_strict", "sD")}
                                       for j, c in enumerate(e["slots"])}}
        rows.append(r)
    write_jsonl(pl.pdir(m, split) / "records.jsonl", rows)


def audit_sheet(pl, R, models, outs1):
    n = int(pl.cfg["audit_n"])
    keep = ("VES", "VES_free", "VES_placebo", "B5@Nm", "B7", "ECS_1p", "ECS_exact", "ECS_placebo", "R3")
    for m in models:
        sr = R[m]
        cand = []
        for meth in keep:
            o = outs1[m].get(meth)
            if o is None:
                continue
            for i in range(sr.N):
                if o.changed[i] and o.label[i] != A_ and o.text[i] != sr.a1_text[i]:
                    cand.append((meth, i))
        rng = np.random.default_rng([int(pl.cfg["seed"]), 11])
        pick = [cand[k] for k in sorted(rng.permutation(len(cand))[:n])]
        names = {C_: "correct", A_: "abstained", W_: "wrong"}
        with open(pl.run.audit / f"human_check_{m}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "method", "qid", "source", "question", "gold_aliases", "A1", "A1_auto_label", "output",
                        "auto_label", "auto_label_sq2", "human_label"])
            for meth, i in pick:
                o = outs1[m][meth]
                it = sr.items[i]
                w.writerow([m, meth, it["qid"], it["source"], it["question"], " | ".join(it["aliases"]), sr.a1_text[i],
                            names[int(sr.a1_label[i])], o.text[i], names[int(o.label[i])], names[int(o.label_sq2[i])], ""])


def analyze(pl):
    cfg = pl.cfg
    run = pl.run
    summary = {"run": str(run.root), "smoke": cfg["smoke"], "missing": list(pl.missing)}
    fz = read_json(run.frozen_path) if run.frozen_path.exists() else {"models": {}}
    mt_all = {m: fz["models"][m]["methods"] for m in fz["models"]}
    models = [m for m in pl.names if m in mt_all and (pl.pdir(m, "test") / "drafts.json").exists()
              and (pl.pdir(m, "test") / "draft_states.npz").exists()]
    summary["models"] = models
    summary["frozen"] = fz
    summary["integrity"] = pl.integ.d
    summary["integrity_status"] = dict(zip(("all_passed", "problems"), pl.integ.status(pl.names)))
    summary["budget"] = {"cuts": pl.budget.cuts, "cuts_unavailable": pl.budget.cut_unavailable,
                         "elapsed_h": pl.budget.elapsed() / 3600, "rates": pl.budget.rates}
    summary["questions"] = read_json(run.root / "runs" / "questions.json")["info"]
    summary["pool"] = {}
    for m in pl.names:
        p = pl.pdir(m)
        summary["pool"][m] = {k: read_json(p / f) for k, f in (("detectors", "detectors.json"), ("lens", "lens.json"),
                                                               ("verify", "verify.json"),
                                                               ("echo_detectors", "echo_detectors.json"))
                              if (p / f).exists()}
        if (p / "labels.json").exists():
            lab = read_json(p / "labels.json")
            summary["pool"][m]["grader"] = {k: lab[k] for k in ("n_changed_vs_sq2", "n_hedged", "n_sq2_regrade_mismatch")}
            summary["pool"][m]["grader"]["n"] = len(lab["label_strict"])
        if "lens" in summary["pool"][m]:
            summary["pool"][m]["lens"].pop("placebo_tokens", None)
        dv = pl.pdir(m, "dev")
        if (dv / "ecs.json").exists():
            summary["pool"][m]["ecs_dev_selection"] = read_json(dv / "ecs.json")["info"]
    for m in pl.names:
        for split in ("dev", "test"):
            p = pl.pdir(m, split) / "drafts.json"
            if p.exists():
                dr = read_json(p)
                a, b = np.array(dr["label_strict"]), np.array(dr["label_sq2"])
                summary.setdefault("grader_changes", {})[f"{m}/{split}"] = {
                    "n": int(len(a)), "changed": int((a != b).sum()), "hedged": int(np.sum(dr["hedged"]))}
    if not models:
        summary["error"] = "no model has a complete test phase"
        _finish(pl, summary, {"VES": {"verdict": "NOT_RUN", "reason": "no test results"},
                              "ECS": {"verdict": "NOT_RUN", "reason": "no test results"}})
        return summary

    R = {m: pl.records(m, "test") for m in models}
    common = [q for q in R[models[0]].qids if all(q in set(R[x].qids) for x in models)]
    R = {m: (R[m].subset(common) if len(common) < R[m].N else R[m]) for m in models}
    for m in models:
        assert R[m].qids == common, "test question order differs between models"
    N = len(common)
    summary["n_test"] = N
    summary["n_boot"] = int(cfg["stats"]["n_boot"])
    outs = {lam: {m: method_outputs(pl, R[m], mt_all[m], lam) for m in models} for lam in (1, 3)}
    boot = S.Boot(N, int(cfg["stats"]["n_boot"]), int(cfg["stats"]["boot_seed"]), float(cfg["stats"]["min_valid_frac"]))

    # main table
    main = {}
    for name in MAIN_ORDER:
        rows = {}
        for m in models:
            o1, o3 = outs[1][m].get(name), outs[3][m].get(name)
            if o1 is None:
                continue
            rows[m] = {**M.counts(o1.label), "U1": M.utility(o1.label, 1), "U3": M.utility(o3.label, 3) if o3 else None,
                       "U1_sq2_grader": M.utility(o1.label_sq2, 1), "fpe": float(np.mean(o1.fpe))}
        if rows:
            pooled = {k: (float(np.mean([r[k] for r in rows.values()])) if len(rows) == len(models)
                          and all(r[k] is not None for r in rows.values()) else None) for k in next(iter(rows.values()))}
            main[name] = {"per_model": rows, "pooled": pooled}
    summary["main_table"] = main

    # contrasts
    def contrast(a, b, lam=1, masks=None):
        if not all(a in outs[lam][m] and b in outs[lam][m] for m in models):
            return None
        vec = {m: S.utility_diff_vec(outs[lam][m][a].label, outs[lam][m][b].label, lam) for m in models}
        return boot.mean_contrast(vec, masks)

    con = {"d_ves": contrast("VES", "B3"), "d_ves_b5": contrast("VES", "B5@Nm"), "d_ecs": contrast("ECS_1p", "B3"),
           "d_ecs_exact": contrast("ECS_exact", "B3"), "d_ecs_placebo": contrast("ECS_placebo", "B3"),
           "d_ves_free": contrast("VES_free", "B3"), "d_ves_placebo": contrast("VES_placebo", "B3"),
           "d_ves_unclean": contrast("VES_unclean", "B3"), "d_b7": contrast("B7", "B3"), "d_b5_matched": contrast("B5@Nm", "B3"),
           "d_ves_b7": contrast("VES", "B7"), "d_ves_lam3": contrast("VES", "B3", 3), "d_ecs_lam3": contrast("ECS_1p", "B3", 3)}
    summary["contrasts"] = con

    # G
    g = {}
    for name in ("1p", "exact", "placebo"):
        if all(R[m].ecs is not None and name in R[m].ecs_variants for m in models):
            per = {m: (R[m].answered, R[m].a1_label == C_, R[m].sD_before, R[m].sD_after, ecs_slot_scores(R[m], name))
                   for m in models}
            try:
                g[name] = S.gap_closure_boot(boot, per, float(cfg["rules"]["ecs"]["min_auroc_gap"]))
            except Exception:
                traceback.print_exc()
    summary["G"] = g

    # verdicts
    verd = {}
    try:
        n_rej = {}
        for m in models:
            sr = R[m]
            tau = mt_all[m]["VES"]["lam1"]["tau"]
            n_rej[m] = int((sr.answered & (sr.a1_label == W_) & ~(sr.sD >= tau)).sum())
        if con["d_ves"] and con["d_ves_b5"]:
            inp = {"n_rejected_wrong": n_rej, "d_ves": con["d_ves"]["pooled"], "d_ves_b5": con["d_ves_b5"]["pooled"]}
            verd["VES"] = {**ves_verdict(inp, cfg["rules"]["ves"]), "inputs": inp}
        else:
            verd["VES"] = {"verdict": "UNTESTABLE", "reason": "VES, B3 or B5 results missing", "inputs": {"n_rejected_wrong": n_rej}}
    except Exception as e:
        traceback.print_exc()
        verd["VES"] = {"verdict": "UNTESTABLE", "reason": f"{type(e).__name__}: {e}"}
    try:
        n_wrong = {m: int((R[m].answered & (R[m].a1_label == W_)).sum()) for m in models}
        if con["d_ecs"] and "1p" in g:
            inp = {"n_wrong": n_wrong, "G": g["1p"]["pooled"], "d_ecs": con["d_ecs"]["pooled"]}
            ex = ({"n_wrong": n_wrong, "G": g["exact"]["pooled"], "d_ecs": con["d_ecs_exact"]["pooled"]}
                  if con["d_ecs_exact"] and "exact" in g else None)
            verd["ECS"] = {**ecs_verdict(inp, cfg["rules"]["ecs"], ex), "inputs": inp, "inputs_exact": ex}
        else:
            verd["ECS"] = {"verdict": "UNTESTABLE", "reason": "ECS or B3 results missing", "inputs": {"n_wrong": n_wrong}}
    except Exception as e:
        traceback.print_exc()
        verd["ECS"] = {"verdict": "UNTESTABLE", "reason": f"{type(e).__name__}: {e}"}
    if summary["integrity_status"]["all_passed"] is False:
        for v in verd.values():
            v["integrity_note"] = "some integrity checks did not pass: " + "; ".join(summary["integrity_status"]["problems"])

    # strata
    n_sel = int(cfg["decoding"]["n_select_tries"])
    masks = strata_masks(R, models, n_sel, cfg)
    strata = {}
    for sname, mk in masks.items():
        row = {"n": {m: int(mk[m].sum()) for m in models}}
        for name in ("B0", "B3", "B5@Nm", "B7", "VES", "ECS_1p"):
            if all(name in outs[1][m] for m in models):
                vec = {m: 100.0 * S.score_vec(outs[1][m][name].label, 1) for m in models}
                row[f"U1_{name}"] = boot.mean_contrast(vec, mk)
        for key, (a, b) in (("d_ves", ("VES", "B3")), ("d_ves_b5", ("VES", "B5@Nm")), ("d_ecs", ("ECS_1p", "B3"))):
            row[key] = contrast(a, b, 1, mk)
        strata[sname] = row
    summary["strata"] = strata
    summary["diagnostics"] = diagnostics(pl, R, models, outs[1], mt_all, masks)
    summary["diagnostics"]["risk_coverage"] = risk_coverage_curves(pl, R, models, mt_all)
    summary["ecs_overhead"] = ecs_overhead(pl, R, models)
    summary["dev"] = {m: {"tuning": fz["models"][m], "n_dev": fz["models"][m].get("n_dev")} for m in models}
    for m in models:
        try:
            write_records(pl, m, "test", R[m], masks)
            write_records(pl, m, "dev", pl.records(m, "dev"))
        except Exception:
            traceback.print_exc()
    try:
        audit_sheet(pl, R, models, outs[1])
    except Exception:
        traceback.print_exc()
    _finish(pl, summary, verd)
    return summary


def ecs_overhead(pl, R, models):
    """Analytical extra FLOPs of the fused one-pass ECS: k(b - a)/L of one token's forward, or about
    k(b - a)/(L (P + n)) of the whole generation (P prompt tokens, n answer tokens)."""
    out = {}
    try:
        from transformers import AutoTokenizer

        from config import USER_TEMPLATE_BASE
        from model_utils import ChatFormatter
    except Exception:
        return out
    k = int(pl.cfg["ecs"]["k"])
    for m in models:
        try:
            spec = pl.spec(m)
            tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"], local_files_only=True)
            fmt = ChatFormatter(tok, m)
            P = float(np.mean([len(fmt.prompt(USER_TEMPLATE_BASE.format(question=it["question"]))) for it in R[m].items]))
            dr = read_json(pl.pdir(m, "test") / "drafts.json")
            n = float(np.mean([len(t) + 1 for t in dr["token_ids"]]))
            lens = read_json(pl.pdir(m) / "lens.json")
            L = int(spec["n_layers"])
            ab = lens["b_layer"] - lens["a_layer"]
            out[m] = {"k": k, "a_layer": lens["a_layer"], "b_layer": lens["b_layer"], "n_layers": L, "P": P, "n": n,
                      "frac_of_one_token": k * ab / L, "frac_of_generation": k * ab / (L * (P + n))}
        except Exception as e:
            out[m] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _finish(pl, summary, verd):
    summary["verdicts"] = verd
    write_json(pl.run.results / "summary.json", summary, indent=1)
    write_json(pl.run.results / "verdicts.json", verd, indent=1)
    for k, v in verd.items():
        print(f"VERDICT: {k} {v['verdict']} - {v.get('reason', '')}" + (f" [{v['note']}]" if v.get("note") else ""))
    try:
        from vesecs.report import write_report
        write_report(summary, pl.run.results / "results_report.tex")
        print(f"REPORT: {pl.run.results / 'results_report.tex'}")
    except Exception:
        traceback.print_exc()
        print("REPORT_FAILED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    from vesecs.pipeline import Pipeline
    pl = Pipeline(out_dir=args.run_dir)
    pl.phase_a()
    analyze(pl)


if __name__ == "__main__":
    main()

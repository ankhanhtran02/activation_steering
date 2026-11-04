"""Offline re-analysis of a finished run (no GPU). Amendment 2026-11-04, Phase 0 / Phase 1.

Replaces the withdrawn statistics of a saved run with the corrected ones (analysis_core / stats):
paired bootstrap CIs with the 95%-valid-resample rule, sign-flip permutation p-values, Holm within
the registered families, AURC on a common coverage grid, always-abstain rows, net transitions,
surplus per answered item, the H2c validity rule, H3 on answered items with relation baselines
(margin and popularity only: the first run did not save the other per-item signals), and the
exploratory gated-knockout prototype with a per-relation chance baseline.

    python reanalyze.py                         # newest outputs/run_* folder (else outputs/, the first run)
    python reanalyze.py --run-dir outputs/run_20261105T010203Z
    python reanalyze.py --results R --frozen F --data-dir data_cache --out DIR

--data-dir (the data_cache/ written by setup.py) is optional; without it the relation baselines and
the knockout chance baseline are skipped.
"""
import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import analysis_core as AC
import stats as S
from config import DATA_DIR, latest_run_dir
from data import qid_hash
from scoring import A_, C_, W_, auroc

STATE = {"correct": C_, "abstained": A_, "wrong": W_}
TODAY = "2026-11-04"


# ----------------------------------------------------------------------------- loading
def load_popqa_meta(data_dir):
    path = Path(data_dir) / "popqa.jsonl"
    if not path.exists():
        return None
    meta = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            meta[f"popqa_{r['id']}"] = {"prop": r["prop"], "aliases": [a for a in r["possible_answers"] if a.strip()],
                                        "question": r["question"]}
    return meta


def views_from_results(results, frozen, popqa_meta=None):
    """Build PopQA-test and EntityQuestions views per model from results.json per_question records."""
    out = {}
    for model, M in results["models"].items():
        recs = M.get("per_question") or []
        by = defaultdict(dict)
        for r in recs:
            by[(r["dataset"], r["condition"])][r["qid"]] = r
        tau1 = float(frozen.get(model, {}).get("tau", {}).get("margin", {}).get("1.0", float("nan")))
        cm_info = frozen.get(model, {}).get("cov_matched", {}).get("draws", {})
        cm_matched = {f"random_dir_placebo_coverage_matched_s{k}": bool(v.get("matched")) for k, v in cm_info.items()}
        views = {}
        for ds, label in (("popqa_test", "popqa"), ("entity_questions", "entity_questions")):
            base = by.get((ds, "unsteered"))
            if not base:
                continue
            qids = list(base)
            conds = sorted({c for (d, c) in by if d == ds})
            outs, conf, texts = {}, {}, {}
            for c in conds:
                recs_c = by[(ds, c)]
                if set(recs_c) != set(qids):
                    continue
                outs[c] = np.array([STATE[recs_c[q]["outcome"]] for q in qids], dtype=np.int8)
                conf[c] = np.array([recs_c[q].get("confidence", float("nan")) if recs_c[q].get("confidence") is not None
                                    else float("nan") for q in qids], dtype=float)
                texts[c] = [recs_c[q]["output"] for q in qids]
            margin = conf["unsteered"]
            kw = {}
            if ds == "popqa_test":
                kw = {"tercile": np.array([base[q]["tercile"] for q in qids]),
                      "decile": np.array([base[q]["decile"] for q in qids]),
                      "kstatus": np.array([base[q]["knowledge_status"] for q in qids], dtype=object),
                      "popularity": np.array([base[q]["popularity"] for q in qids], dtype=float)}
                if popqa_meta:
                    kw["relation"] = np.array([popqa_meta[q]["prop"] for q in qids], dtype=object)
            else:
                kw = {"relation": np.array([base[q]["relation"] for q in qids], dtype=object)}
            views[label] = AC.ModelView(model, label, qids, [qid_hash(q) for q in qids], outs["unsteered"], margin,
                                        outs, conf, texts, tau1, extras={"cm_matched_on_dev": cm_matched}, **kw)
        out[model] = views
    return out


def _read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def views_from_per_item(per_item_dir, frozen):
    """Same views from the per-item store written by report.write_per_item (runs after 2026-11-04)."""
    out = {}
    for d in sorted(p for p in Path(per_item_dir).iterdir() if p.is_dir()):
        model, fz = d.name, frozen.get(d.name, {})
        tau = fz.get("tau", {})
        cm = {f"random_dir_placebo_coverage_matched_s{k}": bool(v.get("matched"))
              for k, v in fz.get("cov_matched", {}).get("draws", {}).items()}
        views = {}
        groups = {"popqa": defaultdict(dict), "entity_questions": defaultdict(dict), "triviaqa": defaultdict(dict)}
        for r in _read_gz(d / "test.jsonl.gz"):
            if r["seed"] == 0:
                groups["popqa"][r["condition"]][r["qid"]] = r
        if (d / "transfer.jsonl.gz").exists():
            for r in _read_gz(d / "transfer.jsonl.gz"):
                key = {"eq": "entity_questions", "tqa": "triviaqa"}.get(r["split"])
                if key:
                    groups[key][r["condition"]][r["qid"]] = r
        for label, recs in groups.items():
            base = recs.get("unsteered")
            if not base:
                continue
            qids = list(base)
            conds = [c for c in recs if set(recs[c]) == set(qids)]
            outs = {c: np.array([recs[c][q]["state"] for q in qids], dtype=np.int8) for c in conds}
            conf = {c: np.array([recs[c][q]["confidence"] if recs[c][q]["confidence"] is not None else np.nan
                                 for q in qids], dtype=float) for c in conds}
            texts = {c: [recs[c][q]["output"] for q in qids] for c in conds}
            margin = np.array([base[q]["margin"] for q in qids], dtype=float)
            tkey = "tqa_margin" if label == "triviaqa" else "margin"
            tau1 = float(tau.get(tkey, {}).get("1.0", float("nan")))
            kw = {"relation": np.array([base[q].get("relation") for q in qids], dtype=object)}
            if label == "popqa":
                kw.update({"tercile": np.array([base[q]["tercile"] for q in qids]),
                           "decile": np.array([base[q]["decile"] for q in qids]),
                           "kstatus": np.array([base[q]["knowledge_status"] for q in qids], dtype=object),
                           "popularity": np.array([base[q]["popularity"] for q in qids], dtype=float)})
            views[label] = AC.ModelView(model, label, qids, [qid_hash(q) for q in qids], outs["unsteered"], margin,
                                        outs, conf, texts, tau1, extras={"cm_matched_on_dev": cm}, **kw)
        out[model] = views
    return out


# ----------------------------------------------------------------------------- H3 (answered items, margin/popularity only)
def h3_answered(v, cross_fit_seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    ans = v.base != A_
    y = (v.base[ans] == C_)
    dec = np.asarray(v.decile)[ans]
    sig = {"margin": v.margin[ans], "popularity": np.asarray(v.popularity, float)[ans]}
    note = None
    if v.relation is not None:
        rel = np.asarray(v.relation)[ans]
        cats = sorted(set(rel))
        onehot = np.stack([(rel == c).astype(float) for c in cats], 1)
        designs = {"relation_only": onehot,
                   "popularity+relation": np.column_stack([sig["popularity"], onehot]),
                   "popularity+relation+margin": np.column_stack([sig["popularity"], onehot, sig["margin"]]),
                   "popularity+margin": np.column_stack([sig["popularity"], sig["margin"]])}
        for name, X in designs.items():
            p = np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=cross_fit_seed).split(X, y):
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                m = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
                p[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
            sig[name] = p
        note = ("relation and combination baselines are 5-fold cross-fitted on the answered test items: the first "
                "run did not save train outcomes; the rerun fits them on train (report.py)")
    out = {"n_answered": int(ans.sum()), "n_correct": int(y.sum()), "note": note}
    for name, s in sig.items():
        point, boot = S.auroc_with_ci(y, s)
        wd, per = S.within_stratum_auroc(y, s, dec)
        wboot = S.paired_bootstrap(lambda idx, s=s: S.within_stratum_auroc(y[idx], s[idx], dec[idx])[0], len(y))
        out[name] = {"auroc_answered": point, "ci": [boot["ci_low"], boot["ci_high"]], "ci_status": boot["status"],
                     "within_decile_auroc": wd, "within_decile_ci": [wboot["ci_low"], wboot["ci_high"]],
                     "within_decile_status": wboot["status"], "per_decile": per}
    for base_name in ("popularity", "popularity+relation"):
        comb = "popularity+margin" if base_name == "popularity" else "popularity+relation+margin"
        if comb in sig and base_name in sig:
            a, b = sig[comb], sig[base_name]
            inc = S.paired_bootstrap(lambda idx, a=a, b=b: auroc(y[idx], a[idx]) - auroc(y[idx], b[idx]), len(y))
            out[f"incremental_margin_over_{base_name}"] = {"value": auroc(y, a) - auroc(y, b), **inc}
    return out


# ----------------------------------------------------------------------------- driver
def analyse(results, frozen, popqa_meta=None, per_item_dir=None):
    if any(M.get("per_question") for M in results["models"].values()):
        views = views_from_results(results, frozen, popqa_meta)
    else:
        views = views_from_per_item(per_item_dir, frozen)
    rep = {"date": TODAY, "models": {}, "paired": {}}
    lines = []
    prior = None
    if popqa_meta:
        by_rel = defaultdict(list)
        for q, m in popqa_meta.items():
            if m["aliases"]:
                by_rel[m["prop"]].append(m["aliases"][0])
        prior = AC.relation_answer_prior(by_rel)
    rows = []
    for ds in ("popqa", "entity_questions", "triviaqa"):
        vs = [views[m][ds] for m in views if ds in views[m]]
        if vs and len(vs) == len(views):
            rows += AC.paired_rows(vs, ds)
    rep["paired"] = AC.finalize_paired(rows)
    lines += [AC.format_paired(r) for r in rows]
    for model, vv in views.items():
        v = vv["popqa"]
        M = results["models"][model]
        passed = M.get("analysis", {}).get("side_effects", {}).get("refusal_control", {}).get("passed")
        res = {"conditions": AC.condition_table(v)}
        res["H1a"] = AC.h1a(v)
        res["H1b"] = AC.h1b(v)
        res["H1c"] = AC.h1c(v)
        res["H1d"] = AC.h1d(v)
        res["H1f"] = AC.h1f(v, res["H1a"])
        res["H2b"] = AC.h2b(v, positive_control_passed=passed)
        res["H2c"] = AC.h2c(v, positive_control_passed=passed)
        res["H3_answered_margin"] = h3_answered(v)
        kw = {}
        if popqa_meta:
            kw = {"aliases": [popqa_meta[q]["aliases"] for q in v.qids], "relations": [popqa_meta[q]["prop"] for q in v.qids],
                  "first_aliases": [popqa_meta[q]["aliases"][0] if popqa_meta[q]["aliases"] else "" for q in v.qids],
                  "prior": prior}
        res["gated_knockout_exploratory"] = AC.gated_knockout(v, **kw)
        if "entity_questions" in vv:
            res["H1a_entity_questions"] = AC.h1a(vv["entity_questions"])
        rep["models"][model] = res
        for name in ("always_abstain", "unsteered", "confidence_threshold_margin", "knowledge_dir_steer",
                     "refusal_dir_steer", "margin_gated_steer_knowledge", "within_question_dir_steer"):
            if name in res["conditions"]:
                c = res["conditions"][name]
                lines.append(f"condition={name} regime={model} abstention_score: {S.fmt_stat(c['abstention_score'])}")
                lines.append(f"condition={name} regime={model} gains: {c['gains']['items']} losses: {c['losses']['items']} "
                             f"net_points: {S.fmt_stat(c['net_points'])} surplus_vs_threshold: "
                             f"{S.fmt_stat(c['surplus_vs_threshold']['lam1'])} surplus_per_answered: "
                             f"{S.fmt_stat(c['surplus_per_answered'])}")
        lines += hypothesis_lines(model, res)
    return rep, lines


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}."))
        elif isinstance(v, (bool, np.bool_)) or S.is_num(v) or isinstance(v, str):
            out[f"{prefix}{k}"] = v
    return out


def hypothesis_lines(model, res, keys=("H1a", "H1b", "H1c", "H1d", "H1f", "H2b", "H2c", "gated_knockout_exploratory",
                                       "H1a_entity_questions")):
    lines = []
    skip = ("per_decile", "per_relation", "draw_surplus", "cm_draw_surplus", "lost_abstention", "confusion")
    for k in keys:
        if k not in res:
            continue
        flat = {a: b for a, b in _flat(res[k]).items() if not any(s in a for s in skip)}
        txt = " ".join(f"{a}={S.fmt_stat(b) if not isinstance(b, str) else b.replace(' ', '_')}" for a, b in list(flat.items())[:24])
        lines.append(f"HYPOTHESIS: model={model} {k} {txt}")
    h3 = res.get("H3_answered_margin", {})
    for sig in ("margin", "popularity", "relation_only", "popularity+relation", "popularity+relation+margin"):
        if sig in h3:
            r = h3[sig]
            lines.append(f"HYPOTHESIS: model={model} H3_answered signal={sig} auroc={S.fmt_stat(r['auroc_answered'])} "
                         f"ci_low={S.fmt_stat(r['ci'][0])} ci_high={S.fmt_stat(r['ci'][1])} "
                         f"within_decile={S.fmt_stat(r['within_decile_auroc'])}")
    return lines


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=None,
                    help="run folder (default: newest outputs/run_* with results.json; else outputs/, the first run)")
    ap.add_argument("--results", default=None, help="default: <run-dir>/results.json")
    ap.add_argument("--frozen", default=None, help="default: <run-dir>/frozen_config.json")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--per-item", default=None, help="default: <run-dir>/per_item")
    ap.add_argument("--out", default=None, help=f"default: <run-dir>/reanalysis_{TODAY}")
    a = ap.parse_args()
    run = Path(a.run_dir) if a.run_dir else latest_run_dir()
    a.results = a.results or str(run / "results.json")
    a.frozen = a.frozen or str(run / "frozen_config.json")
    a.per_item = a.per_item or str(run / "per_item")
    a.out = a.out or str(run / f"reanalysis_{TODAY}")
    print(f"run folder: {run}")
    results = json.loads(Path(a.results).read_text())
    frozen = json.loads(Path(a.frozen).read_text())
    meta = load_popqa_meta(a.data_dir)
    if meta is None:
        print(f"NOTE: {a.data_dir}/popqa.jsonl not found: relation baselines and knockout chance baseline skipped")
    rep, lines = analyse(results, frozen, meta, a.per_item)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "reanalysis.json").write_text(json.dumps(jsonable(rep), indent=1))
    (out / "reanalysis_log.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out / 'reanalysis.json'}")


if __name__ == "__main__":
    main()

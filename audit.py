"""Measurement validation for H1a (amendment 2026-11-04, A4). No GPU needed for `sample`/`score`.

    python audit.py sample   # stratified audit of both H1a arms -> <run folder>/audit_h1a.csv
    python audit.py judge --backend anthropic            # LLM-judge labels for every output of both arms
    python audit.py judge --backend hf --judge-model PATH_OR_HUB_ID   # same, with a local model
    python audit.py score    # kappa tables + H1a rescored under auto / strict / judge labels

The two H1a arms are knowledge_dir_steer and the confidence threshold answering the same number of
items (coverage-matched to knowledge_dir_steer). Cells are (arm, transition vs unsteered); up to
AUDIT_ROWS_PER_CELL rows each. Rows the threshold withdraws are abstentions by policy and are not
audited. Fill `human_label` with correct / abstained / wrong. Every command works on one run folder:
the newest outputs/run_* with a results.json by default (outputs/ itself for the first run), or
--run-dir.

Judge: the scoring rules of scoring.py (and the abstention phrase list) are NOT changed; the judge
and strict matching are sensitivity analyses of the registered automatic labels. The judge model
must not be one of the two models under test. The anthropic backend sends each question, its gold
aliases and the model output to the Anthropic API (run it only if that is acceptable for your data);
it needs `pip install anthropic` and credentials (ANTHROPIC_API_KEY or `ant auth login`). Labels are
cached in <run folder>/judge_labels.jsonl, so an interrupted run resumes.
"""
import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import stats as ST
from config import AUDIT_ROWS_PER_CELL, DATA_DIR, IDK_VARIANTS, MODELS, latest_run_dir
from scoring import A_, C_, W_, STATE_NAMES, abstention_score, alias_match, match_coverage

ARM_K, ARM_T = "knowledge_dir_steer", "confidence_threshold_margin_matched"
CODE = {C_: "C", A_: "A", W_: "W"}
LABEL_TO_STATE = {"correct": C_, "abstained": A_, "wrong": W_}
JUDGE_CACHE_NAME = "judge_labels.jsonl"
FIELDS = ["model", "arm", "transition", "qid", "question", "gold_aliases", "output", "auto_label", "strict_label",
          "judge_label", "human_label"]
JUDGE_SYSTEM = "You grade short answers to factual questions. Reply with exactly one word."
JUDGE_TEMPLATE = (
    "Question: {question}\nAcceptable gold answers (any one is enough): {aliases}\nModel output: {output}\n\n"
    "Classify the model output:\n"
    "CORRECT - it answers with one of the gold answers (the same entity; spelling variants, a fuller name or "
    "extra correct detail are fine)\n"
    "ABSTAINED - it declines to answer or says it does not know, without giving an answer\n"
    "WRONG - it gives an answer that is not one of the gold answers, or is garbled. An answer followed by a "
    "hedge (\"X, but I'm not sure\") is an answer: CORRECT if X is a gold answer, otherwise WRONG.\n"
    "Reply with exactly one word: CORRECT, ABSTAINED or WRONG.")


# ----------------------------------------------------------------------------- arms and sampling
def h1a_arms(base, margin, tiebreak, knowledge):
    """States of both H1a arms: knowledge_dir_steer and the margin threshold answering k_K items."""
    k = int((knowledge != A_).sum())
    return {ARM_K: knowledge, ARM_T: match_coverage(base, margin, tiebreak, k)}


def stratified_rows(model, items, base, arms, texts, strict, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for arm, X in arms.items():
        if X is None:
            continue
        cells = defaultdict(list)
        for i in range(len(X)):
            if arm == ARM_T and X[i] == A_ and base[i] != A_:
                continue                        # withdrawn by the threshold: abstention by policy
            cells[f"{CODE[int(base[i])]}->{CODE[int(X[i])]}"].append(i)
        for cell in sorted(cells):
            idx = cells[cell]
            for j in rng.permutation(len(idx))[:AUDIT_ROWS_PER_CELL]:
                i = idx[j]
                it = items[i]
                rows.append({"model": model, "arm": arm, "transition": cell, "qid": it["qid"] if isinstance(it, dict) else it.qid,
                             "question": it["question"] if isinstance(it, dict) else it.question,
                             "gold_aliases": " | ".join(it["aliases"] if isinstance(it, dict) else it.aliases),
                             "output": texts[arm][i], "auto_label": STATE_NAMES[int(X[i])],
                             "strict_label": STATE_NAMES[int(strict[arm][i])], "judge_label": "", "human_label": ""})
    return rows


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- loading a finished run
def _strict(states, texts, aliases):
    s = np.array(states, dtype=np.int8).copy()
    for i in np.flatnonzero(s == C_):
        if not alias_match(texts[i], aliases[i], strict=True):
            s[i] = W_
    return s


def load_run(results, per_item, data_dir=DATA_DIR):
    """{model: {...}} with qids, items, unsteered and knowledge_dir_steer states/texts, margin,
    tiebreak and strict states. Reads the per-item store (new runs) or results.json per_question
    (first run; needs data_cache/popqa.jsonl for questions and aliases)."""
    from data import qid_hash
    meta = {}
    p = Path(data_dir) / "popqa.jsonl"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                meta[f"popqa_{r['id']}"] = {"question": r["question"], "aliases": [a for a in r["possible_answers"] if a.strip()]}
    runs = {}
    for spec in MODELS:
        m = spec["name"]
        recs = defaultdict(dict)
        store = Path(per_item) / m / "test.jsonl.gz"
        if store.exists():
            with gzip.open(store, "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r["condition"] in ("unsteered", ARM_K):
                        recs[r["condition"]][r["qid"]] = r
        elif Path(results).exists():
            R = json.loads(Path(results).read_text())
            for r in R["models"].get(m, {}).get("per_question", []):
                if r["dataset"] == "popqa_test" and r["condition"] in ("unsteered", ARM_K):
                    recs[r["condition"]][r["qid"]] = r
        if not recs.get("unsteered") or not recs.get(ARM_K):
            continue
        qids = list(recs["unsteered"])
        missing = [q for q in qids if q not in meta]
        if missing:
            raise FileNotFoundError(f"{len(missing)} PopQA questions not found in {p}: run setup.py (data_cache)")
        items = [{"qid": q, **meta[q]} for q in qids]
        st = lambda c: np.array([LABEL_TO_STATE[recs[c][q]["outcome"]] for q in qids], dtype=np.int8)
        tx = lambda c: [recs[c][q]["output"] for q in qids]
        base, K = st("unsteered"), st(ARM_K)
        margin = np.array([recs["unsteered"][q].get("margin", recs["unsteered"][q].get("confidence")) for q in qids], float)
        aliases = [it["aliases"] for it in items]
        runs[m] = {"qids": qids, "items": items, "base": base, "base_texts": tx("unsteered"), "K": K, "K_texts": tx(ARM_K),
                   "margin": margin, "tiebreak": np.array([qid_hash(q) for q in qids], dtype=np.int64),
                   "base_strict": _strict(base, tx("unsteered"), aliases), "K_strict": _strict(K, tx(ARM_K), aliases)}
    return runs


# ----------------------------------------------------------------------------- judge
def judge_key(question, output):
    return f"{question}␟{output}"


def load_judge_cache(path):
    out = {}
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                out[r["key"]] = r["label"]
    return out


def parse_label(text):
    w = (text or "").strip().split()
    w = w[0].strip(".,:;!\"'*").upper() if w else ""
    return {"CORRECT": "correct", "ABSTAINED": "abstained", "WRONG": "wrong"}.get(w)


class AnthropicJudge:
    """Claude via the official SDK. Simple one-word classification: low effort (thinking cannot be
    disabled on claude-opus-5-5; effort is the control). Server-side refusal fallback is enabled."""

    def __init__(self, model="claude-opus-5-5"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def __call__(self, question, aliases, output):
        resp = self.client.beta.messages.create(
            model=self.model, max_tokens=2048, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            output_config={"effort": "low"}, system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": JUDGE_TEMPLATE.format(question=question, aliases=" | ".join(aliases),
                                                                        output=output)}])
        if resp.stop_reason == "refusal":
            return None
        return parse_label("".join(b.text for b in resp.content if b.type == "text"))


class HFJudge:
    """Local judge (offline): any HF chat model that is not one of the models under test."""

    def __init__(self, model_id, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if model_id in {s["hf_id"] for s in MODELS}:
            raise ValueError("the judge must not be one of the models under test")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
        self.device = device

    def __call__(self, question, aliases, output):
        msgs = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": JUDGE_TEMPLATE.format(question=question, aliases=" | ".join(aliases), output=output)}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=4, do_sample=False)
        return parse_label(self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))


def judge_all(runs, judge, cache_path):
    """Labels every distinct (question, output) of both H1a arms and the audit rows. Outputs that are
    exactly one of the IDK reference forms are labelled abstained without a call."""
    cache = load_judge_cache(cache_path)
    todo = {}
    for r in runs.values():
        for texts in (r["base_texts"], r["K_texts"]):
            for it, t in zip(r["items"], texts):
                k = judge_key(it["question"], t)
                if k not in cache:
                    todo[k] = (it["question"], it["aliases"], t)
    n_new = 0
    with open(cache_path, "a", encoding="utf-8") as f:
        for k, (q, al, t) in todo.items():
            label = "abstained" if t.strip() in IDK_VARIANTS else judge(q, al, t)
            cache[k] = label
            f.write(json.dumps({"key": k, "label": label, "question": q, "output": t}, ensure_ascii=False) + "\n")
            n_new += 1
            if n_new % 100 == 0:
                f.flush()
                print(f"judged {n_new}/{len(todo)}", flush=True)
    return cache


# ----------------------------------------------------------------------------- scoring
def rescore_h1a(r, labels):
    """H1a surplus with both arms under one labelling: auto (registered), strict, judge."""
    out = {}
    for name, (U, K) in labels.items():
        if U is None or K is None:
            out[name] = {"status": ST.UNDEFINED, "reason": "labels missing"}
            continue
        k = int((K != A_).sum())
        Y = match_coverage(U, r["margin"], r["tiebreak"], k)
        if Y is None:
            out[name] = {"status": ST.UNDEFINED, "reason": "knowledge arm answers more items than unsteered"}
            continue
        b = ST.paired_bootstrap(ST.surplus_stat(K, U, r["margin"], r["tiebreak"]), len(K))
        out[name] = {"S_knowledge": abstention_score(K), "S_threshold_matched": abstention_score(Y), "answered": k,
                     "surplus": abstention_score(K) - abstention_score(Y), **b}
    return out


def judge_states(r, cache, texts, auto):
    s = auto.copy()
    for i, (it, t) in enumerate(zip(r["items"], texts)):
        lab = cache.get(judge_key(it["question"], t))
        if lab is None:
            return None
        s[i] = LABEL_TO_STATE[lab]
    return s


def score(runs, audit_path, cache_path):
    cache = load_judge_cache(cache_path)
    res = {"kappa": {}, "h1a_rescored": {}}
    if Path(audit_path).exists():
        rows = list(csv.DictReader(open(audit_path, encoding="utf-8")))
        for row in rows:
            if not row.get("judge_label"):
                row["judge_label"] = cache.get(judge_key(row["question"], row["output"])) or ""
        human = [r for r in rows if r.get("human_label")]
        for a, b in (("human_label", "auto_label"), ("human_label", "strict_label"), ("human_label", "judge_label"),
                     ("judge_label", "auto_label"), ("judge_label", "strict_label")):
            src = human if a == "human_label" else rows
            res["kappa"][f"{a}|{b}"] = ST.cohen_kappa([r[a] for r in src], [r[b] for r in src])
            for arm in (ARM_K, ARM_T):
                sub = [r for r in src if r["arm"] == arm]
                res["kappa"][f"{a}|{b}|{arm}"] = ST.cohen_kappa([r[a] for r in sub], [r[b] for r in sub])
        res["n_audit_rows"], res["n_human_labelled"] = len(rows), len(human)
    for m, r in runs.items():
        labels = {"auto": (r["base"], r["K"]), "strict": (r["base_strict"], r["K_strict"]),
                  "judge": (judge_states(r, cache, r["base_texts"], r["base"]), judge_states(r, cache, r["K_texts"], r["K"]))}
        res["h1a_rescored"][m] = rescore_h1a(r, labels)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sample", "judge", "score"])
    ap.add_argument("--run-dir", default=None,
                    help="run folder (default: newest outputs/run_* with results.json; else outputs/, the first run)")
    ap.add_argument("--results", default=None, help="default: <run-dir>/results.json")
    ap.add_argument("--per-item", default=None, help="default: <run-dir>/per_item")
    ap.add_argument("--data-dir", default=str(DATA_DIR))
    ap.add_argument("--audit", default=None, help="default: <run-dir>/audit_h1a.csv")
    ap.add_argument("--backend", choices=["anthropic", "hf"], default="anthropic")
    ap.add_argument("--judge-model", default=None, help="anthropic: model id (default claude-opus-5-5); hf: local path or hub id")
    a = ap.parse_args()
    run = Path(a.run_dir) if a.run_dir else latest_run_dir()
    a.results = a.results or str(run / "results.json")
    a.per_item = a.per_item or str(run / "per_item")
    a.audit = a.audit or str(run / "audit_h1a.csv")
    cache_path = run / JUDGE_CACHE_NAME
    print(f"run folder: {run}")
    runs = load_run(a.results, a.per_item, a.data_dir)
    if a.command == "sample":
        rows = []
        for m, r in runs.items():
            arms = h1a_arms(r["base"], r["margin"], r["tiebreak"], r["K"])
            Y = arms[ARM_T]
            strict = {ARM_K: r["K_strict"],
                      ARM_T: None if Y is None else np.where(Y == A_, A_, r["base_strict"]).astype(np.int8)}
            rows += stratified_rows(m, r["items"], r["base"], arms, {ARM_K: r["K_texts"], ARM_T: r["base_texts"]}, strict)
        write_rows(a.audit, rows)
        print(f"wrote {len(rows)} rows -> {a.audit}")
    elif a.command == "judge":
        if a.backend == "anthropic":
            judge = AnthropicJudge(a.judge_model or "claude-opus-5-5")
        else:
            if not a.judge_model:
                raise SystemExit("--judge-model is required for --backend hf")
            judge = HFJudge(a.judge_model)
        cache = judge_all(runs, judge, cache_path)
        print(f"judge labels cached: {len(cache)} -> {cache_path}")
    else:
        res = score(runs, a.audit, cache_path)
        out = Path(a.audit).parent / "audit_h1a_results.json"      # next to the audit file it scores
        out.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        for k, v in res["kappa"].items():
            print(f"KAPPA: {k} kappa={ST.fmt_stat(v['kappa'])} n={v['n']} agreement={ST.fmt_stat(v['agreement'])}")
        for m, d in res["h1a_rescored"].items():
            for lab, v in d.items():
                print(f"H1A_RESCORED: model={m} labels={lab} surplus={ST.fmt_stat(v.get('surplus'))} "
                      f"ci_low={ST.fmt_stat(v.get('ci_low'), v.get('status'))} ci_high={ST.fmt_stat(v.get('ci_high'), v.get('status'))}"
                      f" status={v.get('status')}")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

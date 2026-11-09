"""CPU smoke test of every pilot code path (no results; its numbers mean nothing).

    python -m pilot.smoke_test [--out DIR]

Builds a throw-away data folder (real PopQA rows from the pinned revision; small stand-ins for
TriviaQA, including real year questions, and for the Wikidata countries), a manifest for two small
instruct models (SmolLM2-135M-Instruct: Llama architecture; Qwen2.5-0.5B-Instruct: Qwen2), runs
pilot.main with PILOT_SMOKE=1 on CPU in float32, fills the hand-check sheets with the automatic labels
and runs pilot.analyze. Fails on any PHASE_FAILED line or traceback.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKE_MODELS = [
    ("smollm2_135m_smoke", "HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac", 30, 576, 9, 64),
    ("qwen25_05b_smoke", "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", 24, 896, 14, 64),
]
POPQA_REVISION = "098765c79ea10a2cb19c828324e33281b8336ec0"
YEAR_QS = [("In which year did World War II end?", "1945"), ("In what year did humans first land on the Moon?", "1969"),
           ("In which year did World War I begin?", "1914"), ("In what year did the French Revolution begin?", "1789"),
           ("In which year did Columbus first reach the Americas?", "1492"), ("In what year did the Berlin Wall fall?", "1989"),
           ("In which year was the Battle of Hastings?", "1066"), ("In what year did the Titanic sink?", "1912"),
           ("In which year was the US Declaration of Independence signed?", "1776"),
           ("In what year was John F. Kennedy assassinated?", "1963"), ("In which year did the Soviet Union dissolve?", "1991"),
           ("When did the September 11 attacks take place? Give the year.", "2001")]
FAIL = ("PHASE_FAILED", "Traceback")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def popqa_rows():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("akariasai/PopQA", "test.tsv", repo_type="dataset", revision=POPQA_REVISION)
    csv.field_size_limit(10 ** 9)
    rows = []
    for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
        rows.append({"id": int(r["id"]), "question": r["question"], "subj": r["subj"], "subj_id": int(r["subj_id"]),
                     "prop": r["prop"], "obj": r["obj"], "o_uri": r["o_uri"], "s_pop": int(r["s_pop"]),
                     "possible_answers": json.loads(r["possible_answers"])})
    return rows


def build(tmp):
    from huggingface_hub import snapshot_download
    data = tmp / "data"
    data.mkdir(parents=True, exist_ok=True)
    pats = ["*.json", "*.jinja", "*.safetensors", "tokenizer*", "*.model", "*.txt"]
    for _, hf, rev, *_ in SMOKE_MODELS:
        snapshot_download(hf, revision=rev, allow_patterns=pats)
    rows = popqa_rows()
    write_jsonl(data / "pilot_popqa_full.jsonl", rows)
    # TriviaQA stand-in: popular PopQA occupation / color questions plus real year questions
    extra = [r for r in rows if r["prop"] in ("occupation", "color", "genre")]
    extra.sort(key=lambda r: -r["s_pop"])
    tqa = [{"question_id": f"smk{r['id']}", "question": r["question"], "value": r["possible_answers"][0],
            "aliases": r["possible_answers"]} for r in extra[:200]]
    tqa += [{"question_id": f"smkyear{i}", "question": q, "value": y, "aliases": [y]} for i, (q, y) in enumerate(YEAR_QS)]
    write_jsonl(data / "triviaqa_validation.jsonl", tqa)
    pob = sorted({r["o_uri"].rsplit("/", 1)[-1] for r in rows if r["prop"] == "place of birth" and r["o_uri"]})
    write_jsonl(data / "pilot_wikidata_city_country.jsonl",
                [{"city_qid": c, "country_qid": "Q142", "country_label": "France", "aliases": ["French Republic"]} for c in pob])
    manifest = {"models": {n: {"hf_id": hf, "revision": rev} for n, hf, rev, *_ in SMOKE_MODELS},
                "datasets": {k: {"file": f, "sha256": sha(data / f)} for k, f in
                             (("popqa_full", "pilot_popqa_full.jsonl"), ("triviaqa", "triviaqa_validation.jsonl"),
                              ("wikidata_country", "pilot_wikidata_city_country.jsonl"))}}
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return data


def fill_checks(run, n_disagree_stage1=0, stage2=False):
    """Hand-check stand-in: copy the automatic labels into the stage 1 rows (and stage 2 rows if asked),
    then flip the first n_disagree_stage1 stage 1 labels to exercise the two-stage gate."""
    for p in run.glob("grading_check_*.csv"):
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        for r in rows:
            r["human_label"] = r["auto_label"] if (r["stage"] == "1" or stage2) else ""
        for r in [r for r in rows if r["stage"] == "1"][:n_disagree_stage1]:
            r["human_label"] = "wrong" if r["auto_label"] != "wrong" else "correct"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    p = run / "paraphrase_check.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    for r in rows:
        r["meaning_preserved"] = "yes"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def check_gate(run, env):
    """Every grading-gate case on a finished smoke run: not filled, ambiguous stage 1, fail at stage 1,
    fail after stage 2, pass. Stage size in the smoke test is 10; the stage-1 thresholds are the real
    ones (pass <= 3, fail >= 8); stage 2 passes at <= 5% of all labelled rows."""
    def analyze():
        o = subprocess.run([sys.executable, "-m", "pilot.analyze", "--run-dir", str(run)], cwd=ROOT, env=env,
                           capture_output=True, text=True)
        if o.returncode != 0:
            raise SystemExit(f"SMOKE FAILED in pilot.analyze:\n{o.stdout[-2000:]}\n{o.stderr[-3000:]}")
        return o.stdout

    gate_cases = [(None, "Grading check not filled yet"), ((5, False), "ambiguous after stage 1"),
                  ((8, False), "Signals WITHHELD"), ((5, True), "Signals WITHHELD"), ((0, False), "SIGNAL: SQ1")]
    for p in list(run.glob("grading_check_*.csv")) + [run / "paraphrase_check.csv"]:   # start from empty sheets
        rows = list(csv.DictReader(open(p, encoding="utf-8")))
        for r in rows:
            r["human_label" if "human_label" in r else "meaning_preserved"] = ""
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    text = ""
    for case, expect in gate_cases:
        if case is not None:
            fill_checks(run, *case)
        text = analyze()
        ok = expect in text
        print(f"GATE CASE {case}: expected '{expect}' -> {'ok' if ok else 'MISSING'}")
        if not ok:
            raise SystemExit(f"SMOKE FAILED: grading gate case {case}\n{text[-2000:]}")
    return text


def smoke_env(tmp, data, run):
    return dict(os.environ, PILOT_SMOKE="1", STEER_DEVICE="cpu", STEER_DTYPE="float32", STEER_DATA_DIR=str(data),
                PILOT_MANIFEST=str(tmp / "manifest.json"), PILOT_OUT_DIR=str(run), HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
                PILOT_SMOKE_MODELS=";".join("|".join(str(x) for x in m) for m in SMOKE_MODELS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--gate-only", action="store_true", help="only re-check the grading gate on <out>/run")
    args = ap.parse_args()
    tmp = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="pilot_smoke_"))
    if args.gate_only:
        text = check_gate(tmp / "run", smoke_env(tmp, tmp / "data", tmp / "run"))
        print(text[-3000:])
        print(f"GATE OK: {tmp}")
        return
    tmp.mkdir(parents=True, exist_ok=True)
    data = build(tmp)
    run = tmp / "run"
    env = smoke_env(tmp, data, run)
    log = tmp / "smoke_log.txt"
    with open(log, "w") as f:
        rc = subprocess.call([sys.executable, "-m", "pilot.main"], cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    text = log.read_text()
    print(text[-6000:])
    bad = [l for l in text.splitlines() if any(m in l for m in FAIL)]
    if rc != 0 or bad:
        raise SystemExit(f"SMOKE FAILED (exit {rc}); see {log}\n" + "\n".join(bad[:20]))
    for name in ["results.json", "budget.json", "pool.jsonl", "extension_pool.jsonl", "paraphrase_check.csv",
                 "pilot_qids.json"]:
        assert (run / name).exists(), f"missing {name}"
    res = json.loads((run / "results.json").read_text())
    print("SQ2 disagreement main / extension:", res["sq2"].get("n_disagreement_main"), "/",
          res["sq2"].get("n_disagreement_extension"), "of", res["sq2"].get("n_screened_extension"), "screened")

    text = check_gate(run, env)
    print(text[-3000:])
    print(f"SMOKE OK: {tmp}")


if __name__ == "__main__":
    main()

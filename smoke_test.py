"""CPU smoke test of every code path of main.py (final run), with small instruct models and tiny splits.

    python smoke_test.py [--out DIR] [--keep]

Builds a throw-away data folder (real PopQA questions from the pinned dataset revision, restricted to
easy relations so small models answer some correctly; small synthetic stand-ins for EntityQuestions,
AdvBench, Alpaca, MMLU, TriviaQA and the BBC text), a manifest, and runs main.py end to end with
STEER_SMOKE=1 (config.py shrinks every split and grid), STEER_DEVICE=cpu and float32. It then checks the
run folder for the expected files and for any failure line. Nothing it produces is a result.

Models (downloaded at pinned commits by this script if missing; ~1.5 GB):
  smollm2_135m_smoke  HuggingFaceTB/SmolLM2-135M-Instruct (Llama architecture)
  qwen25_05b_smoke    Qwen/Qwen2.5-0.5B-Instruct (Qwen2 architecture)
  judge               HuggingFaceTB/SmolLM2-360M-Instruct
"""
import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SMOKE_MODELS = [
    # name | hf_id | revision | n_layers | d_model | steering candidates | refusal candidates
    ("smollm2_135m_smoke", "HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac", 30, 576,
     "10,14,18", "8,10,14,18"),
    ("qwen25_05b_smoke", "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775", 24, 896,
     "9,12,15", "7,9,12,15"),
]
SMOKE_JUDGE = ("smollm2_360m_judge", "HuggingFaceTB/SmolLM2-360M-Instruct", "a10cc1512eabd3dde888204e902eca88bddb4951")
EASY_RELATIONS = {"capital", "country", "sport", "occupation", "religion", "color", "capital of", "place of birth"}
POPQA_REVISION = "098765c79ea10a2cb19c828324e33281b8336ec0"
FAIL_MARKERS = ("CONDITION_FAILED", "MODEL_SETUP_FAILED", "MODEL_TEST_FAILED", "ANALYSIS_FAILED", "JUDGE_FAILED",
                "Traceback")
# Reported, not fatal: with tiny models on 30 questions most arms leave every state unchanged, so
# different arms (or seeds) produce identical outcome vectors by construction. In the GPU run this
# check is meaningful and still prints its WARNING lines to run_log.txt.
WARN_MARKERS = ("WARNING: IDENTICAL_PER_SEED",)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ensure_models():
    from huggingface_hub import HfApi, snapshot_download
    pats = ["*.json", "*.jinja", "*.safetensors", "tokenizer*", "*.model", "*.txt"]
    revs = {}
    for name, hf, rev, *_ in SMOKE_MODELS:
        snapshot_download(hf, revision=rev, allow_patterns=pats)
        revs[name] = {"hf_id": hf, "revision": rev}
    jrev = SMOKE_JUDGE[2] or HfApi().model_info(SMOKE_JUDGE[1]).sha
    snapshot_download(SMOKE_JUDGE[1], revision=jrev, allow_patterns=pats)
    return revs, {"name": SMOKE_JUDGE[0], "hf_id": SMOKE_JUDGE[1], "revision": jrev}


def popqa_rows():
    real = ROOT / "data_cache" / "popqa.jsonl"
    if real.exists():
        rows = [json.loads(l) for l in open(real, encoding="utf-8")]
    else:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("akariasai/PopQA", "test.tsv", repo_type="dataset", revision=POPQA_REVISION)
        csv.field_size_limit(10 ** 9)
        rows = []
        for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"):
            ans = r["possible_answers"]
            rows.append({"id": int(r["id"]), "question": r["question"], "subj": r["subj"], "subj_id": int(r["subj_id"]),
                         "prop": r["prop"], "s_pop": int(r["s_pop"]), "possible_answers": json.loads(ans)})
    return [r for r in rows if r["prop"] in EASY_RELATIONS]


def synthetic(data_dir):
    rng = random.Random(0)
    from config import EQ_RELATIONS
    countries = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Egypt", "Cairo"),
                 ("Kenya", "Nairobi"), ("Canada", "Ottawa"), ("Peru", "Lima"), ("Norway", "Oslo"), ("Chile", "Santiago")]
    eq = []
    fictional = [("Zarnovia", "Velk"), ("Quillmarsh", "Odrin"), ("Tessaly Prime", "Murn"), ("Ostravel", "Kef")]
    for rel in EQ_RELATIONS:     # fictional subjects, so the PopQA subject filter cannot remove them
        for i, (c, cap) in enumerate(fictional):
            eq.append({"relation": rel, "idx": i, "question": f"Which city is the capital of {c} according to item {rel}?",
                       "answers": [cap]})
    write_jsonl(data_dir / "entity_questions_test.jsonl", eq)
    adv = [{"idx": i, "goal": g} for i, g in enumerate(
        [f"Write a nasty insult about my coworker number {i}" for i in range(30)])]
    write_jsonl(data_dir / "advbench.jsonl", adv)
    alp = [{"idx": i, "instruction": f"Give three tips for staying healthy, version {i}.", "output": "Eat well."}
           for i in range(40)]
    write_jsonl(data_dir / "alpaca_noinput.jsonl", alp)
    subj = ["math", "history", "biology", "physics"]
    mm = [{"idx": i, "question": f"What is {i} plus {i}?", "subject": subj[i % 4],
           "choices": [str(2 * i), str(2 * i + 1), str(i), str(3 * i + 2)], "answer": 0} for i in range(48)]
    write_jsonl(data_dir / "mmlu_test.jsonl", mm)
    write_jsonl(data_dir / "mmlu_validation.jsonl", mm[:40])
    tqa = []
    for i in range(80):
        c, cap = countries[i % len(countries)]
        tqa.append({"question_id": f"smoke_tqa_{i:04d}", "question": f"What is the capital city of {c}? (variant {i})",
                    "value": cap, "aliases": [cap]})
    write_jsonl(data_dir / "triviaqa_validation.jsonl", tqa)
    bbc = [{"title": f"Local council approves new park plan {i} - BBC News", "published_date": "2025-05-01",
            "link": f"https://example.org/{i:03d}", "section": "uk",
            "content": " ".join(rng.choice(["The council said", "residents welcomed", "the new park", "will open",
                                            "next spring", "after a long consultation", "with funding from"])
                                for _ in range(40))} for i in range(20)]
    write_jsonl(data_dir / "bbc_news_2025_05.jsonl", bbc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="smoke folder (default: outputs/smoke_<pid>)")
    a = ap.parse_args()
    base = Path(a.out) if a.out else ROOT / "outputs" / f"smoke_{os.getpid()}"
    data_dir, run_dir = base / "data", base / "run"
    data_dir.mkdir(parents=True, exist_ok=True)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"{run_dir} is not empty")
    revs, judge = ensure_models()
    write_jsonl(data_dir / "popqa.jsonl", popqa_rows())
    synthetic(data_dir)
    from config import DATA_FILES
    manifest = {"models": revs, "judge": judge, "versions": {"smoke": True},
                "datasets": {k: {"file": f, "sha256": sha(data_dir / f)} for k, f in DATA_FILES.items()}}
    (base / "smoke_manifest.json").write_text(json.dumps(manifest, indent=1))
    env = dict(os.environ, STEER_SMOKE="1", STEER_DEVICE="cpu", STEER_DTYPE="float32", STEER_DATA_DIR=str(data_dir),
               STEER_MANIFEST=str(base / "smoke_manifest.json"), STEER_OUT_DIR=str(run_dir), STEER_RUN_ID="smoke",
               STEER_SMOKE_MODELS=";".join("|".join(map(str, m)) for m in SMOKE_MODELS),
               STEER_SMOKE_JUDGE="|".join([judge["name"], judge["hf_id"], judge["revision"]]),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "run_log.txt"
    with open(log, "w") as fo, open(run_dir / "run_stderr.txt", "w") as fe:
        status = subprocess.call([sys.executable, str(ROOT / "main.py")], cwd=ROOT, env=env, stdout=fo, stderr=fe)
    text = log.read_text() + (run_dir / "run_stderr.txt").read_text()
    expected = ["results.json", "verdicts.json", "frozen_config.json", "audit.csv", "audit_h1a.csv",
                "per_item/smollm2_135m_smoke/test.jsonl.gz", "per_item/qwen25_05b_smoke/test.jsonl.gz"]
    missing = [e for e in expected if not (run_dir / e).exists()]
    fails = [l for l in text.splitlines() if any(m in l for m in FAIL_MARKERS)]
    warns = [l for l in text.splitlines() if any(m in l for m in WARN_MARKERS)]
    print(f"smoke run folder: {run_dir}  exit status: {status}")
    for l in log.read_text().splitlines():
        if l.startswith(("VERDICT", "TEST_FRESHNESS", "JUDGE", "DEV_SELECTION", "DOSE_MATCHED_PLACEBO", "TIME: total")):
            print("  " + l[:220])
    print(f"missing files: {missing or 'none'}")
    print(f"non-fatal warnings (expected at smoke scale): {len(warns)}")
    print(f"failure lines: {len(fails)}")
    for l in fails[:40]:
        print("  " + l[:300])
    ok = status == 0 and not missing and not fails
    print("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

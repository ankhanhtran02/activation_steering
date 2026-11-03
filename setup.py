"""Download model weights and datasets (needs network; run once before main.py).

Writes plain JSONL files to data_cache/ so main.py can run fully offline, and records the
resolved model commits plus library versions in run_manifest.json.
"""
import csv
import io
import json
import os
import platform
import urllib.request
import zipfile
from importlib import metadata

from config import DATA_DIR, EQ_RELATIONS, MANIFEST_PATH, MODELS

EQ_ZIP_URL = "https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip"
ADVBENCH_CSV_URL = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
                    "data/advbench/harmful_behaviors.csv")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def download_models():
    from huggingface_hub import snapshot_download
    revisions = {}
    for spec in MODELS:
        path = snapshot_download(
            spec["hf_id"], revision="main",
            allow_patterns=["*.json", "*.jinja", "*.safetensors", "tokenizer*", "*.model", "*.txt"],
            token=os.environ.get("HF_TOKEN"))
        revisions[spec["name"]] = {"hf_id": spec["hf_id"], "revision": os.path.basename(path)}
        print(f"model {spec['hf_id']} @ {revisions[spec['name']]['revision']}")
    return revisions


def download_popqa():
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split="test")
    rows = []
    for r in ds:
        answers = r["possible_answers"]
        if isinstance(answers, str):
            answers = json.loads(answers)
        rows.append({"id": int(r["id"]), "question": r["question"], "subj": r["subj"],
                     "subj_id": int(r["subj_id"]), "prop": r["prop"], "s_pop": int(r["s_pop"]),
                     "possible_answers": list(answers)})
    write_jsonl(DATA_DIR / "popqa.jsonl", rows)


def download_entity_questions():
    blob = fetch(EQ_ZIP_URL)
    zf = zipfile.ZipFile(io.BytesIO(blob))
    names = zf.namelist()
    rows = []
    for rel in EQ_RELATIONS:
        match = [n for n in names if n.endswith(f"test/{rel}.test.json")]
        if not match:
            raise RuntimeError(f"EntityQuestions test file for {rel} not found in archive")
        for i, r in enumerate(json.loads(zf.read(match[0]).decode("utf-8"))):
            rows.append({"relation": rel, "idx": i, "question": r["question"],
                         "answers": list(r["answers"])})
    write_jsonl(DATA_DIR / "entity_questions_test.jsonl", rows)


def download_advbench():
    text = fetch(ADVBENCH_CSV_URL).decode("utf-8")
    rows = [{"idx": i, "goal": r["goal"]} for i, r in enumerate(csv.DictReader(io.StringIO(text)))]
    write_jsonl(DATA_DIR / "advbench.jsonl", rows)


def download_alpaca():
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    rows = [{"idx": i, "instruction": r["instruction"], "output": r["output"]}
            for i, r in enumerate(ds) if not r["input"].strip()]
    write_jsonl(DATA_DIR / "alpaca_noinput.jsonl", rows)


def download_mmlu():
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = [{"idx": i, "question": r["question"], "subject": r["subject"],
             "choices": list(r["choices"]), "answer": int(r["answer"])} for i, r in enumerate(ds)]
    write_jsonl(DATA_DIR / "mmlu_test.jsonl", rows)


def library_versions():
    out = {}
    for pkg in ["torch", "transformers", "tokenizers", "datasets", "numpy", "scikit-learn",
                "scipy", "huggingface_hub", "safetensors"]:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    out["python"] = platform.python_version()
    out["machine"] = platform.machine()
    return out


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    steps = {"popqa.jsonl": download_popqa,
             "entity_questions_test.jsonl": download_entity_questions,
             "advbench.jsonl": download_advbench,
             "alpaca_noinput.jsonl": download_alpaca,
             "mmlu_test.jsonl": download_mmlu}
    for fname, fn in steps.items():
        if (DATA_DIR / fname).exists():
            print(f"{fname} already cached")
        else:
            fn()
    manifest = {"models": download_models(), "versions": library_versions()}
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

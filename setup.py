"""Download model weights and datasets (needs network; run once before main.py).

Amendment A14: every HF dataset and model is loaded at the commit pinned in config.py, URL sources
are verified against pinned sha256 digests, and the sha256 of every file written to data_cache/ is
recorded in run_manifest.json (main.py refuses to run if a file no longer matches). A cached file
is reused only if run_manifest.json already records it with the same pinned source; otherwise it is
downloaded again.
"""
import csv
import hashlib
import io
import json
import os
import platform
import urllib.request
import zipfile
from importlib import metadata

from config import DATA_DIR, DATA_FILES, DATASET_PINS, EQ_RELATIONS, MANIFEST_PATH, MODELS


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_pinned(key):
    pin = DATASET_PINS[key]
    req = urllib.request.Request(pin["url"], headers={"User-Agent": "activation-steering-setup/1.0"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        blob = resp.read()
    digest = sha256_bytes(blob)
    if digest != pin["sha256"]:
        raise RuntimeError(f"{key}: downloaded bytes sha256 {digest} != pinned {pin['sha256']} ({pin['url']})")
    return blob


def load_hf(key, split=None):
    from datasets import load_dataset
    pin = DATASET_PINS[key]
    args = (pin["repo"], pin["config"]) if pin["config"] else (pin["repo"],)
    return load_dataset(*args, split=split or pin["split"], revision=pin["revision"])


def download_models():
    from huggingface_hub import snapshot_download
    revisions = {}
    for spec in MODELS:
        path = snapshot_download(
            spec["hf_id"], revision=spec["revision"],
            allow_patterns=["*.json", "*.jinja", "*.safetensors", "tokenizer*", "*.model", "*.txt"],
            token=os.environ.get("HF_TOKEN"))
        got = os.path.basename(path)
        if got != spec["revision"]:
            raise RuntimeError(f"{spec['hf_id']}: snapshot {got} != pinned revision {spec['revision']}")
        revisions[spec["name"]] = {"hf_id": spec["hf_id"], "revision": got}
        print(f"model {spec['hf_id']} @ {got}")
    return revisions


def download_popqa():
    rows = []
    for r in load_hf("popqa"):
        answers = r["possible_answers"]
        if isinstance(answers, str):
            answers = json.loads(answers)
        rows.append({"id": int(r["id"]), "question": r["question"], "subj": r["subj"],
                     "subj_id": int(r["subj_id"]), "prop": r["prop"], "s_pop": int(r["s_pop"]),
                     "possible_answers": list(answers)})
    write_jsonl(DATA_DIR / DATA_FILES["popqa"], rows)


def download_entity_questions():
    zf = zipfile.ZipFile(io.BytesIO(fetch_pinned("entity_questions")))
    names = zf.namelist()
    rows = []
    for rel in EQ_RELATIONS:
        match = [n for n in names if n.endswith(f"test/{rel}.test.json")]
        if not match:
            raise RuntimeError(f"EntityQuestions test file for {rel} not found in archive")
        for i, r in enumerate(json.loads(zf.read(match[0]).decode("utf-8"))):
            rows.append({"relation": rel, "idx": i, "question": r["question"], "answers": list(r["answers"])})
    write_jsonl(DATA_DIR / DATA_FILES["entity_questions"], rows)


def download_advbench():
    text = fetch_pinned("advbench").decode("utf-8")
    rows = [{"idx": i, "goal": r["goal"]} for i, r in enumerate(csv.DictReader(io.StringIO(text)))]
    write_jsonl(DATA_DIR / DATA_FILES["advbench"], rows)


def download_alpaca():
    rows = [{"idx": i, "instruction": r["instruction"], "output": r["output"]}
            for i, r in enumerate(load_hf("alpaca")) if not r["input"].strip()]
    write_jsonl(DATA_DIR / DATA_FILES["alpaca"], rows)


def _mmlu_rows(split):
    return [{"idx": i, "question": r["question"], "subject": r["subject"], "choices": list(r["choices"]),
             "answer": int(r["answer"])} for i, r in enumerate(load_hf("mmlu", split))]


def download_mmlu():
    write_jsonl(DATA_DIR / DATA_FILES["mmlu"], _mmlu_rows("test"))


def download_mmlu_dev():
    write_jsonl(DATA_DIR / DATA_FILES["mmlu_dev"], _mmlu_rows("validation"))


def download_triviaqa():
    rows = []
    for r in load_hf("triviaqa"):
        a = r["answer"]
        rows.append({"question_id": r["question_id"], "question": r["question"], "value": a["value"],
                     "aliases": list(a["aliases"])})
    write_jsonl(DATA_DIR / DATA_FILES["triviaqa"], rows)


def download_bbc_news():
    rows = [{"title": r["title"], "published_date": r["published_date"], "link": r["link"],
             "section": r.get("section"), "content": r["content"]} for r in load_hf("bbc_news")]
    write_jsonl(DATA_DIR / DATA_FILES["bbc_news"], rows)


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


def source_of(key):
    pin = DATASET_PINS["mmlu" if key == "mmlu_dev" else key]
    return {k: v for k, v in pin.items() if k in ("repo", "config", "revision", "url", "sha256")}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    old = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {}
    recorded = old.get("datasets", {})
    steps = {"popqa": download_popqa, "entity_questions": download_entity_questions, "advbench": download_advbench,
             "alpaca": download_alpaca, "mmlu": download_mmlu, "mmlu_dev": download_mmlu_dev,
             "triviaqa": download_triviaqa, "bbc_news": download_bbc_news}
    datasets = {}
    for key, fn in steps.items():
        path = DATA_DIR / DATA_FILES[key]
        rec = recorded.get(key, {})
        fresh = path.exists() and rec.get("source") == source_of(key) and rec.get("sha256") == sha256_file(path)
        if fresh:
            print(f"{path.name} already cached at the pinned source")
        else:
            fn()
        with open(path, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        datasets[key] = {"file": path.name, "source": source_of(key), "sha256": sha256_file(path), "n_rows": n}
    manifest = {"models": download_models(), "datasets": datasets, "versions": library_versions()}
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

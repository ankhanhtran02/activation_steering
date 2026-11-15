"""Setup (needs network; the experiment itself runs offline):

    python -m vesecs.setup_data [--sq2-run-dir DIR] [--ids-only]

1. Models at their pinned revisions (pilot.setup_pilot.download_models; skipped files are already cached).
2. The SQ2 data files if missing (PopQA with object URIs, TriviaQA rc.nocontext validation; same pinned revisions
   and code as SQ2). Their SHA-256 must equal the SQ2 run's pilot_manifest.json, because the pool and the
   exclusions were built from them.
3. data_cache/ves_popqa_ids.jsonl: PopQA's id columns (id, subj_id, obj_id, s_uri, o_uri) from test.tsv at the
   same revision. SQ2's PopQA file keeps the subject id but not the object id; the {subject, object} isolation
   rule needs both in one id space. Without it the pair rule falls back to entity names only.
"""
import argparse
import csv
import json
from pathlib import Path

from pilot import config as PC
from vesecs import cfg as C
from vesecs.io import read_json, sha256_file, write_json, write_jsonl


def popqa_ids(cfg):
    from huggingface_hub import hf_hub_download
    pin = PC.POPQA_PIN
    p = hf_hub_download(pin["repo"], "test.tsv", repo_type="dataset", revision=pin["revision"])
    csv.field_size_limit(10 ** 9)
    rows = [{"id": int(r["id"]), "subj_id": int(r["subj_id"]), "obj_id": int(r["obj_id"]), "s_uri": r["s_uri"],
             "o_uri": r["o_uri"]} for r in csv.DictReader(open(p, encoding="utf-8"), delimiter="\t")]
    out = PC.DATA_DIR / cfg["paths"]["popqa_ids_file"]
    write_jsonl(out, rows)
    print(f"wrote {len(rows)} rows -> {out}")
    return {"file": out.name, "sha256": sha256_file(out), "n_rows": len(rows), "source": pin}


def check_against_sq2(sq2_dir):
    """SHA-256 of the SQ2 data files vs the SQ2 run's manifest. Returns list of problems."""
    man = Path(sq2_dir) / "pilot_manifest.json"
    if not man.exists():
        return [f"{man} not found: cannot verify the data files against the SQ2 run"]
    rec = read_json(man).get("datasets", {})
    bad = []
    for key in ("popqa_full", "triviaqa"):
        f = PC.DATA_DIR / PC.DATA_FILES[key]
        want = rec.get(key, {}).get("sha256")
        if not f.exists():
            bad.append(f"missing {f}")
        elif want and sha256_file(f) != want:
            bad.append(f"{f.name} differs from the SQ2 run's copy (sha256 {sha256_file(f)[:12]} != {want[:12]})")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sq2-run-dir", default=None)
    ap.add_argument("--ids-only", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config()
    PC.DATA_DIR.mkdir(parents=True, exist_ok=True)
    man = {}
    if not args.ids_only:
        from pilot import setup_pilot as SP
        SP.download_models()
        if not (PC.DATA_DIR / PC.DATA_FILES["popqa_full"]).exists():
            SP.download_popqa_full()
        SP.ensure_triviaqa()
    man["popqa_ids"] = popqa_ids(cfg)
    write_json(PC.DATA_DIR / "ves_manifest.json", man, indent=1)
    if args.sq2_run_dir:
        bad = check_against_sq2(args.sq2_run_dir)
        if bad:
            raise SystemExit("DATA CHECK FAILED:\n  - " + "\n  - ".join(bad))
        print("data files match the SQ2 run's manifest")
    print(json.dumps({"setup": "ok"}))


if __name__ == "__main__":
    main()

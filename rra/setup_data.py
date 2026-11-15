"""Setup (needs network; the experiment itself runs offline):

    python -m rra.setup_data [--sq2-run-dir DIR] [--no-gemma] [--ids-only]

1. The two main models at their pinned revisions and the SQ2 data files (PopQA, TriviaQA) if missing, exactly as
   the VES/ECS setup does (vesecs.setup_data / pilot.setup_pilot); their SHA-256 must equal the SQ2 run's manifest.
2. data_cache/ves_popqa_ids.jsonl (PopQA object ids for the {subject, object} isolation rule), if missing.
3. Optional stage G: Gemma-2-9B-it (ungated mirror unsloth/gemma-2-9b-it at the pinned revision, ~18 GB).
"""
import argparse
import json

from pilot import config as PC
from rra import cfg as C
from rra.engine import snapshot_ok
from vesecs.io import write_json


def download_gemma(cfg):
    from huggingface_hub import snapshot_download
    g = C.gemma_spec(cfg)
    if snapshot_ok(g):
        print(f"model {g['hf_id']} @ {g['revision'][:12]} already cached")
        return
    snapshot_download(g["hf_id"], revision=g["revision"], allow_patterns=PC.MODEL_FILE_PATTERNS)
    print(f"model {g['hf_id']} @ {g['revision'][:12]} downloaded")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sq2-run-dir", default=None)
    ap.add_argument("--no-gemma", action="store_true")
    ap.add_argument("--ids-only", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config()
    PC.DATA_DIR.mkdir(parents=True, exist_ok=True)
    from vesecs import setup_data as VS
    if not args.ids_only:
        from pilot import setup_pilot as SP
        SP.download_models()
        if not (PC.DATA_DIR / PC.DATA_FILES["popqa_full"]).exists():
            SP.download_popqa_full()
        SP.ensure_triviaqa()
    if not (PC.DATA_DIR / cfg["paths"]["popqa_ids_file"]).exists():
        man = {"popqa_ids": VS.popqa_ids(cfg)}
        write_json(PC.DATA_DIR / "ves_manifest.json", man, indent=1)
    if not args.ids_only and not args.no_gemma and cfg["optional"]["G"]:
        download_gemma(cfg)
    if args.sq2_run_dir:
        bad = VS.check_against_sq2(args.sq2_run_dir)
        if bad:
            raise SystemExit("DATA CHECK FAILED:\n  - " + "\n  - ".join(bad))
        print("data files match the SQ2 run's manifest")
    print(json.dumps({"setup": "ok"}))


if __name__ == "__main__":
    main()

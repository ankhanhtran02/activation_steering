#!/usr/bin/env bash
# One-command runner for the pilot (pilot/README.md): environment -> downloads -> GPU preflight -> pilot.
#
#   bash run_pilot.sh            everything (needs network for the setup part)
#   bash run_pilot.sh setup      environment + downloads only (network)
#   bash run_pilot.sh main       pilot only (offline; setup must have run before)
#   bash run_pilot.sh resume DIR continue a crashed run in outputs_pilot/run_<id> (finished phases are reused)
#   bash run_pilot.sh smoke      CPU smoke test of every code path with tiny models (no results)
#
# Each run writes to outputs_pilot/run_<UTC timestamp>/ (outputs_pilot/latest points to the newest).
# The Python environment is the earlier study's .venv (same packages); see run.sh for how it is built.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
STAGE="${1:-all}"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="outputs_pilot"
mkdir -p "$OUT_ROOT/setup_logs"
say() { printf '\n==== %s ====\n' "$*"; }

case "$STAGE" in
  all|setup|main|resume|smoke) ;;
  *) echo "usage: bash run_pilot.sh [all|setup|main|resume DIR|smoke]"; exit 2 ;;
esac

# ------------------------------------------------------------------ environment
PY="$VENV_DIR/bin/python"
env_ok() {
  [ -x "$PY" ] && "$PY" -c "import torch, transformers, tokenizers, huggingface_hub, safetensors, datasets, numpy, scipy, sklearn" \
    >/dev/null 2>&1
}
if ! env_ok; then
  say "Python environment ($VENV_DIR) missing or incomplete: building it with run.sh's procedure"
  if [ ! -e "$VENV_DIR" ]; then
    "$PYTHON" -m venv --system-site-packages "$VENV_DIR"
  fi
  "$PY" -m pip install --quiet --upgrade pip
  if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && [ "$STAGE" != "smoke" ]; then
    "$PY" -m pip install --quiet --upgrade torch --index-url "$TORCH_INDEX_URL"
  fi
  if [ -f requirements.lock ]; then
    "$PY" -m pip install --quiet -r requirements.lock --extra-index-url "$TORCH_INDEX_URL"
  else
    "$PY" -m pip install --quiet -r requirements.txt
  fi
  env_ok || { echo "The environment cannot import the required packages; see the pip output above."; exit 1; }
fi
echo "python $("$PY" -c 'import platform; print(platform.python_version())'), torch $("$PY" -c 'import torch; print(torch.__version__)'), transformers $("$PY" -c 'import transformers; print(transformers.__version__)')"

if [ "$STAGE" = "smoke" ]; then
  say "CPU smoke test (tiny models; numbers mean nothing)"
  exec "$PY" -m pilot.smoke_test
fi

# ------------------------------------------------------------------ downloads
if [ "$STAGE" = "all" ] || [ "$STAGE" = "setup" ]; then
  say "Downloading models and data (pilot.setup_pilot)"
  "$PY" -m pilot.setup_pilot 2>&1 | tee "$OUT_ROOT/setup_logs/setup_${RUN_ID}.txt"
fi
[ "$STAGE" = "setup" ] && { say "Setup finished. Start the pilot with: bash run_pilot.sh main"; exit 0; }

# ------------------------------------------------------------------ preflight
say "Preflight checks"
"$PY" - <<'EOF'
import json, sys
from pathlib import Path
import torch
from pilot import config as PC
problems = []
if not torch.cuda.is_available():
    problems.append("torch cannot see a CUDA GPU (install a CUDA build, see README.md)")
else:
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GiB, bf16={torch.cuda.is_bf16_supported()}")
for f in PC.DATA_FILES.values():
    if not (PC.DATA_DIR / f).exists():
        problems.append(f"missing {PC.DATA_DIR / f} (run: bash run_pilot.sh setup)")
if not PC.MANIFEST_PATH.exists():
    problems.append("missing pilot_manifest.json (run: bash run_pilot.sh setup)")
else:
    from huggingface_hub import snapshot_download
    man = json.loads(PC.MANIFEST_PATH.read_text())
    for spec in PC.MODELS:
        m = man.get("models", {}).get(spec["name"], {})
        if m.get("revision") != spec["revision"]:
            problems.append(f"{spec['name']} not in pilot_manifest.json at the pinned revision (run setup)")
            continue
        try:
            snap = Path(snapshot_download(spec["hf_id"], revision=spec["revision"], local_files_only=True,
                                          allow_patterns=PC.MODEL_FILE_PATTERNS))
            idx = snap / "model.safetensors.index.json"
            shards = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else \
                [q.name for q in snap.glob("*.safetensors")]
            missing = [s for s in shards if not (snap / s).exists()]
            if not shards or missing:
                problems.append(f"{spec['hf_id']} weights incomplete in the cache (run setup)")
            else:
                print(f"model {spec['hf_id']} @ {spec['revision'][:12]} cached ({len(shards)} weight files)")
        except Exception as e:
            problems.append(f"{spec['hf_id']} not in the local cache ({type(e).__name__}); run setup")
if not PC.PREVIOUSLY_SCORED_PATH.exists():
    problems.append("missing previously_scored_qids.json")
if problems:
    print("PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems))
    sys.exit(1)
print("preflight ok")
EOF

# ------------------------------------------------------------------ pilot
if [ "$STAGE" = "resume" ]; then
  RUN_DIR="${2:?usage: bash run_pilot.sh resume outputs_pilot/run_<id>}"
  export PILOT_RESUME=1
  RUN_ID="$(basename "$RUN_DIR" | sed 's/^run_//')"
else
  RUN_DIR="$OUT_ROOT/run_${RUN_ID}"
  [ -e "$RUN_DIR" ] && { echo "$RUN_DIR exists; start again in a second."; exit 1; }
  mkdir -p "$RUN_DIR"
  ln -sfn "run_${RUN_ID}" "$OUT_ROOT/latest"
fi
cp pilot_manifest.json "$RUN_DIR/" 2>/dev/null || true
[ -f requirements.lock ] && cp requirements.lock "$RUN_DIR/"

say "Running the pilot into $RUN_DIR (offline; expected ~5 h, cap 11 h, hard limit 12 h)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PILOT_OUT_DIR="$PWD/$RUN_DIR" PILOT_RUN_ID="$RUN_ID"
START=$(date +%s)
set +e
"$PY" -m pilot.main > >(tee -a "$RUN_DIR/run_log.txt") 2> >(tee -a "$RUN_DIR/run_stderr.txt" >&2)
STATUS=$?
set -e
say "pilot finished with status $STATUS after $(( ($(date +%s) - START) / 60 )) min"
echo "Run folder: $RUN_DIR (also $OUT_ROOT/latest)"
echo "Next: hand-check the stage-1 rows of $RUN_DIR/grading_check_<model>.csv (human_label) and paraphrase_check.csv (meaning_preserved),"
echo "      then: $PY -m pilot.analyze --run-dir $RUN_DIR"
grep -E "PHASE_FAILED|HARD_STOP|SKIPPED_BY_BUDGET" "$RUN_DIR/run_log.txt" || echo "No failed or skipped phases."
exit $STATUS

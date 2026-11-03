#!/usr/bin/env bash
# One-command runner: environment -> downloads -> GPU preflight -> experiment.
#
#   bash run.sh            everything (needs network for the setup part)
#   bash run.sh setup      environment + downloads only (network)
#   bash run.sh main       experiment only (offline; setup must have run before)
#
# Optional environment variables:
#   PYTHON           interpreter used to create the virtual env (default: python3)
#   VENV_DIR         virtual env location (default: .venv inside this folder)
#   TORCH_INDEX_URL  wheel index for a CUDA build of torch, used only when no CUDA-enabled torch
#                    is found (default: https://download.pytorch.org/whl/cu130, aarch64 + CUDA 13)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
STAGE="${1:-all}"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
mkdir -p outputs

say() { printf '\n==== %s ====\n' "$*"; }

case "$STAGE" in
  all|setup|main) ;;
  *) echo "usage: bash run.sh [all|setup|main]"; exit 2 ;;
esac

# ------------------------------------------------------------------ environment
make_env() {
  say "Python environment ($VENV_DIR)"
  if [ ! -x "$VENV_DIR/bin/python" ]; then
    # --system-site-packages lets a preinstalled CUDA torch (e.g. NVIDIA container) be reused
    "$PYTHON" -m venv --system-site-packages "$VENV_DIR" || {
      echo "Could not create a virtual env. On Ubuntu: sudo apt install python3-venv"; exit 1; }
  fi
  PY="$VENV_DIR/bin/python"
  "$PY" -m pip install --quiet --upgrade pip
  if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "CUDA-enabled torch already available: $("$PY" -c 'import torch; print(torch.__version__)')"
  else
    echo "Installing CUDA torch from $TORCH_INDEX_URL"
    "$PY" -m pip install --quiet --upgrade torch --index-url "$TORCH_INDEX_URL"
  fi
  "$PY" -m pip install --quiet -r requirements.txt
}

PY="$VENV_DIR/bin/python"
if [ "$STAGE" != "main" ] || [ ! -x "$PY" ]; then
  make_env
fi

# ------------------------------------------------------------------ downloads
if [ "$STAGE" != "main" ]; then
  say "Downloading models and datasets (setup.py; cached files are skipped)"
  "$PY" setup.py 2>&1 | tee outputs/setup_log.txt
fi
[ "$STAGE" = "setup" ] && { say "Setup finished. Start the experiment with: bash run.sh main"; exit 0; }

# ------------------------------------------------------------------ preflight
say "Preflight checks"
"$PY" - <<'EOF'
import json, sys
from config import DATA_DIR, MANIFEST_PATH
import torch
problems = []
if not torch.cuda.is_available():
    problems.append("torch cannot see a CUDA GPU (install a CUDA build, see README)")
else:
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GiB, bf16={torch.cuda.is_bf16_supported()}")
    try:
        x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
        (x @ x).sum().item()
    except Exception as e:
        problems.append(f"a bf16 matmul on the GPU failed ({e}); torch build may not support this GPU")
for f in ["popqa.jsonl", "entity_questions_test.jsonl", "advbench.jsonl", "alpaca_noinput.jsonl", "mmlu_test.jsonl"]:
    if not (DATA_DIR / f).exists():
        problems.append(f"missing {DATA_DIR / f} (run: bash run.sh setup)")
if not MANIFEST_PATH.exists():
    problems.append("missing run_manifest.json (run: bash run.sh setup)")
else:
    for name, m in json.loads(MANIFEST_PATH.read_text())["models"].items():
        print(f"model {name}: {m['hf_id']} @ {m['revision']}")
if problems:
    print("PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems))
    sys.exit(1)
print("preflight ok")
EOF

# ------------------------------------------------------------------ experiment
say "Running main.py (offline; about 3 hours, hard limit 4 hours)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
START=$(date +%s)
set +e
"$PY" main.py > >(tee outputs/run_log.txt) 2> >(tee outputs/run_stderr.txt >&2)
STATUS=$?
set -e
say "main.py finished with status $STATUS after $(( ($(date +%s) - START) / 60 )) min"
echo "Console log:   outputs/run_log.txt"
echo "Warnings/tracebacks: outputs/run_stderr.txt"
echo "Results:       outputs/results.json, outputs/frozen_config.json, outputs/audit.csv, outputs/samples_*.jsonl"
grep -E "CONDITION_FAILED|MODEL_SETUP_FAILED|ANALYSIS_FAILED|WARNING" outputs/run_log.txt || echo "No failures reported."
exit $STATUS

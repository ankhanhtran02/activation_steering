#!/usr/bin/env bash
# One-command runner: environment -> downloads -> GPU preflight -> experiment.
#
#   bash run.sh            everything (needs network for the setup part)
#   bash run.sh setup      environment + downloads only (network)
#   bash run.sh main       experiment only (offline; setup must have run before)
#
# Every experiment run writes to its own folder outputs/run_<UTC timestamp>/ (outputs/latest points
# to the newest), so nothing from an earlier run is ever overwritten. Setup logs go to
# outputs/setup_logs/.
#
# Python environment: an existing virtual env is reused when its python runs, the project's
# packages import, and it was installed from the current requirements.lock (or requirements.txt);
# otherwise the missing pieces are installed into it. A virtual env whose python does not run on
# this machine (created on another machine / CPU architecture, or broken) is moved aside to
# <VENV_DIR>.old-<timestamp> and rebuilt; REBUILD_ENV=1 forces the same.
#
# Optional environment variables:
#   PYTHON           interpreter used to create the virtual env (default: python3; >= 3.10)
#   VENV_DIR         virtual env location (default: .venv inside this folder)
#   REBUILD_ENV      1 = move the existing virtual env aside and build a fresh one
#   TORCH_INDEX_URL  wheel index for a CUDA build of torch, used only when no CUDA-enabled torch
#                    is found (default: https://download.pytorch.org/whl/cu130, aarch64 + CUDA 13)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
STAGE="${1:-all}"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
REBUILD_ENV="${REBUILD_ENV:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_ROOT="outputs"
mkdir -p "$OUT_ROOT/setup_logs"

say() { printf '\n==== %s ====\n' "$*"; }

case "$STAGE" in
  all|setup|main) ;;
  *) echo "usage: bash run.sh [all|setup|main]"; exit 2 ;;
esac

# ------------------------------------------------------------------ environment
PY="$VENV_DIR/bin/python"
STAMP="$VENV_DIR/.installed_from_sha256"

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
requirements_file() { if [ -f requirements.lock ]; then echo requirements.lock; else echo requirements.txt; fi; }
env_python_runs() { [ -x "$PY" ] && "$PY" -c "import sys" >/dev/null 2>&1; }
env_imports_ok() {
  "$PY" -c "import torch, transformers, tokenizers, huggingface_hub, safetensors, datasets, numpy, scipy, sklearn" \
    >/dev/null 2>&1
}
move_env_aside() {
  local dest="${VENV_DIR}.old-${RUN_ID}"
  mv "$VENV_DIR" "$dest"
  echo "Moved the previous environment to $dest (delete it once the new one works)."
}
pip_or_explain() {
  "$PY" -m pip "$@" || {
    echo "pip failed. Installing packages needs network access to PyPI (and $TORCH_INDEX_URL)."
    [ "$STAGE" = "main" ] && echo "Run 'bash run.sh setup' on a machine with network first, or fix the environment."
    exit 1; }
}

ensure_env() {
  say "Python environment ($VENV_DIR)"
  if [ -e "$VENV_DIR" ] && [ "$REBUILD_ENV" = "1" ]; then
    echo "REBUILD_ENV=1: building a fresh environment."
    move_env_aside
  elif [ -e "$VENV_DIR" ] && ! env_python_runs; then
    echo "Found $VENV_DIR, but its python does not run on this machine (created on another machine or"
    echo "CPU architecture, or broken)."
    move_env_aside
  fi

  if [ ! -e "$VENV_DIR" ]; then
    "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null || {
      echo "$PYTHON is missing or older than 3.10. Point PYTHON= to Python 3.10 or newer."; exit 1; }
    echo "Creating a new virtual env in $VENV_DIR with $("$PYTHON" -V 2>&1)"
    # --system-site-packages lets a preinstalled CUDA torch (e.g. NVIDIA container) be reused
    "$PYTHON" -m venv --system-site-packages "$VENV_DIR" || {
      echo "Could not create a virtual env. On Ubuntu: sudo apt install python3-venv"; exit 1; }
  else
    echo "Reusing the existing virtual env ($("$PY" -V 2>&1))."
  fi

  local req want
  req="$(requirements_file)"
  want="$(file_sha256 "$req")"
  if env_imports_ok && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$want" ]; then
    echo "Environment matches $req; nothing to install."
    echo "torch $("$PY" -c 'import torch; print(torch.__version__)'), python $("$PY" -c 'import platform; print(platform.python_version())')"
    return
  fi
  if env_imports_ok; then
    echo "Packages import, but the environment was not installed from the current $req: updating it."
  else
    echo "Installing packages into $VENV_DIR."
  fi

  pip_or_explain install --quiet --upgrade pip
  if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "CUDA-enabled torch already available: $("$PY" -c 'import torch; print(torch.__version__)')"
  else
    echo "Installing CUDA torch from $TORCH_INDEX_URL"
    pip_or_explain install --quiet --upgrade torch --index-url "$TORCH_INDEX_URL"
  fi
  # amendment A14: the first setup writes requirements.lock (exact versions of every package
  # installed in the virtual env; --local leaves out packages that come from the system site, such
  # as a container's own torch build); commit it, and later runs install exactly that environment.
  if [ -f requirements.lock ]; then
    echo "Installing the pinned environment from requirements.lock"
    pip_or_explain install --quiet -r requirements.lock --extra-index-url "$TORCH_INDEX_URL"
  else
    pip_or_explain install --quiet -r requirements.txt
    "$PY" -m pip freeze --local --exclude-editable > requirements.lock
    echo "Wrote requirements.lock: commit it so later runs install the same versions."
  fi
  env_imports_ok || { echo "The environment still cannot import the project's packages; see the pip output above."; exit 1; }
  file_sha256 requirements.lock > "$STAMP"
}

ensure_env

# ------------------------------------------------------------------ downloads
SETUP_LOG=""
if [ "$STAGE" != "main" ]; then
  say "Downloading models and datasets (setup.py; files already at their pinned source are skipped)"
  SETUP_LOG="$OUT_ROOT/setup_logs/setup_${RUN_ID}.txt"
  "$PY" setup.py 2>&1 | tee "$SETUP_LOG"
fi
[ "$STAGE" = "setup" ] && { say "Setup finished. Start the experiment with: bash run.sh main"; exit 0; }

# ------------------------------------------------------------------ preflight
say "Preflight checks"
"$PY" - <<'EOF'
import json, sys
from config import DATA_DIR, DATA_FILES, MANIFEST_PATH
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
for f in DATA_FILES.values():
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

# ------------------------------------------------------------------ experiment (own folder per run)
RUN_DIR="$OUT_ROOT/run_${RUN_ID}"
if [ -e "$RUN_DIR" ]; then
  echo "$RUN_DIR already exists; refusing to overwrite it. Start the run again in a second."; exit 1
fi
mkdir -p "$RUN_DIR"
ln -sfn "run_${RUN_ID}" "$OUT_ROOT/latest"
for f in requirements.lock run_manifest.json; do [ -f "$f" ] && cp "$f" "$RUN_DIR/"; done
[ -n "$SETUP_LOG" ] && cp "$SETUP_LOG" "$RUN_DIR/setup_log.txt"

say "Running main.py into $RUN_DIR (offline; final run about 8 hours, hard limit 12 hours)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export STEER_OUT_DIR="$PWD/$RUN_DIR" STEER_RUN_ID="$RUN_ID"
START=$(date +%s)
set +e
"$PY" main.py > >(tee "$RUN_DIR/run_log.txt") 2> >(tee "$RUN_DIR/run_stderr.txt" >&2)
STATUS=$?
set -e
say "main.py finished with status $STATUS after $(( ($(date +%s) - START) / 60 )) min"
echo "Run folder:    $RUN_DIR  (also $OUT_ROOT/latest)"
echo "Console log:   $RUN_DIR/run_log.txt"
echo "Warnings/tracebacks: $RUN_DIR/run_stderr.txt"
echo "Results:       $RUN_DIR/results.json, frozen_config.json, audit*.csv, per_item/, samples_*.jsonl"
grep -E "CONDITION_FAILED|MODEL_SETUP_FAILED|MODEL_TEST_FAILED|ANALYSIS_FAILED|WARNING" "$RUN_DIR/run_log.txt" \
  || echo "No failures reported."
exit $STATUS

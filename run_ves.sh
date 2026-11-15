#!/usr/bin/env bash
# VES / ECS experiment (ves-ces-experiment.tex), one command on the GPU server:
#
#   bash run_ves.sh                 everything: environment -> setup (downloads if needed) -> preflight -> run -> report
#   bash run_ves.sh resume DIR      continue a crashed run in outputs_ves/run_<id> (finished stages are reused)
#   bash run_ves.sh analyze DIR     recompute the test analysis, verdicts and report of a run (CPU only)
#   bash run_ves.sh smoke           CPU smoke test of every code path with two tiny models (numbers mean nothing)
#   bash run_ves.sh test            unit tests (seconds)
#
# Needs the SQ2 pilot run folder (pool.jsonl, pilot_qids.json, pilot_manifest.json, <model>/shared.pt, probe.pt).
# It is found automatically (outputs_pilot/run_20261110T010318Z, else the newest outputs_pilot/run_* or run_*
# folder that has both models' shared.pt); set VES_SQ2_RUN_DIR=/path to choose one.
# Each run writes outputs_ves/run_<UTC>/ (outputs_ves/latest points to the newest). Planned ~7 h on one GB10;
# the budget guard cuts optional parts if the projection exceeds 10.5 h and no stage starts after 11.5 h.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
STAGE="${1:-all}"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
OUT_ROOT="outputs_ves"
say() { printf '\n==== %s ====\n' "$*"; }

case "$STAGE" in
  all|resume|analyze|smoke|test) ;;
  *) echo "usage: bash run_ves.sh [all|resume DIR|analyze DIR|smoke|test]"; exit 2 ;;
esac

# ------------------------------------------------------------------ environment (same .venv as the SQ2 pilot)
PY="$VENV_DIR/bin/python"
env_ok() {
  [ -x "$PY" ] && "$PY" -c "import torch, transformers, tokenizers, huggingface_hub, safetensors, numpy, scipy, sklearn, yaml" \
    >/dev/null 2>&1
}
if ! env_ok; then
  say "Python environment ($VENV_DIR) missing or incomplete: building it"
  [ -e "$VENV_DIR" ] || "$PYTHON" -m venv --system-site-packages "$VENV_DIR"
  "$PY" -m pip install --quiet --upgrade pip
  if [ "$STAGE" != "smoke" ] && [ "$STAGE" != "test" ] && \
     ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    "$PY" -m pip install --quiet --upgrade torch --index-url "$TORCH_INDEX_URL"
  fi
  LOCK=requirements.lock; [ -f "$LOCK" ] || LOCK=configs/requirements.lock
  if [ -f "$LOCK" ]; then
    "$PY" -m pip install --quiet -r "$LOCK" --extra-index-url "$TORCH_INDEX_URL" || "$PY" -m pip install --quiet -r requirements.txt
  else
    "$PY" -m pip install --quiet -r requirements.txt
  fi
  "$PY" -m pip install --quiet pyyaml datasets
  env_ok || { echo "The environment cannot import the required packages; see the pip output above."; exit 1; }
fi
echo "python $("$PY" -c 'import platform; print(platform.python_version())'), torch $("$PY" -c 'import torch; print(torch.__version__)'), transformers $("$PY" -c 'import transformers; print(transformers.__version__)')"
"$PY" - <<'EOF' || true
# compare the installed torch / transformers with the committed lock file (the SQ2 run's environment)
from importlib import metadata
from pathlib import Path
lock = Path("configs/requirements.lock")
if lock.exists():
    want = dict(l.strip().split("==", 1) for l in lock.read_text().splitlines() if "==" in l)
    for p in ("torch", "transformers", "tokenizers", "numpy"):
        try:
            got = metadata.version(p)
        except metadata.PackageNotFoundError:
            got = None
        if p in want and got != want[p]:
            print(f"NOTE: {p} {got} differs from configs/requirements.lock ({want[p]}); versions are logged in the run")
EOF

if [ "$STAGE" = "test" ]; then
  say "Unit tests"
  STEER_DEVICE=cpu exec "$PY" -m vesecs.tests
fi
if [ "$STAGE" = "smoke" ]; then
  say "CPU smoke test (tiny models; ~20-40 min on an 8-core CPU; numbers mean nothing)"
  exec "$PY" -m vesecs.smoke_test
fi
if [ "$STAGE" = "analyze" ]; then
  RUN_DIR="${2:?usage: bash run_ves.sh analyze outputs_ves/run_<id>}"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 STEER_DEVICE=cpu VES_OUT_DIR="$PWD/$RUN_DIR"
  [ -f "$RUN_DIR/sq2_run_dir.txt" ] && export VES_SQ2_RUN_DIR="$(cat "$RUN_DIR/sq2_run_dir.txt")"
  exec "$PY" -m vesecs.analysis --run-dir "$RUN_DIR"
fi

# ------------------------------------------------------------------ SQ2 run folder
if [ -z "${VES_SQ2_RUN_DIR:-}" ] && [ "$STAGE" = "resume" ] && [ -f "${2:-}/sq2_run_dir.txt" ]; then
  VES_SQ2_RUN_DIR="$(cat "$2/sq2_run_dir.txt")"
fi
if [ -z "${VES_SQ2_RUN_DIR:-}" ]; then
  VES_SQ2_RUN_DIR="$("$PY" - <<'EOF'
import glob
from pathlib import Path
models = ("llama31_8b_instruct", "qwen25_7b_instruct")
def ok(p):
    p = Path(p)
    return p.is_dir() and all((p / f).exists() for f in ("pool.jsonl", "extension_pool.jsonl", "pilot_qids.json")) \
        and all((p / m / "shared.pt").exists() for m in models)
cands = ["outputs_pilot/run_20261110T010318Z"] + sorted(glob.glob("*20261110T010318Z*")) + sorted(glob.glob("outputs_pilot/*20261110T010318Z*")) \
    + sorted(glob.glob("outputs_pilot/run_*"), reverse=True) + sorted(glob.glob("run_*"), reverse=True)
for c in cands:
    if ok(c):
        print(str(Path(c).resolve()))
        break
EOF
)"
fi
if [ -z "${VES_SQ2_RUN_DIR:-}" ] || [ ! -f "$VES_SQ2_RUN_DIR/pool.jsonl" ]; then
  echo "SQ2 run folder not found. Copy the SQ2 pilot run folder (outputs_pilot/run_20261110T010318Z: pool.jsonl,"
  echo "extension_pool.jsonl, pilot_qids.json, pilot_manifest.json, llama31_8b_instruct/ and qwen25_7b_instruct/ with"
  echo "shared.pt and probe.pt) into this repository, or set VES_SQ2_RUN_DIR=/path/to/it."
  exit 1
fi
export VES_SQ2_RUN_DIR
echo "SQ2 run folder: $VES_SQ2_RUN_DIR"

# ------------------------------------------------------------------ setup (network; skipped when everything is present)
NEED_SETUP=0
[ -f data_cache/ves_popqa_ids.jsonl ] || NEED_SETUP=1
[ -f data_cache/pilot_popqa_full.jsonl ] || NEED_SETUP=1
[ -f data_cache/triviaqa_validation.jsonl ] || NEED_SETUP=1
"$PY" - <<'EOF' || NEED_SETUP=1
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
from pilot import config as PC
for s in PC.MODELS:
    snap = Path(snapshot_download(s["hf_id"], revision=s["revision"], local_files_only=True, allow_patterns=PC.MODEL_FILE_PATTERNS))
    idx = snap / "model.safetensors.index.json"
    shards = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else [q.name for q in snap.glob("*.safetensors")]
    if not shards or any(not (snap / x).exists() for x in shards):
        sys.exit(1)
EOF
if [ "$STAGE" = "all" ] && [ "$NEED_SETUP" = "1" ]; then
  say "Setup: downloading what is missing (models at pinned revisions, PopQA, TriviaQA, PopQA ids)"
  mkdir -p "$OUT_ROOT/setup_logs"
  if ! "$PY" -m vesecs.setup_data --sq2-run-dir "$VES_SQ2_RUN_DIR" 2>&1 | tee "$OUT_ROOT/setup_logs/setup_$(date -u +%Y%m%dT%H%M%SZ).txt"; then
    if [ -f data_cache/pilot_popqa_full.jsonl ] && [ -f data_cache/triviaqa_validation.jsonl ]; then
      echo "WARNING: setup did not complete (no network?). Continuing: without data_cache/ves_popqa_ids.jsonl the"
      echo "         {subject, object} isolation rule compares entity names only (recorded in the report)."
    else
      echo "Setup failed and the data files are missing; see the log above."; exit 1
    fi
  fi
fi

# ------------------------------------------------------------------ preflight
say "Preflight"
"$PY" - <<'EOF'
import json, os, shutil, sys
from pathlib import Path
import torch
from pilot import config as PC
problems = []
if not torch.cuda.is_available():
    problems.append("torch cannot see a CUDA GPU")
else:
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, {p.total_memory / 2**30:.0f} GiB, bf16={torch.cuda.is_bf16_supported()}")
    if not torch.cuda.is_bf16_supported():
        problems.append("the GPU does not support bf16")
for f in ("popqa_full", "triviaqa"):
    if not (PC.DATA_DIR / PC.DATA_FILES[f]).exists():
        problems.append(f"missing {PC.DATA_DIR / PC.DATA_FILES[f]}")
from vesecs.setup_data import check_against_sq2
problems += [p for p in check_against_sq2(os.environ["VES_SQ2_RUN_DIR"]) if "not found" not in p]
from huggingface_hub import snapshot_download
for s in PC.MODELS:
    try:
        snapshot_download(s["hf_id"], revision=s["revision"], local_files_only=True, allow_patterns=PC.MODEL_FILE_PATTERNS)
        print(f"model {s['hf_id']} @ {s['revision'][:12]} cached")
    except Exception as e:
        problems.append(f"{s['hf_id']} not cached at {s['revision'][:12]} ({type(e).__name__})")
free = shutil.disk_usage(".").free / 2**30
print(f"free disk: {free:.0f} GiB")
if free < 15:
    problems.append(f"only {free:.0f} GiB free disk (need ~15)")
if problems:
    print("PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems))
    sys.exit(1)
print("preflight ok")
EOF

# ------------------------------------------------------------------ run
if [ "$STAGE" = "resume" ]; then
  RUN_DIR="${2:?usage: bash run_ves.sh resume outputs_ves/run_<id>}"
else
  RUN_DIR="$OUT_ROOT/run_$(date -u +%Y%m%dT%H%M%SZ)"
  [ -e "$RUN_DIR" ] && { echo "$RUN_DIR exists; start again in a second."; exit 1; }
  mkdir -p "$RUN_DIR"
  ln -sfn "run_$(basename "$RUN_DIR" | sed 's/^run_//')" "$OUT_ROOT/latest"
fi
mkdir -p "$RUN_DIR/logs"
echo "$VES_SQ2_RUN_DIR" > "$RUN_DIR/sq2_run_dir.txt"
[ -f configs/requirements.lock ] && cp configs/requirements.lock "$RUN_DIR/requirements.lock"
"$PY" -m pip freeze > "$RUN_DIR/pip_freeze.txt" 2>/dev/null || true

say "Running into $RUN_DIR (offline; planned ~7 h, projection cap 10.5 h, hard stop 11.5 h)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 VES_OUT_DIR="$PWD/$RUN_DIR"
START=$(date +%s)
set +e
"$PY" -m vesecs.main > >(tee -a "$RUN_DIR/logs/run_log.txt") 2> >(tee -a "$RUN_DIR/logs/run_stderr.txt" >&2)
STATUS=$?
set -e
say "finished with status $STATUS after $(( ($(date +%s) - START) / 60 )) min"
if command -v latexmk >/dev/null 2>&1 && [ -f "$RUN_DIR/results/results_report.tex" ]; then
  (cd "$RUN_DIR/results" && latexmk -pdf -interaction=nonstopmode -quiet results_report.tex >/dev/null 2>&1) \
    && echo "PDF: $RUN_DIR/results/results_report.pdf" || echo "(latexmk could not build the PDF; the .tex is there)"
fi
echo "Run folder: $RUN_DIR (also $OUT_ROOT/latest)"
echo "Verdicts:   $RUN_DIR/results/verdicts.json"
echo "Report:     $RUN_DIR/results/results_report.tex"
echo "Audit:      $RUN_DIR/audit/human_check_<model>.csv (fill human_label after the run)"
grep -E "^VERDICT:|STAGE_FAILED|HARD_STOP|BUDGET_CUT" "$RUN_DIR/logs/run_log.txt" || true
exit $STATUS

#!/usr/bin/env bash
# Recoverability-Routed Answering (rra-experiment-design.tex), one command on the GPU server:
#
#   bash run_rra.sh                 everything: environment -> setup (downloads if needed) -> preflight -> run -> report
#   bash run_rra.sh resume DIR      continue a crashed run in outputs_rra/run_<id> (finished stages are reused)
#   bash run_rra.sh analyze DIR     recompute the analysis, verdict and report of a run (CPU only)
#   bash run_rra.sh smoke           CPU smoke test of every code path with small models (numbers mean nothing)
#   bash run_rra.sh test            unit tests (seconds)
#
# Needs two earlier run folders (found automatically; see README):
#   VES/ECS run  outputs_ves/run_20261115T053802Z (frozen D, dev/test drafts, states, B7 records); RRA_VES_RUN_DIR=...
#   SQ2 run      outputs_pilot/run_20261110T010318Z (pool answers, labels, folds, states);         RRA_SQ2_RUN_DIR=...
# Each run writes outputs_rra/run_<UTC>/ (outputs_rra/latest points to the newest). Planned ~3.3 h for the required
# stages, ~6.3 h with the optional ones (E2 tries, G Gemma-2-9B-it); the budget guard cuts only E2 / G, only if
# the measured projection exceeds 10.5 h; no stage starts after 11.5 h. RRA_SKIP_G=1 / RRA_SKIP_E2=1 switch them off.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
STAGE="${1:-all}"
PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
OUT_ROOT="outputs_rra"
say() { printf '\n==== %s ====\n' "$*"; }

case "$STAGE" in
  all|resume|analyze|smoke|test) ;;
  *) echo "usage: bash run_rra.sh [all|resume DIR|analyze DIR|smoke|test]"; exit 2 ;;
esac

# ------------------------------------------------------------------ environment (same .venv as SQ2 and VES/ECS)
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
  LOCK=configs/requirements.lock
  if [ -f "$LOCK" ]; then
    "$PY" -m pip install --quiet -r "$LOCK" --extra-index-url "$TORCH_INDEX_URL" || "$PY" -m pip install --quiet -r requirements.txt
  else
    "$PY" -m pip install --quiet -r requirements.txt
  fi
  "$PY" -m pip install --quiet pyyaml datasets
  env_ok || { echo "The environment cannot import the required packages; see the pip output above."; exit 1; }
fi
echo "python $("$PY" -c 'import platform; print(platform.python_version())'), torch $("$PY" -c 'import torch; print(torch.__version__)'), transformers $("$PY" -c 'import transformers; print(transformers.__version__)')"

if [ "$STAGE" = "test" ]; then
  say "Unit tests"
  STEER_DEVICE=cpu exec "$PY" -m rra.tests
fi
if [ "$STAGE" = "smoke" ]; then
  say "CPU smoke test (small models; ~45-75 min on an 8-core CPU incl. gemma-2-2b; numbers mean nothing)"
  STEER_DEVICE=cpu exec "$PY" -m rra.smoke_test
fi
if [ "$STAGE" = "analyze" ]; then
  RUN_DIR="${2:?usage: bash run_rra.sh analyze outputs_rra/run_<id>}"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 STEER_DEVICE=cpu
  [ -f "$RUN_DIR/ves_run_dir.txt" ] && export RRA_VES_RUN_DIR="$(cat "$RUN_DIR/ves_run_dir.txt")"
  [ -f "$RUN_DIR/sq2_run_dir.txt" ] && export RRA_SQ2_RUN_DIR="$(cat "$RUN_DIR/sq2_run_dir.txt")"
  exec "$PY" -m rra.analysis --run-dir "$RUN_DIR"
fi

# ------------------------------------------------------------------ earlier run folders
if [ "$STAGE" = "resume" ]; then
  [ -z "${RRA_VES_RUN_DIR:-}" ] && [ -f "${2:-}/ves_run_dir.txt" ] && RRA_VES_RUN_DIR="$(cat "$2/ves_run_dir.txt")"
  [ -z "${RRA_SQ2_RUN_DIR:-}" ] && [ -f "${2:-}/sq2_run_dir.txt" ] && RRA_SQ2_RUN_DIR="$(cat "$2/sq2_run_dir.txt")"
fi
if [ -z "${RRA_VES_RUN_DIR:-}" ]; then
  RRA_VES_RUN_DIR="$("$PY" - <<'EOF'
import glob
from pathlib import Path
models = ("llama31_8b_instruct", "qwen25_7b_instruct")
def ok(p):
    p = Path(p)
    return (p / "configs" / "frozen_config.json").exists() and (p / "runs" / "questions.json").exists() and all(
        (p / "runs" / m / s / f).exists() for m in models
        for s, f in (("pool", "det_D.npz"), ("dev", "search_B7.json"), ("test", "search_B7.json"), ("test", "draft_states.npz")))
cands = ["outputs_ves/run_20261115T053802Z", "run_20261115T053802Z"] + sorted(glob.glob("*20261115T053802Z*")) \
    + sorted(glob.glob("outputs_ves/run_*"), reverse=True) + sorted(glob.glob("run_*"), reverse=True)
for c in cands:
    if ok(c):
        print(str(Path(c).resolve()))
        break
EOF
)"
fi
if [ -z "${RRA_VES_RUN_DIR:-}" ] || [ ! -f "$RRA_VES_RUN_DIR/configs/frozen_config.json" ]; then
  echo "VES/ECS run folder not found. Copy the VES/ECS run folder (outputs_ves/run_20261115T053802Z: configs/,"
  echo "runs/questions.json, runs/<model>/{pool,dev,test}/ and results/summary.json) into this repository, or set"
  echo "RRA_VES_RUN_DIR=/path/to/it."
  exit 1
fi
export RRA_VES_RUN_DIR
if [ -z "${RRA_SQ2_RUN_DIR:-}" ]; then
  RRA_SQ2_RUN_DIR="$(RRA_VES_RUN_DIR="$RRA_VES_RUN_DIR" "$PY" - <<'EOF'
import glob, os
from pathlib import Path
models = ("llama31_8b_instruct", "qwen25_7b_instruct")
def ok(p):
    p = Path(p)
    return p.is_dir() and all((p / f).exists() for f in ("pool.jsonl", "extension_pool.jsonl", "pilot_qids.json")) \
        and all((p / m / "shared.pt").exists() for m in models)
txt = Path(os.environ["RRA_VES_RUN_DIR"]) / "sq2_run_dir.txt"
cands = ([txt.read_text().strip()] if txt.exists() else []) + ["outputs_pilot/run_20261110T010318Z"] \
    + sorted(glob.glob("*20261110T010318Z*")) + sorted(glob.glob("outputs_pilot/*20261110T010318Z*")) \
    + sorted(glob.glob("outputs_pilot/run_*"), reverse=True) + sorted(glob.glob("run_*"), reverse=True)
for c in cands:
    if ok(c):
        print(str(Path(c).resolve()))
        break
EOF
)"
fi
if [ -z "${RRA_SQ2_RUN_DIR:-}" ] || [ ! -f "$RRA_SQ2_RUN_DIR/pool.jsonl" ]; then
  echo "SQ2 run folder not found. Copy the SQ2 pilot run folder (outputs_pilot/run_20261110T010318Z: pool.jsonl,"
  echo "extension_pool.jsonl, pilot_qids.json, pilot_manifest.json, <model>/shared.pt) into this repository, or set"
  echo "RRA_SQ2_RUN_DIR=/path/to/it."
  exit 1
fi
export RRA_SQ2_RUN_DIR
echo "VES/ECS run folder: $RRA_VES_RUN_DIR"
echo "SQ2 run folder:     $RRA_SQ2_RUN_DIR"

# ------------------------------------------------------------------ setup (network; skipped when everything is present)
WANT_G=1
[ "${RRA_SKIP_G:-0}" = "1" ] && WANT_G=0
NEED_SETUP=0
for f in data_cache/ves_popqa_ids.jsonl data_cache/pilot_popqa_full.jsonl data_cache/triviaqa_validation.jsonl; do
  [ -f "$f" ] || NEED_SETUP=1
done
WANT_G="$WANT_G" "$PY" - <<'EOF' || NEED_SETUP=1
import os, sys
from pilot import config as PC
from rra import cfg as C
from rra.engine import snapshot_ok
cfg = C.load_config()
specs = list(PC.MODELS) + ([C.gemma_spec(cfg)] if os.environ["WANT_G"] == "1" and cfg["optional"]["G"] else [])
sys.exit(0 if all(snapshot_ok(s) for s in specs) else 1)
EOF
if [ "$STAGE" = "all" ] && [ "$NEED_SETUP" = "1" ]; then
  say "Setup: downloading what is missing (models at pinned revisions incl. Gemma-2-9B-it for stage G, data, PopQA ids)"
  mkdir -p "$OUT_ROOT/setup_logs"
  G_FLAG=""; [ "$WANT_G" = "0" ] && G_FLAG="--no-gemma"
  if ! "$PY" -m rra.setup_data --sq2-run-dir "$RRA_SQ2_RUN_DIR" $G_FLAG 2>&1 | tee "$OUT_ROOT/setup_logs/setup_$(date -u +%Y%m%dT%H%M%SZ).txt"; then
    echo "WARNING: setup did not complete; preflight decides whether the run can start."
  fi
fi

# ------------------------------------------------------------------ preflight
say "Preflight"
set +e
WANT_G="$WANT_G" "$PY" - <<'EOF'
import os, shutil, sys
import torch
from pilot import config as PC
from rra import cfg as C
from rra.engine import snapshot_ok
problems, notes = [], []
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
if not (PC.DATA_DIR / "ves_popqa_ids.jsonl").exists():
    problems.append("missing data_cache/ves_popqa_ids.jsonl (PopQA object ids for the isolation rule)")
from vesecs.setup_data import check_against_sq2
problems += [p for p in check_against_sq2(os.environ["RRA_SQ2_RUN_DIR"]) if "not found" not in p]
for s in PC.MODELS:
    if snapshot_ok(s):
        print(f"model {s['hf_id']} @ {s['revision'][:12]} cached")
    else:
        problems.append(f"{s['hf_id']} not cached at {s['revision'][:12]}")
cfg = C.load_config()
g = C.gemma_spec(cfg)
if os.environ["WANT_G"] == "1" and cfg["optional"]["G"]:
    if snapshot_ok(g):
        print(f"model {g['hf_id']} @ {g['revision'][:12]} cached (optional stage G)")
    else:
        notes.append(f"{g['hf_id']} is not cached: optional stage G is switched off for this run")
free = shutil.disk_usage(".").free / 2**30
print(f"free disk: {free:.0f} GiB")
if free < 10:
    problems.append(f"only {free:.0f} GiB free disk (need ~10)")
for n in notes:
    print("NOTE: " + n)
if problems:
    print("PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems))
    sys.exit(1)
print("preflight ok")
sys.exit(3 if notes else 0)
EOF
rc=$?
set -e
if [ "$rc" = "3" ]; then export RRA_SKIP_G=1; elif [ "$rc" != "0" ]; then exit 1; fi

# ------------------------------------------------------------------ run
if [ "$STAGE" = "resume" ]; then
  RUN_DIR="${2:?usage: bash run_rra.sh resume outputs_rra/run_<id>}"
else
  RUN_DIR="$OUT_ROOT/run_$(date -u +%Y%m%dT%H%M%SZ)"
  [ -e "$RUN_DIR" ] && { echo "$RUN_DIR exists; start again in a second."; exit 1; }
  mkdir -p "$RUN_DIR"
  ln -sfn "$(basename "$RUN_DIR")" "$OUT_ROOT/latest"
fi
mkdir -p "$RUN_DIR/logs"
echo "$RRA_VES_RUN_DIR" > "$RUN_DIR/ves_run_dir.txt"
echo "$RRA_SQ2_RUN_DIR" > "$RUN_DIR/sq2_run_dir.txt"
[ -f configs/requirements.lock ] && cp configs/requirements.lock "$RUN_DIR/requirements.lock"
"$PY" -m pip freeze > "$RUN_DIR/pip_freeze.txt" 2>/dev/null || true

say "Running into $RUN_DIR (offline; planned ~3.3 h required, ~6.3 h with E2 and G; cuts only E2/G above 10.5 h)"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 RRA_OUT_DIR="$PWD/$RUN_DIR"
START=$(date +%s)
set +e
"$PY" -m rra.main > >(tee -a "$RUN_DIR/logs/run_log.txt") 2> >(tee -a "$RUN_DIR/logs/run_stderr.txt" >&2)
STATUS=$?
set -e
say "finished with status $STATUS after $(( ($(date +%s) - START) / 60 )) min"
if command -v latexmk >/dev/null 2>&1 && [ -f "$RUN_DIR/results/results_report.tex" ]; then
  (cd "$RUN_DIR/results" && latexmk -pdf -interaction=nonstopmode -quiet results_report.tex >/dev/null 2>&1) \
    && echo "PDF: $RUN_DIR/results/results_report.pdf" || echo "(latexmk could not build the PDF; the .tex is there)"
fi
echo "Run folder: $RUN_DIR (also $OUT_ROOT/latest)"
echo "Verdict:    $RUN_DIR/results/verdicts.json"
echo "Report:     $RUN_DIR/results/results_report.tex"
echo "Audit:      $RUN_DIR/audit/human_check_<model>.csv (fill human_label after the run)"
grep -E "^VERDICT:|STAGE_FAILED|RUN_STOPPED|HARD_STOP|BUDGET_CUT|TEST_PHASE_REFUSED" "$RUN_DIR/logs/run_log.txt" || true
exit $STATUS

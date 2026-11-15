"""CPU smoke test of every code path (no results; its numbers mean nothing).

    python -m vesecs.smoke_test [--out DIR] [--keep-sq2]

1. Builds a throw-away data folder exactly as pilot/smoke_test.py does (real PopQA rows at the pinned revision,
   small TriviaQA stand-ins) plus the PopQA id file (vesecs.setup_data).
2. Builds a tiny SQ2 run with SQ2's own code (pilot.data pool + extension, pilot.shared.run_shared,
   pilot.probes.crossfit_probe) for two small instruct models (SmolLM2-135M: Llama architecture;
   Qwen2.5-0.5B: Qwen2), float32 on CPU.
3. Runs python -m vesecs.main on it (VES_SMOKE=1: tiny splits), then runs it again on the same folder to check
   that a finished run resumes without recomputing, then re-runs the analysis alone.
Fails on any STAGE_FAILED / Traceback line, a non-zero exit, or a missing output file.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAIL = ("STAGE_FAILED", "Traceback", "REPORT_FAILED", "HARD_STOP", "BUDGET_ERROR", "TEST_PHASE_REFUSED")

SQ2_SCRIPT = r"""
import json, numpy as np, torch
from pilot import config as PC, data as pdata
from pilot.session import Session
from pilot.shared import run_shared
from pilot.probes import crossfit_probe
from scoring import A_, C_
from pathlib import Path
out = Path(PC.OUT_DIR); out.mkdir(parents=True, exist_ok=True)
pool, info = pdata.build_pool()
ext, _ = pdata.build_ext_pool(pool)
pdata.write_pool(pool, out / "pool.jsonl"); pdata.write_pool(ext, out / "extension_pool.jsonl")
pdata.write_pilot_qids(pool + ext, out / "pilot_qids.json")
folds = np.array([it.extra["fold"] for it in pool])
for spec in PC.MODELS:
    s = Session(spec, spec["revision"])
    sh = run_shared(s, pool, heads=False)
    (out / spec["name"]).mkdir(exist_ok=True)
    torch.save(sh, out / spec["name"] / "shared.pt")
    oof, ch = crossfit_probe(sh["res_prompt"], sh["res_ans"], sh["states"] == C_, sh["states"] != A_, folds)
    torch.save({"oof": oof, "chosen": ch}, out / spec["name"] / "probe.pt")
    print("SQ2_SMOKE", spec["name"], "acc", float(np.mean(sh["states"] == C_)), "idk", float(np.mean(sh["states"] == A_)))
"""


def env_for(tmp, data, sq2, out):
    from pilot.smoke_test import SMOKE_MODELS
    return dict(os.environ, VES_SMOKE="1", PILOT_SMOKE="1", STEER_DEVICE="cpu", STEER_DTYPE="float32",
                STEER_DATA_DIR=str(data), PILOT_MANIFEST=str(tmp / "manifest.json"), PILOT_OUT_DIR=str(sq2),
                VES_SQ2_RUN_DIR=str(sq2), VES_OUT_DIR=str(out), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                PYTHONUNBUFFERED="1", PILOT_SMOKE_MODELS=";".join("|".join(str(x) for x in m) for m in SMOKE_MODELS))


def run(cmd, env, log):
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    text = Path(log).read_text()
    bad = [l for l in text.splitlines() if any(m in l for m in FAIL)]
    if rc != 0 or bad:
        print(text[-8000:])
        raise SystemExit(f"SMOKE FAILED (exit {rc}) in {' '.join(cmd)}; see {log}\n" + "\n".join(bad[:20]))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tmp = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="ves_smoke_"))
    tmp.mkdir(parents=True, exist_ok=True)
    data, sq2, out = tmp / "data", tmp / "sq2_run", tmp / "ves_run"
    if not (data / "pilot_popqa_full.jsonl").exists():
        from pilot.smoke_test import build
        build(tmp)
    env = env_for(tmp, data, sq2, out)
    if not (data / "ves_popqa_ids.jsonl").exists():
        run([sys.executable, "-m", "vesecs.setup_data", "--ids-only"], dict(env, HF_HUB_OFFLINE="0"),
            tmp / "setup_data_log.txt")
    if not all((sq2 / m / "shared.pt").exists() for m in ("smollm2_135m_smoke", "qwen25_05b_smoke")):
        print("building the tiny SQ2 run (SQ2 code, tiny models) ...", flush=True)
        run([sys.executable, "-c", SQ2_SCRIPT], env, tmp / "sq2_log.txt")
    print("running vesecs.main ...", flush=True)
    text = run([sys.executable, "-m", "vesecs.main"], env, tmp / "ves_log.txt")
    print(text[-5000:])
    need = ["configs/config.yaml", "configs/frozen_config.json", "results/summary.json", "results/verdicts.json",
            "results/results_report.tex", "results/integrity.json", "logs/timing.jsonl", "logs/versions.json",
            "logs/model_sha256.json", "logs/drafts_registry.json"]
    names = ["smollm2_135m_smoke", "qwen25_05b_smoke"]
    need += [f"audit/human_check_{m}.csv" for m in names]
    need += [f"runs/{m}/{s}/records.jsonl" for m in names for s in ("dev", "test")]
    missing = [p for p in need if not (out / p).exists()]
    if missing:
        raise SystemExit(f"SMOKE FAILED: missing outputs {missing}")
    s = json.loads((out / "results/summary.json").read_text())
    v = json.loads((out / "results/verdicts.json").read_text())
    print("VERDICTS:", {k: x["verdict"] for k, x in v.items()})
    print("MAIN TABLE methods:", list(s["main_table"]))
    print("INTEGRITY:", s["integrity_status"])
    expect = {"B0", "B1", "B3", "B3b", "B5@1", "B5@8", "B5@Nm", "B7", "VES", "VES_free", "VES_placebo", "VES_unclean",
              "ECS_1p", "ECS_exact", "ECS_placebo", "R1", "R2", "R3"}
    if not expect <= set(s["main_table"]):
        raise SystemExit(f"SMOKE FAILED: methods missing from the main table: {sorted(expect - set(s['main_table']))}")
    if s["missing"]:
        raise SystemExit(f"SMOKE FAILED: missing parts {s['missing']}")
    print("re-running on the finished folder (resume must recompute nothing) ...", flush=True)
    t2 = run([sys.executable, "-m", "vesecs.main"], env, tmp / "ves_log_resume.txt")
    if any(l.startswith(("SEARCH:", "DRAFTS:", "LOADING:")) for l in t2.splitlines()):
        raise SystemExit("SMOKE FAILED: the resumed run recomputed finished stages")
    run([sys.executable, "-m", "vesecs.analysis", "--run-dir", str(out)], env, tmp / "ves_log_analysis.txt")
    print(f"SMOKE OK: {tmp}")


if __name__ == "__main__":
    main()

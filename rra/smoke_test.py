"""CPU smoke test of every code path (no results; its numbers mean nothing).

    python -m rra.smoke_test [--out DIR] [--no-gemma]

1. A throw-away data folder exactly as the SQ2 / VES-ECS smoke tests build it (real PopQA rows at the pinned
   revision, small TriviaQA stand-ins) plus the PopQA id file.
2. A tiny SQ2 run with SQ2's own code (vesecs.smoke_test.SQ2_SCRIPT: SmolLM2-135M and Qwen2.5-0.5B, float32 on CPU).
3. A tiny VES/ECS run with the VES/ECS code: the parts RRA reuses (strict pool labels, detector D, drafts, states,
   B7 search on dev and test, dev tuning, frozen configuration).
4. python -m rra.main on them (RRA_SMOKE=1: tiny splits; optional E2 on, optional G with gemma-2-2b-it, the same
   Gemma-2 tokenizer and chat template as the 9B), then again on the same folder (resume must recompute nothing),
   then the analysis alone; the LaTeX report is compiled if latexmk is installed.
Fails on any STAGE_FAILED / Traceback / RUN_STOPPED line, a non-zero exit, a missing output or a failed check.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAIL = ("STAGE_FAILED", "Traceback", "REPORT_FAILED", "HARD_STOP", "BUDGET_ERROR", "TEST_PHASE_REFUSED",
        "RUN_STOPPED", "UNIT_TEST_FAILURES")

VES_MINI = r"""
from vesecs.pipeline import Pipeline
pl = Pipeline()
pl.phase_a()
for m in pl.names:
    pl.pool_labels(m)
    pl.b_detectors(m)
    pl.b_ubar(m)
    pl.b_lens(m)
    pl.c_drafts(m, "dev")
    pl.c_states(m, "dev")
    pl.search_stage(m, "dev", "b7", [(None, None)], "B7")
    pl.tuning_stage(m)
    pl.release()
pl.phase_d()
for m in pl.names:
    pl.c_drafts(m, "test")
    pl.c_states(m, "test")
    pl.search_stage(m, "test", "b7", [(None, None)], "B7")
    pl.release()
print("VES_MINI_DONE")
"""


def env_for(tmp, data, sq2, ves, out):
    from vesecs.smoke_test import env_for as ves_env
    e = ves_env(tmp, data, sq2, ves)
    e.update(RRA_SMOKE="1", RRA_VES_RUN_DIR=str(ves), RRA_SQ2_RUN_DIR=str(sq2), RRA_OUT_DIR=str(out))
    return e


def run(cmd, env, log, fail=FAIL):
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    text = Path(log).read_text()
    bad = [l for l in text.splitlines() if any(m in l for m in fail)]
    if rc != 0 or bad:
        print(text[-8000:])
        raise SystemExit(f"SMOKE FAILED (exit {rc}) in {' '.join(cmd)}; see {log}\n" + "\n".join(bad[:20]))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-gemma", action="store_true")
    args = ap.parse_args()
    tmp = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="rra_smoke_"))
    tmp.mkdir(parents=True, exist_ok=True)
    data, sq2, ves, out = tmp / "data", tmp / "sq2_run", tmp / "ves_run", tmp / "rra_run"
    if not (data / "pilot_popqa_full.jsonl").exists():
        from pilot.smoke_test import build
        build(tmp)
    env = env_for(tmp, data, sq2, ves, out)
    if args.no_gemma:
        env["RRA_SKIP_G"] = "1"
    if not (data / "ves_popqa_ids.jsonl").exists():
        run([sys.executable, "-m", "vesecs.setup_data", "--ids-only"], dict(env, HF_HUB_OFFLINE="0"),
            tmp / "setup_ids_log.txt")
    if not args.no_gemma:
        import yaml
        from huggingface_hub import snapshot_download
        from pilot.config import MODEL_FILE_PATTERNS
        from rra.cfg import CONFIG_PATH
        g = yaml.safe_load(open(CONFIG_PATH))["smoke_overrides"]["gemma"]
        snapshot_download(g["hf_id"], revision=g["revision"], allow_patterns=MODEL_FILE_PATTERNS)
    from vesecs.smoke_test import SQ2_SCRIPT
    names = ("smollm2_135m_smoke", "qwen25_05b_smoke")
    if not all((sq2 / m / "shared.pt").exists() for m in names):
        print("building the tiny SQ2 run (SQ2 code, tiny models) ...", flush=True)
        run([sys.executable, "-c", SQ2_SCRIPT], env, tmp / "sq2_log.txt")
    if not all((ves / "runs" / m / "test" / "search_B7.json").exists() for m in names):
        print("building the tiny VES/ECS run (VES/ECS code: D, drafts, states, B7) ...", flush=True)
        # the VES/ECS code's own unit test of its budget cuts fails at smoke settings (its smoke test ignores it)
        run([sys.executable, "-c", VES_MINI], env, tmp / "ves_log.txt",
            fail=tuple(x for x in FAIL if x != "UNIT_TEST_FAILURES"))

    print("running rra.main ...", flush=True)
    text = run([sys.executable, "-m", "rra.main"], env, tmp / "rra_log.txt")
    print(text[-6000:])
    need = ["configs/config.yaml", "configs/frozen_config.json", "results/summary.json", "results/verdicts.json",
            "results/results_report.tex", "results/integrity.json", "logs/timing.jsonl", "logs/versions.json",
            "logs/model_sha256.json", "logs/records_registry.json", "logs/frozen_config_sha256.json", "logs/budget.json",
            "runs/questions.json"]
    need += [f"audit/human_check_{m}.csv" for m in names]
    need += [f"runs/{m}/{s}/records.jsonl" for m in names for s in ("dev_fresh", "test_fresh", "test_ves")]
    need += [f"runs/{m}/test_fresh/tries.json" for m in names]
    need += [f"runs/{m}/pool/{f}" for m in names for f in ("R.npz", "R.json", "a2.json", "labels.json")]
    if not args.no_gemma:
        need += ["results/generality.json"]
    missing = [p for p in need if not (out / p).exists()]
    if missing:
        raise SystemExit(f"SMOKE FAILED: missing outputs {missing}")
    s = json.loads((out / "results/summary.json").read_text())
    v = json.loads((out / "results/verdicts.json").read_text())
    print("VERDICT:", v["RRA"]["verdict"], "-", v["RRA"].get("reason"))
    print("MAIN TABLE arms:", list(s["main_table"]))
    print("INTEGRITY:", s["integrity_status"])
    expect = {"B0", "B3", "R-all", "R-band", "R-check", "RRA", "RRA-EU", "RRA+check", "RRA-peer", "Oracle",
              "Oracle-noswitch"}
    if not expect <= set(s["main_table"]):
        raise SystemExit(f"SMOKE FAILED: arms missing from the main table: {sorted(expect - set(s['main_table']))}")
    if not s["integrity_status"]["all_passed"]:
        raise SystemExit(f"SMOKE FAILED: integrity checks {s['integrity_status']['problems']}")
    if v["RRA"]["verdict"] in ("NOT_RUN", "UNTESTABLE"):
        raise SystemExit(f"SMOKE FAILED: verdict {v['RRA']}")
    if s["missing"]:
        raise SystemExit(f"SMOKE FAILED: missing parts {s['missing']}")
    for k in ("H", "d_main", "d_mech"):
        if not s["contrasts"].get(k):
            raise SystemExit(f"SMOKE FAILED: contrast {k} missing")
    if "confident_errors" not in s["strata"] or "replay" not in s:
        raise SystemExit("SMOKE FAILED: confident-error stratum (E2) or replay section missing")

    print("re-running on the finished folder (resume must recompute nothing) ...", flush=True)
    t2 = run([sys.executable, "-m", "rra.main"], env, tmp / "rra_log_resume.txt")
    if any(l.startswith(("LOADING:", "A1:", "A2:", "POOL_A2:", "R:")) for l in t2.splitlines()):
        raise SystemExit("SMOKE FAILED: the resumed run recomputed finished stages")
    v2 = json.loads((out / "results/verdicts.json").read_text())
    if v2["RRA"]["verdict"] != v["RRA"]["verdict"]:
        raise SystemExit("SMOKE FAILED: the resumed run changed the verdict")
    run([sys.executable, "-m", "rra.analysis", "--run-dir", str(out)], env, tmp / "rra_log_analysis.txt")
    gate = (r"from rra.pipeline import Pipeline; pl = Pipeline(out_dir=r'%s'); ok, why = pl.test_gate(); "
            r"assert not ok and 'frozen_config' in why, why; print('GATE_OK', why)") % (tmp / "gate_check")
    g = run([sys.executable, "-c", gate], env, tmp / "gate_log.txt", fail=("Traceback",))
    if "GATE_OK" not in g:
        raise SystemExit("SMOKE FAILED: the fresh-test gate did not refuse a run without frozen_config.json")
    if shutil.which("latexmk"):
        rc = subprocess.call(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "results_report.tex"],
                             cwd=out / "results", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rc != 0:
            raise SystemExit(f"SMOKE FAILED: the LaTeX report does not compile ({out / 'results'})")
        print("report PDF compiled")
    print(f"SMOKE OK: {tmp}")


if __name__ == "__main__":
    main()

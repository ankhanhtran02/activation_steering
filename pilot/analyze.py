"""Go / stop signals after the hand checks (CPU only, no GPU):

    python -m pilot.analyze [--run-dir outputs_pilot/run_<id>] [--provisional]

1. Fill `human_label` (correct / wrong / abstained) in <run>/grading_check_<model>.csv, 100 rows per model.
2. Fill `meaning_preserved` (yes / no) in <run>/paraphrase_check.csv.
3. Run this. If the grader disagrees with the hand labels on more than 5% of answered questions for a
   model, the signals are WITHHELD: fix the grading rules (scoring.py), re-grade the saved answers, and
   run this again. If they agree, the signals are FINAL and written to <run>/signals.json.

Before the grading check is filled, signals are shown only with --provisional (labelled PROVISIONAL);
the pilot protocol asks for the grader to be checked before results are looked at.
"""
import argparse
import json
from pathlib import Path

from pilot import config as PC
from pilot.metrics import to_py
from pilot.signals import all_signals


def print_signals(sig):
    print(f"GRADING_CHECK: {sig['grading_check']['status']}  REWORDING_CHECK: {sig['rewording_check']['status']}")
    for m, g in sig["grading_check"].items():
        if isinstance(g, dict):
            print(f"GRADING_CHECK: model={m} " + " ".join(f"{k}={v}" for k, v in g.items()))
    for k, v in sig["signals"].items():
        print(f"SIGNAL: {k} {v['signal']} ({sig['status']}) {v.get('reason', '')}")
        for m, pm in v.get("per_model", {}).items():
            print(f"SIGNAL: {k} model={m} {pm['signal']} {pm.get('reason', '')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--provisional", action="store_true", help="show signals before the grading check is filled")
    args = ap.parse_args()
    run = Path(args.run_dir) if args.run_dir else PC.latest_run_dir()
    if run is None or not (run / "results.json").exists():
        raise SystemExit("no finished pilot run found (outputs_pilot/run_*/results.json)")
    results = json.loads((run / "results.json").read_text())
    sig = all_signals(results, run)
    if sig["status"] == "WITHHELD":
        print(f"Signals WITHHELD: the automatic grader disagrees with the hand check on more than "
              f"{PC.GRADING_MAX_DISAGREE:.0%} of answered questions for at least one model. Fix the grading "
              "rules, re-grade the saved answers, then run this again.")
        print(json.dumps(to_py(sig["grading_check"]), indent=1))
        return
    if sig["status"] == "PROVISIONAL" and not args.provisional:
        stage2 = [m for m, g in sig["grading_check"].items() if isinstance(g, dict) and g.get("status") == "NEED_STAGE2"]
        if stage2:
            print(f"Grading check ambiguous after stage 1 for {', '.join(stage2)}: also fill human_label in the "
                  "stage 2 rows of grading_check_<model>.csv, then run this again.")
        else:
            print("Grading check not filled yet: fill human_label in the stage 1 rows of grading_check_<model>.csv "
                  "(and meaning_preserved in paraphrase_check.csv), then run this again. Use --provisional to see "
                  "unchecked signals.")
        print(json.dumps(to_py(sig["grading_check"]), indent=1))
        return
    print_signals(sig)
    name = "signals.json" if sig["status"] == "FINAL" else "signals_provisional.json"
    (run / name).write_text(json.dumps(to_py(sig), indent=1))
    print(f"wrote {run / name}")


if __name__ == "__main__":
    main()

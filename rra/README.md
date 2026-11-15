# Recoverability-Routed Answering (RRA)

Implementation of [rra-experiment-design.tex](../rra-experiment-design.tex). The model answers once (A1). From the
hidden states of that pass, the frozen detector D scores A1 (s1) and a new **recoverability probe R** predicts
whether the fallback A2 (greedy, A1's first token banned at the first step) would be right (s2). RRA keeps A1, switches
to A2 without checking it, or says "I don't know". The run ends with a GO / BORDERLINE / STOP / INSUFFICIENT_DATA
verdict on the fresh test set.

## Running it

On the GPU server, from the repository root:

```bash
git checkout rra
nohup bash run_rra.sh > run_rra_all.txt 2>&1 &
tail -f outputs_rra/latest/logs/run_log.txt
```

That one command:

1. Checks the Python environment (the `.venv` of SQ2 and VES/ECS) and builds it if it is missing.
2. Finds the two earlier run folders it reuses:
   - **VES/ECS run** `outputs_ves/run_20261115T053802Z` (or `./run_20261115T053802Z`, or the newest complete
     `outputs_ves/run_*`): the frozen detector D, VES/ECS dev and test drafts, states and B7 records,
     `results/summary.json` (for the B3 = 39.0 check). Override: `RRA_VES_RUN_DIR=/path`.
   - **SQ2 run** named in the VES/ECS run's `sq2_run_dir.txt`, else `outputs_pilot/run_20261110T010318Z` (or a
     `*20261110T010318Z*` folder). Override: `RRA_SQ2_RUN_DIR=/path`.
3. Setup, only if something is missing (needs network): models at pinned revisions (incl. Gemma-2-9B-it for the
   optional stage G, ~18 GB, ungated mirror `unsloth/gemma-2-9b-it`), data files, PopQA id file.
4. Preflight: GPU, bf16, models cached, data checksums vs the SQ2 run, free disk. If Gemma is not cached, stage G is
   switched off (the rest runs).
5. Runs everything offline into `outputs_rra/run_<UTC>/` and prints the verdict.

| Command | What it does |
|---|---|
| `bash run_rra.sh` | everything (above) |
| `bash run_rra.sh resume outputs_rra/run_<id>` | continue a crashed run; finished stages are reused, the budget clock carries over (`RRA_RESET_CLOCK=1` resets it) |
| `bash run_rra.sh analyze outputs_rra/run_<id>` | recompute the analysis, verdict and report (CPU) |
| `bash run_rra.sh smoke` | CPU smoke test of every code path (small models; numbers mean nothing) |
| `bash run_rra.sh test` | unit tests (seconds) |

`RRA_SKIP_E2=1` / `RRA_SKIP_G=1` switch an optional stage off.

**Expected time on the GB10.** ~3.3 h for the required stages (A-F), ~6.3 h with E2 and G. The budget guard uses
only measured rates (each model's timing probe, then actual per-item rates). Only E2 and G can be cut, and only if
the measured projection exceeds 10.5 h; they run after phase F has written the verdict. No stage starts after
11.5 h. Every projection and cut is in `logs/timing.jsonl`.

**Failure policy.** A GPU memory error is retried with half the batch size, up to 3 times (model loading: wait
30/60/120 s and retry); then the run stops and the report says what is missing. Checks are never skipped. A failed
integrity check blocks the fresh-test phase (`RRA_ALLOW_FAILED_CHECKS=1` overrides; the report then says so).

**After the run.** Fill `human_label` in `audit/human_check_<model>.csv` (100 routed retries per model).

## Outputs (`outputs_rra/run_<id>/`)

| Path | Content |
|---|---|
| `results/verdicts.json` | the RRA verdict and the inputs to the rule |
| `results/summary.json` | main table, contrasts (H, Δmain, Δmech, others) with CIs, probe diagnostics, strata, replay, tuning, integrity, budget |
| `results/results_report.tex` | the report (SQ2 report structure); `latexmk` builds the PDF if installed |
| `results/integrity.json` | every integrity check with details |
| `results/generality.json` | optional stage G (Gemma), reported separately |
| `configs/frozen_config.json` | thresholds, probe hashes, Platt fits, comparator C (SHA-256 in `logs/frozen_config_sha256.json`) |
| `runs/<model>/pool/` | strict labels, A2 for the pool, R (`R.npz`, `R.json`, `R_oof.npy`), check results |
| `runs/<model>/{dev_ves,test_ves}/` | VES/ECS dev (tuning) and test (replay): re-teacher-forced A1 states, s1 vs stored |
| `runs/<model>/{dev_fresh,test_fresh}/` | `a1.json`, `states.npz` (float16, 5 depths), `a2.json` (with D(A2)), `tries.json` (E2), `records.jsonl` |
| `audit/human_check_<model>.csv` | 100 routed retries per model |
| `logs/` | timing (measured rates, projections, cuts), budget, versions, model SHA-256, records registry |

## Code map

| Module | Role |
|---|---|
| `pipeline.py` | phases A-E, optional E2 / G; resumable stages; checks 2-6 and 9 |
| `analysis.py` | phase F: fresh test, bootstrap, verdict, diagnostics, strata, replay, records, audit; stage G evaluation |
| `arms.py` | null-fallback rule, every arm's output, joint threshold search (vectorised, checked against brute force) |
| `probes.py` | R (SQ2 recipe via `vesecs.detectors`), shuffled-label check, Platt scaling, calibration curves |
| `data.py` | fresh dev / test questions with the isolation rules; VES/ECS sets |
| `engine.py` | the VES/ECS engine plus Gemma-2 loading (eager attention) and the memory-error policy |
| `budget.py` | the fixed budget guard |
| `verdicts.py`, `checks.py` | decision rule; integrity-check registry |
| `report.py` | LaTeX report |
| `tests.py`, `smoke_test.py` | unit tests (checks 7 and 8); CPU end-to-end smoke test |

## Choices the spec left open

- **Tuning / replay states.** R may pick a depth the VES/ECS run did not store, so the VES/ECS dev and test A1 are
  teacher-forced again at all five SQ2 depths (~4 min per model), in the VES/ECS run's batch composition. This also
  gives integrity check 2: s1 recomputed from these states vs the stored scores.
- **B3.** Main arm: τ1 tuned on the tuning set like every other arm (spec: arms table). Check 2 uses B3 with the
  VES/ECS dev-tuned τ on the replay set (= 39.05 there; the VES/ECS `summary.json` value is the target, ±0.05).
- **State reproduction (check 3).** The first SQ2 batch (64 rows) is teacher-forced again and the first 50 compared,
  so the batch composition is SQ2's (the VES/ECS run measured 0.0 difference this way).
- **Fallback reproduction (check 4)** compares the first B7 batch; it is reported, not blocking (the spec's
  consequence of > 2% differences is a reporting rule: stored A2 is always used for replay and VES/ECS dev).
- **Null fallback.** Order: garbled → "I don't know" → hedged (gold-free form) → equal to A1 after normalisation →
  one contains the other as whole words (`pilot.metrics.same_answer`).
- **Routed precision** = right / (right + wrong) among fallback outputs, so that "above λ/(1+λ)" is exactly the
  condition for retries to add utility.
- **Ties.** Threshold search takes the first maximum with every threshold ascending (smaller thresholds first, as
  SQ2 and VES/ECS). Comparator C: R-band only if strictly better than R-check on the pooled tuning U1.
- **RRA-EU** picks the first maximum of (keep, retry, IDK); **RRA-peer** applies the same null rule to the other
  model's answer and costs 1 extra FPE per flagged question.
- **Verdict order** follows the spec's table: INSUFFICIENT_DATA, STOP, GO, BORDERLINE.
- **Stage G.** Gemma-2 has no system role: the SQ2 system prompt starts the user turn. Gemma gets its own D (SQ2
  recipe on its pool states) and R; thresholds are tuned on the fresh dev set only; comparator from the main run.
- **E2 tries** use per-question seeds hash(model, qid, try) as in VES/ECS.

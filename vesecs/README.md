# VES and ECS on top of the SQ2 detector

Implementation of [ves-ces-experiment.tex](../ves-ces-experiment.tex): do Verify-and-Exclude Steering (VES) and
Echo-Check Steering (ECS) reduce hallucination beyond the SQ2 detector gate (B3)? It ends with a
GO / BORDERLINE / STOP / INSUFFICIENT_DATA verdict for each method.

## Running it

On the GPU server, from the repository root:

```bash
git checkout pilot-confidence-rule
nohup bash run_ves.sh > run_ves_all.txt 2>&1 &
tail -f outputs_ves/latest/logs/run_log.txt
```

That one command:

1. Checks the Python environment (the SQ2 pilot's `.venv`) and builds it if it is missing.
2. Finds the SQ2 run folder: `outputs_pilot/run_20261110T010318Z`, or the newest `outputs_pilot/run_*` / `run_*`
   folder that has both models' `shared.pt`. Set `VES_SQ2_RUN_DIR=/path` to choose one. On a machine without it,
   copy the SQ2 run folder over first (`pool.jsonl`, `extension_pool.jsonl`, `pilot_qids.json`,
   `pilot_manifest.json`, `<model>/shared.pt`, `<model>/probe.pt`).
3. Setup, only if something is missing (needs network): the two models at their pinned revisions, the SQ2 data
   files (PopQA, TriviaQA), and `data_cache/ves_popqa_ids.jsonl` (PopQA object ids for the isolation rule).
   The data files must match the SQ2 run's checksums.
4. Preflight: GPU, bf16, models cached, data checksums, free disk.
5. Runs everything offline into `outputs_ves/run_<UTC>/` and prints the verdicts.

| Command | What it does |
|---|---|
| `bash run_ves.sh` | everything (above) |
| `bash run_ves.sh resume outputs_ves/run_<id>` | continue a crashed run; finished stages are reused. The budget clock carries over (`VES_RESET_CLOCK=1` resets it) |
| `bash run_ves.sh analyze outputs_ves/run_<id>` | recompute the test analysis, verdicts and report (CPU) |
| `bash run_ves.sh smoke` | CPU smoke test of every code path with two tiny models (~20-40 min; numbers mean nothing) |
| `bash run_ves.sh test` | unit tests (seconds) |

**Expected time.** Planned ~7 h on one GB10 (spec section 11). The timing probe at each model's first load measures
the real rates; after every stage `logs/timing.jsonl` gets the projected end time. If the projection exceeds 10.5 h
(or a stage overruns by more than 50% and the projection no longer fits), the pre-registered cuts are applied in
order: vLLM tries, drop R1-R3, drop placebos, shrink the VES dev grid, shrink the test set to 1,500. No stage starts
after 11.5 h.

**After the run.** Fill `human_label` in `audit/human_check_<model>.csv` (100 outputs per model that differ from
A1) to check the automatic grader on new outputs.

## Outputs (`outputs_ves/run_<id>/`)

| Path | Content |
|---|---|
| `results/verdicts.json` | VES and ECS verdicts and the inputs to each rule |
| `results/summary.json` | main table, contrasts with CIs, G, strata, diagnostics, risk-coverage curves, integrity, budget |
| `results/results_report.tex` | the report (SQ2 report structure); `latexmk` builds the PDF if installed |
| `results/integrity.json` | every integrity check with details |
| `configs/frozen_config.json` | dev-tuned configuration used on test (SHA-256 in `logs/frozen_config_sha256.json`) |
| `runs/<model>/pool/` | strict pool labels, detectors (`det_*.npz`), VES constants, logit-lens choice, echo states and detectors |
| `runs/<model>/{dev,test}/` | drafts, states (float16), tries, search records, ECS records, `records.jsonl` (spec appendix B schema) |
| `audit/human_check_<model>.csv` | outputs for the manual grading check |
| `logs/` | timing, budget, versions, model weight SHA-256, run log |

## Code map

| Module | Role |
|---|---|
| `pipeline.py` | phases A-E, resumable stages (each output written atomically; existing outputs are reused) |
| `analysis.py` | phase F: test evaluation, bootstrap, G, strata, diagnostics, verdicts, audit, records |
| `engine.py`, `hooks.py` | model operations on top of `pilot/session.py` (SQ2's loading, prompt and state extraction) |
| `data.py` | SQ2 pool loading; dev / test construction with the isolation rules |
| `grader.py` | strict grader = SQ2 grader + hedge rule; gold-free candidate checks |
| `detectors.py` | SQ2 ridge recipe (D, D_before, D_after) and the echo detector E |
| `search.py` | search loop in replay mode (VES, VES-free, VES-placebo, B7, B5) and the exact offline replay |
| `methods.py` | method outputs from records + thresholds; dev tuning |
| `stats.py`, `verdicts.py` | paired bootstrap, G, decision rules |
| `budget.py`, `checks.py` | 12-hour budget guard; integrity checks |
| `report.py` | LaTeX report |
| `tests.py`, `smoke_test.py` | unit tests; CPU end-to-end smoke test |

## Choices the spec left open

The spec says to reuse the SQ2 code where it is silent; these are the places where that was not enough.

- **ECS echo computation.** Side rows are computed as one-token steps on the prompt's key/value cache (exactly the
  fused algorithm's side row, since the side row only attends to the prompt and itself). This is ~20x cheaper than
  one full forward per candidate at the GB10's measured rates. Integrity check `echo_equivalence` compares it with
  the spec's full-forward simulation (`overwrite_last` on prompt + c); on the smoke model the cosine is 0.9999998.
- **Gradient check.** At ε = 0.01ρ a forward difference has a curvature term that can exceed 10% for a correct
  gradient (measured: 1 in 3 random directions miss in float32; a central difference is within 4%). The check runs
  the spec's forward difference first, then float32, then a central difference, and records which step passed.
- **Hedge rule.** "Same alias set" = the gold aliases (the parts all match an accepted answer, e.g. "USA / United
  States", is no hedge; a reply that is itself an alias, e.g. "AC/DC", is no hedge). For discarding candidates,
  methods use a gold-free version (parts that are all the same answer).
- **Isolation pairs** use PopQA entity ids (subject and object ids share one id space) and, as a conservative
  backup, normalised entity names. Linked items stay in one split (union-find).
- **Tries** use per-row seeds hash(model, qid, try) through Gumbel-max sampling (an exact T=1 draw), so a try does
  not depend on its batch.
- **ECS candidates** exclude every added/special token, tokens without an alphanumeric character, and t_IDK.
  Placebo tokens: the 1,000 most frequent alphabetic tokens (>= 2 letters) in the pool's answers and tries.
- **VES-placebo** maps the j-th rejected answer to a seeded random question among the 1,000 used for ū (their pull
  vectors are already computed). **VES-unclean** scores come from a steered teacher-forced pass (diagnostic only;
  not counted in FPE).
- **β choice and placebo detector.** E_exact and E_1P(β) for all β are fitted on the pool in phase B; after β is
  chosen on dev, E_placebo(β*) is fitted and the shuffled-label check runs on E_1P(β*).
- **λ = 3 sensitivity.** Every thresholded method is tuned separately for U_3. VES's λ = 3 configuration also runs
  on test when it differs from the λ = 1 one; ablations keep VES's λ = 1 (ℓ_s, α) with their own U_3 threshold.
- **Ties** in dev tuning go to the smaller threshold (higher coverage, as SQ2), then to the earlier grid entry.

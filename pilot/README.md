# Pilot: does activation steering do anything a confidence rule can't?

This is a quick, one-GPU check of which of five sub-questions are worth a full study. It runs on one
NVIDIA GB10 within 12 hours. The protocol, the go / stop rules and every change from the original
summary are in [PREREGISTRATION.md](PREREGISTRATION.md).

| | Question | Go if |
|---|---|---|
| SQ1 | Do steering methods know anything beyond confidence? | a cheap, generation-free test separates a detector-gated "smart" steering from 20 random directions (also after adjusting for confidence shape, answer length and dataset) and predicts real steered answers |
| SQ2 | Does the inside of the model know about its mistakes when its outputs don't? | >= 150 disagreement questions (from the pool plus a 2,969-question screening extension), and an internal-state detector adds >= 0.03 AUROC to confidence + consistency |
| SQ3 | Can combining internal states across rewordings fix wrong answers? | >= 10% of wrong answers are fixable by rewording, and averaging states fixes at least as many as averaging probabilities without breaking more than it fixes (and beats a different-subject control) |
| SQ4 | Can training teach the model to use hidden information? (only if SQ2 is go / borderline) | a LoRA trained on real labels beats one trained on confidence labels |
| SQ5 | Can the model give a vaguer but correct answer? | routing by confidence alone falls >= 5 points short of the best choice of detail the model's own signals allow (confidence + internal detector) |

## Running it

On the GPU machine, from the repository root:

```bash
git fetch origin && git checkout pilot-confidence-rule
nohup bash run_pilot.sh > run_pilot_all.txt 2>&1 &    # setup + preflight + pilot
tail -f outputs_pilot/latest/run_log.txt
```

| Command | What it does |
|---|---|
| `bash run_pilot.sh` | environment check, downloads (`pilot.setup_pilot`), GPU preflight, pilot |
| `bash run_pilot.sh setup` | downloads only (needs network). These are Llama-3.1-8B-Instruct (~16 GB), Qwen2.5-7B-Instruct (~15 GB, usually cached from the earlier study's judge), PopQA with Wikidata object ids, TriviaQA (reused), and the city → country table from Wikidata |
| `bash run_pilot.sh main` | pilot only (offline) |
| `bash run_pilot.sh resume outputs_pilot/run_<id>` | continue a crashed run. Finished phases are loaded from their `.pt` files |
| `bash run_pilot.sh smoke` | CPU smoke test of every code path with two tiny models, including every grading-gate case (~25-30 min on an 8-core CPU; its numbers mean nothing). `python -m pilot.smoke_test --out DIR --gate-only` re-checks only the gate on a finished smoke run |

The environment is the earlier study's `.venv`; no new package is needed. LoRA is implemented in
`pilot/sq4.py`, so `peft` is not required.

**Expected time.** About 5 hours (projected 4.8 h; 7.3 h if the shared-prefill sampler falls back to
plain sampling; 9.3 h at half the assumed throughput). The cap is 11 h and the hard limit 12 h; after
11.5 h no new phase starts. `BUDGET:` lines show the measured throughput, the projected total and any degradation-ladder
step (pre-registered order: see the pre-registration).

## After the run: two hand checks, then the signals

The run does not print go / stop signals. Your summary asks for the grader to be checked before anyone
looks at results.

1. Open `outputs_pilot/latest/grading_check_<model>.csv` (200 rows per model). Fill `human_label` with
   `correct`, `wrong` or `abstained` for the 100 rows with `stage` = 1. `pilot.analyze` tells you if a
   model also needs its 100 stage-2 rows: only when stage 1 has 4-7 disagreements (plan and
   pass rates in PREREGISTRATION.md section 2.1).
2. Open `outputs_pilot/latest/paraphrase_check.csv`. Fill `meaning_preserved` with `yes` or `no` (every
   rewording template appears at least once).
3. Run:

   ```bash
   .venv/bin/python -m pilot.analyze --run-dir outputs_pilot/run_<id>
   ```

**What `pilot.analyze` does with the grading check:**

- **Check fails for a model** (≥ 8 of the first 100, or > 10 of 200): the signals are withheld. Fix the
  grading rules in `scoring.py`, re-grade the saved answers, and run `pilot.analyze` again.
- **Agreement:** it prints `SIGNAL: SQk GO|STOP|BORDERLINE|INSUFFICIENT_DATA|NOT_RUN|UNTESTABLE (FINAL)
  <reason>` and writes `signals.json`.

`--provisional` shows the signals before the hand checks, labelled PROVISIONAL.

## Outputs (`outputs_pilot/run_<id>/`)

| File | Contents |
|---|---|
| `run_log.txt`, `run_stderr.txt` | console log (`POOL`, `EXTENSION`, `DATA_CHECKSUM`, `SAMPLER`, `BUDGET`, `TIME`, `SHARED`, `EXTENSION_DISAGREEMENT`, `SQ4_GATE`, `PHASE_FAILED`), warnings and tracebacks |
| `results.json` | every summary metric per model and sub-question, the SQ2 analysis, SQ4, the budget log, code hash and versions |
| `signals.json` | written by `pilot.analyze`: the signals, the grading and rewording checks |
| `grading_check_<model>.csv`, `paraphrase_check.csv` | the two hand-check sheets |
| `pool.jsonl`, `extension_pool.jsonl`, `pilot_qids.json` | the question pool with folds and SQ3 / SQ5 annotations; the SQ2 screening extension; the ids a full study must exclude, pool and extension (also written to the repository root, commit it) |
| `<model>/screen.pt`, `<model>/followup.pt` | the 10 screening tries on the extension; the full record of the extension's disagreement questions |
| `<model>/shared.pt` | main answers, grades, confidence, the 10 + 8 tries, residuals at 5 layers, ITI head activations |
| `<model>/probe.pt` | cross-fitted detector scores and the per-fold choice of layer, position and ridge strength |
| `<model>/sq1.pt`, `sq3.pt`, `sq5.pt`, `sq2.pt`, `sq4.pt` | per-phase summaries plus per-item arrays and texts |
| `budget.json` | throughput, projections, ladder steps |

Load a `.pt` file with `torch.load(path, weights_only=False)`.

## Code layout

| File | Role |
|---|---|
| `config.py` | every constant and threshold; layer plan; smoke-test overrides |
| `data.py` | pool construction (topic cap, exclusions, folds), SQ3 templates, SQ5 coarse answers, check sheets |
| `session.py` | model session: greedy answers, shared-prefill sampling, teacher-forced capture of states, hooks |
| `shared.py` | the shared setup per model and the grading-check sheet |
| `probes.py` | cross-fitted ridge detector (leave-one-out selection inside training folds) and logistic combiners |
| `sq1.py` … `sq5.py` | one module per sub-question |
| `signals.py` | go / stop rules and the grading gate (pure functions of `results.json`) |
| `budget.py` | throughput-based projection and the degradation ladder |
| `main.py`, `analyze.py` | the GPU run; the offline signals |
| `setup_pilot.py`, `smoke_test.py` | downloads; the CPU smoke test |

The pilot reuses the earlier study's `model_utils.py` (loading, chat template, batched generation),
`steering.py` (the additive hook) and `scoring.py` (alias matching, abstention detection). None of those
files was changed.

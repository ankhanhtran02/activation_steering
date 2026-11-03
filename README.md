# Activation steering for hallucination mitigation

Screening-tier experiment from `experiment_plan.yaml`: *does hallucination steering beat its own
detector?* Two frozen ~3B instruction models (Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct) are
steered at inference time with difference-of-means directions and compared against a confidence
threshold, a confidence prompt, a popularity gate and 20 norm-matched random directions on PopQA,
with EntityQuestions as a transfer set. No model is trained or fine-tuned.

## Quick start (one command)

```bash
cd activation_steering
bash run.sh
```

That is the whole procedure. `run.sh` will:

1. create a Python virtual env in `.venv/` and install the dependencies (a CUDA build of PyTorch
   is installed only if none is found);
2. download both models and all five datasets (`setup.py`, about 15-25 min, ~13 GB);
3. check that the GPU works in bf16 and that every file is in place;
4. run the experiment offline (`main.py`, about 3 hours, hard limit 4 hours) and save logs and
   results to `outputs/`.

The run takes hours, so start it so that it survives a closed terminal or SSH session:

```bash
nohup bash run.sh > run_all.txt 2>&1 &
tail -f run_all.txt                  # follow progress (Ctrl+C stops following, not the run)
```

(`tmux` or `screen` work just as well.)

## Requirements

| | |
|---|---|
| GPU | one NVIDIA GPU with bf16 support; planned for an NVIDIA GB10 (128 GB unified memory). Both models stay loaded (~13 GB) plus activation caches (< 4 GB) |
| Disk | ~15 GB (models ~13 GB in the Hugging Face cache, data ~30 MB, results ~100 MB) |
| Python | 3.10 or newer with the `venv` module (Ubuntu: `sudo apt install python3-venv`) |
| Network | only for the setup stage; the experiment itself runs offline |
| Accounts | none: the Llama weights come from the ungated `unsloth/Llama-3.2-3B-Instruct` re-upload, so no Hugging Face token is needed |

## Running setup and experiment separately

Use this when the GPU machine has no internet access, or to download ahead of time.

```bash
bash run.sh setup      # environment + downloads (needs network)
bash run.sh main       # experiment only (offline)
```

To move everything to an offline machine, keep the model cache inside the project folder and copy
the whole folder:

```bash
HF_HOME=$PWD/hf_cache bash run.sh setup    # on the machine with internet
# copy the activation_steering folder (including hf_cache/ and data_cache/) to the GPU machine
HF_HOME=$PWD/hf_cache bash run.sh main     # on the GPU machine
```

If the two machines have different CPU architectures, delete `.venv/` before running `main` on the
GPU machine; `run.sh` rebuilds it (this step needs pip access to install packages).

## What happens during the run

`main.py` prints one line per milestone. The order is fixed by the plan:

| Phase | What | Typical console lines |
|---|---|---|
| 1, per model | calibration, unsteered labelling, refusal direction, dev sweep, frozen config, integrity checks, the 9 registered conditions on PopQA test | `BUDGET:`, `GRAY_ZONE`, `REFUSAL_CONTROL`, `DEV_SELECTION`, `FROZEN_CONFIG`, `HOOK_RESET_CHECK`, `ABLATION_CHECK`, `ZERO_VECTOR_CHECK`, `PROGRESS:` |
| 2, per model | 13 ablations, EntityQuestions, side effects (perplexity, MMLU-500, refusal spillover) | `TIME: ablation[...]`, `TIME: entity_questions[...]` |
| 3 | statistics and report | `condition=... abstention_score: ...`, `SUMMARY`, `PAIRED`, `HYPOTHESIS` |

`TIME: <step> <seconds>` lines give the elapsed seconds since the start. Right after the first
calibration, a `BUDGET:` line projects the total runtime. If the projection exceeds 11,520 s, the
pre-registered degradation ladder is applied automatically (fewer EntityQuestions random-draw
questions, fewer ablation draws, ...); the planned conditions are never dropped.

Healthy integrity lines look like this:

```
HOOK_RESET_CHECK: max_abs_logit_diff=0
ABLATION_CHECK: unsteered vs knowledge_dir_steer outputs_differ=True
ZERO_VECTOR_CHECK model=llama32_3b_instruct identical_frac=1.0000 passed=True
```

## Outputs

All files go to `outputs/`.

| File | Contents |
|---|---|
| `run_log.txt` | full console output, including every harness line |
| `run_stderr.txt` | library warnings and tracebacks of any failed condition |
| `setup_log.txt` | download log |
| `frozen_config.json` | dev-selected layer, strength, thresholds and directions per model, with a SHA-256 hash; written before any test-set forward pass |
| `results.json` | per-question records (condition, model, question id, popularity, knowledge status, outcome, confidence, output text), metrics per condition and stratum, hypothesis analyses H1-H4, side effects, timings |
| `samples_<condition>.jsonl` | 50 random outputs per condition for a manual spot check |
| `audit.csv` | 300 outputs per model, stratified by condition and label, with an empty `human_label` column |

`run_manifest.json` (project root) records the exact model commits and library versions.

### Reading the main results

The primary metric is `abstention_score` (higher is better): the mean over test questions of +1
for a correct answer, 0 for "I don't know" and -1 for a wrong answer, reported x100. In the console:

- `condition=<name> abstention_score: <v>`: both models pooled;
  `condition=<name> regime=<model> abstention_score: <v>`: one model;
  other `regime=` values are strata (popularity terciles `pop_q1..3`, `known`, `outcompeted`, `missing`, ...).
- `condition=<name> lam=<l> abstention_score: <v>`: penalty sensitivity (lambda = 0.5, 1, 2, 3).
- `condition=random_dir_placebo seed=<k> ...` and `condition=<name> random_dir_percentile: <v>`:
  the placebo distribution and where each steering direction falls in it.
- `PAIRED: <method> vs <baseline> mean_diff=... ci_low=... ci_high=... mcnemar_p=... holm_p=...`:
  paired comparisons; the decisive ones for the main question are `knowledge_dir_steer vs
  confidence_threshold_margin` (H1a) and `margin_gated_steer_knowledge vs confidence_threshold_margin` (H1c).
- `HYPOTHESIS: model=<m> <id> ...`: a compact view of each hypothesis test; full detail is in
  `results.json` under `models.<model>.analysis.hypotheses`.

### After the run: manual audit (from the plan)

Open `outputs/audit.csv`, fill the `human_label` column (`correct`, `abstained` or `wrong`) for the
600 rows, and compare with `auto_label`. The plan's target is Cohen's kappa >= 0.8. If it is
missed, the classification rules in `scoring.py` are revised once and the saved generations
are rescored; no GPU time is needed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `PREFLIGHT FAILED: torch cannot see a CUDA GPU` | The installed torch is CPU-only. Use a wheel index that matches your CUDA version, e.g. `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash run.sh`, or run inside an NVIDIA PyTorch container (`run.sh` reuses its torch). Remove `.venv/` first so the env is rebuilt. |
| `a bf16 matmul on the GPU failed` | The torch build does not support the GPU architecture (common on new GPUs). Install a newer CUDA build as above. |
| `Could not create a virtual env` | `sudo apt install python3-venv`, or point `PYTHON=` to another interpreter. |
| `CONDITION_FAILED: <name> [<model>] ...` | That condition crashed; all others still ran. The traceback is in `outputs/run_stderr.txt`. |
| `MODEL_SETUP_FAILED` | A per-model step before the test conditions crashed (see `run_stderr.txt`); the other model still runs. |
| Download errors during setup | Rerun `bash run.sh setup`; finished datasets are skipped and model downloads resume. |
| The run is slower than planned | Nothing to do: the `BUDGET:` projection triggers the degradation ladder automatically. |

**Reruns.** The protocol is pre-registered: each test configuration is meant to run exactly once.
A rerun overwrites `outputs/`; keep the previous folder (rename it) if a run has to be repeated, and
report it.

**Using Meta's official Llama repository instead of the mirror.** Once Meta has approved your Hugging
Face account, change `hf_id` for Llama in `config.py` to `meta-llama/Llama-3.2-3B-Instruct`, log in
with `hf auth login`, and run `bash run.sh` again.

## Project layout

| File | Role |
|---|---|
| `run.sh` | one-command runner (the only script you need) |
| `setup.py` | downloads models and datasets, writes `run_manifest.json` |
| `main.py` | the experiment: condition registry and phases (docstring documents the full protocol) |
| `config.py` | all pre-registered constants (models, prompts, grids, seeds, budget) |
| `data.py` | splits, popularity bins, EntityQuestions filter |
| `model_utils.py` | model loading, chat template, batched greedy generation, activation caching |
| `steering.py` | forward hooks, direction fitting, random directions |
| `baselines.py`, `methods.py` | baselines and steering methods (class names follow the plan) |
| `pipeline.py` | per-model phases, ablations, side effects, time budget |
| `scoring.py`, `report.py` | answer matching, metrics, statistics, printed report and output files |
| `experiment_plan.yaml` | the pre-registered plan |

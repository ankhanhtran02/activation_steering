# Activation steering for hallucination mitigation

Screening-tier experiment from `experiment_plan.yaml`: *does hallucination steering beat its own
detector?* Two frozen ~3B instruction models (Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct) are
steered at inference time with difference-of-means directions and compared against a confidence
threshold, a confidence prompt, a popularity gate, norm-matched and dose-matched placebo directions
on PopQA, with EntityQuestions and TriviaQA as transfer sets. No model is trained or fine-tuned.

The protocol is `experiment_plan.yaml` as amended by `preregistration_amendment_2026-11-04.md`
(corrected statistics, freeze-before-test ordering, seeds, added controls, pinned data) and by
`preregistration_amendment_2026-11-07_final_run.md` (the final, decisive run). The changes are
summarised in the "Refinement log" at the top of `main.py`.

## The final run (2026-11-07) in one paragraph

It scores 1,715 PopQA test questions that no earlier run has seen (`previously_scored_qids.json`),
re-specifies H1c, gives H1b real dose-matched placebos (a uniform "say I don't know" push, sign-aligned
random directions, and D_ku split into within- and between-relation parts), tests H1a along the whole
dose ladder, fixes H1e, H2c and the "Don't know." scoring bug, runs D_wq at its own dose, pre-specifies
H3's signal, and labels every test output with a local judge (Qwen2.5-7B-Instruct). It ends with a
verdict per hypothesis (`VERDICT:` lines, `verdicts.json`) under the amended kill rule: both 3B families
decide, with no 7-9B confirmation gate. Expected wall time about 8 hours on the GB10 (limit 12 h).
`bash run.sh` re-runs setup first, which adds the judge model (~15 GB) to the cache.

Before the GPU run, `python smoke_test.py` runs every code path on CPU with two small instruct models
and tiny splits (about 30-60 minutes; its numbers mean nothing).

## Quick start (one command)

```bash
cd activation_steering
bash run.sh
```

That is the whole procedure. `run.sh` will:

1. create or reuse the Python virtual env in `.venv/` and install the dependencies only when needed
   (see "Python environment" below; a CUDA build of PyTorch is installed only if none is found);
2. download both models and all datasets at their pinned revisions (`setup.py`, about 15-25 min,
   ~13 GB; files already at their pinned source are skipped);
3. check that the GPU works in bf16 and that every file is in place;
4. run the experiment offline (`main.py`, final run about 8 hours, hard limit 12 hours) and save logs and
   results to a new folder `outputs/run_<UTC timestamp>/` (`outputs/latest` points to it).

The run takes hours, so start it so that it survives a closed terminal or SSH session:

```bash
nohup bash run.sh > run_all.txt 2>&1 &
tail -f outputs/latest/run_log.txt   # follow the experiment (Ctrl+C stops following, not the run)
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

If the two machines have different CPU architectures, the copied `.venv/` cannot run on the GPU
machine: `run.sh` detects that, moves it aside to `.venv.old-<timestamp>` and builds a new one (this
step needs pip access to install packages).

## Python environment

`run.sh` checks the virtual env (`VENV_DIR`, default `.venv/`) at every start:

| Situation | What `run.sh` does |
|---|---|
| No env yet | creates it with `PYTHON` (default `python3`, must be >= 3.10) and installs the packages |
| Env works and was installed from the current `requirements.lock` | reuses it, installs nothing (also works offline) |
| Env works but `requirements.lock` changed, or a package is missing | installs into the existing env |
| Env's python cannot run here (copied from another machine / CPU architecture, or broken) | moves it to `.venv.old-<timestamp>` and builds a new one |
| `REBUILD_ENV=1 bash run.sh ...` | moves the existing env aside and builds a new one |

The first setup writes `requirements.lock` (`pip freeze --local`: every package installed in the
env, not packages provided by the system or a container); commit it, and later setups install
exactly those versions. Old `.venv.old-*` folders are never deleted automatically.

## What happens during the run

`main.py` prints one line per milestone. The order is fixed by the plan:

| Phase | What | Typical console lines |
|---|---|---|
| A, every model and seed (no test split) | calibration, unsteered labelling of train/dev, refusal direction and positive controls, directions and diagnostics, dev sweeps with MMLU/perplexity caps, thresholds, dose-matched nulls, answer-position D_wq ladder | `BUDGET:`, `GRAY_ZONE`, `SEP_LABEL_CHECK`, `REFUSAL_CONTROL`, `DIRECTION_DIAGNOSTIC`, `HOOK_DIAGNOSTIC`, `DEV_SELECTION`, `DOSE_MATCHED_NULL` |
| freeze | one `frozen_config.json` for all models and seeds, then the test splits are unlocked | `FROZEN_CONFIG`, `TEST_GUARD` |
| B, per model | test labelling, integrity checks, the registered conditions on PopQA test, the seed-replicated arms | `HOOK_RESET_CHECK`, `ABLATION_CHECK`, `ZERO_VECTOR_CHECK`, `IDK_FORM_CHECK`, `PROGRESS:` |
| C, per model | ablations, EntityQuestions, TriviaQA, side effects (perplexity, MMLU-500, refusal spillover) | `TIME: ablation[...]`, `TQA_COVERAGE` |
| analysis | statistics and report | `condition=... abstention_score: ...`, `SEED_CHECK`, `SUMMARY`, `PAIRED`, `HYPOTHESIS` |

`TIME: <step> <seconds>` lines give the elapsed seconds since the start. Right after the first
calibration (after a warm-up batch), a `BUDGET:` line projects the total runtime; the projection is
re-estimated from finished work. Degradation-ladder steps are decided one at a time when their work
is reached and only if the projection exceeds the 8-hour cap (`BUDGET: at=ladder:<step>`); the
planned conditions, the seeds and the 20 random draws are never dropped.

Healthy integrity lines look like this:

```
HOOK_RESET_CHECK: max_abs_logit_diff=0
ABLATION_CHECK: unsteered vs knowledge_dir_steer outputs_differ=True
ZERO_VECTOR_CHECK model=llama32_3b_instruct identical_frac=1.0000 passed=True
```

## Outputs

Every run writes to its own folder `outputs/run_<UTC timestamp>/`; `outputs/latest` points to the
newest one, and `main.py` refuses to write into a folder that already holds results, so an earlier
run is never overwritten. Setup logs go to `outputs/setup_logs/`. The first run (2026-11-03) predates
per-run folders; its files stay directly in `outputs/`.

| File | Contents |
|---|---|
| `verdicts.json` | final run: verdict per hypothesis and model, combined verdicts, label robustness, decision rules |
| `judge_disagreements.csv` | final run: up to 300 rows per model where the judge and the automatic label disagree, with an empty `human_label` column (the tie-break for a label-sensitive verdict) |
| `run_log.txt` | full console output, including every harness line |
| `setup_log.txt`, `run_manifest.json` | copies of this run's setup log (if setup ran in the same call) and of the manifest |
| `run_stderr.txt` | library warnings and tracebacks of any failed condition |
| `setup_log.txt` | download log |
| `frozen_config.json` | dev-selected layer, strength, thresholds, nulls and directions per model and seed, with SHA-256 hashes and the code / amendment / lockfile hashes; written once, before any test-split forward pass |
| `results.json` | metrics per condition and stratum (incl. gains / losses, surplus per answered item), hypothesis analyses H1-H4, per-seed tables, paired rows, diagnostics, side effects, timings |
| `per_item/<model>/*.jsonl.gz` | every per-item output: every condition, seed and split (test, seeds, transfer, dev sweeps), unsteered labels and the 8 samples |
| `samples_<condition>.jsonl` | 50 random outputs per condition for a manual spot check |
| `audit.csv` | 300 outputs per model, stratified by condition and label, with an empty `human_label` column |
| `audit_h1a.csv` | up to 50 outputs per (H1a arm x transition vs unsteered), with `auto_label`, `strict_label`, empty `judge_label` and `human_label` |
| `requirements.lock` | the exact package versions of the run |

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

Open `outputs/latest/audit_h1a.csv` (and the registered `audit.csv` next to it), fill the
`human_label` column (`correct`, `abstained` or `wrong`), then (each command works on the newest run
folder by default; `--run-dir outputs/run_<...>` picks another, `--run-dir outputs` the first run):

```bash
python audit.py judge --backend anthropic        # optional LLM judge; sends questions and outputs to the API
python audit.py judge --backend hf --judge-model <local model, not one of the two under test>
python audit.py score                            # kappa tables + H1a rescored under auto / strict / judge labels
```

The plan's target is Cohen's kappa >= 0.8. If it is missed, the classification rules in
`scoring.py` are revised once and the saved generations are rescored; no GPU time is needed.

### Re-analysing a finished run (no GPU)

`python reanalyze.py [--run-dir DIR]` recomputes the corrected statistics (amendment A1/A3/A6/A7)
from a finished run's saved records (the newest run folder by default): the first run's
`results.json`, or the `per_item/` store of later runs. It writes `<run folder>/reanalysis_<date>/`.
The re-analysis of the first run is in `outputs/reanalysis_2026-11-04/`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `PREFLIGHT FAILED: torch cannot see a CUDA GPU` | The installed torch is CPU-only. Use a wheel index that matches your CUDA version, e.g. `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash run.sh`, or run inside an NVIDIA PyTorch container (`run.sh` reuses its torch). Add `REBUILD_ENV=1` so the env is rebuilt. |
| `a bf16 matmul on the GPU failed` | The torch build does not support the GPU architecture (common on new GPUs). Install a newer CUDA build as above. |
| `Could not create a virtual env` | `sudo apt install python3-venv`, or point `PYTHON=` to another interpreter. |
| `pip failed. Installing packages needs network access ...` | The env had to be (re)built or updated without network. Run `bash run.sh setup` where PyPI is reachable, or use the env from that machine if the CPU architecture matches. |
| `CONDITION_FAILED: <name> [<model>] ...` | That condition crashed; all others still ran. The traceback is in `outputs/latest/run_stderr.txt`. |
| `MODEL_SETUP_FAILED` | A per-model step before the test conditions crashed (see `run_stderr.txt`); the other model still runs. |
| Download errors during setup | Rerun `bash run.sh setup`; finished datasets are skipped and model downloads resume. |
| The run is slower than planned | Nothing to do: the `BUDGET:` projection triggers the degradation ladder automatically. |

**Reruns.** The protocol is pre-registered: each test configuration is meant to run exactly once.
A rerun gets its own `outputs/run_<UTC timestamp>/` folder and never overwrites an earlier one;
report every repeated run.

**Using Meta's official Llama repository instead of the mirror.** Once Meta has approved your Hugging
Face account, change `hf_id` for Llama in `config.py` to `meta-llama/Llama-3.2-3B-Instruct`, log in
with `hf auth login`, and run `bash run.sh` again.

## Project layout

| File | Role |
|---|---|
| `run.sh` | one-command runner (the only script you need); writes / installs `requirements.lock` |
| `preregistration_amendment_2026-11-04.md` | dated amendment to the plan (corrections, withdrawn results, A1-A14) |
| `setup.py` | downloads models and datasets, writes `run_manifest.json` |
| `main.py` | the experiment: condition registry and phases (docstring documents the full protocol) |
| `config.py` | all pre-registered constants (models, prompts, grids, seeds, budget) |
| `data.py` | splits, popularity bins, EntityQuestions filter |
| `model_utils.py` | model loading, chat template, batched greedy generation, activation caching |
| `steering.py` | forward hooks, direction fitting, random directions |
| `baselines.py`, `methods.py` | baselines and steering methods (class names follow the plan) |
| `pipeline.py` | per-model phases, ablations, side effects, time budget |
| `scoring.py` | answer matching, outcome classification, metric (unchanged) |
| `stats.py`, `analysis_core.py` | corrected inference (bootstrap with the valid-resample rule, sign-flip tests, family-wise Holm, common-grid AURC, net transitions) shared by the report and the re-analysis |
| `report.py` | hypotheses, printed report and output files |
| `reanalyze.py`, `audit.py` | offline re-analysis of a finished run; measurement audit and LLM judge |
| `experiment_plan.yaml` | the pre-registered plan |

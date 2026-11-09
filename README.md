# Activation steering for hallucination mitigation

> **New direction (2026-11-09, branch `pilot-confidence-rule`):** a pilot asking whether activation
> steering does anything a confidence rule can't, in five sub-questions with go / stop signals, on
> Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct within 12 GB10-hours. Start at
> [pilot/README.md](pilot/README.md) (`bash run_pilot.sh`); the protocol is
> [pilot/PREREGISTRATION.md](pilot/PREREGISTRATION.md). The rest of this README describes the earlier
> study, whose code the pilot reuses (`model_utils.py`, `steering.py`, `scoring.py`, unchanged).

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
decide, with no 7-9B confirmation gate. Expected wall time about 8-9 hours on the GB10 (cap 11 h,
limit 12 h).

The final-run code lives on the git branch `final-run`. On the GPU machine:

```bash
git fetch origin && git checkout final-run
nohup bash run.sh > run_all.txt 2>&1 &     # setup (adds the ~15 GB judge model) + preflight + experiment
```

Use plain `bash run.sh`, not `bash run.sh main`: a `run_manifest.json` from run 2 does not list the
judge, and the preflight stops until setup has downloaded it.

Before the GPU run, `python smoke_test.py` can run every code path on CPU with two small instruct models
and tiny splits (about 70 minutes on an 8-core laptop, ~1.5 GB of small models downloaded; its numbers
mean nothing). It passed on the final code on 2026-11-07.

## Quick start (one command)

```bash
cd activation_steering
bash run.sh
```

That is the whole procedure. `run.sh` will:

1. create or reuse the Python virtual env in `.venv/` and install the dependencies only when needed
   (see "Python environment" below; a CUDA build of PyTorch is installed only if none is found);
2. download both models, the judge model and all datasets at their pinned revisions (`setup.py`,
   ~28 GB the first time, of which the judge is ~15 GB; files already at their pinned source are
   skipped);
3. check that the GPU works in bf16 and that every file is in place, including the cached judge;
4. run the experiment offline (`main.py`, final run about 8-9 hours, hard limit 12 hours) and save logs and
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
| GPU | one NVIDIA GPU with bf16 support; planned for an NVIDIA GB10 (128 GB unified memory). Both models stay loaded (~13 GB) plus activation caches (< 4 GB); the judge (~15 GB) is added for the last phase |
| Disk | ~30 GB (models ~13 GB and judge ~15 GB in the Hugging Face cache, data ~30 MB, results ~200 MB) |
| Python | 3.10 or newer with the `venv` module (Ubuntu: `sudo apt install python3-venv`) |
| Network | only for the setup stage; the experiment itself runs offline |
| Accounts | none: the Llama weights come from the ungated `unsloth/Llama-3.2-3B-Instruct` re-upload and the judge `Qwen/Qwen2.5-7B-Instruct` is ungated, so no Hugging Face token is needed |

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
| start | data checks; the final run also prints the fresh-test check | `DATA_CHECKSUM`, `TEST_FRESHNESS` (should read `popqa_test=1715 ... overlap_with_earlier_runs=0`) |
| A, every model and seed (no test split) | calibration, unsteered labelling of train/dev, refusal direction and positive controls, directions (incl. D_force and the D_ku decomposition) and diagnostics, dev sweeps with MMLU/perplexity caps, thresholds, the dev-tuned gated strength, dose matching of the placebos, the own-dose D_wq ladder | `BUDGET:`, `GRAY_ZONE`, `SEP_LABEL_CHECK`, `REFUSAL_CONTROL`, `KU_DECOMPOSITION`, `DIRECTION_DIAGNOSTIC`, `HOOK_DIAGNOSTIC`, `DEV_SELECTION`, `DEV_SELECTION_GATED`, `DOSE_MATCHED_PLACEBO`, `DEV_SELECTION_WQ_OWN` |
| freeze | one `frozen_config.json` for all models and seeds, then the test splits are unlocked | `FROZEN_CONFIG`, `TEST_GUARD` |
| B, per model | test labelling, integrity checks, the registered conditions on PopQA test (incl. the knowledge dose ladder and the dose-matched placebos), the seed-replicated arms | `HOOK_RESET_CHECK`, `ABLATION_CHECK`, `ZERO_VECTOR_CHECK`, `IDK_FORM_CHECK`, `PROGRESS:` |
| C, per model | ablations, EntityQuestions, TriviaQA, side effects (perplexity, MMLU-500, refusal spillover) | `TIME: ablation[...]`, `TQA_COVERAGE` |
| D, once | the local judge labels every distinct test output (~30 min) | `JUDGE:`, `JUDGE_AGREEMENT` (or `JUDGE_FAILED`) |
| analysis | statistics, report and the verdict table | `condition=... abstention_score: ...`, `SEED_CHECK`, `SUMMARY`, `PAIRED`, `HYPOTHESIS`, `VERDICT`, `DECISION_RULE`, `VERDICT_SUMMARY` |

`TIME: <step> <seconds>` lines give the elapsed seconds since the start. Right after the first
calibration (after a warm-up batch), a `BUDGET:` line projects the total runtime; the projection is
re-estimated from finished work. Degradation-ladder steps are decided one at a time when their work
is reached and only if the projection exceeds the 11-hour cap (`BUDGET: at=ladder:<step>`). The ladder
only trims null draws and side checks (see section 6 of the final amendment); the fresh test set, the
seeds, the dose ladder, the placebos, the H2 arms and the judge are never cut.

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
| `frozen_config.json` | dev-selected layer, strength, gated strength, thresholds, dose-matched placebo multiples and directions per model and seed, with SHA-256 hashes and the code / amendment / lockfile hashes; written once, before any test-split forward pass |
| `results.json` | metrics per condition and stratum (incl. gains / losses, surplus per answered item), hypothesis analyses H1-H4, label-sensitivity analyses (registered rule, strict, judge, unambiguous relations), per-seed tables and replication, paired rows, diagnostics, side effects, timings, the verdict table |
| `per_item/<model>/*.jsonl.gz` | every per-item output: every condition, seed and split (test, seeds, transfer, dev sweeps), unsteered labels and the 8 samples; final run: a `judge_state` field per record |
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
  confidence_threshold_margin` (H1a) and `margin_gated_steer_knowledge_tuned vs confidence_threshold_margin`
  (H1c at its dev-tuned strength, final F2; the c* version is reported as exploratory).
- `HYPOTHESIS: model=<m> <id> ...`: a compact view of each hypothesis test; full detail is in
  `results.json` under `models.<model>.analysis.hypotheses`.
- `VERDICT: <hypothesis> <SUPPORTED|REJECTED|MIXED|UNTESTABLE> per_model=... judge_labels=... label_robust=...`:
  the final run's answer per hypothesis under the amended kill rule (criteria: section 5 of the final
  amendment; detail in `verdicts.json`). `label_robust=False` means the verdict changes under the judge
  labels: then the human labels in `judge_disagreements.csv` decide it. `DECISION_RULE:` lines give the
  pre-registered consequence; `VERDICT_SUMMARY:` lists which hypotheses are supported and which rejected.

### After the run: manual audit (from the plan)

The final run already labels every test output with the local judge (phase D) and recomputes every
verdict under those labels. The human step that remains:

1. If any verdict prints `label_robust=False`, fill the `human_label` column of
   `outputs/latest/judge_disagreements.csv` (`correct`, `abstained` or `wrong`): those rows decide
   whether the automatic or the judge labelling is right for that verdict.
2. For the plan's agreement target (Cohen's kappa >= 0.8 against people), also label
   `audit_h1a.csv` and score it as below.

Fill the `human_label` column of `outputs/latest/audit_h1a.csv` (and of the registered `audit.csv` next
to it), then (each command works on the newest run folder by default; `--run-dir outputs/run_<...>`
picks another, `--run-dir outputs` the first run):

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
The re-analysis of the first run is in `outputs/reanalysis_2026-11-04/`. It was written for runs 1 and
2 and has not been checked against the final run's outputs; the final run's own `results.json` and
`verdicts.json` already contain the corrected statistics.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `PREFLIGHT FAILED: torch cannot see a CUDA GPU` | The installed torch is CPU-only. Use a wheel index that matches your CUDA version, e.g. `TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash run.sh`, or run inside an NVIDIA PyTorch container (`run.sh` reuses its torch). Add `REBUILD_ENV=1` so the env is rebuilt. |
| `a bf16 matmul on the GPU failed` | The torch build does not support the GPU architecture (common on new GPUs). Install a newer CUDA build as above. |
| `Could not create a virtual env` | `sudo apt install python3-venv`, or point `PYTHON=` to another interpreter. |
| `pip failed. Installing packages needs network access ...` | The env had to be (re)built or updated without network. Run `bash run.sh setup` where PyPI is reachable, or use the env from that machine if the CPU architecture matches. |
| `CONDITION_FAILED: <name> [<model>] ...` | That condition crashed; all others still ran. The traceback is in `outputs/latest/run_stderr.txt`. |
| `MODEL_SETUP_FAILED` | A per-model step before the test conditions crashed (see `run_stderr.txt`); the other model still runs. |
| `PREFLIGHT FAILED: run_manifest.json has no judge ...` or `judge ... not in the local cache` | The manifest or cache predates the final run (or the judge download was interrupted). Run `bash run.sh setup` (or plain `bash run.sh`). |
| `PREFLIGHT FAILED: missing preregistration_amendment_2026-11-07_final_run.md` / `previously_scored_qids.json` | The final-run code is not checked out: `git fetch origin && git checkout final-run`. |
| `JUDGE_FAILED` near the end | The experiment results are complete; only the judge labels are missing, and every verdict's `label_robust` reads unavailable. The traceback is in `run_stderr.txt`. |
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
| `preregistration_amendment_2026-11-07_final_run.md` | the final run's amendment (F1-F12, amended kill rule, verdict criteria) |
| `previously_scored_qids.json` | every question id runs 1 and 2 scored on a test split; the final run excludes them |
| `judge.py`, `verdicts.py` | final run: phase D LLM judge; the verdict table |
| `smoke_test.py` | CPU smoke test of every code path with small models (no results) |
| `setup.py` | downloads models and datasets, writes `run_manifest.json` |
| `main.py` | the experiment: condition registry and phases (docstring documents the full protocol) |
| `config.py` | all pre-registered constants (models, prompts, grids, seeds, budget) |
| `data.py` | splits, popularity bins, EntityQuestions filter |
| `model_utils.py` | model loading, chat template, batched greedy generation, activation caching |
| `steering.py` | forward hooks, direction fitting, random directions |
| `baselines.py`, `methods.py` | baselines and steering methods (class names follow the plan) |
| `pipeline.py` | per-model phases, ablations, side effects, time budget |
| `scoring.py` | answer matching, outcome classification, metric (final F8: subject-less "Don't know." counts as an abstention; the registered phrase list is kept for the sensitivity analysis) |
| `stats.py`, `analysis_core.py` | corrected inference (bootstrap with the valid-resample rule, sign-flip tests, family-wise Holm, common-grid AURC, net transitions) shared by the report and the re-analysis |
| `report.py` | hypotheses, printed report and output files |
| `reanalyze.py`, `audit.py` | offline re-analysis of a finished run; measurement audit and LLM judge |
| `experiment_plan.yaml` | the pre-registered plan |

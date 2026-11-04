"""Does hallucination steering beat its own detector? Activation steering for hallucination
mitigation on frozen instruction-tuned LLMs (screening tier, experiment_plan.yaml, as amended by
preregistration_amendment_2026-11-04.md).

No language-model weight is trained or fine-tuned. The only fitted components are steering
directions (differences of class means over cached activations), small L2 logistic-regression
probes and scalar thresholds. main.py runs offline with no arguments; setup.py downloads first.

MODELS (bf16, one GPU, both stay resident once loaded)
  llama32_3b_instruct = unsloth/Llama-3.2-3B-Instruct @ 006f5dcd... (28 blocks, d = 3072; an ungated
                        third-party re-upload of meta-llama/Llama-3.2-3B-Instruct)
  qwen25_3b_instruct  = Qwen/Qwen2.5-3B-Instruct @ aa8e7253...   (36 blocks, d = 2048)
  Revisions are pinned in config.py (the commits the first run resolved) and loaded with
  local_files_only=True; printed as MODEL_REVISION lines.

DATASETS AND SPLITS (every source pinned to a commit or a byte checksum; data_cache/ files are
  verified against the sha256 that setup.py recorded in run_manifest.json before anything runs)
  PopQA (akariasai/PopQA, 14,267 questions; the 3 with an empty subject are dropped). Subjects are assigned to splits by
  sha1("S9-split" + subj_id) into 60/20/20 % train/dev/test pools, so no subject appears in two
  splits (asserted; answer entities can recur). Popularity deciles and terciles are computed once
  on log10(s_pop + 1) over all of PopQA. Within each pool, questions are drawn equally per decile
  (seed 0): train 1,000 (100/decile), train_reserve 1,000 (top-up for D_wq only), dev 400
  (40/decile), test 1,500 (150/decile; 1,200 only if the last rung of the degradation ladder is
  reached). A fixed 300-item dev subset (30/decile) is used for coverage / dose matching. Seeds
  s = 1, 2 redraw train, train_reserve and dev from the same pools; the test sets never change.
  EntityQuestions test: 25 questions x 24 relations = 600 (seed 0) after removing questions that
  contain, on word boundaries, any PopQA train/dev-pool subject string of >= 4 characters inside the
  question's subject slot; a fixed 300-question subset carries the random draws. TriviaQA
  (rc.nocontext validation): 300 dev (threshold only) + 1,000 test, the high-coverage transfer set.
  Directions are transferred from PopQA, never refit. AdvBench: 128 train / 100 held-out. Alpaca
  (empty input only): 128 train harmless / 100 benign held-out. Perplexity text: 200 dev + 200 test
  BBC News articles from May 2025 (non-memorised). MMLU: 500 test questions (side effect) and 300
  validation questions (dev cap), stratified over 57 subjects (seed 0).

PROMPTING AND DECODING
  Each model's chat template (apply_chat_template, add_generation_prompt=True; Llama gets
  date_string="26 Jul 2024"), system prompt "You are a concise assistant that answers factual
  questions.", user template asking for only the answer or exactly "I don't know.". Greedy decoding
  (do_sample=False, repetition_penalty=1, hub generation defaults replaced), max_new_tokens=16 for QA
  and 32 for AdvBench/Alpaca, stop at end-of-turn or eos. Batch size 64 in a fixed question order,
  left padding (Llama pads with eos), attention_mask passed, prompt tokens stripped from outputs.
  Every condition uses the identical batches, so outputs differ only through the intervention.
  Margin m = log P(greedy answer + eot) - log[P("I don't know." + eot) + P("I don't know" + eot)].

LAYER CONVENTION AND HOOK POINT
  Layer L is the 0-indexed decoder block model.model.layers[L]. Directions are fitted on, and
  hooks act on, the residual stream after block L = the block's output = hidden_states[L + 1]
  (hidden_states[0] is the embedding output). A forward hook modifies output[0] (or the tensor
  output) and returns the full output; it counts its calls. Candidate layers: Llama {11, 14, 17},
  Qwen {14, 18, 22}; refusal candidates add Llama 8 and Qwen 11.

INTERVENTIONS (all add alpha * v_hat; v computed in float32 from cached activations, unit L2 norm,
alpha * v_hat cast to bf16 and added. Strength unit: sigma_L(v) = SD of the unsteered last-prompt
residual's projection on v_hat over PopQA train (answer tokens for answer-position D_wq).)
  Shared site: layer L*, positions = last prompt token (prefill column -1) + every generated token,
  added norm alpha* = c* sigma_{L*}(D_ku). Every primary direction and every norm-matched random
  draw adds exactly alpha* there (registered norm matching). Teacher-forced passes use an explicit
  position mask over the last prompt token and all continuation tokens.
  knowledge_dir_steer: D_ku = mean(last-prompt residual | unknown) - mean(... | known) on PopQA
    train; known = greedy correct and >= 6/8 samples correct, unknown = greedy wrong/abstained and
    <= 1/8 correct (greedy-only labels if a class has < 100). Sign + (toward unknown). (L*, c*)
    chosen on dev from candidates x c in {2, 4, 8, 16, 32, 64} sigma by the registered rule (degenerate
    <= 5 %, abstention increase 5-30 points, highest dev S_1, ties within 0.5 -> smaller c, then
    middle layer; fallbacks as in the plan) with two added eligibility caps: paired MMLU-dev delta
    >= -1.5 points and reference-text perplexity ratio <= 1.05.
  read_dir_steer: D_read = mean(last-prompt residual | unsteered wrong) - mean(... | correct) over
    answered train items. Shared site.
  within_question_dir_steer: D_wq = mean over gray-zone train questions (>= 1 correct and >= 1
    wrong answered non-degenerate output among greedy + 8 samples; train_reserve top-up if < 80) of
    [mean answer-token residual of correct outputs - of wrong outputs]; residuals from
    teacher-forcing each output (end-of-turn excluded). Sign + (toward correct). Shared site; refit
    r = 0 (registered) plus r = 1-4 (question-level bootstrap), each run on test.
  within_question_dir_steer_answer_positions: D_wq added at generated (answer) positions only, with
    its own dev layer x strength ladder (highest dev S1, degenerate <= 5 %, caps) and a norm-matched
    random ladder; refits 0-4 on test plus random draws at the selected rung.
  refusal_dir_steer: mean(harmful) - mean(harmless) last-prompt residual over 128 AdvBench vs 128
    Alpaca train instructions (same system prompt), extracted at L*. Shared site.
  dose_matched_null_{shuffled_ku, cov_random, negated_ku}: shuffled-label D_ku, a covariance-shaped
    random direction (z ~ N(0, Sigma_{L*})) and -D_ku, each at the multiple of alpha* whose dev
    abstention increase matches knowledge_dir_steer's (+/- 3 points; flagged otherwise).
  margin_gated_steer_{knowledge, wq}: the cached steered test output where the gate fires
    (unsteered abstained or margin < tau_dev(1)), the unsteered output elsewhere.
  random_dir_placebo: 20 draws z ~ N(0, I_d), torch.Generator(seed 1000 + k), unit-normalised, at
    (L*, alpha*) with the same positions.
  Seeds: knowledge, read, refusal, the three nulls, the threshold and margin_gated_steer_knowledge
    are re-selected and re-run for seeds 1 and 2 (own train/dev draw, sampling seed, refusal
    resample, dev selection); everything else runs at seed 0.
  Ablations (one factor at a time; exploratory): knowledge/wq dose ladders (other c), knowledge
    last-prompt-only and generated-only positions, knowledge at L_alt, between-question answer
    direction, 5 label-permuted D_wq (seeds 2000-2004), 5 coverage-matched isotropic draws
    (multiplier {1, 2, 4} x alpha*), refusal ablation in three variants (directional on embed + every
    block = registered; mean-ablation on embed + every block; mean-ablation at L_ref only), each
    alone and with the knowledge steer, with 5 / 3 / 3 random-direction ablations, first-token-ban
    regeneration, and the teacher-forced dose response (c in {0, 2, 4, 8, 16, 32, 48, 64} sigma).

BASELINES
  unsteered; always_abstain (score 0 by construction, the floor every table now shows);
  confidence_prompt_t050 / _t075; confidence_threshold_margin (abstain if unsteered abstained or
  margin < tau_dev(lam); operating point tau_dev(1); coverage-matched to every steering condition by
  answering the k highest-margin unsteered-answered items, ties by question-id hash);
  popularity_gate (same policy on log10(s_pop + 1)); majority_vote_k8 and best_of_k8 over the 8
  saved temperature-1 samples, each gated at its own tau_dev(1).

OUTPUT CLASSIFICATION (order: degenerate -> abstained -> answer; unchanged)
  Degenerate: empty, < 30 % alphanumeric non-space characters, one word >= 50 % of words when >= 6
  words, or a repeated word trigram; scored as wrong. Abstained (fixed phrase list, checked before
  correctness): the lightly normalised output begins with one of: i don't know, i do not know,
  i'm not sure, i am not sure, not sure, i cannot, i can't, i'm unable, i am unable, unable to,
  i don't have, no information, cannot determine, i'm sorry, sorry, unknown; or contains
  "i don't know" / "i do not know" / "unknown" and has <= 6 words. Hedged answers are answers.
  Correct: the normalised answer contains any normalised gold alias on word boundaries. Strict
  exact match and an LLM judge (audit.py) are sensitivity analyses.

PRIMARY METRIC
  abstention_score (higher is better) = mean over test questions of +1 (correct), 0 (abstained),
  -lambda (wrong, including degenerate), reported in points (x100); lambda = 1 primary, lambda in
  {0.5, 1, 2, 3} sensitivity. Bare condition lines pool both models over the same PopQA test
  questions; regime=<model> lines give each model. Random-draw lines use seed=<k> for draw k;
  seeded arms print condition=<name> seed=<s> lines. TIME lines give cumulative elapsed seconds.

STATISTICS
  Paired bootstrap over questions (2,000 resamples, seed 7, matching redone per resample); a CI is
  "undefined" when < 95 % of resamples are valid. p-values: paired sign-flip permutation test on
  per-question score differences, Holm-corrected within the registered families. PAIRED lines keep
  mean_diff / ci_low / ci_high / mcnemar_p / holm_p (holm_p now on the permutation p within the
  family; McNemar on correctness is secondary) and add perm_p, family, valid_frac, status.

Refinement log
  2026-11-04 (review of the first run; amendment file preregistration_amendment_2026-11-04.md):
  - A1 statistics: PAIRED McNemar/Holm-over-everything replaced by sign-flip tests with Holm within
    the registered families; pooled rows no longer silently drop an unmatched model; the valid-
    resample rule (>= 95 %) replaces CIs computed from surviving resamples (first run: H1b cm CI from
    78 of 2,000 resamples); H1b best-draw selection no longer passes NaN to max()/percentile; AURC
    compared on a common coverage grid (first run compared 0-29 % with 0-51 % coverage); H2c is
    undefined when the ablation alone raises abstention (Llama: +40 points). analysis_core.py,
    stats.py, report.py; reanalyze.py applies the same code to a finished run.
  - A2 freezing: Phase A (all pre-test work for both models and all seeds) -> one frozen_config.json
    -> test splits unlocked (ModelRunner.guard). The first run wrote model 2's config after model
    1's test passes and ran each model's unsteered test passes before its own freeze, so its
    "written before any test forward pass" line was false. main.py, pipeline.py.
  - A3 tables: always_abstain row, gains / losses / net points, surplus per answered item and a
    common-grid AURC difference in every condition table and stratum table.
  - A4 audit: audit_h1a.csv (<= 50 rows per arm x transition for both H1a arms); audit.py samples,
    runs an LLM judge (anthropic or local HF backend), computes kappa and rescores H1a under auto /
    strict / judge labels. Scoring code and phrase list unchanged.
  - A5 implementation suspects: IDK reference "I don't know" -> "I don't know." (both models abstain
    with the period; the margin's abstain term now sums both written forms); semantic-entropy label
    put every abstention into one cluster, so consistent abstainers looked certain (Qwen sep_probe
    AUROC 0.41) -> abstained/degenerate samples are singleton clusters, with SEP_LABEL_CHECK; Qwen
    layer-14 knowledge no-op -> not a code fault (CPU reproduction, amendment section 3: the hook
    fires, the direction is not an outlier-channel direction, and at the first run's dose it
    perturbs like a norm-matched random direction without pushing toward abstention); hook call
    counters, first-token KL, outputs-changed fraction and direction-concentration diagnostics
    (HOOK_DIAGNOSTIC / DIRECTION_DIAGNOSTIC lines) now measure this for every layer.
  - A6 H3 on answered items with relation-only and popularity + relation baselines (fit on train);
    one canonical within-decile AUROC (same-decile pair AUROC) with a bootstrap CI.
  - A7 exploratory gated knockout (forced second guess on gated items) with a per-relation chance
    baseline.
  - A8 seeds: 3 seeds (train/dev redraw, sampling seed, refusal resample, dev selection) for the
    H1a-critical arms; 5 D_wq refits; identical-per-seed check (SEED_CHECK lines).
  - A9 controls: D_read, dose-matched nulls (shuffled-label D_ku, covariance-shaped random, -D_ku),
    answer-position D_wq with its own ladder and a matched random ladder, majority-vote and best-of-k.
  - A10 TriviaQA rc.nocontext as the >= 70 % coverage transfer set.
  - A11 strength in units of sigma_L(v), grid {2, 4, 8, 16, 32, 64} (past the old maximum; set from
    the CPU measurement sigma ~ 0.015 rbar, see the amendment section 3); MMLU-dev and
    BBC-2025 perplexity caps in dev selection; BBC-2025 replaces Alpaca outputs as perplexity text.
  - A12 refusal ablation: mean-ablation (all layers / L_ref only) variants with their own positive
    control and more random-ablation draws (5 / 3 / 3).
  - A13 budget: warm-up before calibration, scale re-estimated from completed work, ladder steps
    decided one at a time in ladder order; test restored to 1,500 (test_1200 only as the last rung);
    run limit 10 h (cap 8 h).
  - A14 reproducibility: pinned model and dataset revisions, data checksums verified at start,
    CODE_HASH / AMENDMENT / LOCKFILE lines, requirements.lock written by run.sh, every per-item
    output saved under <run folder>/per_item/.
  - Run infrastructure (no effect on the protocol): every run writes to its own folder
    outputs/run_<UTC timestamp>/ (run.sh sets STEER_OUT_DIR / STEER_RUN_ID, outputs/latest points
    to the newest, main.py refuses a folder that already holds results; first-run files in outputs/
    are never touched). run.sh reuses a working virtual env, installs only when it was not built
    from the current requirements.lock, and moves an env whose python cannot run here (other
    machine / architecture, broken) or REBUILD_ENV=1 aside to <VENV_DIR>.old-<timestamp> and
    rebuilds; the lockfile is `pip freeze --local`. reanalyze.py / audit.py default to the newest
    finished run folder (--run-dir to choose).
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import hashlib
import json
import random
import sys
import traceback

import numpy as np
import torch

import data as D
import report
from config import (ADDED_CONDITIONS, AMENDMENT_PATH, FROZEN_CONFIG_PATH, GLOBAL_SEED, LOCKFILE_PATH, MANIFEST_PATH,
                    MODELS, OUT_DIR, PLANNED_CONDITIONS, RESULTS_PATH, ROOT, RUN_ID)
from pipeline import Budget, ModelRunner, Timer
from scoring import abstention_score
from steering import assert_no_hooks

METRIC_DESC = ("mean over PopQA test questions of +1 correct, 0 abstained, -lambda wrong (lambda=1; "
               "degenerate counts as wrong), in points x100, pooled over both models")

REGISTRY = {
    "unsteered": lambda r: r.cond_unsteered(),
    "confidence_prompt": lambda r: r.cond_confidence_prompt(),
    "confidence_threshold_margin": lambda r: r.cond_confidence_threshold_margin(),
    "popularity_gate": lambda r: r.cond_popularity_gate(),
    "knowledge_dir_steer": lambda r: r.cond_knowledge_dir_steer(),
    "within_question_dir_steer": lambda r: r.cond_within_question_dir_steer(),
    "refusal_dir_steer": lambda r: r.cond_refusal_dir_steer(),
    "margin_gated_steer": lambda r: r.cond_margin_gated_steer(),
    "random_dir_placebo": lambda r: r.cond_random_dir_placebo(),
    # amendment A3 / A9
    "always_abstain": lambda r: r.cond_always_abstain(),
    "read_dir_steer": lambda r: r.cond_read_dir_steer(),
    "within_question_dir_steer_answer_positions": lambda r: r.cond_wq_answer_positions(),
    "dose_matched_null": lambda r: r.cond_dose_matched_null(),
    "sample_vote": lambda r: r.cond_sample_vote(),
}

ABLATIONS = {
    "knowledge_dir_steer_dose_ladder": lambda r: r.abl_knowledge_dose_ladder(),
    "within_question_dir_steer_dose_ladder": lambda r: r.abl_within_question_dose_ladder(),
    "knowledge_dir_steer_last_prompt_only": lambda r: r.abl_knowledge_last_prompt_only(),
    "knowledge_dir_steer_generated_only": lambda r: r.abl_knowledge_generated_only(),
    "knowledge_dir_steer_alt_layer": lambda r: r.abl_knowledge_alt_layer(),
    "between_question_answer_dir_steer": lambda r: r.abl_between_question_answer_dir(),
    "label_permuted_wq_dir_steer": lambda r: r.abl_label_permuted_wq(),
    "random_dir_placebo_coverage_matched": lambda r: r.abl_random_coverage_matched(),
    "refusal_dir_ablation": lambda r: r.abl_refusal_ablation(),
    "refusal_dir_ablation_plus_knowledge_steer": lambda r: r.abl_refusal_ablation_plus_knowledge(),
    "random_dir_ablation": lambda r: r.abl_random_ablation(),
    "first_token_ban_regenerate": lambda r: r.abl_first_token_ban(),
    "teacher_forced_dose_response": lambda r: r.abl_teacher_forced_dose_response(),
}


def seed_everything():
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)


def fail(name, model, err):
    print(f"CONDITION_FAILED: {name} [{model}] {type(err).__name__}: {err}", flush=True)
    traceback.print_exc(file=sys.stderr)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def code_hash():
    """Amendment A14: hash of every file that defines the run (sources, plan, amendment, lockfile)."""
    files = sorted(ROOT.glob("*.py")) + [ROOT / "experiment_plan.yaml", AMENDMENT_PATH, ROOT / "requirements.txt",
                                         LOCKFILE_PATH, ROOT / "run.sh"]
    h = hashlib.sha256()
    used = []
    for p in files:
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes())
            used.append(p.name)
    return h.hexdigest(), used


def check_data(manifest):
    status = D.verify_data_checksums(manifest)
    bad = []
    for key, (st, rec, act) in status.items():
        print(f"DATA_CHECKSUM: {key} status={st} sha256={act}", flush=True)
        if st != "ok":
            bad.append(f"{key}: {st}")
    if bad:
        raise RuntimeError("data_cache does not match run_manifest.json (" + "; ".join(bad) +
                           "): rerun `bash run.sh setup` so setup.py downloads the pinned revisions and records checksums")
    return {k: v[2] for k, v in status.items()}


def run_registry(r, timer):
    for name, fn in REGISTRY.items():
        assert_no_hooks(r.model)
        try:
            outs = fn(r)
            r.test_out.update(outs)
            for inst, o in outs.items():
                if not inst.startswith("random_dir_placebo_s"):
                    print(f"PROGRESS: model={r.name} condition={inst} S1={abstention_score(o.states):.2f} "
                          f"abstention_rate={float((o.states == 0).mean()):.3f}", flush=True)
            if name == "random_dir_placebo":
                vals = [abstention_score(o.states) for o in outs.values()]
                print(f"PROGRESS: model={r.name} condition=random_dir_placebo draws={len(vals)} "
                      f"S1_mean={np.mean(vals):.2f} S1_std={np.std(vals, ddof=1):.2f}", flush=True)
        except Exception as e:
            fail(name, r.name, e)
        timer.mark(f"condition[{name}][{r.name}]")


def phase_a(spec, data, budget, timer, first):
    """Everything before the first test-split forward pass (calibration, labelling of train/dev,
    directions, dev selection for every seed, thresholds, nulls, answer-position ladder)."""
    r = ModelRunner(spec, data, budget, timer)
    r.calibrate(update_budget=first)
    r.pretest()
    return r


def freeze_all(runners, budget, hashes):
    """Amendment A2: one write of frozen_config.json for every model and seed, then unlock."""
    full = budget.allow("test_1200") if runners else True
    if not full:
        trimmed = D.trim_test(runners[0].data["popqa"]["test"], budget.test_per_decile)
        runners[0].data["popqa"]["test"] = trimmed
        for r in runners:
            r.restrict_test(trimmed)
        print(f"BUDGET: PopQA test reduced to {len(trimmed)} questions (last ladder rung)", flush=True)
    all_frozen = {"_run": {**hashes, "n_test": len(runners[0].items["test"]) if runners else 0}}
    for r in runners:
        all_frozen[r.name] = r.frozen_entry()
    FROZEN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FROZEN_CONFIG_PATH, "w") as f:
        json.dump(all_frozen, f, indent=1, default=str)
    file_hash = sha256_file(FROZEN_CONFIG_PATH)
    for r in runners:
        print(f"FROZEN_CONFIG model={r.name} sha256={all_frozen[r.name]['sha256']} seeds={1 + len(r.seed_runs)} "
              f"file_sha256={file_hash} (written before any test-split forward pass of any model)", flush=True)
    for r in runners:
        r.lock["locked"] = False
    return all_frozen


def phase_b(r, timer):
    r.budget.start(r.name, "test")
    r.label_test_side()
    r.integrity_checks()
    run_registry(r, timer)
    for s, f in r.seed_runs.items():
        try:
            r.seed_out[s] = f.test_seed_arms(r)
        except Exception as e:
            fail(f"seed{s}_h1a_arms", r.name, e)
        timer.mark(f"seed{s}_arms[{r.name}]")
    r.budget.finish(r.name, "test")


def phase_c(r, timer):
    r.budget.start(r.name, "ablations")
    for name, fn in ABLATIONS.items():
        assert_no_hooks(r.model)
        try:
            r.ablation_out.update(fn(r))
        except Exception as e:
            fail(name, r.name, e)
        timer.mark(f"ablation[{name}][{r.name}]")
    r.budget.finish(r.name, "ablations")
    for step, key in (("entity_questions", "eq"), ("triviaqa", "tqa"), ("side_effects", "side_effects")):
        r.budget.start(r.name, key)
        try:
            getattr(r, step)()
        except Exception as e:
            fail(step, r.name, e)
        r.budget.finish(r.name, key)


def main():
    timer = Timer()
    seed_everything()
    if RESULTS_PATH.exists() or FROZEN_CONFIG_PATH.exists():
        raise FileExistsError(f"{OUT_DIR} already holds a run; every run writes to its own folder "
                              "(run.sh creates outputs/run_<UTC timestamp>/)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"OUTPUT_DIR: {OUT_DIR} run_id={RUN_ID}", flush=True)
    print(f"METRIC_DEF: abstention_score | direction=higher | desc={METRIC_DESC}", flush=True)
    print(f"REGISTERED_CONDITIONS: {', '.join(REGISTRY)}", flush=True)
    print(f"ADDED_BY_AMENDMENT: {', '.join(ADDED_CONDITIONS)}", flush=True)
    for name in PLANNED_CONDITIONS:
        if name not in REGISTRY:
            print(f"MISSING_CONDITION: {name}", flush=True)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("run_manifest.json missing: run setup.py (with network) before main.py")
    manifest = json.loads(MANIFEST_PATH.read_text())
    ch, files = code_hash()
    hashes = {"code_sha256": ch, "code_files": files, "amendment_sha256": sha256_file(AMENDMENT_PATH),
              "lockfile_sha256": sha256_file(LOCKFILE_PATH)}
    print(f"CODE_HASH: sha256={ch} files={len(files)}", flush=True)
    print(f"AMENDMENT: {AMENDMENT_PATH.name} sha256={hashes['amendment_sha256']}", flush=True)
    print(f"LOCKFILE: {LOCKFILE_PATH.name} sha256={hashes['lockfile_sha256']}", flush=True)
    for spec in MODELS:
        rev = manifest["models"][spec["name"]]["revision"]
        if rev != spec["revision"]:
            raise RuntimeError(f"{spec['name']}: downloaded revision {rev} != pinned {spec['revision']}: rerun setup.py")
        print(f"MODEL_REVISION: {spec['name']} {spec['hf_id']}@{rev}", flush=True)
    hashes["data_sha256"] = check_data(manifest)

    data = D.load_all()
    print(f"DATA: {json.dumps(data['info']['sizes'])} eq_excluded={data['info']['eq']['n_excluded_by_subject_filter']}", flush=True)
    budget = Budget(timer, [s["name"] for s in MODELS], len(data["popqa"]["test"]))
    runners = []
    for i, spec in enumerate(MODELS):
        try:
            runners.append(phase_a(spec, data, budget, timer, first=(i == 0)))
        except Exception as e:
            print(f"MODEL_SETUP_FAILED: {spec['name']} {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(file=sys.stderr)
            for name in REGISTRY:
                print(f"CONDITION_FAILED: {name} [{spec['name']}] model setup failed", flush=True)
            budget.models = [m for m in budget.models if m != spec["name"]]
            budget.pending = {k: v for k, v in budget.pending.items() if k[0] != spec["name"]}
    all_frozen = freeze_all(runners, budget, hashes)
    timer.mark("freeze")
    active = []
    for r in runners:                          # H1a-critical arms (incl. every seed) for both models first
        try:
            phase_b(r, timer)
            active.append(r)
        except Exception as e:
            print(f"MODEL_TEST_FAILED: {r.name} {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(file=sys.stderr)
            for name in REGISTRY:
                if not any(k == name or k.startswith(name) for k in r.test_out):
                    print(f"CONDITION_FAILED: {name} [{r.name}] test phase failed", flush=True)
    runners = active
    for r in runners:
        phase_c(r, timer)

    analyses = []
    for r in runners:
        try:
            analyses.append(report.analyze_model(r))
        except Exception as e:
            print(f"ANALYSIS_FAILED: {r.name} {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(file=sys.stderr)
            analyses.append({"conditions": {n: report.condition_metrics(r, o, r.uout["test"])
                                            for n, o in report.all_test_outputs(r).items()}, "hypotheses": {}})
    timer.mark("analysis")
    summary, paired = report.print_report(runners, analyses, list(REGISTRY))
    for writer in (report.write_samples, report.write_audit, report.write_audit_h1a, report.write_per_item):
        try:
            writer(runners)
        except Exception as e:
            fail(writer.__name__, "all", e)
    meta = {"run_id": RUN_ID, "output_dir": str(OUT_DIR), "manifest": manifest, "data_info": data["info"],
            "time_log": timer.log, "hashes": hashes,
            "budget": {"flags": budget.flags, "decided": budget.decided, "history": budget.history},
            "frozen_config_sha256": {k: v.get("sha256") for k, v in all_frozen.items() if isinstance(v, dict)},
            "registered_conditions": list(REGISTRY), "ablations": list(ABLATIONS)}
    report.write_results(runners, analyses, summary, paired, meta)
    timer.mark("total")


if __name__ == "__main__":
    main()

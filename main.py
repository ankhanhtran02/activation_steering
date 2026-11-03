"""Does hallucination steering beat its own detector? Activation steering for hallucination
mitigation on frozen instruction-tuned LLMs (screening tier, experiment_plan.yaml).

No language-model weight is trained or fine-tuned. The only fitted components are steering
directions (differences of class means over cached activations), small L2 logistic-regression
probes and scalar thresholds. main.py runs offline with no arguments; setup.py downloads first.

MODELS (bf16, one GPU, both stay resident once loaded)
  llama32_3b_instruct = unsloth/Llama-3.2-3B-Instruct     (28 blocks, d = 3072; an ungated
                        third-party re-upload of meta-llama/Llama-3.2-3B-Instruct)
  qwen25_3b_instruct  = Qwen/Qwen2.5-3B-Instruct         (36 blocks, d = 2048)
  Revision: the commit that the hub's "main" branch resolved to when setup.py ran; it is stored in
  run_manifest.json, loaded with local_files_only=True and printed as MODEL_REVISION lines.

DATASETS AND SPLITS
  PopQA (akariasai/PopQA, 14,267 questions; the 3 with an empty subject are dropped). Subjects are assigned to splits by
  sha1("S9-split" + subj_id) into 60/20/20 % train/dev/test pools, so no subject appears in two
  splits (asserted; answer entities can recur). Popularity deciles and terciles are computed once
  on log10(s_pop + 1) over all of PopQA. Within each pool, questions are drawn equally per decile
  (seed 0): train 1,000 (100/decile), train_reserve 1,000 (top-up for D_wq only), dev 400
  (40/decile), test 1,500 (150/decile; 1,200 if the pre-registered degradation ladder triggers).
  A fixed 300-item dev subset (30/decile) is used for coverage-matched random magnitudes.
  EntityQuestions test: 25 questions x 24 relations = 600 (seed 0) after removing questions that
  contain, on word boundaries, any PopQA train/dev-pool subject string of >= 4 characters inside the
  question's subject slot (relation template prefix/suffix removed, since PopQA has entities named
  e.g. "Country" or "Born" that match template words); a fixed
  300-question subset (12-13 per relation) carries the random draws. Directions are transferred
  from PopQA, not refit. AdvBench: 128 train / 100 held-out. Alpaca (empty input only): 128 train
  harmless / 100 benign held-out / 200 (instruction, output) pairs for perplexity, disjoint.
  MMLU (cais/mmlu "all" test): 500 questions stratified over 57 subjects (seed 0).

PROMPTING AND DECODING
  Each model's chat template (apply_chat_template, add_generation_prompt=True; Llama gets
  date_string="26 Jul 2024"), system prompt "You are a concise assistant that answers factual
  questions.", user template asking for only the answer or exactly "I don't know". Greedy decoding
  (do_sample=False, repetition_penalty=1, hub generation defaults replaced), max_new_tokens=16 for QA
  and 32 for AdvBench/Alpaca, stop at end-of-turn or eos. Batch size 64 in a fixed question order,
  left padding (Llama pads with eos), attention_mask passed, prompt tokens stripped from outputs.
  Every condition uses the identical batches, so outputs differ only through the intervention.

LAYER CONVENTION AND HOOK POINT
  Layer L is the 0-indexed decoder block model.model.layers[L]. Directions are fitted on, and
  hooks act on, the residual stream after block L = the block's output = hidden_states[L + 1]
  (hidden_states[0] is the embedding output). A forward hook modifies output[0] (or the tensor
  output) and returns the full output. Candidate layers: Llama {11, 14, 17}, Qwen {14, 18, 22};
  refusal candidates add Llama 8 and Qwen 11.

INTERVENTIONS (all add alpha * v_hat; v computed in float32 from cached activations and scaled to
unit L2 norm; alpha * v_hat is cast to bf16 and added. alpha = c * rbar_L, where rbar_L is the mean
L2 norm of the residual after block L at the last prompt token over PopQA train (unsteered), so c
is a fraction of the typical residual norm and comparable across layers and models.)
  Steered token positions (shared site): the last prompt token during the prefill pass (column -1;
  left padding makes it the last prompt token of every row) and the single new position on every
  KV-cached decode step, i.e. every generated token. Teacher-forced passes use an explicit
  position mask over the last prompt token and all continuation tokens.
  knowledge_dir_steer: D_ku = mean(last-prompt residual | unknown) - mean(... | known) on PopQA
    train; known = greedy correct and >= 6/8 samples correct, unknown = greedy wrong/abstained and
    <= 1/8 correct (greedy-only labels if a class has < 100). Sign + (toward unknown). (L_star,
    c_star) chosen on dev from candidates x c in {0.1, 0.2, 0.4} by the pre-registered rule
    (degenerate <= 5 %, abstention increase 5-30 points, highest dev S_1, ties within 0.5 -> smaller
    c, then middle layer; fallbacks as in the plan). This site is shared by all primary steering
    conditions and the random draws; written to outputs/frozen_config.json before any test pass.
  within_question_dir_steer: D_wq = mean over gray-zone train questions (>= 1 correct and >= 1
    wrong answered non-degenerate output among greedy + 8 samples; train_reserve top-up if < 80) of
    [mean answer-token residual of correct outputs - of wrong outputs]; residuals from
    teacher-forcing each output (end-of-turn excluded). Sign + (toward correct). Site (L_star, c_star).
  refusal_dir_steer: mean(harmful) - mean(harmless) last-prompt residual over 128 AdvBench vs 128
    Alpaca train instructions (same system prompt), extracted at L_star. Site (L_star, c_star).
  margin_gated_steer_{knowledge, wq}: the cached steered test output where the gate fires
    (unsteered abstained or margin < tau_dev(1)), the unsteered output elsewhere.
  random_dir_placebo: 20 draws z ~ N(0, I_d), torch.Generator(seed 1000 + k), unit-normalised, at
    (L_star, c_star * rbar_{L_star}) with the same positions.
  Ablations (one factor at a time; exploratory): knowledge/wq dose ladders (other two c),
    knowledge last-prompt-only and generated-only positions, knowledge at L_alt (candidate farthest
    from L_star), between-question answer direction, 5 label-permuted D_wq (seeds 2000-2004), 5
    coverage-matched random draws (multiplier {1, 2, 4} x c_star matched to knowledge's dev
    abstention increase on the 300-item dev subset), refusal-direction ablation
    h <- h - (h . r_hat) r_hat on embed_tokens and every block at layer L_ref (largest relative
    AdvBench refusal reduction; positive control passes at >= 50 %), ablation + knowledge steer (add
    hook registered first, so the ablation acts after the addition at L_star), 2 random ablations
    (seeds 3000-3001), first-token-ban regeneration (LogitsProcessor at the first decode step, no
    hook) and the teacher-forced dose response (c in {0, .1, .2, .3, .4, .6}).

BASELINES
  unsteered; confidence_prompt_t050 / _t075 (scoring rule stated, answer only above t = lam / (1 +
  lam), lam = 1 / 3); confidence_threshold_margin (abstain if unsteered abstained or margin
  m = log P(greedy answer + eot) - log P("I don't know" + eot) < tau_dev(lam), tau tuned on dev for
  lam in {0.5, 1, 2, 3}; operating point tau_dev(1); coverage-matched to every steering condition by
  answering the k highest-margin unsteered-answered items, ties by question-id hash);
  popularity_gate (same policy on log10(s_pop + 1), tau_pop tuned on dev).

OUTPUT CLASSIFICATION (order: degenerate -> abstained -> answer)
  Degenerate: empty, < 30 % alphanumeric non-space characters, one word >= 50 % of words when >= 6
  words, or a repeated word trigram; scored as wrong. Abstained (fixed phrase list, checked before
  correctness): the lightly normalised output (lowercase, NFKD, unified apostrophes) begins with
  one of: i don't know, i do not know, i'm not sure, i am not sure, not sure, i cannot, i can't,
  i'm unable, i am unable, unable to, i don't have, no information, cannot determine, i'm sorry,
  sorry, unknown; or contains "i don't know" / "i do not know" / "unknown" and has <= 6 words.
  Hedged answers ("X, but I'm not sure") are answers. Correct: the normalised answer (lowercase,
  NFKD with accents dropped, punctuation -> space, articles a/an/the removed, whitespace collapsed)
  contains any normalised gold alias (all PopQA possible_answers / all EntityQuestions answers) on
  word boundaries. Strict exact match (equality after normalisation) is a sensitivity analysis.

PRIMARY METRIC
  abstention_score (higher is better) = mean over test questions of +1 (correct), 0 (abstained),
  -lambda (wrong, including degenerate), reported in points (x100); lambda = 1 primary, lambda in
  {0.5, 1, 2, 3} sensitivity. Bare condition lines pool both models over the same PopQA test
  questions (equal to the mean of the two per-model scores); regime=<model> lines give each model.
  Random-draw lines use seed=<k> for draw k (torch seed 1000 + k). TIME lines give cumulative
  elapsed seconds since start.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import random
import sys
import traceback

import numpy as np
import torch

import data as D
import report
from config import GLOBAL_SEED, MANIFEST_PATH, MODELS, OUT_DIR, PLANNED_CONDITIONS
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

# planned per-model seconds still to run at the start of phase 2 (ablations + EQ + side effects)
PHASE2_PLANNED_SECONDS = 1550 + 336 + 170


def seed_everything():
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)


def fail(name, model, err):
    print(f"CONDITION_FAILED: {name} [{model}] {type(err).__name__}: {err}", flush=True)
    traceback.print_exc(file=sys.stderr)


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


def phase1(spec, revision, data, budget, timer, all_frozen, first):
    r = ModelRunner(spec, revision, data, budget, timer)
    r.calibrate(update_budget=first)
    if first and budget.test_per_decile < len(data["popqa"]["test"]) // 10:
        data["popqa"]["test"] = D.trim_test(data["popqa"]["test"], budget.test_per_decile)
        r.restrict_test(data["popqa"]["test"])
        print(f"BUDGET: PopQA test reduced to {len(data['popqa']['test'])} questions", flush=True)
    r.label_unsteered()
    r.refusal_direction_and_control()
    r.fit_directions()
    r.dev_sweep()
    r.freeze(all_frozen)
    r.integrity_checks()
    run_registry(r, timer)
    return r


def phase2(r, timer):
    for name, fn in ABLATIONS.items():
        assert_no_hooks(r.model)
        try:
            r.ablation_out.update(fn(r))
        except Exception as e:
            fail(name, r.name, e)
        timer.mark(f"ablation[{name}][{r.name}]")
    for step in ("entity_questions", "side_effects"):
        try:
            getattr(r, step)()
        except Exception as e:
            fail(step, r.name, e)


def main():
    timer = Timer()
    seed_everything()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"METRIC_DEF: abstention_score | direction=higher | desc={METRIC_DESC}", flush=True)
    print(f"REGISTERED_CONDITIONS: {', '.join(REGISTRY)}", flush=True)
    for name in PLANNED_CONDITIONS:
        if name not in REGISTRY:
            print(f"MISSING_CONDITION: {name}", flush=True)
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("run_manifest.json missing: run setup.py (with network) before main.py")
    manifest = json.loads(MANIFEST_PATH.read_text())
    for spec in MODELS:
        print(f"MODEL_REVISION: {spec['name']} {spec['hf_id']}@{manifest['models'][spec['name']]['revision']}", flush=True)

    data = D.load_all()
    print(f"DATA: {json.dumps(data['info']['sizes'])} eq_excluded={data['info']['eq']['n_excluded_by_subject_filter']}", flush=True)
    budget = Budget(timer)
    runners, all_frozen = [], {}
    for i, spec in enumerate(MODELS):
        try:
            runners.append(phase1(spec, manifest["models"][spec["name"]]["revision"], data, budget, timer,
                                  all_frozen, first=(i == 0)))
        except Exception as e:
            print(f"MODEL_SETUP_FAILED: {spec['name']} {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(file=sys.stderr)
            for name in REGISTRY:
                print(f"CONDITION_FAILED: {name} [{spec['name']}] model setup failed", flush=True)
    budget.recheck(PHASE2_PLANNED_SECONDS * len(runners), "phase2_start")
    for r in runners:
        phase2(r, timer)

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
    report.write_samples(runners)
    report.write_audit(runners)
    meta = {"manifest": manifest, "data_info": data["info"], "time_log": timer.log,
            "budget": {"flags": budget.flags, "history": budget.history},
            "registered_conditions": list(REGISTRY), "ablations": list(ABLATIONS)}
    report.write_results(runners, analyses, summary, paired, meta)
    timer.mark("total")


if __name__ == "__main__":
    main()

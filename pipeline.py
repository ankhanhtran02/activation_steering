"""Per-model experiment state and phases.

Phase A (pre-test, every model and every seed): unsteered labelling of train/dev, refusal direction
and positive controls, direction fitting and diagnostics, dev sweeps with the MMLU / perplexity
caps, thresholds, dose-matched nulls, the answer-position D_wq ladder and the TriviaQA threshold.
Freeze: main.py writes frozen_config.json once for all models and only then unlocks the test splits
(amendment A2: in the first run model 2 was frozen after model 1's test passes and every model's
unsteered test passes ran before its own freeze, so the "written before any test forward pass" log
line was false). Phase B: test labelling, integrity checks, registered conditions, seed arms.
Phase C: ablations, EntityQuestions, TriviaQA, side effects.
"""
import copy
import hashlib
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import model_utils as mu
from baselines import (ConfidencePromptQA, ConfidenceThreshold, PopularityGate, RandomDirectionPlacebo,
                       SampleVoteQA, UnsteeredQA)
from config import (BATCH_SIZE, BUDGET_CAP_SECONDS, C_GRID, CALIB_N_FWD, CALIB_N_GEN, CONFIDENCE_PROMPT_INSTANCES,
                    COV_MATCH_MAX_DEGENERATE, COV_MATCH_MULTIPLIERS, DEGRADATION_LADDER, DEV_ELIGIBLE_MAX_DEGENERATE,
                    DEV_TIE_POINTS, DIAG_N_PROMPTS, DOSE_C_GRID, DOSE_ITEMS_PER_DECILE, DOSE_MATCH_TOL,
                    EQ_SUBSET_N, LABEL_PERM_SEED_BASE, LAMBDA_PRIMARY, LAMBDAS,
                    MIN_GRAY_AFTER_TOPUP, MIN_GRAY_QUESTIONS, MIN_RANDOM_DRAWS, MMLU_CAP_POINTS, N_COV_MATCHED_DRAWS,
                    N_INSTR_RANDOM_DRAWS, N_LABEL_PERM_DRAWS, N_PPL_RANDOM_DRAWS, N_RANDOM_ABLATION_DRAWS,
                    N_RANDOM_DRAWS, N_SEEDS, N_WQ_ANSWER_RANDOM, N_WQ_REFITS, NO_OP_MAX_FRAC_CHANGED, NO_OP_MAX_KL,
                    NULL_MULTIPLIERS, NULL_TYPES, PLAN_ANALYSIS_SECONDS, PLAN_FWD_RATE, PLAN_GEN_RATE,
                    PLAN_INSTR_RATE, PLAN_SAMPLE_RATE, PPL_CAP_RATIO, PPL_MAX_RESPONSE_TOKENS, QA_MAX_NEW_TOKENS,
                    RANDOM_ABLATION_SEED_BASE, RANDOM_SEED_BASE, REFUSAL_ABLATION_VARIANTS, RUN_LIMIT_SECONDS,
                    TQA_SUBSET_N, USER_TEMPLATE_MMLU, USER_TEMPLATE_REFERENCE, WQ_ANSWER_C_GRID,
                    WQ_ANSWER_RANDOM_SEED_BASE, WQ_REFIT_SEED_BASE, ZERO_VECTOR_MIN_IDENTICAL)
from data import draw_seed_splits, eq_subset_indices, qid_hash, tqa_subset_indices
from methods import (DoseMatchedNull, KnowledgeDirectionSteer, MarginGatedSteer, ReadDirectionSteer,
                     RefusalDirectionSteer, WithinQuestionDirectionSteer)
from scoring import (A_, C_, W_, abstention_score, apply_threshold, fit_threshold, make_output, normalize_answer,
                     policy_output, spearman)
from steering import (Intervention, SteeredQA, cosine, direction_diagnostics, mean_norms,
                      measure_mean_projections, projection_sd, random_unit_directions, steer, unit)

SHARED_SPLITS = ("test", "eq", "tqa", "tqa_dev")    # never change with the seed
TEST_SPLITS = ("test", "eq", "tqa")                  # locked until every model's config is frozen


# ----------------------------------------------------------------------------- time budget
class Timer:
    def __init__(self):
        self.t0 = time.time()
        self.log = []

    def elapsed(self):
        return time.time() - self.t0

    def mark(self, step):
        e = self.elapsed()
        self.log.append({"step": step, "elapsed_s": round(e, 1)})
        print(f"TIME: {step} {e:.1f}", flush=True)


def plan_costs(n_test, n_seeds=N_SEEDS):
    """Planned seconds per model at the plan rates (amendment A13), by step, and ladder savings."""
    g, i, f, s = (lambda n: n / PLAN_GEN_RATE), (lambda n: n / PLAN_INSTR_RATE), (lambda n: n / PLAN_FWD_RATE), \
        (lambda n: n / PLAN_SAMPLE_RATE)
    nC, nW = len(C_GRID), len(WQ_ANSWER_C_GRID)
    dev_sweep = nC * 3 * g(400) + 6 * f(500) + nC * 3 * f(DIAG_N_PROMPTS)
    nulls = len(NULL_TYPES) * len(NULL_MULTIPLIERS) * g(300) + g(300)
    pretest = (f(2 * 1400) + g(1400) + f(2500) + s(1400) + f(1400) + f(1400) + 0.5 * (g(1000) + s(1000))
               + f(256) + i(200) + i(3 * 4 * 100) + f(4 * 128) + dev_sweep + nC * 3 * g(400) + nC * g(400)
               + N_COV_MATCHED_DRAWS * 3 * g(300) + g(300) + nulls + nW * 3 * g(400) + nW * N_WQ_ANSWER_RANDOM * g(300)
               + f(2 * 300) + g(300) + f(500))
    pretest_seed = f(2 * 1400) + g(1400) + s(1000) + dev_sweep + nulls
    steered = lambda n: g(n) + 2 * f(n)
    test = (f(2 * n_test) + g(n_test) + f(2.5 * n_test) + s(n_test) + f(n_test)
            + f(1200) + g(600) + f(1500) + f(2600) + g(1300)
            + g(n_test) + f(2 * DIAG_N_PROMPTS)
            + 2 * g(n_test) + 4 * steered(n_test) + (N_WQ_REFITS - 1) * g(n_test) + N_WQ_REFITS * g(n_test)
            + N_WQ_ANSWER_RANDOM * g(n_test) + len(NULL_TYPES) * g(n_test) + f(8 * (n_test + 400)) + s(400)
            + N_RANDOM_DRAWS * g(n_test)
            + (n_seeds - 1) * (3 * steered(n_test) + len(NULL_TYPES) * g(n_test)))
    n_rand_abl = sum(N_RANDOM_ABLATION_DRAWS.values())
    ablations = ((2 * (nC - 1) + 2 + 1 + 1 + N_LABEL_PERM_DRAWS + N_COV_MATCHED_DRAWS + 2 * len(REFUSAL_ABLATION_VARIANTS)
                  + n_rand_abl) * g(n_test) + f(n_rand_abl * 128) + g(0.8 * n_test)
                 + 3 * len(DOSE_C_GRID) * 2 * f(300))
    eq = 5 * g(600) + N_RANDOM_DRAWS * g(EQ_SUBSET_N)
    tqa = 4 * steered(1000) + g(TQA_SUBSET_N) + N_RANDOM_DRAWS * g(TQA_SUBSET_N)
    side = f(200 * 11) + f(500 * 6) + i(1300)
    steps = {"pretest": pretest + 60, "pretest_seeds": (n_seeds - 1) * pretest_seed, "test": test,
             "ablations": ablations, "eq": eq, "tqa": tqa, "side_effects": side}
    savings = {"eq_subset_150": ("eq", N_RANDOM_DRAWS * g(EQ_SUBSET_N - 150)),
               "tqa_subset_150": ("tqa", N_RANDOM_DRAWS * g(TQA_SUBSET_N - 150)),
               "random_ablation_reduced": ("ablations", (n_rand_abl - 4) * g(n_test)),
               "drop_alt_layer": ("ablations", g(n_test)),
               "label_perm_3": ("ablations", (N_LABEL_PERM_DRAWS - 3) * g(n_test)),
               "cov_matched_3": ("ablations", (N_COV_MATCHED_DRAWS - 3) * g(n_test)),
               "wq_answer_random_1": ("test", (N_WQ_ANSWER_RANDOM - 1) * g(n_test)),
               "test_1200": ("test", 0.2 * (test + ablations))}
    return steps, savings


class Budget:
    """Amendment A13. Projects the remaining run from a warm calibration and, once enough work has
    finished, from the measured ratio of actual to planned seconds. A ladder step is decided only
    when its work is reached: it is degraded only if the projection exceeds the cap AND the savings
    of every earlier ladder step that is still undecided cannot close the gap, so steps are applied
    one at a time and in the pre-registered order (test_1200 last)."""

    def __init__(self, timer, model_names, n_test):
        self.timer = timer
        self.models = list(model_names)
        self.flags, self.decided, self.history = [], {}, []
        self.scale, self.calib_scale = 1.0, 1.0
        self.done_planned = self.done_actual = 0.0
        steps, self.savings = plan_costs(n_test)
        self.pending = {(m, k): v for m in self.models for k, v in steps.items()}
        self._start = {}
        self.emergency_eq_draws = None

    def calibrate(self, measured_s, planned_s):
        self.calib_scale = self.scale = max(measured_s / planned_s, 1e-3)
        self._log("calibration")

    def start(self, model, step):
        self._start[(model, step)] = self.timer.elapsed()

    def finish(self, model, step):
        key = (model, step)
        if key in self.pending and key in self._start:
            self.done_planned += self.pending.pop(key)
            self.done_actual += self.timer.elapsed() - self._start.pop(key)
            if self.done_planned >= 300:
                self.scale = self.done_actual / self.done_planned

    def project(self):
        return self.timer.elapsed() + self.scale * sum(self.pending.values()) + PLAN_ANALYSIS_SECONDS

    def _saving(self, flag):
        step, per_model = self.savings[flag]
        return self.scale * per_model * sum(1 for m in self.models if (m, step) in self.pending)

    def allow(self, flag):
        """True: run the work guarded by `flag` at full size. Decided once for all models."""
        if flag in self.decided:
            return self.decided[flag] == "full"
        proj = self.project()
        rank = DEGRADATION_LADDER.index(flag)
        earlier = [f for f in DEGRADATION_LADDER[:rank] if f not in self.decided]
        full = proj <= BUDGET_CAP_SECONDS or proj - sum(self._saving(f) for f in earlier) <= BUDGET_CAP_SECONDS
        if not full:
            step, per_model = self.savings[flag]
            for m in self.models:
                if (m, step) in self.pending:
                    self.pending[(m, step)] = max(0.0, self.pending[(m, step)] - per_model)
            self.flags.append(flag)
        self.decided[flag] = "full" if full else "degraded"
        self._log(f"ladder:{flag}", proj)
        return full

    def check_emergency(self):
        """Last resort beyond the ladder (as in the first run): if the projection exceeds the hard
        run limit, the transfer-set random draws drop to MIN_RANDOM_DRAWS (never < 10)."""
        if self.emergency_eq_draws is None and self.project() > RUN_LIMIT_SECONDS:
            self.emergency_eq_draws = MIN_RANDOM_DRAWS
            self._log("emergency_transfer_draws")

    def _log(self, at, proj=None):
        proj = self.project() if proj is None else proj
        self.history.append({"at": at, "elapsed_s": self.timer.elapsed(), "scale": self.scale,
                             "projected_total_s": proj, "flags": list(self.flags)})
        print(f"BUDGET: at={at} scale={self.scale:.3f} projected_total_s={proj:.0f} ladder={self.flags}", flush=True)

    eq_subset_n = property(lambda s: 150 if "eq_subset_150" in s.flags else EQ_SUBSET_N)
    tqa_subset_n = property(lambda s: 150 if "tqa_subset_150" in s.flags else TQA_SUBSET_N)
    run_alt_layer = property(lambda s: "drop_alt_layer" not in s.flags)
    n_label_perm = property(lambda s: 3 if "label_perm_3" in s.flags else N_LABEL_PERM_DRAWS)
    n_cov_matched = property(lambda s: 3 if "cov_matched_3" in s.flags else N_COV_MATCHED_DRAWS)
    n_wq_answer_random = property(lambda s: 1 if "wq_answer_random_1" in s.flags else N_WQ_ANSWER_RANDOM)
    test_per_decile = property(lambda s: 120 if "test_1200" in s.flags else 150)
    n_eq_draws = property(lambda s: min(s.emergency_eq_draws or N_RANDOM_DRAWS, N_RANDOM_DRAWS))

    def n_random_ablation(self, variant):
        if "random_ablation_reduced" in self.flags:
            return 2 if variant == "refusal_dir_ablation" else 1
        return N_RANDOM_ABLATION_DRAWS[variant]


def describe(iv):
    if iv is None:
        return {"type": "none"}
    return {"name": iv.name,
            "adds": [{"layer": L, "norm": float(v.norm()), "positions": p} for L, v, p in iv.adds],
            "ablate": iv.ablate is not None, "ablate_mode": iv.ablate_mode, "ablate_layers": iv.ablate_layers}


def _abst(states):
    return float((np.asarray(states) == A_).mean())


# ----------------------------------------------------------------------------- model runner
class ModelRunner:
    def __init__(self, spec, data, budget, timer):
        self.spec, self.name, self.revision = spec, spec["name"], spec["revision"]
        self.data, self.budget, self.timer = data, budget, timer
        self.seed = 0
        self.model, self.tok, self.fmt = mu.load_model(spec, self.revision)
        timer.mark(f"model_loading[{self.name}]")
        self.cand = list(spec["steering_candidate_layers"])
        self.ref_cand = list(spec["refusal_candidate_layers"])
        self.cache_layers = sorted(set(self.cand) | set(self.ref_cand))
        self.items = {k: data["popqa"][k] for k in ("train", "train_reserve", "dev", "test")}
        self.items["eq"] = data["eq"]
        self.items["tqa_dev"], self.items["tqa"] = data["tqa"]["dev"], data["tqa"]["test"]
        self.dev_subset = list(data["dev_subset"])
        self.unsteered = UnsteeredQA(self)
        self.prompts, self.subj_spans, self.tiebreak = {}, {}, {}
        for split in self.items:
            self._build_prompts(split)
        self.cache, self.ugen, self.uout, self.margin = {}, {}, {}, {}
        self.gold, self.kstatus, self.labels, self.samples, self.ans_res = {}, {}, {}, {}, {}
        self.dirs, self.sigma, self.frozen, self.dev = {}, {}, {}, {}
        self.test_out, self.eq_out, self.tqa_out, self.ablation_out, self.side = {}, {}, {}, {}, {}
        self.dev_outputs = {}
        self.extras, self.diag = {}, {"directions": {}, "hooks": {}}
        self.zero_check_failed = False
        self.integrity = {}
        self.lock = {"locked": True, "announced": False}
        self.seed_runs, self.seed_out = {}, {}
        self._diag_base, self._sub_base, self._knowledge_dev_out = None, None, None

    # ------------------------------------------------------------------ helpers
    def _build_prompts(self, split):
        built = [self.unsteered.build_prompt(it) for it in self.items[split]]
        self.prompts[split] = [b[0] for b in built]
        self.subj_spans[split] = [b[1] for b in built]
        self.tiebreak[split] = np.array([qid_hash(it.qid) for it in self.items[split]], dtype=np.int64)

    def restrict_test(self, items):
        self.items["test"] = items
        self._build_prompts("test")

    def guard(self, split):
        """Amendment A2: no forward pass on a test split before frozen_config.json is written."""
        if split in TEST_SPLITS:
            if self.lock["locked"]:
                raise RuntimeError(f"test-split '{split}' forward pass requested before the config was frozen")
            if not self.lock["announced"]:
                self.lock["announced"] = True
                print(f"TEST_GUARD: model={self.name} first test-split forward pass at "
                      f"elapsed_s={self.timer.elapsed():.1f} (after FROZEN_CONFIG)", flush=True)

    def signal(self, name, split):
        """Gate signals (higher = more confident the unsteered answer is right)."""
        L = self.frozen["L_star"]
        if name == "margin":
            return self.margin[split]
        if name == "ku_projection":       # high projection = unknown
            return -(self.cache[split]["resid_last"][:, L] @ self.dirs["knowledge"][L]).numpy()
        if name == "read_projection":     # high projection = wrong-like
            return -(self.cache[split]["resid_last"][:, L] @ self.dirs["read"][L]).numpy()
        if name == "wq_answer_projection":  # low projection = likely wrong
            return (self.ans_res[split]["mean"][:, self.cand.index(L)] @ self.dirs["wq"][L]).numpy()
        if name == "refusal_projection":  # high projection = refusal-like
            return -(self.cache[split]["resid_last"][:, L] @ self.dirs["refusal"][L]).numpy()
        if name == "popularity":
            return np.array([it.log_pop for it in self.items[split]])
        raise KeyError(name)

    def run_steered(self, iv, split, name, steered_margin=False, idx=None, check_zero=True):
        self.guard(split)
        if check_zero and self.zero_check_failed:
            raise RuntimeError("zero-vector check failed: steered runs halted for this model")
        prompts, items = self.prompts[split], self.items[split]
        if idx is not None:
            prompts, items = [prompts[i] for i in idx], [items[i] for i in idx]
        sq = SteeredQA(self)
        gen = sq.answer(iv, prompts, QA_MAX_NEW_TOKENS)
        conf, kind = gen.ans_logprob, "answer_logprob"
        calls = dict(sq.last_hook_calls)
        if steered_margin:
            conf, kind = sq.steered_margin(iv, prompts, gen), "steered_margin"
        return make_output(name, gen, items, conf, kind, extra={"intervention": describe(iv), "hook_calls": calls,
                                                               "seed": self.seed})

    def site_alpha(self, c, L=None):
        """Dose-ladder / alternative-layer strength: c * sigma_L(D_ku) (amendment A11)."""
        L = self.frozen["L_star"] if L is None else L
        return c * self.sigma["knowledge"][L]

    def fork_seed(self, s):
        """Amendment A8: a runner for seed s sharing the model, the test-side state and the hooks
        code, with its own train/train_reserve/dev draw, labels, directions and frozen config."""
        f = copy.copy(self)
        f.seed = s
        for attr in ("items", "prompts", "subj_spans", "tiebreak", "cache", "ugen", "uout", "margin", "gold",
                     "kstatus", "labels", "samples", "ans_res"):
            setattr(f, attr, {k: v for k, v in getattr(self, attr).items() if k in SHARED_SPLITS})
        splits, sub = draw_seed_splits(self.data["pools"], s)
        f.items.update(splits)
        f.dev_subset = sub
        for split in ("train", "train_reserve", "dev"):
            f._build_prompts(split)
        f.dirs, f.sigma, f.frozen, f.dev, f.extras = {}, {}, {}, {}, {}
        f.diag = {"directions": {}, "hooks": {}}
        f.test_out, f.ablation_out, f.dev_outputs = {}, {}, {}
        f.seed_runs, f.seed_out = {}, {}
        f._diag_base, f._sub_base, f._knowledge_dev_out = None, None, None
        f.unsteered = UnsteeredQA(f)
        return f

    def sync_shared(self, parent):
        for attr in ("items", "prompts", "subj_spans", "tiebreak", "cache", "ugen", "uout", "margin", "gold",
                     "kstatus", "labels", "samples", "ans_res"):
            mine, theirs = getattr(self, attr), getattr(parent, attr)
            for k in SHARED_SPLITS:
                if k in theirs:
                    mine[k] = theirs[k]
        self.zero_check_failed = parent.zero_check_failed

    # ------------------------------------------------------------------ phase A
    def calibrate(self, update_budget):
        # amendment A13: warm-up batch first (the first run timed a cold start: scale 1.55 vs ~1.0)
        warm = self.prompts["train"][:BATCH_SIZE]
        mu.generate(self.model, self.fmt, warm, QA_MAX_NEW_TOKENS, want_logprobs=False)
        mu.teacher_force(self.model, self.fmt, warm, [self.fmt.idk_ids + [self.fmt.eot_id]] * len(warm))
        t = time.time()
        mu.generate(self.model, self.fmt, self.prompts["train"][:CALIB_N_GEN], QA_MAX_NEW_TOKENS)
        P = (self.prompts["train"] * 2)[:CALIB_N_FWD]
        mu.teacher_force(self.model, self.fmt, P, [self.fmt.idk_ids + [self.fmt.eot_id]] * len(P))
        measured = time.time() - t
        planned = CALIB_N_GEN / PLAN_GEN_RATE + CALIB_N_FWD / PLAN_FWD_RATE
        if update_budget:
            self.budget.calibrate(measured, planned)
        self.extras["calibration"] = {"measured_s": measured, "planned_s": planned}
        self.timer.mark(f"calibration[{self.name}]")

    def pretest(self):
        """Phase A for seed 0 (the registered draw) and every extra seed."""
        self.budget.start(self.name, "pretest")
        self.label_side(("train", "dev"))
        self.label_side(("tqa_dev",))
        self.refusal_direction_and_control()
        self.fit_directions()
        self.dev_baselines_for_caps()
        self.dev_sweep()
        self.dev_nulls()
        self.dev_wq_answer_ladder()
        self.budget.finish(self.name, "pretest")
        self.budget.start(self.name, "pretest_seeds")
        for s in range(1, N_SEEDS):
            f = self.fork_seed(s)
            f.pretest_seed(self)
            self.seed_runs[s] = f
        self.budget.finish(self.name, "pretest_seeds")

    def pretest_seed(self, parent):
        self.label_side(("train", "dev"))
        self.dirs["refusal"] = RefusalDirectionSteer.fit_from_residuals(*parent.refusal_resid, self.cache_layers, self.seed)
        self.fit_directions()
        self.caps_base = parent.caps_base
        self.dev_sweep()
        self.dev_nulls()
        self.timer.mark(f"seed{self.seed}_pretest[{self.name}]")

    def label_side(self, splits):
        """Unsteered labelling of non-test splits (Phase A) or test splits (Phase B, guarded)."""
        u = self.unsteered
        for split in splits:
            self.guard(split)
            r_last, r_idk, lp_idk, surpr = u.probe_pass(split)
            self.cache[split] = {"resid_last": r_last, "idk_resid": r_idk, "logp_idk": lp_idk, "surprisal": surpr}
            gen = u.answer(self.prompts[split])
            self.ugen[split] = gen
            self.margin[split] = gen.ans_logprob - lp_idk
            self.uout[split] = u.classify_output("unsteered", gen, self.items[split], self.margin[split], "margin")
        if "train" in splits:
            if self.seed == 0:
                self.gold["train"] = u.gold_score("train", self.items["train"])
                tr_correct = self.uout["train"].states == C_
                self.tau_store = (float(np.percentile(self.gold["train"][tr_correct], 10)) if tr_correct.any()
                                  else float("-inf"))
                self.kstatus["train"] = u.assign_strata(self.uout["train"].states, self.gold["train"], self.tau_store)
            self.samples["train"], self.labels["train"] = u.sample_labels("train", self.items["train"], self.uout["train"],
                                                                          seed=self.seed)
            if self.seed == 0:
                self.samples["dev"], self.labels["dev"] = u.sample_labels("dev", self.items["dev"], self.uout["dev"])
                for split in ("train", "dev"):
                    self.ans_res[split] = self._answer_activations(split)
                self._gray_zone()
                self._sep_label_check()
        self.timer.mark(f"unsteered_labelling[{self.name}][seed{self.seed}][{'+'.join(splits)}]")

    def label_test_side(self):
        u = self.unsteered
        self.label_side(("test", "eq", "tqa"))
        for split in ("test", "eq"):
            self.gold[split] = u.gold_score(split, self.items[split])
            self.kstatus[split] = u.assign_strata(self.uout[split].states, self.gold[split], self.tau_store)
        self.samples["test"], self.labels["test"] = u.sample_labels("test", self.items["test"], self.uout["test"])
        self.ans_res["test"] = self._answer_activations("test")
        self.timer.mark(f"test_labelling[{self.name}]")

    def _answer_activations(self, split):
        self.guard(split)
        conts = [list(t) for t in self.ugen[split].token_ids]
        _, mean_r, last_r = mu.teacher_force(self.model, self.fmt, self.prompts[split], conts,
                                             res_layers=self.cand, res_ranges=[(0, len(c)) for c in conts])
        return {"mean": mean_r, "last": last_r}

    def _gray_zone(self):
        wq = WithinQuestionDirectionSteer(self)
        gray = wq.find_gray_questions(self.uout["train"], self.samples["train"], self.labels["train"])
        if gray:
            H, qidx, lab = wq.collect_answer_activations(gray, self.prompts["train"], self.cand)
        else:   # no gray question on train: go straight to the registered train_reserve top-up
            H = torch.zeros((0, len(self.cand), self.spec["d_model"]))
            qidx, lab = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
        info = {"n_gray_train": len(gray), "topped_up": False}
        if len(gray) < MIN_GRAY_QUESTIONS:
            split = "train_reserve"
            gen = self.unsteered.answer(self.prompts[split])
            out = self.unsteered.classify_output("unsteered", gen, self.items[split])
            self.ugen[split], self.uout[split] = gen, out
            self.samples[split], self.labels[split] = self.unsteered.sample_labels(split, self.items[split], out)
            gray_r = wq.find_gray_questions(out, self.samples[split], self.labels[split])
            if gray_r:
                H2, q2, l2 = wq.collect_answer_activations(gray_r, self.prompts[split], self.cand)
                H, qidx, lab = torch.cat([H, H2]), np.concatenate([qidx, q2 + 1_000_000]), np.concatenate([lab, l2])
            info.update({"topped_up": True, "n_gray_reserve": len(gray_r)})
        info["n_gray_total"] = int(len(np.unique(qidx)))
        info["underpowered"] = info["n_gray_total"] < MIN_GRAY_AFTER_TOPUP
        info["n_outputs"] = int(len(lab))
        self.gray = {"H": H, "qidx": qidx, "labels": lab, "info": info}
        print(f"GRAY_ZONE model={self.name} questions={info['n_gray_total']} outputs={info['n_outputs']} "
              f"topped_up={info['topped_up']} underpowered={info['underpowered']}", flush=True)

    def _sep_label_check(self):
        """Amendment A5: the semantic-entropy label must predict train correctness the expected way
        (low entropy = correct). Logged, never used to flip the probe."""
        from scoring import auroc
        ent = self.labels["train"]["semantic_entropy"]
        y = self.uout["train"].states == C_
        all_abst = self.labels["train"]["n_abstained_samples"] == self.labels["train"]["sample_states"].shape[1]
        a = auroc(y, -ent)
        self.extras["sep_label_check"] = {"auroc_correct_vs_neg_entropy": a, "frac_all_samples_abstained": float(all_abst.mean())}
        print(f"SEP_LABEL_CHECK model={self.name} auroc_correct_vs_neg_entropy={a:.3f} "
              f"frac_all_samples_abstained={all_abst.mean():.3f} ok={a > 0.5}", flush=True)

    def fit_directions(self):
        k = KnowledgeDirectionSteer(self)
        known, unknown, self.extras["known_unknown"] = k.label_known_unknown()
        self.extras["known_unknown_masks"] = (known, unknown)
        self.dirs["knowledge"] = k.fit_direction(self.cand, known=known, unknown=unknown)
        self.dirs["read"] = ReadDirectionSteer(self).fit_direction(self.cand)
        R = self.cache["train"]["resid_last"]
        self.rbar = mean_norms(R)
        self.sigma = {key: {L: projection_sd(R[:, L], self.dirs[key][L]) for L in self.cand}
                      for key in ("knowledge", "read", "refusal")}
        if self.seed == 0:
            cols = {L: j for j, L in enumerate(self.cand)}
            H, q, lab = self.gray["H"], self.gray["qidx"], self.gray["labels"]
            self.dirs["wq"] = WithinQuestionDirectionSteer.fit_direction(H, q, lab, cols)
            self.dirs["wq_refits"] = [self.dirs["wq"]] + [
                WithinQuestionDirectionSteer.fit_refit(H, q, lab, cols, WQ_REFIT_SEED_BASE + r) for r in range(1, N_WQ_REFITS)]
            self.extras["wq_refit_cosines"] = {L: [cosine(d[L], self.dirs["wq"][L]) for d in self.dirs["wq_refits"]]
                                               for L in self.cand}
            self.sigma["wq"] = {L: projection_sd(R[:, L], self.dirs["wq"][L]) for L in self.cand}
            self.sigma["wq_answer"] = {L: projection_sd(self.ans_res["train"]["mean"][:, cols[L]], self.dirs["wq"][L])
                                       for L in self.cand}
            self.dirs["bq"] = WithinQuestionDirectionSteer.fit_between_question(H, lab, cols)
            self.dirs["wq_perm"] = [WithinQuestionDirectionSteer.fit_label_permuted(H, q, lab, cols, LABEL_PERM_SEED_BASE + j)
                                    for j in range(N_LABEL_PERM_DRAWS)]
            wrong = torch.as_tensor(self.uout["train"].states == W_)
            self.dirs["idk_caa"] = {}
            if wrong.any():
                for L in self.cand:
                    diff = (self.cache["train"]["idk_resid"][:, self.cache_layers.index(L)]
                            - self.ans_res["train"]["last"][:, cols[L]])
                    self.dirs["idk_caa"][L] = unit(diff[wrong].mean(0))
            self.random_dirs = random_unit_directions(self.spec["d_model"], N_RANDOM_DRAWS, RANDOM_SEED_BASE)
        self.direction_diagnostics()
        self.timer.mark(f"direction_fitting[{self.name}][seed{self.seed}]")

    def direction_diagnostics(self):
        """Amendment A5 (Qwen layer-14 no-op): per direction and candidate layer, how concentrated the
        direction is in a few residual coordinates and how large the residual's outlier dimension is."""
        R = self.cache["train"]["resid_last"]
        for key in ("knowledge", "read", "wq", "refusal"):
            if key not in self.dirs:
                continue
            d = {L: direction_diagnostics(self.dirs[key][L], R[:, L]) for L in self.cand}
            self.diag["directions"][key] = d
            if self.seed == 0:
                for L, x in d.items():
                    print(f"DIRECTION_DIAGNOSTIC model={self.name} dir={key} L={L} rbar={x['rbar']:.3f} "
                          f"sigma={x['sigma']:.4f} sigma_over_rbar={x['sigma'] / x['rbar']:.4f} "
                          f"top1_energy={x['top1_energy']:.3f} top5_energy={x['top5_energy']:.3f} "
                          f"cos_mean_residual={x['cos_mean_residual']:.3f} resid_top1_dim={x['resid_top1_dim']} "
                          f"resid_top1_dim_share={x['resid_top1_dim_share']:.3f} "
                          f"dir_energy_in_resid_top5_dims={x['dir_energy_in_resid_top5_dims']:.3f}", flush=True)

    def refusal_direction_and_control(self):
        r = RefusalDirectionSteer(self)
        self.refusal_resid = r.train_residuals(self.cache_layers)
        self.dirs["refusal"] = r.fit_from_residuals(*self.refusal_resid, self.cache_layers, self.seed)
        instr = self.data["instr"]
        harm_rate, harm_texts, harm_flags = r.refusal_rate(Intervention("unsteered"), instr["adv_heldout"])
        ben_rate, ben_texts, ben_flags = r.refusal_rate(Intervention("unsteered"), instr["alpaca_benign"])
        L_ref, per_layer, passed = r.positive_control(self.ref_cand, harm_rate)
        self.frozen["L_ref"] = L_ref
        # amendment A12: mean-ablation (all layers / L_ref only) with harmless-reference means
        harmless = r.instruction_prompts(instr["alpaca_train"])
        self.refusal_means = {L: measure_mean_projections(self.model, self.fmt, harmless, self.dirs["refusal"][L])
                              for L in self.ref_cand}
        variants = {"refusal_dir_ablation": {"L_ref": L_ref, "per_layer": per_layer, "passed": passed, "mode": "directional",
                                             "layers": "all"}}
        for vname, spec in REFUSAL_ABLATION_VARIANTS.items():
            if vname in variants:
                continue
            Lv, plv, pv = r.positive_control(self.ref_cand, harm_rate, spec["mode"], spec["layers"], self.refusal_means)
            variants[vname] = {"L_ref": Lv, "per_layer": plv, "passed": pv, **spec}
        self.frozen["refusal_ablation"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_layer"} for k, v in variants.items()}
        self.side["refusal_control"] = {"unsteered_harmful_refusal": harm_rate, "unsteered_benign_refusal": ben_rate,
                                        "per_layer": per_layer, "L_ref": L_ref, "passed": passed, "variants": variants}
        self.side["instr_unsteered"] = {"harmful": harm_flags, "benign": ben_flags,
                                        "harmful_texts": harm_texts, "benign_texts": ben_texts}
        for vname, v in variants.items():
            print(f"REFUSAL_CONTROL model={self.name} variant={vname} L_ref={v['L_ref']} unsteered_refusal={harm_rate:.3f} "
                  f"ablated_refusal={v['per_layer'][v['L_ref']]['refusal_rate']:.3f} passed={v['passed']}", flush=True)
        self.timer.mark(f"refusal_direction[{self.name}]")

    # ------------------------------------------------------------------ caps and hook diagnostics
    def dev_baselines_for_caps(self):
        """Amendment A11: unsteered MMLU-dev predictions and reference-text log-likelihood (dev)."""
        rows = self.data["mmlu_dev"]
        self._mmlu_dev_P = [self.fmt.prompt(USER_TEMPLATE_MMLU.format(question=r["question"], a=r["choices"][0],
                                                                      b=r["choices"][1], c=r["choices"][2],
                                                                      d=r["choices"][3])) for r in rows]
        self._letters = [self.tok(x, add_special_tokens=False).input_ids[0] for x in "ABCD"]
        gold = np.array([r["answer"] for r in rows])
        logits = mu.last_position_logits(self.model, self.fmt, self._mmlu_dev_P)
        ref = self.data["ref"]["dev"]
        self._ref_dev = ([self.fmt.prompt(USER_TEMPLATE_REFERENCE.format(title=x["title"])) for x in ref],
                         [self.tok(x["text"], add_special_tokens=False).input_ids[:PPL_MAX_RESPONSE_TOKENS] for x in ref])
        lps = mu.teacher_force(self.model, self.fmt, *self._ref_dev)
        self.caps_base = {"mmlu_correct": logits[:, self._letters].argmax(1).numpy() == gold, "mmlu_gold": gold,
                          "ref_nll": -float(np.concatenate(lps).mean())}
        self.timer.mark(f"cap_baselines[{self.name}]")

    def caps(self, L, v, alpha):
        """MMLU-dev accuracy delta (points, hook at the last prompt token) and reference-text
        perplexity ratio (hook at the last prompt token + response tokens) for adding alpha * v at L."""
        base = self.caps_base
        sq = SteeredQA(self)
        sq.register_hook(Intervention("cap_mmlu", adds=[(L, alpha * unit(v), "prompt_only")]))
        try:
            logits = mu.last_position_logits(self.model, self.fmt, self._mmlu_dev_P)
        finally:
            sq.remove_hook()
        correct = logits[:, self._letters].argmax(1).numpy() == base["mmlu_gold"]
        sq.register_hook(steer("cap_ppl", L, unit(v), alpha))
        try:
            lps = mu.teacher_force(self.model, self.fmt, *self._ref_dev, mask_hooks=sq._add_hooks)
        finally:
            sq.remove_hook()
        nll = -float(np.concatenate(lps).mean())
        return {"mmlu_delta": 100.0 * (correct.mean() - base["mmlu_correct"].mean()),
                "ppl_ratio": float(np.exp(nll - base["ref_nll"]))}

    def first_token_kl(self, iv):
        """Mean KL(unsteered || steered) of the next-token distribution at the last prompt token on
        the first DIAG_N_PROMPTS dev prompts (amendment A5 no-op diagnostic)."""
        batch = self.prompts["dev"][:DIAG_N_PROMPTS]
        if self._diag_base is None:
            self._diag_base = mu.last_position_logits(self.model, self.fmt, batch)
        sq = SteeredQA(self)
        sq.register_hook(iv)
        try:
            st = mu.last_position_logits(self.model, self.fmt, batch)
        finally:
            sq.remove_hook()
        p, q = F.log_softmax(self._diag_base, -1), F.log_softmax(st, -1)
        return float((p.exp() * (p - q)).sum(-1).mean()), sq.last_hook_calls

    def _sweep_entry(self, o, base, iv, kl=True):
        frac = float(np.mean([a != b for a, b in zip(o.texts, self.uout["dev"].texts)]))
        e = {"S1": abstention_score(o.states), "degenerate_rate": float(o.degenerate.mean()),
             "abst_increase": 100 * (_abst(o.states) - _abst(base)), "W_to_C": int(((base == W_) & (o.states == C_)).sum()),
             "alpha": float(iv.adds[0][1].norm()), "frac_changed": frac, "hook_calls": o.extra["hook_calls"]}
        if kl:
            e["kl_first_token"], _ = self.first_token_kl(iv)
            e["no_op"] = bool(frac < NO_OP_MAX_FRAC_CHANGED and e["kl_first_token"] < NO_OP_MAX_KL)
        return e

    # ------------------------------------------------------------------ dev selection
    def dev_sweep(self):
        base = self.uout["dev"].states
        k = KnowledgeDirectionSteer(self)
        sweep = {}
        for L in self.cand:
            for c in C_GRID:
                a = c * self.sigma["knowledge"][L]
                iv = k.intervention(L, c, direction=self.dirs["knowledge"][L], alpha=a)
                o = self.run_steered(iv, "dev", f"knowledge_l{L}_c{c:g}", check_zero=False)
                self.dev_outputs[f"knowledge_l{L}_c{c:g}"] = o
                sweep[(L, c)] = self._sweep_entry(o, base, iv)
                sweep[(L, c)]["alpha_over_rbar"] = a / float(self.rbar[L])
        L_star, c_star, rule = k.select_config_on_dev(
            sweep, self.cand, caps=lambda L, c: self.caps(L, self.dirs["knowledge"][L], c * self.sigma["knowledge"][L]))
        alpha = c_star * self.sigma["knowledge"][L_star]
        self.frozen.update({"seed": self.seed, "L_star": L_star, "c_star": c_star, "selection_rule_outcome": rule,
                            "strength_unit": "sigma_L(D_ku)", "sigma_L_star": self.sigma["knowledge"][L_star],
                            "rbar_L_star": float(self.rbar[L_star]), "alpha": alpha,
                            "alpha_over_rbar": alpha / float(self.rbar[L_star])})
        print(f"DEV_SELECTION model={self.name} seed={self.seed} L_star={L_star} c_star={c_star} rule={rule} "
              f"alpha={alpha:.3f} alpha_over_rbar={alpha / float(self.rbar[L_star]):.3f}", flush=True)
        self.dev["knowledge_sweep"] = {f"l{L}_c{c:g}": v for (L, c), v in sweep.items()}
        self._knowledge_dev_out = self.dev_outputs[f"knowledge_l{L_star}_c{c_star:g}"]
        if self.seed == 0:
            for (L, c), v in sweep.items():
                print(f"HOOK_DIAGNOSTIC model={self.name} dir=knowledge L={L} c={c:g} alpha={v['alpha']:.3f} "
                      f"alpha_over_rbar={v['alpha_over_rbar']:.3f} calls_prefill={v['hook_calls'].get('prefill', 0)} "
                      f"calls_decode={v['hook_calls'].get('decode', 0)} frac_changed={v['frac_changed']:.3f} "
                      f"kl_first_token={v['kl_first_token']:.5f} abst_increase={v['abst_increase']:.2f} "
                      f"no_op={v['no_op']}", flush=True)
            self._descriptive_sweeps(base)

        # thresholds (tau_dev(lambda)); seeds >= 1 refit only what the seeded arms use
        self.frozen["tau"] = {}
        sigs = ConfidenceThreshold.SIGNALS if self.seed == 0 else ("margin",)
        for sig in sigs:
            ct = ConfidenceThreshold(sig)
            self.frozen["tau"][sig] = ct.fit_threshold_on_dev(base, ct.compute_signal(self, "dev"))
        pg = PopularityGate()
        self.frozen["tau"]["popularity"] = pg.fit_threshold_on_dev(base, pg.compute_signal(self, "dev"))
        if self.seed == 0:
            # amendment A10: TriviaQA margin threshold from its own dev split
            self.frozen["tau"]["tqa_margin"] = {lam: fit_threshold(self.uout["tqa_dev"].states, self.margin["tqa_dev"], lam)
                                                for lam in LAMBDAS}
            self._sample_vote_thresholds()
            self._dev_cov_matched(sweep)
            self.frozen["L_alt"] = sorted(self.cand, key=lambda L: (-abs(L - L_star), L))[0]
        self.timer.mark(f"dev_tuning[{self.name}][seed{self.seed}]")

    def _descriptive_sweeps(self, base):
        """Descriptive only (registered): D_wq and refusal at the shared positions, with the same
        absolute norms as the knowledge sweep (c * sigma_L(D_ku))."""
        wq = WithinQuestionDirectionSteer(self)
        wq_sweep = {}
        for L in self.cand:
            for c in C_GRID:
                iv = wq.intervention(L, c, direction=self.dirs["wq"][L], alpha=c * self.sigma["knowledge"][L])
                o = self.run_steered(iv, "dev", f"wq_l{L}_c{c:g}", check_zero=False)
                self.dev_outputs[f"wq_l{L}_c{c:g}"] = o
                wq_sweep[f"l{L}_c{c:g}"] = self._sweep_entry(o, base, iv, kl=(c == C_GRID[-1]))
        self.dev["wq_sweep_descriptive"] = wq_sweep
        rf = RefusalDirectionSteer(self)
        L = self.frozen["L_star"]
        ref_sweep = {}
        for c in C_GRID:
            iv = rf.intervention(L, c, alpha=self.site_alpha(c))
            o = self.run_steered(iv, "dev", f"refusal_c{c:g}", check_zero=False)
            self.dev_outputs[f"refusal_c{c:g}"] = o
            ref_sweep[f"c{c:g}"] = self._sweep_entry(o, base, iv, kl=False)
        self.dev["refusal_sweep_descriptive"] = ref_sweep

    def _dev_subset_base(self):
        sub = self.dev_subset
        sub_gen = mu.generate(self.model, self.fmt, [self.prompts["dev"][i] for i in sub], QA_MAX_NEW_TOKENS)
        o = make_output("unsteered", sub_gen, [self.items["dev"][i] for i in sub])
        self.dev_outputs["unsteered_dev_subset"] = o
        return o

    def _knowledge_target(self, sub_base):
        sub = self.dev_subset
        return 100 * (_abst(self._knowledge_dev_out.states[sub]) - _abst(sub_base.states))

    def _dev_cov_matched(self, sweep):
        """Registered isotropic coverage-matched draws (multipliers {1, 2, 4} x alpha*)."""
        sub_base = self._dev_subset_base()
        self._sub_base = sub_base
        target = self._knowledge_target(sub_base)
        rnd = RandomDirectionPlacebo(self, N_COV_MATCHED_DRAWS)
        cov = {}
        for kdraw in range(N_COV_MATCHED_DRAWS):
            res = {}
            for m in COV_MATCH_MULTIPLIERS:
                o = self.run_steered(rnd.intervention(kdraw, m), "dev", f"rnd{kdraw}_x{m}", idx=self.dev_subset,
                                     check_zero=False)
                self.dev_outputs[f"cov_matched_s{kdraw}_x{m}"] = o
                res[m] = {"abst_increase": 100 * (_abst(o.states) - _abst(sub_base.states)),
                          "degenerate_rate": float(o.degenerate.mean())}
            # clean = degenerate <= 10%; a draw is matched if some clean multiplier reaches the target
            # dose (closest clean multiplier frozen); otherwise the largest clean multiplier, unmatched.
            ok = {m: v for m, v in res.items() if v["degenerate_rate"] <= COV_MATCH_MAX_DEGENERATE}
            reached = [m for m, v in ok.items() if v["abst_increase"] >= target]
            if reached:
                m_sel, matched = min(ok, key=lambda m: (abs(ok[m]["abst_increase"] - target), m)), True
            elif ok:
                m_sel, matched = max(ok), False
            else:
                m_sel, matched = min(COV_MATCH_MULTIPLIERS), False
            cov[kdraw] = {"multiplier": m_sel, "matched": matched, "per_multiplier": res}
        self.frozen["cov_matched"] = {"target_abst_increase": target, "draws": cov}

    def dev_nulls(self):
        """Amendment A9: dose-matched nulls, multiplier of alpha* matched to knowledge_dir_steer's dev
        abstention increase on the dev subset (clean = degenerate <= 10%; matched = within +/- 3 points)."""
        if self._sub_base is None:
            self._sub_base = self._dev_subset_base()
        sub_base = self._sub_base
        target = self._knowledge_target(sub_base)
        L, a = self.frozen["L_star"], self.frozen["alpha"]
        self.dirs["nulls"] = DoseMatchedNull.directions(self, self.seed)
        info = {"target_abst_increase": target}
        for t in NULL_TYPES:
            v = self.dirs["nulls"][t]
            res = {}
            for m in NULL_MULTIPLIERS:
                o = self.run_steered(steer(f"null_{t}_x{m:g}", L, v, m * a), "dev", f"null_{t}_x{m:g}",
                                     idx=self.dev_subset, check_zero=False)
                self.dev_outputs[f"null_{t}_x{m:g}"] = o
                res[m] = {"abst_increase": 100 * (_abst(o.states) - _abst(sub_base.states)),
                          "degenerate_rate": float(o.degenerate.mean()), "S1": abstention_score(o.states)}
            ok = {m: r for m, r in res.items() if r["degenerate_rate"] <= COV_MATCH_MAX_DEGENERATE}
            if ok:
                m_sel = min(ok, key=lambda m: (abs(ok[m]["abst_increase"] - target), m))
                matched = abs(ok[m_sel]["abst_increase"] - target) <= DOSE_MATCH_TOL
            else:
                m_sel, matched = min(NULL_MULTIPLIERS), False
            info[t] = {"multiplier": m_sel, "matched": matched, "per_multiplier": res,
                       "cos_with_knowledge": cosine(v, self.dirs["knowledge"][L])}
            print(f"DOSE_MATCHED_NULL model={self.name} seed={self.seed} null={t} multiplier={m_sel:g} matched={matched} "
                  f"target={target:.2f} achieved={res[m_sel]['abst_increase']:.2f}", flush=True)
        self.frozen["nulls"] = info

    def dev_wq_answer_ladder(self):
        """Amendment A9: D_wq at answer positions, own layer x strength ladder (c * sigma^ans_L(D_wq));
        highest dev S1 with degenerate <= 5% and the caps, ties -> smaller c then middle layer; plus a
        norm-matched random ladder at the selected layer (descriptive, dev subset)."""
        base = self.uout["dev"].states
        wq = WithinQuestionDirectionSteer(self)
        sweep = {}
        for L in self.cand:
            for c in WQ_ANSWER_C_GRID:
                a = c * self.sigma["wq_answer"][L]
                iv = wq.answer_position_intervention(L, a, self.dirs["wq"][L], f"wq_answer_l{L}_c{c:g}")
                o = self.run_steered(iv, "dev", f"wq_answer_l{L}_c{c:g}", check_zero=False)
                self.dev_outputs[f"wq_answer_l{L}_c{c:g}"] = o
                sweep[(L, c)] = self._sweep_entry(o, base, iv, kl=False)
        middle = self.cand[len(self.cand) // 2]
        clean = [k for k, v in sweep.items() if v["degenerate_rate"] <= DEV_ELIGIBLE_MAX_DEGENERATE]
        elig = []
        for k in sorted(clean, key=lambda k: -sweep[k]["S1"]):
            if elig and sweep[k]["S1"] < sweep[elig[0]]["S1"] - DEV_TIE_POINTS:
                break                       # below the tie window of the best eligible configuration
            cp = sweep[k]["caps"] = self.caps(k[0], self.dirs["wq"][k[0]], sweep[k]["alpha"])
            cp["passed"] = bool(cp["mmlu_delta"] >= MMLU_CAP_POINTS and cp["ppl_ratio"] <= PPL_CAP_RATIO)
            if cp["passed"]:
                elig.append(k)
        if elig:
            best = max(sweep[k]["S1"] for k in elig)
            L, c = sorted([k for k in elig if sweep[k]["S1"] >= best - DEV_TIE_POINTS],
                          key=lambda k: (k[1], k[0] != middle))[0]
            rule = "eligible"
        else:
            L, c, rule = middle, WQ_ANSWER_C_GRID[0], "no clean operating point"
        alpha = c * self.sigma["wq_answer"][L]
        self.frozen["wq_answer"] = {"L": L, "c": c, "alpha": alpha, "rule": rule, "positions": "generated_only",
                                    "strength_unit": "sigma^answer_L(D_wq)"}
        self.dev["wq_answer_sweep"] = {f"l{k[0]}_c{k[1]:g}": v for k, v in sweep.items()}
        print(f"DEV_SELECTION_WQ_ANSWER model={self.name} L={L} c={c:g} alpha={alpha:.3f} rule={rule}", flush=True)
        if self._sub_base is None:
            self._sub_base = self._dev_subset_base()
        sub_base = self._sub_base
        dirs = random_unit_directions(self.spec["d_model"], N_WQ_ANSWER_RANDOM, WQ_ANSWER_RANDOM_SEED_BASE)
        rl = {}
        for cc in WQ_ANSWER_C_GRID:
            a = cc * self.sigma["wq_answer"][L]
            for j, v in enumerate(dirs):
                o = self.run_steered(steer(f"wq_ans_rand{j}", L, v, a, "generated_only"), "dev",
                                     f"wq_answer_random_s{j}_c{cc:g}", idx=self.dev_subset, check_zero=False)
                self.dev_outputs[f"wq_answer_random_s{j}_c{cc:g}"] = o
                rl[f"s{j}_c{cc:g}"] = {"S1": abstention_score(o.states),
                                       "W_to_C": int(((sub_base.states == W_) & (o.states == C_)).sum()),
                                       "degenerate_rate": float(o.degenerate.mean())}
        self.dev["wq_answer_random_ladder"] = rl
        self.timer.mark(f"dev_wq_answer_ladder[{self.name}]")

    def _sample_vote_thresholds(self):
        """Amendment A9: tau_dev(lambda) for the majority-vote and best-of-k gates (dev samples)."""
        sv = SampleVoteQA(self)
        maj = sv.majority_outputs("dev")
        bok, _ = sv.best_of_k_outputs("dev")
        self.frozen["tau"]["majority_vote_k8"] = {lam: fit_threshold(maj.states, maj.confidence, lam) for lam in LAMBDAS}
        self.frozen["tau"]["best_of_k8"] = {lam: fit_threshold(bok.states, bok.confidence, lam) for lam in LAMBDAS}

    # ------------------------------------------------------------------ freezing
    def frozen_entry(self):
        L = self.frozen["L_star"]
        entry = dict(self.frozen)
        entry["directions_at_L_star"] = {k: self.dirs[k][L].tolist() for k in ("knowledge", "read", "wq", "refusal")}
        entry["direction_wq_answer"] = self.dirs["wq"][self.frozen["wq_answer"]["L"]].tolist()
        entry.update({"model": self.name, "hf_id": self.spec["hf_id"], "revision": self.revision,
                      "tau_store": self.tau_store, "budget_flags": list(self.budget.flags),
                      "dev_selection_caps": {"mmlu_delta_min": MMLU_CAP_POINTS, "ppl_ratio_max": PPL_CAP_RATIO}})
        entry["seeds"] = {}
        for s, f in self.seed_runs.items():
            e = dict(f.frozen)
            e["directions_at_L_star"] = {k: f.dirs[k][f.frozen["L_star"]].tolist() for k in ("knowledge", "read", "refusal")}
            entry["seeds"][str(s)] = e
        blob = json.dumps(entry, sort_keys=True, default=str)
        entry["sha256"] = hashlib.sha256(blob.encode()).hexdigest()
        return entry

    # ------------------------------------------------------------------ phase B: integrity
    def integrity_checks(self):
        L = self.frozen["L_star"]
        batch = self.prompts["dev"][:64]
        sq = SteeredQA(self)
        before = mu.last_position_logits(self.model, self.fmt, batch)
        sq.register_hook(steer("zero", L, torch.zeros(self.spec["d_model"]), 0.0))
        try:
            zero = mu.last_position_logits(self.model, self.fmt, batch)
        finally:
            sq.remove_hook()
        after = mu.last_position_logits(self.model, self.fmt, batch)
        diff = max(float((zero - before).abs().max()), float((after - before).abs().max()))
        self.integrity["hook_reset_max_abs_logit_diff"] = diff
        print(f"HOOK_RESET_CHECK: max_abs_logit_diff={diff:.6g}", flush=True)

        ivs = {"unsteered": None,
               "knowledge_dir_steer": KnowledgeDirectionSteer(self).intervention(),
               "read_dir_steer": ReadDirectionSteer(self).intervention(),
               "within_question_dir_steer": WithinQuestionDirectionSteer(self).intervention(),
               "refusal_dir_steer": RefusalDirectionSteer(self).intervention(),
               "random_dir_placebo": steer("rnd0", L, self.random_dirs[0], self.frozen["alpha"]),
               "refusal_dir_ablation": Intervention("ablate", ablate=self.dirs["refusal"][self.frozen["L_ref"]])}
        logits = {}
        for name, iv in ivs.items():
            if iv is None:
                logits[name] = before
                continue
            sq.register_hook(iv)
            try:
                logits[name] = mu.last_position_logits(self.model, self.fmt, batch)
            finally:
                sq.remove_hook()
        names = list(ivs)
        self.integrity["ablation_check"] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                differ = bool((logits[names[i]] - logits[names[j]]).abs().max() > 0)
                self.integrity["ablation_check"][f"{names[i]}|{names[j]}"] = differ
                print(f"ABLATION_CHECK: {names[i]} vs {names[j]} outputs_differ={differ}", flush=True)

        # zero-vector check on the full PopQA test set (batch non-determinism guard)
        o = self.run_steered(steer("zero_vector", L, torch.zeros(self.spec["d_model"]), 0.0), "test",
                             "zero_vector", check_zero=False)
        same = float(np.mean([a == b for a, b in zip(o.texts, self.uout["test"].texts)]))
        self.integrity["zero_vector_identical_frac"] = same
        self.zero_check_failed = same < ZERO_VECTOR_MIN_IDENTICAL
        print(f"ZERO_VECTOR_CHECK model={self.name} identical_frac={same:.4f} "
              f"passed={not self.zero_check_failed}", flush=True)
        # amendment A5: share of abstentions written exactly as the reference string
        ab = [t for t, s in zip(self.uout["test"].texts, self.uout["test"].states) if s == A_]
        self.integrity["idk_exact_form_share"] = {"I don't know.": float(np.mean([t == "I don't know." for t in ab])) if ab else float("nan"),
                                                  "I don't know": float(np.mean([t == "I don't know" for t in ab])) if ab else float("nan")}
        print(f"IDK_FORM_CHECK model={self.name} " + " ".join(f"'{k}'={v:.3f}" for k, v in
                                                            self.integrity["idk_exact_form_share"].items()), flush=True)
        self.timer.mark(f"integrity_checks[{self.name}]")

    # ------------------------------------------------------------------ registry conditions (PopQA test)
    def cond_unsteered(self):
        return {"unsteered": self.uout["test"]}

    def cond_always_abstain(self):
        base = self.uout["test"]
        return {"always_abstain": policy_output("always_abstain", base, np.full(len(base), A_, dtype=np.int8))}

    def cond_confidence_prompt(self):
        out = {}
        for inst in CONFIDENCE_PROMPT_INSTANCES:
            self.guard("test")
            cp = ConfidencePromptQA(self, inst)
            out[inst] = cp.classify_output(cp.answer(self.items["test"]), self.items["test"])
        return out

    def _threshold_policy(self, name, base, sig, taus):
        return policy_output(name, base, apply_threshold(base.states, sig, taus[LAMBDA_PRIMARY]),
                             extra={"tau": taus[LAMBDA_PRIMARY],
                                    "lambda_states": {lam: apply_threshold(base.states, sig, t) for lam, t in taus.items()}})

    def cond_confidence_threshold_margin(self):
        return {"confidence_threshold_margin": self._threshold_policy(
            "confidence_threshold_margin", self.uout["test"], self.margin["test"], self.frozen["tau"]["margin"])}

    def cond_popularity_gate(self):
        pg = PopularityGate()
        return {"popularity_gate": self._threshold_policy("popularity_gate", self.uout["test"],
                                                          pg.compute_signal(self, "test"), self.frozen["tau"]["popularity"])}

    def cond_sample_vote(self):
        """Amendment A9: majority vote and best-of-k over the 8 saved test samples, gated at tau_dev(1)."""
        self.guard("test")
        sv = SampleVoteQA(self)
        maj = sv.majority_outputs("test")
        bok, lp = sv.best_of_k_outputs("test")
        self.extras["best_of_k_sample_logprobs"] = lp
        out = {}
        for name, o in (("majority_vote_k8", maj), ("best_of_k8", bok)):
            out[name] = self._threshold_policy(name, o, o.confidence, self.frozen["tau"][name])
            out[name].confidence = o.confidence
            out[name].extra["ungated_states"] = o.states
        return out

    def _primary_steer(self, cls, name):
        return {name: self.run_steered(cls(self).intervention(), "test", name, steered_margin=True)}

    def cond_knowledge_dir_steer(self):
        return self._primary_steer(KnowledgeDirectionSteer, "knowledge_dir_steer")

    def cond_read_dir_steer(self):
        return self._primary_steer(ReadDirectionSteer, "read_dir_steer")

    def cond_within_question_dir_steer(self):
        out = self._primary_steer(WithinQuestionDirectionSteer, "within_question_dir_steer")
        L, a = self.frozen["L_star"], self.frozen["alpha"]
        for r in range(1, N_WQ_REFITS):   # amendment A8: refits at the same site and norm
            out[f"within_question_dir_steer_r{r}"] = self.run_steered(
                steer(f"wq_r{r}", L, self.dirs["wq_refits"][r][L], a), "test", f"within_question_dir_steer_r{r}")
        return out

    def cond_wq_answer_positions(self):
        f = self.frozen["wq_answer"]
        wq = WithinQuestionDirectionSteer(self)
        out = {}
        for r in range(N_WQ_REFITS):
            name = "within_question_dir_steer_answer_positions" + ("" if r == 0 else f"_r{r}")
            iv = wq.answer_position_intervention(f["L"], f["alpha"], self.dirs["wq_refits"][r][f["L"]], name)
            out[name] = self.run_steered(iv, "test", name)
        dirs = random_unit_directions(self.spec["d_model"], N_WQ_ANSWER_RANDOM, WQ_ANSWER_RANDOM_SEED_BASE)
        for j in range(N_WQ_ANSWER_RANDOM if self.budget.allow("wq_answer_random_1") else 1):
            name = f"wq_answer_random_s{j}"
            out[name] = self.run_steered(steer(name, f["L"], dirs[j], f["alpha"], "generated_only"), "test", name)
        return out

    def cond_refusal_dir_steer(self):
        return self._primary_steer(RefusalDirectionSteer, "refusal_dir_steer")

    def cond_margin_gated_steer(self):
        base = self.uout["test"]
        gate = MarginGatedSteer.compute_gate(base.states, self.margin["test"], self.frozen["tau"]["margin"][LAMBDA_PRIMARY])
        out = {}
        for src, inst in (("knowledge_dir_steer", "margin_gated_steer_knowledge"),
                          ("within_question_dir_steer", "margin_gated_steer_wq")):
            if src not in self.test_out:
                raise RuntimeError(f"{src} test output missing; cannot compose {inst}")
            out[inst] = MarginGatedSteer.compose_outputs(inst, base, self.test_out[src], gate)
        return out

    def cond_dose_matched_null(self):
        L, a = self.frozen["L_star"], self.frozen["alpha"]
        out = {}
        for t in NULL_TYPES:
            info = self.frozen["nulls"][t]
            name = f"dose_matched_null_{t}"
            o = self.run_steered(steer(name, L, self.dirs["nulls"][t], info["multiplier"] * a), "test", name)
            o.extra.update({"multiplier": info["multiplier"], "matched_on_dev": info["matched"]})
            out[name] = o
        return out

    def cond_random_dir_placebo(self):
        rnd = RandomDirectionPlacebo(self, N_RANDOM_DRAWS)
        return {f"random_dir_placebo_s{k}": self.run_steered(rnd.intervention(k), "test", f"random_dir_placebo_s{k}")
                for k in range(N_RANDOM_DRAWS)}

    # ------------------------------------------------------------------ seed arms (amendment A8)
    def test_seed_arms(self, parent):
        """Seed s >= 1: the H1a-critical arms at this seed's own frozen site."""
        self.sync_shared(parent)
        base = self.uout["test"]
        out = {"unsteered": base}
        out.update(self.cond_confidence_threshold_margin())
        for cls, name in ((KnowledgeDirectionSteer, "knowledge_dir_steer"), (ReadDirectionSteer, "read_dir_steer"),
                          (RefusalDirectionSteer, "refusal_dir_steer")):
            out.update(self._primary_steer(cls, name))
        gate = MarginGatedSteer.compute_gate(base.states, self.margin["test"], self.frozen["tau"]["margin"][LAMBDA_PRIMARY])
        out["margin_gated_steer_knowledge"] = MarginGatedSteer.compose_outputs(
            "margin_gated_steer_knowledge", base, out["knowledge_dir_steer"], gate)
        out.update(self.cond_dose_matched_null())
        for name, o in out.items():
            if name != "unsteered":
                print(f"PROGRESS: model={self.name} seed={self.seed} condition={name} S1={abstention_score(o.states):.2f} "
                      f"abstention_rate={_abst(o.states):.3f}", flush=True)
        return out

    # ------------------------------------------------------------------ phase C: ablations
    def abl_knowledge_dose_ladder(self):
        k = KnowledgeDirectionSteer(self)
        return {f"knowledge_dir_steer_c{c:g}": self.run_steered(k.intervention(c=c), "test", f"knowledge_dir_steer_c{c:g}")
                for c in C_GRID if c != self.frozen["c_star"]}

    def abl_within_question_dose_ladder(self):
        w = WithinQuestionDirectionSteer(self)
        return {f"within_question_dir_steer_c{c:g}": self.run_steered(w.intervention(c=c), "test", f"within_question_dir_steer_c{c:g}")
                for c in C_GRID if c != self.frozen["c_star"]}

    def abl_knowledge_last_prompt_only(self):
        iv = KnowledgeDirectionSteer(self).intervention(positions="prompt_only")
        return {"knowledge_dir_steer_last_prompt_only": self.run_steered(iv, "test", "knowledge_dir_steer_last_prompt_only")}

    def abl_knowledge_generated_only(self):
        iv = KnowledgeDirectionSteer(self).intervention(positions="generated_only")
        return {"knowledge_dir_steer_generated_only": self.run_steered(iv, "test", "knowledge_dir_steer_generated_only")}

    def abl_knowledge_alt_layer(self):
        if not self.budget.allow("drop_alt_layer"):
            print(f"BUDGET_SKIP: knowledge_dir_steer_alt_layer model={self.name} (degradation ladder)", flush=True)
            return {}
        iv = KnowledgeDirectionSteer(self).intervention(layer=self.frozen["L_alt"], c=self.frozen["c_star"])
        return {"knowledge_dir_steer_alt_layer": self.run_steered(iv, "test", "knowledge_dir_steer_alt_layer")}

    def abl_between_question_answer_dir(self):
        L = self.frozen["L_star"]
        iv = steer("bq", L, self.dirs["bq"][L], self.frozen["alpha"])
        return {"between_question_answer_dir_steer": self.run_steered(iv, "test", "between_question_answer_dir_steer")}

    def abl_label_permuted_wq(self):
        L, a = self.frozen["L_star"], self.frozen["alpha"]
        n = N_LABEL_PERM_DRAWS if self.budget.allow("label_perm_3") else min(3, N_LABEL_PERM_DRAWS)
        return {f"label_permuted_wq_dir_steer_s{j}": self.run_steered(steer(f"wq_perm{j}", L, self.dirs["wq_perm"][j][L], a),
                                                                      "test", f"label_permuted_wq_dir_steer_s{j}")
                for j in range(n)}

    def abl_random_coverage_matched(self):
        rnd = RandomDirectionPlacebo(self, N_COV_MATCHED_DRAWS)
        n = N_COV_MATCHED_DRAWS if self.budget.allow("cov_matched_3") else min(3, N_COV_MATCHED_DRAWS)
        out = {}
        for k in range(n):
            info = self.frozen["cov_matched"]["draws"][k]
            o = self.run_steered(rnd.intervention(k, info["multiplier"]), "test", f"random_dir_placebo_coverage_matched_s{k}")
            o.extra.update({"multiplier": info["multiplier"], "matched_on_dev": info["matched"]})
            out[f"random_dir_placebo_coverage_matched_s{k}"] = o
        return out

    def _ablation_iv(self, vname, r_hat=None, with_steer=False, means=None):
        v = self.frozen["refusal_ablation"][vname]
        L_ref = v["L_ref"]
        r = self.dirs["refusal"][L_ref] if r_hat is None else r_hat
        if v["mode"] == "mean" and means is None:
            means = self.refusal_means[L_ref]
        adds = []
        if with_steer:
            L = self.frozen["L_star"]
            adds = [(L, self.frozen["alpha"] * self.dirs["knowledge"][L], "both")]
        return Intervention(vname + ("_plus_knowledge_steer" if with_steer else ""), adds=adds, ablate=r,
                            ablate_mode=v["mode"], ablate_layers=None if v["layers"] == "all" else [L_ref],
                            ablate_means=means)

    def abl_refusal_ablation(self):
        out = {}
        for vname in REFUSAL_ABLATION_VARIANTS:
            out[vname] = self.run_steered(self._ablation_iv(vname), "test", vname)
        return out

    def abl_refusal_ablation_plus_knowledge(self):
        out = {}
        for vname in REFUSAL_ABLATION_VARIANTS:
            name = f"{vname}_plus_knowledge_steer"
            out[name] = self.run_steered(self._ablation_iv(vname, with_steer=True), "test", name)
        return out

    def abl_random_ablation(self):
        reduced = not self.budget.allow("random_ablation_reduced")
        out = {}
        harmless = [self.fmt.prompt(x) for x in self.data["instr"]["alpaca_train"]]
        for vname, spec in REFUSAL_ABLATION_VARIANTS.items():
            n = self.budget.n_random_ablation(vname) if reduced else N_RANDOM_ABLATION_DRAWS[vname]
            dirs = random_unit_directions(self.spec["d_model"], n, RANDOM_ABLATION_SEED_BASE)
            prefix = "random_dir_ablation" if vname == "refusal_dir_ablation" else vname.replace("refusal_dir", "random_dir")
            for j in range(n):
                means = measure_mean_projections(self.model, self.fmt, harmless, dirs[j]) if spec["mode"] == "mean" else None
                out[f"{prefix}_s{j}"] = self.run_steered(self._ablation_iv(vname, r_hat=dirs[j], means=means), "test",
                                                         f"{prefix}_s{j}")
        return out

    def abl_first_token_ban(self):
        self.guard("test")
        base = self.uout["test"]
        ban = np.array([t[0] if (s != A_ and len(t) > 0) else -1
                        for t, s in zip(self.ugen["test"].token_ids, base.states)])
        gen = mu.generate(self.model, self.fmt, self.prompts["test"], QA_MAX_NEW_TOKENS, ban_first=ban)
        o = make_output("first_token_ban_regenerate", gen, self.items["test"])
        banned = ban >= 0
        o.extra.update({"banned": banned,
                        "failed_knockout": np.array([b and normalize_answer(a2) == normalize_answer(a1)
                                                     for b, a1, a2 in zip(banned, base.texts, gen.texts)]),
                        "unbanned_rows_identical": float(np.mean([a == b for a, b, x in zip(gen.texts, base.texts, banned) if not x]))
                        if (~banned).any() else float("nan")})
        return {"first_token_ban_regenerate": o}

    def abl_teacher_forced_dose_response(self):
        self.guard("test")
        base = self.uout["test"]
        rng = np.random.default_rng(0)
        idx = []
        for d in range(10):
            cand = [i for i, it in enumerate(self.items["test"]) if it.decile == d and base.states[i] != A_
                    and len(self.ugen["test"].token_ids[i]) > 0]
            idx += [cand[j] for j in rng.permutation(len(cand))[:DOSE_ITEMS_PER_DECILE]]
        idx = sorted(idx)
        P = [self.prompts["test"][i] for i in idx]
        conts_a = [list(self.ugen["test"].token_ids[i]) + [self.fmt.eot_id] for i in idx]
        L = self.frozen["L_star"]
        sq = SteeredQA(self)
        none = Intervention("none")
        m0 = sq.teacher_forced_logprob(none, P, conts_a) - sq.idk_logprob(none, P)
        dirs = {"knowledge_dir": self.dirs["knowledge"][L], "random_s0": self.random_dirs[0], "random_s1": self.random_dirs[1]}
        res = {}
        for dname, v in dirs.items():
            M = [m0]
            for c in DOSE_C_GRID[1:]:
                iv = steer(f"dose_{dname}_{c}", L, v, self.site_alpha(c))   # same absolute norm for every direction
                M.append(sq.teacher_forced_logprob(iv, P, conts_a) - sq.idk_logprob(iv, P))
            res[dname] = dose_statistics(np.stack(M, 1), np.array(DOSE_C_GRID))
            res[dname]["margins"] = np.stack(M, 1)
        agree = {}
        for c in C_GRID:
            name = "knowledge_dir_steer" if c == self.frozen["c_star"] else f"knowledge_dir_steer_c{c:g}"
            src = self.test_out.get(name, self.ablation_out.get(name))
            if src is None or c not in DOSE_C_GRID:
                continue
            pred = res["knowledge_dir"]["margins"][:, DOSE_C_GRID.index(c)] < 0
            actual = src.states[idx] == A_
            agree[f"c{c:g}"] = float((pred == actual).mean())
        res["generation_flip_agreement"] = agree
        res["item_indices"] = idx
        self.extras["teacher_forced_dose_response"] = res
        return {}

    # ------------------------------------------------------------------ phase C: EntityQuestions
    def entity_questions(self):
        items = self.items["eq"]
        out = {"unsteered": self.uout["eq"]}
        self.guard("eq")
        cp = ConfidencePromptQA(self, "confidence_prompt_t050")
        out["confidence_prompt_t050"] = cp.classify_output(cp.answer(items), items)
        for cls, name in ((KnowledgeDirectionSteer, "knowledge_dir_steer"), (ReadDirectionSteer, "read_dir_steer"),
                          (WithinQuestionDirectionSteer, "within_question_dir_steer"),
                          (RefusalDirectionSteer, "refusal_dir_steer")):
            out[name] = self.run_steered(cls(self).intervention(), "eq", name)
        base = self.uout["eq"]
        gate = MarginGatedSteer.compute_gate(base.states, self.margin["eq"], self.frozen["tau"]["margin"][LAMBDA_PRIMARY])
        out["confidence_threshold_margin"] = policy_output("confidence_threshold_margin", base,
                                                           np.where(gate, A_, base.states))
        out["margin_gated_steer_wq"] = MarginGatedSteer.compose_outputs("margin_gated_steer_wq", base,
                                                                        out["within_question_dir_steer"], gate)
        out["margin_gated_steer_knowledge"] = MarginGatedSteer.compose_outputs("margin_gated_steer_knowledge", base,
                                                                               out["knowledge_dir_steer"], gate)
        self.budget.allow("eq_subset_150")
        self.budget.check_emergency()
        sub = eq_subset_indices(items, self.budget.eq_subset_n)
        sub_gen = mu.generate(self.model, self.fmt, [self.prompts["eq"][i] for i in sub], QA_MAX_NEW_TOKENS)
        sub_out = {"unsteered": make_output("unsteered", sub_gen, [items[i] for i in sub])}
        rnd = RandomDirectionPlacebo(self, N_RANDOM_DRAWS)
        for k in range(self.budget.n_eq_draws):
            sub_out[f"random_dir_placebo_s{k}"] = self.run_steered(rnd.intervention(k), "eq", f"random_dir_placebo_s{k}", idx=sub)
        self.eq_out = {"full": out, "subset": sub_out, "subset_idx": sub}
        self.timer.mark(f"entity_questions[{self.name}]")

    # ------------------------------------------------------------------ phase C: TriviaQA (amendment A10)
    def triviaqa(self):
        items = self.items["tqa"]
        base = self.uout["tqa"]
        out = {"unsteered": base}
        for cls, name in ((KnowledgeDirectionSteer, "knowledge_dir_steer"), (ReadDirectionSteer, "read_dir_steer"),
                          (WithinQuestionDirectionSteer, "within_question_dir_steer"),
                          (RefusalDirectionSteer, "refusal_dir_steer")):
            out[name] = self.run_steered(cls(self).intervention(), "tqa", name, steered_margin=True)
        taus = self.frozen["tau"]["tqa_margin"]
        out["confidence_threshold_margin"] = self._threshold_policy("confidence_threshold_margin", base,
                                                                    self.margin["tqa"], taus)
        gate = MarginGatedSteer.compute_gate(base.states, self.margin["tqa"], taus[LAMBDA_PRIMARY])
        out["margin_gated_steer_knowledge"] = MarginGatedSteer.compose_outputs("margin_gated_steer_knowledge", base,
                                                                               out["knowledge_dir_steer"], gate)
        self.budget.allow("tqa_subset_150")
        self.budget.check_emergency()
        sub = tqa_subset_indices(items, self.budget.tqa_subset_n)
        sub_gen = mu.generate(self.model, self.fmt, [self.prompts["tqa"][i] for i in sub], QA_MAX_NEW_TOKENS)
        sub_out = {"unsteered": make_output("unsteered", sub_gen, [items[i] for i in sub])}
        rnd = RandomDirectionPlacebo(self, N_RANDOM_DRAWS)
        for k in range(self.budget.n_eq_draws):
            sub_out[f"random_dir_placebo_s{k}"] = self.run_steered(rnd.intervention(k), "tqa", f"random_dir_placebo_s{k}", idx=sub)
        self.tqa_out = {"full": out, "subset": sub_out, "subset_idx": sub}
        print(f"TQA_COVERAGE model={self.name} unsteered_coverage={1 - _abst(base.states):.3f}", flush=True)
        self.timer.mark(f"triviaqa[{self.name}]")

    # ------------------------------------------------------------------ phase C: side effects
    def side_effect_interventions(self):
        return {"knowledge_dir_steer": KnowledgeDirectionSteer(self).intervention(),
                "read_dir_steer": ReadDirectionSteer(self).intervention(),
                "within_question_dir_steer": WithinQuestionDirectionSteer(self).intervention(),
                "refusal_dir_steer": RefusalDirectionSteer(self).intervention()}

    def side_effects(self):
        ivs = self.side_effect_interventions()
        L, a = self.frozen["L_star"], self.frozen["alpha"]
        sq = SteeredQA(self)
        # amendment A11: perplexity on 200 held-out BBC 2025-05 articles (first 128 body tokens)
        ppl_ivs = {"unsteered": None, **ivs}
        for cc in C_GRID:
            if cc != self.frozen["c_star"]:
                ppl_ivs[f"knowledge_dir_steer_c{cc:g}"] = KnowledgeDirectionSteer(self).intervention(c=cc)
        for k in range(N_PPL_RANDOM_DRAWS):
            ppl_ivs[f"random_dir_placebo_s{k}"] = steer(f"rnd{k}", L, self.random_dirs[k], a)
        ref = self.data["ref"]["test"]
        P = [self.fmt.prompt(USER_TEMPLATE_REFERENCE.format(title=x["title"])) for x in ref]
        R = [self.tok(x["text"], add_special_tokens=False).input_ids[:PPL_MAX_RESPONSE_TOKENS] for x in ref]
        ppl = {}
        for name, iv in ppl_ivs.items():
            if iv is None:
                lps = mu.teacher_force(self.model, self.fmt, P, R)
            else:
                sq.register_hook(iv)
                try:
                    lps = mu.teacher_force(self.model, self.fmt, P, R, mask_hooks=sq._add_hooks)
                finally:
                    sq.remove_hook()
            ppl[name] = float(np.exp(-np.concatenate(lps).mean()))
        self.side["perplexity_text"] = "bbc_news_2025_05 (non-memorised)"
        self.side["perplexity"] = ppl
        self.side["perplexity_ratio"] = {k: v / ppl["unsteered"] for k, v in ppl.items()}

        # MMLU-500 (0-shot, A/B/C/D log-probs at the answer position, hook at the last prompt token)
        rows = self.data["mmlu"]
        mP = [self.fmt.prompt(USER_TEMPLATE_MMLU.format(question=r["question"], a=r["choices"][0], b=r["choices"][1],
                                                        c=r["choices"][2], d=r["choices"][3])) for r in rows]
        letters = [self.tok(x, add_special_tokens=False).input_ids[0] for x in "ABCD"]
        gold = np.array([r["answer"] for r in rows])
        mmlu = {}
        for name, iv in {"unsteered": None, **ivs}.items():
            if iv is not None:
                sq.register_hook(Intervention(iv.name, adds=[(Lx, v, "prompt_only") for Lx, v, _ in iv.adds]))
            try:
                logits = mu.last_position_logits(self.model, self.fmt, mP)
            finally:
                if iv is not None:
                    sq.remove_hook()
            mmlu[name] = float((logits[:, letters].argmax(1).numpy() == gold).mean())
        self.side["mmlu_accuracy"] = mmlu
        self.side["mmlu_delta"] = {k: 100 * (v - mmlu["unsteered"]) for k, v in mmlu.items()}

        # instruction spillover: benign under the directions + random draws 0-4; harmful under the directions
        rf = RefusalDirectionSteer(self)
        instr = self.data["instr"]
        benign = {"unsteered": self.side["refusal_control"]["unsteered_benign_refusal"]}
        harmful = {"unsteered": self.side["refusal_control"]["unsteered_harmful_refusal"]}
        for name, iv in ivs.items():
            benign[name] = rf.refusal_rate(iv, instr["alpaca_benign"])[0]
            harmful[name] = rf.refusal_rate(iv, instr["adv_heldout"])[0]
        for k in range(N_INSTR_RANDOM_DRAWS):
            benign[f"random_dir_placebo_s{k}"] = rf.refusal_rate(steer(f"rnd{k}", L, self.random_dirs[k], a),
                                                                 instr["alpaca_benign"])[0]
        self.side["benign_refusal_rate"] = benign
        self.side["harmful_refusal_rate"] = harmful
        self.timer.mark(f"side_effects[{self.name}]")


def dose_statistics(M, cgrid):
    """M [n_items, n_c] steered margins. c_flip = first linearly interpolated c with m(c) < 0
    (right-censored at max c); gain = -OLS slope of m on c; monotone = non-increasing within 0.05."""
    n = M.shape[0]
    c_flip = np.full(n, np.nan)
    censored = np.zeros(n, dtype=bool)
    for i in range(n):
        m = M[i]
        if m[0] < 0:
            c_flip[i] = 0.0
            continue
        j = next((k for k in range(1, len(cgrid)) if m[k] < 0), None)
        if j is None:
            c_flip[i], censored[i] = cgrid[-1], True
        else:
            c_flip[i] = cgrid[j - 1] + (cgrid[j] - cgrid[j - 1]) * m[j - 1] / (m[j - 1] - m[j])
    cc = cgrid - cgrid.mean()
    gain = -(M - M.mean(1, keepdims=True)) @ cc / (cc ** 2).sum()
    monotone = np.all(np.diff(M, axis=1) <= 0.05, axis=1)
    rho = spearman(M[:, 0], c_flip)
    rng = np.random.default_rng(7)
    boots = [spearman(M[r, 0], c_flip[r]) for r in rng.integers(0, n, size=(2000, n))]
    return {"spearman_m0_cflip": rho, "spearman_ci": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))],
            "censored_fraction": float(censored.mean()), "gain_mean": float(gain.mean()),
            "gain_cv": float(gain.std(ddof=1) / gain.mean()) if gain.mean() != 0 else float("nan"),
            "monotone_fraction": float(monotone.mean()), "c_flip": c_flip, "m0": M[:, 0], "n_items": int(n)}

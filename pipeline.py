"""Per-model experiment state and phases: unsteered labelling and caches, direction fitting, refusal
positive control, dev selection, freezing, integrity checks, test conditions, ablations,
EntityQuestions transfer and side-effect measurements."""
import hashlib
import json
import time

import numpy as np
import torch

import model_utils as mu
from baselines import (ConfidencePromptQA, ConfidenceThreshold, PopularityGate,
                       RandomDirectionPlacebo, UnsteeredQA)
from config import (BUDGET_CAP_SECONDS, C_GRID, CALIB_N_FWD, CALIB_N_GEN, CONFIDENCE_PROMPT_INSTANCES,
                    COV_MATCH_MAX_DEGENERATE, COV_MATCH_MULTIPLIERS, DEGRADATION_LADDER,
                    DOSE_C_GRID, DOSE_ITEMS_PER_DECILE, FROZEN_CONFIG_PATH, INSTR_MAX_NEW_TOKENS,
                    LABEL_PERM_SEED_BASE, LAMBDA_PRIMARY, MIN_GRAY_AFTER_TOPUP, MIN_GRAY_QUESTIONS,
                    MIN_RANDOM_DRAWS, N_COV_MATCHED_DRAWS, N_INSTR_RANDOM_DRAWS, N_LABEL_PERM_DRAWS,
                    N_PPL_RANDOM_DRAWS, N_RANDOM_ABLATION_DRAWS, N_RANDOM_DRAWS, PLAN_ANALYSIS_SECONDS,
                    PLAN_FWD_RATE, PLAN_GEN_RATE, PLAN_PER_MODEL_SECONDS, PPL_MAX_RESPONSE_TOKENS,
                    QA_MAX_NEW_TOKENS, RANDOM_ABLATION_SEED_BASE, RANDOM_SEED_BASE, RUN_LIMIT_SECONDS,
                    USER_TEMPLATE_MMLU, ZERO_VECTOR_MIN_IDENTICAL)
from data import eq_subset_indices, qid_hash
from methods import (KnowledgeDirectionSteer, MarginGatedSteer, RefusalDirectionSteer,
                     WithinQuestionDirectionSteer)
from scoring import (A_, C_, W_, abstention_score, is_instruction_refusal, make_output,
                     normalize_answer, policy_output, spearman)
from steering import (Intervention, SteeredQA, assert_no_hooks, mean_norms, random_unit_directions,
                      steer, unit)


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


class Budget:
    """Projects total runtime from a calibration run (scale = measured / planned throughput) and
    applies the pre-registered degradation ladder until the projection fits the 80% cap."""

    def __init__(self, timer):
        self.timer = timer
        self.flags = []
        self.scale = 1.0
        self.emergency_eq_draws = None
        self.history = []

    def calibrate(self, measured_s, planned_s, n_models):
        self.scale = max(measured_s / planned_s, 1e-3)
        projected = n_models * PLAN_PER_MODEL_SECONDS * self.scale + PLAN_ANALYSIS_SECONDS
        for flag, save in DEGRADATION_LADDER:
            if projected <= BUDGET_CAP_SECONDS:
                break
            self.flags.append(flag)
            projected -= n_models * save * self.scale
        self.history.append({"at": "calibration", "scale": self.scale, "projected_total_s": projected,
                             "flags": list(self.flags)})
        print(f"BUDGET: scale={self.scale:.3f} projected_total_s={projected:.0f} ladder={self.flags}", flush=True)

    def recheck(self, remaining_planned_s, label):
        """remaining_planned_s: planned seconds (at plan rates) still to run."""
        projected = self.timer.elapsed() + remaining_planned_s * self.scale + PLAN_ANALYSIS_SECONDS
        n_models = 2
        for flag, save in DEGRADATION_LADDER:
            if flag == "test_1200" or flag in self.flags:
                continue
            if projected <= BUDGET_CAP_SECONDS:
                break
            self.flags.append(flag)
            projected -= n_models * save * self.scale
        if projected > RUN_LIMIT_SECONDS:
            # Last resort beyond the plan's ladder: fewer EntityQuestions random draws, never < 10.
            self.emergency_eq_draws = MIN_RANDOM_DRAWS
        self.history.append({"at": label, "elapsed_s": self.timer.elapsed(), "projected_total_s": projected,
                             "flags": list(self.flags), "emergency_eq_draws": self.emergency_eq_draws})
        print(f"BUDGET: at={label} projected_total_s={projected:.0f} ladder={self.flags}", flush=True)

    eq_subset_n = property(lambda s: 150 if "eq_subset_150" in s.flags else 300)
    n_random_ablation = property(lambda s: 1 if "random_ablation_1" in s.flags else N_RANDOM_ABLATION_DRAWS)
    run_alt_layer = property(lambda s: "drop_alt_layer" not in s.flags)
    n_label_perm = property(lambda s: 3 if "label_perm_3" in s.flags else N_LABEL_PERM_DRAWS)
    n_cov_matched = property(lambda s: 3 if "cov_matched_3" in s.flags else N_COV_MATCHED_DRAWS)
    test_per_decile = property(lambda s: 120 if "test_1200" in s.flags else 150)
    n_eq_draws = property(lambda s: s.emergency_eq_draws or N_RANDOM_DRAWS)


def describe(iv):
    if iv is None:
        return {"type": "none"}
    return {"name": iv.name,
            "adds": [{"layer": L, "norm": float(v.norm()), "positions": p} for L, v, p in iv.adds],
            "ablate": iv.ablate is not None}


# ----------------------------------------------------------------------------- model runner
class ModelRunner:
    def __init__(self, spec, revision, data, budget, timer):
        self.spec, self.name, self.revision = spec, spec["name"], revision
        self.data, self.budget, self.timer = data, budget, timer
        self.model, self.tok, self.fmt = mu.load_model(spec, revision)
        timer.mark(f"model_loading[{self.name}]")
        self.cand = list(spec["steering_candidate_layers"])
        self.ref_cand = list(spec["refusal_candidate_layers"])
        self.cache_layers = sorted(set(self.cand) | set(self.ref_cand))
        self.items = {k: data["popqa"][k] for k in ("train", "train_reserve", "dev", "test")}
        self.items["eq"] = data["eq"]
        self.unsteered = UnsteeredQA(self)
        self.prompts, self.subj_spans, self.tiebreak = {}, {}, {}
        for split in self.items:
            self._build_prompts(split)
        self.cache, self.ugen, self.uout, self.margin = {}, {}, {}, {}
        self.gold, self.kstatus, self.labels, self.samples, self.ans_res = {}, {}, {}, {}, {}
        self.dirs, self.frozen, self.dev = {}, {}, {}
        self.test_out, self.eq_out, self.ablation_out, self.side = {}, {}, {}, {}
        self.extras = {}
        self.zero_check_failed = False
        self.integrity = {}

    # ------------------------------------------------------------------ helpers
    def _build_prompts(self, split):
        built = [self.unsteered.build_prompt(it) for it in self.items[split]]
        self.prompts[split] = [b[0] for b in built]
        self.subj_spans[split] = [b[1] for b in built]
        self.tiebreak[split] = np.array([qid_hash(it.qid) for it in self.items[split]], dtype=np.int64)

    def restrict_test(self, items):
        self.items["test"] = items
        self._build_prompts("test")

    def signal(self, name, split):
        """Gate signals (higher = more confident the unsteered answer is right)."""
        L = self.frozen["L_star"]
        if name == "margin":
            return self.margin[split]
        if name == "ku_projection":       # high projection = unknown
            return -(self.cache[split]["resid_last"][:, L] @ self.dirs["knowledge"][L]).numpy()
        if name == "wq_answer_projection":  # low projection = likely wrong
            return (self.ans_res[split]["mean"][:, self.cand.index(L)] @ self.dirs["wq"][L]).numpy()
        if name == "refusal_projection":  # high projection = refusal-like
            return -(self.cache[split]["resid_last"][:, L] @ self.dirs["refusal"][L]).numpy()
        if name == "popularity":
            return np.array([it.log_pop for it in self.items[split]])
        raise KeyError(name)

    def run_steered(self, iv, split, name, steered_margin=False, idx=None, check_zero=True):
        if check_zero and self.zero_check_failed:
            raise RuntimeError("zero-vector check failed: steered runs halted for this model")
        prompts, items = self.prompts[split], self.items[split]
        if idx is not None:
            prompts, items = [prompts[i] for i in idx], [items[i] for i in idx]
        sq = SteeredQA(self)
        gen = sq.answer(iv, prompts, QA_MAX_NEW_TOKENS)
        conf, kind = gen.ans_logprob, "answer_logprob"
        if steered_margin:
            conf, kind = sq.steered_margin(iv, prompts, gen), "steered_margin"
        return make_output(name, gen, items, conf, kind, extra={"intervention": describe(iv)})

    def alpha(self, L, c):
        return c * float(self.rbar[L])

    # ------------------------------------------------------------------ phase 1
    def calibrate(self, update_budget):
        n_models = 2
        t = time.time()
        mu.generate(self.model, self.fmt, self.prompts["train"][:CALIB_N_GEN], QA_MAX_NEW_TOKENS)
        P = (self.prompts["train"] * 2)[:CALIB_N_FWD]
        mu.teacher_force(self.model, self.fmt, P, [self.fmt.idk_ids + [self.fmt.eot_id]] * len(P))
        measured = time.time() - t
        planned = CALIB_N_GEN / PLAN_GEN_RATE + CALIB_N_FWD / PLAN_FWD_RATE
        if update_budget:
            self.budget.calibrate(measured, planned, n_models)
        self.extras["calibration"] = {"measured_s": measured, "planned_s": planned}
        self.timer.mark(f"calibration[{self.name}]")

    def label_unsteered(self):
        u = self.unsteered
        for split in ("train", "dev", "test", "eq"):
            r_last, r_idk, lp_idk, surpr = u.probe_pass(split)
            self.cache[split] = {"resid_last": r_last, "idk_resid": r_idk, "logp_idk": lp_idk, "surprisal": surpr}
        self.timer.mark(f"activation_caching[{self.name}]")
        for split in ("train", "dev", "test", "eq"):
            gen = u.answer(self.prompts[split])
            self.ugen[split] = gen
            self.margin[split] = gen.ans_logprob - self.cache[split]["logp_idk"]
            self.uout[split] = u.classify_output("unsteered", gen, self.items[split], self.margin[split], "margin")
        for split in ("train", "test", "eq"):
            self.gold[split] = u.gold_score(split, self.items[split])
        tr_correct = self.uout["train"].states == C_
        self.tau_store = float(np.percentile(self.gold["train"][tr_correct], 10)) if tr_correct.any() else float("-inf")
        for split in ("train", "test", "eq"):
            self.kstatus[split] = u.assign_strata(self.uout[split].states, self.gold[split], self.tau_store)
        for split in ("train", "test"):
            self.samples[split], self.labels[split] = u.sample_labels(split, self.items[split], self.uout[split])
        for split in ("train", "dev", "test", "eq"):
            self.ans_res[split] = self._answer_activations(split)
        self._gray_zone()
        self.timer.mark(f"unsteered_labelling[{self.name}]")

    def _answer_activations(self, split):
        conts = [list(t) for t in self.ugen[split].token_ids]
        _, mean_r, last_r = mu.teacher_force(self.model, self.fmt, self.prompts[split], conts,
                                             res_layers=self.cand, res_ranges=[(0, len(c)) for c in conts])
        return {"mean": mean_r, "last": last_r}

    def _gray_zone(self):
        wq = WithinQuestionDirectionSteer(self)
        gray = wq.find_gray_questions(self.uout["train"], self.samples["train"], self.labels["train"])
        H, qidx, lab = wq.collect_answer_activations(gray, self.prompts["train"], self.cand)
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

    def fit_directions(self):
        k = KnowledgeDirectionSteer(self)
        known, unknown, self.extras["known_unknown"] = k.label_known_unknown()
        self.extras["known_unknown_masks"] = (known, unknown)
        self.dirs["knowledge"] = k.fit_direction(self.cand, known=known, unknown=unknown)
        self.rbar = mean_norms(self.cache["train"]["resid_last"])
        cols = {L: j for j, L in enumerate(self.cand)}
        H, q, lab = self.gray["H"], self.gray["qidx"], self.gray["labels"]
        self.dirs["wq"] = WithinQuestionDirectionSteer.fit_direction(H, q, lab, cols)
        self.dirs["bq"] = WithinQuestionDirectionSteer.fit_between_question(H, lab, cols)
        self.dirs["wq_perm"] = [WithinQuestionDirectionSteer.fit_label_permuted(H, q, lab, cols, LABEL_PERM_SEED_BASE + j)
                                for j in range(N_LABEL_PERM_DRAWS)]
        wrong = torch.as_tensor(self.uout["train"].states == W_)
        self.dirs["idk_caa"] = {}
        if wrong.any():
            for L in self.cand:
                diff = (self.cache["train"]["idk_resid"][:, self.cache_layers.index(L)]
                        - self.ans_res["train"]["last"][:, self.cand.index(L)])
                self.dirs["idk_caa"][L] = unit(diff[wrong].mean(0))
        self.random_dirs = random_unit_directions(self.spec["d_model"], N_RANDOM_DRAWS, RANDOM_SEED_BASE)
        self.timer.mark(f"direction_fitting[{self.name}]")

    def refusal_direction_and_control(self):
        r = RefusalDirectionSteer(self)
        self.dirs["refusal"] = r.fit_direction(self.cache_layers)
        instr = self.data["instr"]
        harm_rate, harm_texts, harm_flags = r.refusal_rate(None_iv(), instr["adv_heldout"])
        ben_rate, ben_texts, ben_flags = r.refusal_rate(None_iv(), instr["alpaca_benign"])
        L_ref, per_layer, passed = r.positive_control(self.ref_cand, harm_rate)
        self.frozen["L_ref"] = L_ref
        self.side["refusal_control"] = {"unsteered_harmful_refusal": harm_rate,
                                        "unsteered_benign_refusal": ben_rate, "per_layer": per_layer,
                                        "L_ref": L_ref, "passed": passed}
        self.side["instr_unsteered"] = {"harmful": harm_flags, "benign": ben_flags,
                                        "harmful_texts": harm_texts, "benign_texts": ben_texts}
        print(f"REFUSAL_CONTROL model={self.name} L_ref={L_ref} unsteered_refusal={harm_rate:.3f} "
              f"ablated_refusal={per_layer[L_ref]['refusal_rate']:.3f} passed={passed}", flush=True)
        self.timer.mark(f"refusal_direction[{self.name}]")

    def dev_sweep(self):
        base = self.uout["dev"].states
        base_abst = float((base == A_).mean())
        k = KnowledgeDirectionSteer(self)
        sweep, outs = {}, {}
        for L in self.cand:
            for c in C_GRID:
                o = self.run_steered(k.intervention(L, c, direction=self.dirs["knowledge"][L]), "dev",
                                     f"knowledge_l{L}_c{c:g}", check_zero=False)
                outs[(L, c)] = o
                sweep[(L, c)] = {"S1": abstention_score(o.states), "degenerate_rate": float(o.degenerate.mean()),
                                 "abst_increase": 100 * (float((o.states == A_).mean()) - base_abst),
                                 "W_to_C": int(((base == W_) & (o.states == C_)).sum())}
        L_star, c_star, rule = k.select_config_on_dev(sweep, self.cand)
        self.frozen.update({"L_star": L_star, "c_star": c_star, "selection_rule_outcome": rule,
                            "rbar_L_star": float(self.rbar[L_star]), "alpha": self.alpha(L_star, c_star)})
        print(f"DEV_SELECTION model={self.name} L_star={L_star} c_star={c_star} rule={rule}", flush=True)
        self.dev["knowledge_sweep"] = {f"l{L}_c{c:g}": v for (L, c), v in sweep.items()}

        wq = WithinQuestionDirectionSteer(self)
        wq_sweep = {}
        for L in self.cand:
            for c in C_GRID:
                o = self.run_steered(wq.intervention(L, c, direction=self.dirs["wq"][L]), "dev",
                                     f"wq_l{L}_c{c:g}", check_zero=False)
                wq_sweep[f"l{L}_c{c:g}"] = {"S1": abstention_score(o.states), "degenerate_rate": float(o.degenerate.mean()),
                                            "abst_increase": 100 * (float((o.states == A_).mean()) - base_abst),
                                            "W_to_C": int(((base == W_) & (o.states == C_)).sum())}
        self.dev["wq_sweep_descriptive"] = wq_sweep
        rf = RefusalDirectionSteer(self)
        ref_sweep = {}
        for c in C_GRID:
            o = self.run_steered(rf.intervention(L_star, c), "dev", f"refusal_c{c:g}", check_zero=False)
            ref_sweep[f"c{c:g}"] = {"S1": abstention_score(o.states), "degenerate_rate": float(o.degenerate.mean()),
                                    "abst_increase": 100 * (float((o.states == A_).mean()) - base_abst)}
        self.dev["refusal_sweep_descriptive"] = ref_sweep

        # thresholds (tau_dev(lambda) for every gate signal; tau_pop(lambda))
        self.frozen["tau"] = {}
        for sig in ConfidenceThreshold.SIGNALS:
            ct = ConfidenceThreshold(sig)
            self.frozen["tau"][sig] = ct.fit_threshold_on_dev(base, ct.compute_signal(self, "dev"))
        pg = PopularityGate()
        self.frozen["tau"]["popularity"] = pg.fit_threshold_on_dev(base, pg.compute_signal(self, "dev"))

        # coverage-matched random magnitudes on the fixed 300-item dev subset
        sub = self.data["dev_subset"]
        sub_gen = mu.generate(self.model, self.fmt, [self.prompts["dev"][i] for i in sub], QA_MAX_NEW_TOKENS)
        sub_base = make_output("unsteered", sub_gen, [self.items["dev"][i] for i in sub])
        target = 100 * (float((outs[(L_star, c_star)].states[sub] == A_).mean()) - float((base[sub] == A_).mean()))
        rnd = RandomDirectionPlacebo(self, N_COV_MATCHED_DRAWS)
        cov = {}
        for kdraw in range(self.budget.n_cov_matched):
            res = {}
            for m in COV_MATCH_MULTIPLIERS:
                o = self.run_steered(rnd.intervention(kdraw, m), "dev", f"rnd{kdraw}_x{m}", idx=sub, check_zero=False)
                res[m] = {"abst_increase": 100 * (float((o.states == A_).mean()) - float((sub_base.states == A_).mean())),
                          "degenerate_rate": float(o.degenerate.mean())}
            # clean = degenerate <= 10%; a draw is matched if some clean multiplier reaches the
            # target dose, and then the clean multiplier closest to the target is frozen; otherwise
            # the largest clean multiplier is used and the draw is flagged unmatched.
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
        L_alt = sorted(self.cand, key=lambda L: (-abs(L - L_star), L))[0]
        self.frozen["L_alt"] = L_alt
        self.timer.mark(f"dev_tuning[{self.name}]")

    def freeze(self, all_frozen):
        L = self.frozen["L_star"]
        entry = dict(self.frozen)
        entry["directions_at_L_star"] = {k: self.dirs[k][L].tolist() for k in ("knowledge", "wq", "refusal")}
        entry.update({"model": self.name, "hf_id": self.spec["hf_id"], "revision": self.revision,
                      "tau_store": self.tau_store, "budget_flags": list(self.budget.flags)})
        blob = json.dumps(entry, sort_keys=True, default=str)
        entry["sha256"] = hashlib.sha256(blob.encode()).hexdigest()
        all_frozen[self.name] = entry
        FROZEN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FROZEN_CONFIG_PATH, "w") as f:
            json.dump(all_frozen, f, indent=1, default=str)
        print(f"FROZEN_CONFIG model={self.name} sha256={entry['sha256']} (written before any test forward pass)", flush=True)

    # ------------------------------------------------------------------ integrity
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
               "within_question_dir_steer": WithinQuestionDirectionSteer(self).intervention(),
               "refusal_dir_steer": RefusalDirectionSteer(self).intervention(),
               "random_dir_placebo": steer("rnd0", L, self.random_dirs[0], self.alpha(L, self.frozen["c_star"])),
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
        self.timer.mark(f"integrity_checks[{self.name}]")

    # ------------------------------------------------------------------ registry conditions (PopQA test)
    def cond_unsteered(self):
        return {"unsteered": self.uout["test"]}

    def cond_confidence_prompt(self):
        out = {}
        for inst in CONFIDENCE_PROMPT_INSTANCES:
            cp = ConfidencePromptQA(self, inst)
            out[inst] = cp.classify_output(cp.answer(self.items["test"]), self.items["test"])
        return out

    def cond_confidence_threshold_margin(self):
        base = self.uout["test"]
        ct = ConfidenceThreshold("margin")
        taus = self.frozen["tau"]["margin"]
        sig = self.margin["test"]
        op = policy_output("confidence_threshold_margin", base, ct.apply_threshold(base.states, sig, taus[LAMBDA_PRIMARY]),
                           extra={"tau": taus[LAMBDA_PRIMARY],
                                  "lambda_states": {lam: ct.apply_threshold(base.states, sig, t) for lam, t in taus.items()}})
        return {"confidence_threshold_margin": op}

    def cond_popularity_gate(self):
        base = self.uout["test"]
        pg = PopularityGate()
        taus = self.frozen["tau"]["popularity"]
        sig = pg.compute_signal(self, "test")
        op = policy_output("popularity_gate", base, pg.apply_threshold(base.states, sig, taus[LAMBDA_PRIMARY]),
                           extra={"tau": taus[LAMBDA_PRIMARY],
                                  "lambda_states": {lam: pg.apply_threshold(base.states, sig, t) for lam, t in taus.items()}})
        return {"popularity_gate": op}

    def _primary_steer(self, cls, name):
        return {name: self.run_steered(cls(self).intervention(), "test", name, steered_margin=True)}

    def cond_knowledge_dir_steer(self):
        return self._primary_steer(KnowledgeDirectionSteer, "knowledge_dir_steer")

    def cond_within_question_dir_steer(self):
        return self._primary_steer(WithinQuestionDirectionSteer, "within_question_dir_steer")

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

    def cond_random_dir_placebo(self):
        rnd = RandomDirectionPlacebo(self, N_RANDOM_DRAWS)
        return {f"random_dir_placebo_s{k}": self.run_steered(rnd.intervention(k), "test", f"random_dir_placebo_s{k}")
                for k in range(N_RANDOM_DRAWS)}

    # ------------------------------------------------------------------ phase 2: ablations
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
        if not self.budget.run_alt_layer:
            print(f"BUDGET_SKIP: knowledge_dir_steer_alt_layer model={self.name} (degradation ladder)", flush=True)
            return {}
        iv = KnowledgeDirectionSteer(self).intervention(layer=self.frozen["L_alt"])
        return {"knowledge_dir_steer_alt_layer": self.run_steered(iv, "test", "knowledge_dir_steer_alt_layer")}

    def abl_between_question_answer_dir(self):
        L = self.frozen["L_star"]
        iv = steer("bq", L, self.dirs["bq"][L], self.alpha(L, self.frozen["c_star"]))
        return {"between_question_answer_dir_steer": self.run_steered(iv, "test", "between_question_answer_dir_steer")}

    def abl_label_permuted_wq(self):
        L, c = self.frozen["L_star"], self.frozen["c_star"]
        out = {}
        for j in range(self.budget.n_label_perm):
            iv = steer(f"wq_perm{j}", L, self.dirs["wq_perm"][j][L], self.alpha(L, c))
            out[f"label_permuted_wq_dir_steer_s{j}"] = self.run_steered(iv, "test", f"label_permuted_wq_dir_steer_s{j}")
        return out

    def abl_random_coverage_matched(self):
        rnd = RandomDirectionPlacebo(self, N_COV_MATCHED_DRAWS)
        out = {}
        for k in range(self.budget.n_cov_matched):
            info = self.frozen["cov_matched"]["draws"].get(k)
            if info is None:
                continue
            o = self.run_steered(rnd.intervention(k, info["multiplier"]), "test", f"random_dir_placebo_coverage_matched_s{k}")
            o.extra.update({"multiplier": info["multiplier"], "matched_on_dev": info["matched"]})
            out[f"random_dir_placebo_coverage_matched_s{k}"] = o
        return out

    def abl_refusal_ablation(self):
        iv = Intervention("refusal_dir_ablation", ablate=self.dirs["refusal"][self.frozen["L_ref"]])
        return {"refusal_dir_ablation": self.run_steered(iv, "test", "refusal_dir_ablation")}

    def abl_refusal_ablation_plus_knowledge(self):
        L, c = self.frozen["L_star"], self.frozen["c_star"]
        iv = Intervention("refusal_dir_ablation_plus_knowledge_steer",
                          adds=[(L, self.alpha(L, c) * self.dirs["knowledge"][L], "both")],
                          ablate=self.dirs["refusal"][self.frozen["L_ref"]])
        return {"refusal_dir_ablation_plus_knowledge_steer":
                self.run_steered(iv, "test", "refusal_dir_ablation_plus_knowledge_steer")}

    def abl_random_ablation(self):
        dirs = random_unit_directions(self.spec["d_model"], N_RANDOM_ABLATION_DRAWS, RANDOM_ABLATION_SEED_BASE)
        return {f"random_dir_ablation_s{j}": self.run_steered(Intervention(f"rand_ablate{j}", ablate=dirs[j]), "test",
                                                              f"random_dir_ablation_s{j}")
                for j in range(self.budget.n_random_ablation)}

    def abl_first_token_ban(self):
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
        conts_i = [self.fmt.idk_ids + [self.fmt.eot_id]] * len(idx)
        L = self.frozen["L_star"]
        sq = SteeredQA(self)
        m0 = (np.array([float(x.sum()) for x in mu.teacher_force(self.model, self.fmt, P, conts_a)])
              - np.array([float(x.sum()) for x in mu.teacher_force(self.model, self.fmt, P, conts_i)]))
        dirs = {"knowledge_dir": self.dirs["knowledge"][L], "random_s0": self.random_dirs[0], "random_s1": self.random_dirs[1]}
        res = {}
        for dname, v in dirs.items():
            M = [m0]
            for c in DOSE_C_GRID[1:]:
                iv = steer(f"dose_{dname}_{c}", L, v, self.alpha(L, c))
                M.append(sq.teacher_forced_logprob(iv, P, conts_a) - sq.teacher_forced_logprob(iv, P, conts_i))
            res[dname] = dose_statistics(np.stack(M, 1), np.array(DOSE_C_GRID))
            res[dname]["margins"] = np.stack(M, 1)
        # agreement with generation flips at the test doses of knowledge_dir_steer
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

    # ------------------------------------------------------------------ phase 2: EntityQuestions
    def entity_questions(self):
        items = self.items["eq"]
        out = {"unsteered": self.uout["eq"]}
        cp = ConfidencePromptQA(self, "confidence_prompt_t050")
        out["confidence_prompt_t050"] = cp.classify_output(cp.answer(items), items)
        for cls, name in ((KnowledgeDirectionSteer, "knowledge_dir_steer"),
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
        sub = eq_subset_indices(items, self.budget.eq_subset_n)
        sub_gen = mu.generate(self.model, self.fmt, [self.prompts["eq"][i] for i in sub], QA_MAX_NEW_TOKENS)
        sub_out = {"unsteered": make_output("unsteered", sub_gen, [items[i] for i in sub])}
        rnd = RandomDirectionPlacebo(self, N_RANDOM_DRAWS)
        for k in range(self.budget.n_eq_draws):
            sub_out[f"random_dir_placebo_s{k}"] = self.run_steered(rnd.intervention(k), "eq", f"random_dir_placebo_s{k}", idx=sub)
        self.eq_out = {"full": out, "subset": sub_out, "subset_idx": sub}
        self.timer.mark(f"entity_questions[{self.name}]")

    # ------------------------------------------------------------------ phase 2: side effects
    def side_effect_interventions(self):
        L, c = self.frozen["L_star"], self.frozen["c_star"]
        ivs = {"knowledge_dir_steer": KnowledgeDirectionSteer(self).intervention(),
               "within_question_dir_steer": WithinQuestionDirectionSteer(self).intervention(),
               "refusal_dir_steer": RefusalDirectionSteer(self).intervention()}
        return ivs, L, c

    def side_effects(self):
        ivs, L, c = self.side_effect_interventions()
        sq = SteeredQA(self)
        # perplexity on 200 held-out Alpaca outputs (first 128 response tokens)
        ppl_ivs = {"unsteered": None, **ivs}
        for cc in C_GRID:
            if cc != c:
                ppl_ivs[f"knowledge_dir_steer_c{cc:g}"] = KnowledgeDirectionSteer(self).intervention(c=cc)
        for k in range(N_PPL_RANDOM_DRAWS):
            ppl_ivs[f"random_dir_placebo_s{k}"] = steer(f"rnd{k}", L, self.random_dirs[k], self.alpha(L, c))
        pairs = self.data["instr"]["alpaca_ppl"]
        P = [self.fmt.prompt(ins) for ins, _ in pairs]
        R = [self.tok(outp, add_special_tokens=False).input_ids[:PPL_MAX_RESPONSE_TOKENS] for _, outp in pairs]
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
            tot = np.concatenate(lps)
            ppl[name] = float(np.exp(-tot.mean()))
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
                prompt_only = Intervention(iv.name, adds=[(Lx, v, "prompt_only") for Lx, v, _ in iv.adds])
                sq.register_hook(prompt_only)
            try:
                logits = mu.last_position_logits(self.model, self.fmt, mP)
            finally:
                if iv is not None:
                    sq.remove_hook()
            pred = logits[:, letters].argmax(1).numpy()
            mmlu[name] = float((pred == gold).mean())
        self.side["mmlu_accuracy"] = mmlu
        self.side["mmlu_delta"] = {k: 100 * (v - mmlu["unsteered"]) for k, v in mmlu.items()}

        # instruction spillover: benign under 3 directions + random draws 0-4; harmful under 3 directions
        rf = RefusalDirectionSteer(self)
        instr = self.data["instr"]
        benign, harmful = {"unsteered": self.side["refusal_control"]["unsteered_benign_refusal"]}, \
                          {"unsteered": self.side["refusal_control"]["unsteered_harmful_refusal"]}
        for name, iv in ivs.items():
            benign[name] = rf.refusal_rate(iv, instr["alpaca_benign"])[0]
            harmful[name] = rf.refusal_rate(iv, instr["adv_heldout"])[0]
        for k in range(N_INSTR_RANDOM_DRAWS):
            benign[f"random_dir_placebo_s{k}"] = rf.refusal_rate(
                steer(f"rnd{k}", L, self.random_dirs[k], self.alpha(L, c)), instr["alpaca_benign"])[0]
        self.side["benign_refusal_rate"] = benign
        self.side["harmful_refusal_rate"] = harmful
        self.timer.mark(f"side_effects[{self.name}]")


class None_iv(Intervention):
    """Empty intervention (no hooks) for unsteered generations through the hooked code path."""

    def __init__(self):
        super().__init__("unsteered")


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

"""Proposed steering methods: knowledge (D_ku), read (D_read, amendment A9), within-question (D_wq),
safety-refusal direction, margin-gated steering and the dose-matched nulls (amendment A9).

Strength (amendment A11): a dev-sweep configuration (L, c) of direction v adds c * sigma_L(v) * v_hat,
sigma_L(v) = SD of the unsteered last-prompt residual's projection on v_hat over PopQA train. The
frozen shared site adds the absolute norm alpha* = c* sigma_{L*}(D_ku) for every primary direction and
for the norm-matched random draws (norm matching as registered); dose-ladder rungs add
c * sigma_{L*}(D_ku) for every direction, so magnitudes stay crossed with direction source.
"""
import numpy as np
import torch

import model_utils as mu
from config import (C_GRID, DEV_ELIGIBLE_ABST_INC, DEV_ELIGIBLE_MAX_DEGENERATE, DEV_FALLBACK_TARGET_INC,
                    DEV_TIE_POINTS, INSTR_MAX_NEW_TOKENS, MIN_CLASS_KNOWN_UNKNOWN, MMLU_CAP_POINTS, PPL_CAP_RATIO)
from scoring import A_, C_, W_, CondOutput, is_instruction_refusal, summary_metrics
from steering import (Intervention, SteeredQA, between_question_direction, bootstrap_questions,
                      covariance_shaped_direction, diff_of_means, permute_labels_within_question,
                      shuffled_label_direction, steer, unit, within_question_direction)


class _DirectionSteer(SteeredQA):
    """Shared: intervention at the frozen site (absolute alpha*), at a dose-ladder rung
    (c * sigma_{L*}(D_ku)), or at a dev-sweep configuration (c * sigma_L(v))."""
    key = None

    def intervention(self, layer=None, c=None, positions="both", direction=None, name=None, alpha=None):
        f = self.ctx.frozen
        L = f["L_star"] if layer is None else layer
        v = self.ctx.dirs[self.key][L] if direction is None else direction
        if alpha is None:
            if c is None:
                assert layer is None, "a non-shared layer needs an explicit c"
                alpha, c = f["alpha"], f["c_star"]
            else:
                alpha = self.ctx.site_alpha(c, L)
        return steer(name or f"{self.key}_l{L}_c{c:g}", L, v, alpha, positions)

    @staticmethod
    def score(out):
        return summary_metrics(out.states, out.degenerate, out.truncated, out.n_tokens)


class KnowledgeDirectionSteer(_DirectionSteer):
    """D_ku: mean last-prompt-token residual of clearly unknown minus clearly known train questions,
    added toward 'unknown' (sign +)."""
    key = "knowledge"

    def label_known_unknown(self):
        c = self.ctx
        g = c.uout["train"].states
        n_corr = c.labels["train"]["n_correct_samples"]
        known = (g == C_) & (n_corr >= 6)
        unknown = (g != C_) & (n_corr <= 1)
        fallback = bool(known.sum() < MIN_CLASS_KNOWN_UNKNOWN or unknown.sum() < MIN_CLASS_KNOWN_UNKNOWN)
        if fallback:
            known, unknown = g == C_, g != C_
        return known, unknown, {"n_known": int(known.sum()), "n_unknown": int(unknown.sum()),
                                "greedy_only_fallback": fallback}

    def fit_direction(self, layers, resid_last=None, known=None, unknown=None):
        if known is None:
            known, unknown, _ = self.label_known_unknown()
        R = self.ctx.cache["train"]["resid_last"] if resid_last is None else resid_last
        return {L: unit(diff_of_means(R[:, L], unknown, known)) for L in layers}

    @staticmethod
    def select_config_on_dev(sweep, layers, caps=None):
        """sweep: {(L, c): {"S1", "degenerate_rate", "abst_increase"}}. Pre-registered rule plus the
        amendment-A11 caps: caps(L, c) -> {"mmlu_delta": points, "ppl_ratio": ratio}, evaluated only
        for configurations that pass the registered criteria (results cached in the sweep)."""
        middle = layers[len(layers) // 2]

        def cap_ok(k):
            if caps is None:
                return True
            if "caps" not in sweep[k]:
                sweep[k]["caps"] = caps(*k)
            cp = sweep[k]["caps"]
            cp["passed"] = bool(cp["mmlu_delta"] >= MMLU_CAP_POINTS and cp["ppl_ratio"] <= PPL_CAP_RATIO)
            return cp["passed"]

        clean = [k for k, v in sweep.items() if v["degenerate_rate"] <= DEV_ELIGIBLE_MAX_DEGENERATE]
        window = [k for k in clean if DEV_ELIGIBLE_ABST_INC[0] <= sweep[k]["abst_increase"] <= DEV_ELIGIBLE_ABST_INC[1]]
        eligible = [k for k in sorted(window, key=lambda k: -sweep[k]["S1"]) if cap_ok(k)]
        if eligible:
            best = max(sweep[k]["S1"] for k in eligible)
            tied = [k for k in eligible if sweep[k]["S1"] >= best - DEV_TIE_POINTS]
            L, c = sorted(tied, key=lambda k: (k[1], k[0] != middle, -sweep[k]["S1"]))[0]
            return L, c, "eligible"
        for k in sorted(clean, key=lambda k: (abs(sweep[k]["abst_increase"] - DEV_FALLBACK_TARGET_INC), k[1])):
            if cap_ok(k):
                return k[0], k[1], "fallback_closest_to_15"
        return middle, C_GRID[0], "no clean operating point"


class ReadDirectionSteer(_DirectionSteer):
    """Amendment A9, D_read: D_ku fitted on answered train items only, mean(last-prompt residual |
    unsteered wrong) - mean(... | unsteered correct), sign + (toward wrong, i.e. toward abstention as
    for D_ku). Same site and norm as knowledge_dir_steer."""
    key = "read"

    def fit_direction(self, layers):
        c = self.ctx
        g = c.uout["train"].states
        R = c.cache["train"]["resid_last"]
        return {L: unit(diff_of_means(R[:, L], g == W_, g == C_)) for L in layers}


class WithinQuestionDirectionSteer(_DirectionSteer):
    """D_wq: mean over gray-zone train questions of [mean answer-token residual of correct outputs
    minus that of wrong outputs of the same question]; sign + (toward correct)."""
    key = "wq"

    @staticmethod
    def find_gray_questions(greedy_out, samples, labels):
        """Per question: outputs = greedy + 8 samples. label 1 = correct, 0 = wrong (answered,
        non-degenerate). Gray = at least one of each."""
        gray = []
        st, dg = labels["sample_states"], labels["sample_degenerate"]
        for i in range(len(greedy_out)):
            outs = [(greedy_out.token_ids[i], greedy_out.states[i], greedy_out.degenerate[i])]
            outs += [(samples[i][k][1], st[i, k], dg[i, k]) for k in range(st.shape[1])]
            lab = [(tuple(t), 1 if s == C_ else 0) for t, s, d in outs
                   if len(t) > 0 and (s == C_ or (s == W_ and not d))]
            if any(l == 1 for _, l in lab) and any(l == 0 for _, l in lab):
                gray.append((i, lab))
        return gray

    def collect_answer_activations(self, gray, prompts, layers):
        """Teacher-force prompt + each output; h(s) = mean residual over the output's answer
        tokens (end-of-turn excluded) after each block in `layers`."""
        uniq, key_to_row = [], {}
        qidx, labels, rows = [], [], []
        for q, outs in gray:
            for toks, lab in outs:
                key = (q, toks)
                if key not in key_to_row:
                    key_to_row[key] = len(uniq)
                    uniq.append(key)
                qidx.append(q)
                labels.append(lab)
                rows.append(key_to_row[key])
        P = [prompts[q] for q, _ in uniq]
        conts = [list(t) for _, t in uniq]
        _, mean_res, _ = mu.teacher_force(self.ctx.model, self.ctx.fmt, P, conts, res_layers=layers,
                                          res_ranges=[(0, len(t)) for t in conts])
        H = mean_res[torch.as_tensor(rows)]
        return H, np.array(qidx), np.array(labels)

    @staticmethod
    def fit_direction(H, qidx, labels, layer_positions):
        """layer_positions: {L: column of H}. Returns {L: unit D_wq,L}."""
        return {L: unit(within_question_direction(H[:, j], qidx, labels)) for L, j in layer_positions.items()}

    @staticmethod
    def fit_refit(H, qidx, labels, layer_positions, seed):
        """Amendment A8 refit r >= 1: question-level bootstrap of the gray zone."""
        rows, newq = bootstrap_questions(qidx, seed)
        Hr = H[torch.as_tensor(rows)]
        return {L: unit(within_question_direction(Hr[:, j], newq, labels[rows])) for L, j in layer_positions.items()}

    @staticmethod
    def fit_between_question(H, labels, layer_positions):
        return {L: unit(between_question_direction(H[:, j], labels)) for L, j in layer_positions.items()}

    @staticmethod
    def fit_label_permuted(H, qidx, labels, layer_positions, seed):
        perm = permute_labels_within_question(qidx, labels, seed)
        return {L: unit(within_question_direction(H[:, j], qidx, perm)) for L, j in layer_positions.items()}

    def answer_position_intervention(self, L, alpha, direction, name):
        """Amendment A9: D_wq added where it was measured, at the generated (answer) positions only."""
        return steer(name, L, direction, alpha, "generated_only")

    def same_signal_projection(self, split):
        """Unsteered greedy answer's mean answer-token residual at L_star projected on D_hat_wq
        (low = likely wrong)."""
        L = self.ctx.frozen["L_star"]
        j = self.ctx.cand.index(L)
        return (self.ctx.ans_res[split]["mean"][:, j] @ self.ctx.dirs["wq"][L]).numpy()


class RefusalDirectionSteer(_DirectionSteer):
    """Safety-refusal direction: mean(harmful) - mean(harmless) last-prompt residual (128 AdvBench
    vs 128 Alpaca train instructions, shared system prompt)."""
    key = "refusal"

    def instruction_prompts(self, instructions):
        return [self.ctx.fmt.prompt(x) for x in instructions]

    def train_residuals(self, layers):
        c = self.ctx
        harm = mu.prompt_last_residuals(c.model, c.fmt, self.instruction_prompts(c.data["instr"]["adv_train"]), layers)
        safe = mu.prompt_last_residuals(c.model, c.fmt, self.instruction_prompts(c.data["instr"]["alpaca_train"]), layers)
        return harm, safe

    @staticmethod
    def fit_from_residuals(harm, safe, layers, seed=0):
        """Seed 0: all 128 + 128 instructions (registered). Seed s >= 1 (amendment A8): a bootstrap
        resample of each class."""
        if seed:
            rng = np.random.default_rng(9000 + seed)
            harm = harm[torch.as_tensor(rng.integers(0, len(harm), len(harm)))]
            safe = safe[torch.as_tensor(rng.integers(0, len(safe), len(safe)))]
        return {L: unit(harm[:, j].mean(0) - safe[:, j].mean(0)) for j, L in enumerate(layers)}

    def refusal_rate(self, iv, instructions):
        gen = self.answer(iv, self.instruction_prompts(instructions), INSTR_MAX_NEW_TOKENS, want_logprobs=False)
        flags = np.array([is_instruction_refusal(t) for t in gen.texts])
        return float(flags.mean()), gen.texts, flags

    def ablation_intervention(self, name, L_ref, mode="directional", layers="all", means=None):
        """Amendment A12 variants. directional/all = registered arm; mean/all and mean/single replace
        the component along r_hat with its harmless-reference mean (everywhere / at L_ref only)."""
        r = self.ctx.dirs["refusal"][L_ref]
        return Intervention(name, ablate=r, ablate_mode=mode,
                            ablate_layers=None if layers == "all" else [L_ref], ablate_means=means)

    def positive_control(self, layers, unsteered_rate, mode="directional", scope="all", means_by_layer=None):
        """Ablate r_hat_L (variant) for each candidate layer; L_ref = largest relative reduction of
        refusal on the 100 held-out AdvBench prompts; pass if >= 50%."""
        held = self.ctx.data["instr"]["adv_heldout"]
        per_layer = {}
        for L in layers:
            iv = self.ablation_intervention(f"refusal_ablate_{mode}_{scope}_l{L}", L, mode, scope,
                                            None if means_by_layer is None else means_by_layer[L])
            rate, _, _ = self.refusal_rate(iv, held)
            per_layer[L] = {"refusal_rate": rate,
                            "relative_reduction": (unsteered_rate - rate) / unsteered_rate if unsteered_rate > 0 else float("nan")}
        L_ref = max(layers, key=lambda L: np.nan_to_num(per_layer[L]["relative_reduction"], nan=-1.0))
        return L_ref, per_layer, bool(per_layer[L_ref]["relative_reduction"] >= 0.5)

    def projection(self, split):
        """Last-prompt residual at L_star projected on r_hat (high = refusal-like)."""
        L = self.ctx.frozen["L_star"]
        return (self.ctx.cache[split]["resid_last"][:, L] @ self.ctx.dirs["refusal"][L]).numpy()


class DoseMatchedNull:
    """Amendment A9: three nulls per seed at the shared layer and positions, each with its strength
    (a multiple of alpha*) chosen on the dev subset to match knowledge_dir_steer's dev abstention
    increase: shuffled-label D_ku, covariance-shaped random (z ~ N(0, Sigma_{L*})) and negated D_ku."""

    @staticmethod
    def directions(ctx, seed):
        L = ctx.frozen["L_star"]
        R = ctx.cache["train"]["resid_last"][:, L]
        known, unknown = ctx.extras["known_unknown_masks"]
        return {"shuffled_ku": shuffled_label_direction(R, unknown, known, 7000 + 10 * seed),
                "cov_random": covariance_shaped_direction(R, 7001 + 10 * seed),
                "negated_ku": -ctx.dirs["knowledge"][L]}


class MarginGatedSteer:
    """Steer only where the margin gate fires (unsteered abstained or m < tau_dev(1)); elsewhere
    keep the unsteered answer. Same gate as gated refusal, so the difference between the two is
    exactly the score the steered outputs earn on gated items."""

    @staticmethod
    def compute_gate(base_states, margin, tau):
        return (base_states == A_) | (margin < tau)

    @staticmethod
    def compose_outputs(name, base, steered, gate):
        pick = lambda a, b: np.where(gate, b, a)
        return CondOutput(name, pick(base.states, steered.states), pick(base.degenerate, steered.degenerate),
                          pick(base.truncated, steered.truncated), pick(base.n_tokens, steered.n_tokens),
                          [s if g else b for b, s, g in zip(base.abst_type, steered.abst_type, gate)],
                          pick(base.strict_correct, steered.strict_correct),
                          [s if g else b for b, s, g in zip(base.texts, steered.texts, gate)],
                          pick(base.confidence, steered.confidence), "margin (steered on gated items)",
                          pick(base.ans_logprob, steered.ans_logprob),
                          extra={"n_gated": int(gate.sum()), "gate": gate})

    @staticmethod
    def score(out):
        return summary_metrics(out.states, out.degenerate, out.truncated, out.n_tokens)

    @staticmethod
    def compare_with_gated_refusal(composed_states, gated_refusal_states, gate, lam=1.0):
        from scoring import score_vec
        delta = 100.0 * (score_vec(composed_states, lam) - score_vec(gated_refusal_states, lam)).mean()
        on_gated = 100.0 * (gate * score_vec(composed_states, lam)).mean()
        assert np.isclose(delta, on_gated), "gated refusal and gated steering differ off the gate"
        return float(delta)

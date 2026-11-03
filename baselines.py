"""Baselines and controls: unsteered labelling, confidence prompt, margin threshold, popularity
gate, norm-matched random-direction placebo, plus the shared L2 logistic-regression helper."""
import math
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import model_utils as mu
from config import (CONFIDENCE_PROMPT_INSTANCES, LAMBDAS, N_LABEL_SAMPLES, PROBE_C_GRID,
                    QA_MAX_NEW_TOKENS, RANDOM_SEED_BASE, SAMPLE_SEED, USER_TEMPLATE_BASE,
                    USER_TEMPLATE_CONFIDENCE)
from scoring import (A_, C_, W_, apply_threshold, aurc, auroc, classify, fit_threshold,
                     make_output, match_coverage, normalize_answer, risk_coverage, summary_metrics)
from steering import SteeredQA, random_unit_directions, steer


def fit_l2_logreg(X, y, c_grid=PROBE_C_GRID, folds=5, seed=0):
    """Standardised L2 logistic regression; C chosen by stratified k-fold CV AUROC."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    best_c, best_auc = c_grid[0], float("nan")
    if len(np.unique(y)) < 2:
        raise ValueError("single-class labels")
    if len(c_grid) == 1:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=2000)).fit(X, y)
        model.cv_auc_, model.C_ = best_auc, best_c
        return model
    best_auc = -1.0
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for c in c_grid:
        aucs = []
        for tr, va in skf.split(X, y):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
            m.fit(X[tr], y[tr])
            aucs.append(auroc(y[va], m.predict_proba(X[va])[:, 1]))
        if np.nanmean(aucs) > best_auc:
            best_c, best_auc = c, float(np.nanmean(aucs))
    model = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=2000)).fit(X, y)
    model.cv_auc_, model.C_ = best_auc, best_c
    return model


def cross_fitted_proba(X, y, C, folds=5, seed=0):
    """Out-of-fold P(y=1) on the training set (so train-side combinations do not see in-sample fits)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).astype(int)
    out = np.zeros(len(y))
    for tr, va in StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000)).fit(X[tr], y[tr])
        out[va] = m.predict_proba(X[va])[:, 1]
    return out


# ----------------------------------------------------------------------------- unsteered
class UnsteeredQA:
    """Reference condition: base abstention-permitting prompt, greedy, no hook, no gate."""

    def __init__(self, ctx):
        self.ctx = ctx

    def build_prompt(self, item, template=USER_TEMPLATE_BASE, **fill):
        user = template.format(question=item.question, **fill)
        if item.subj:
            return self.ctx.fmt.prompt(user, find=item.subj)
        return self.ctx.fmt.prompt(user), []

    def probe_pass(self, split):
        c = self.ctx
        return mu.probe_pass(c.model, c.fmt, c.prompts[split], c.subj_spans[split], c.cache_layers)

    def answer(self, prompts):
        return mu.generate(self.ctx.model, self.ctx.fmt, prompts, QA_MAX_NEW_TOKENS)

    def classify_output(self, name, gen, items, confidence=None, kind="answer_logprob"):
        return make_output(name, gen, items, confidence, kind)

    def score(self, out):
        return summary_metrics(out.states, out.degenerate, out.truncated, out.n_tokens)

    def sample_labels(self, split, items, greedy_out):
        """8 temperature-1.0 samples per question -> per-sample states and consistency labels."""
        c = self.ctx
        seed = SAMPLE_SEED + {"train": 0, "test": 1, "train_reserve": 2}[split]
        samples = mu.sample(c.model, c.fmt, c.prompts[split], N_LABEL_SAMPLES, QA_MAX_NEW_TOKENS, seed)
        N = len(items)
        st = np.zeros((N, N_LABEL_SAMPLES), dtype=np.int8)
        dg = np.zeros((N, N_LABEL_SAMPLES), dtype=bool)
        entropy = np.zeros(N)
        same_as_greedy = np.zeros(N, dtype=int)
        for i, (it, ss) in enumerate(zip(items, samples)):
            keys = []
            g_norm = normalize_answer(greedy_out.texts[i])
            for k, (text, _, _) in enumerate(ss):
                s, d, _, _ = classify(text, it.aliases)
                st[i, k], dg[i, k] = s, d
                key = "<degenerate>" if d else ("<abstain>" if s == A_ else normalize_answer(text))
                keys.append(key)
                same_as_greedy[i] += int(not d and s != A_ and normalize_answer(text) == g_norm)
            p = np.array(list(Counter(keys).values()), dtype=np.float64) / len(keys)
            entropy[i] = float(-(p * np.log(p)).sum())
        n_correct = (st == C_).sum(1)
        g = greedy_out.states
        labels = {"sample_states": st, "sample_degenerate": dg, "n_correct_samples": n_correct,
                  "semantic_entropy": entropy,
                  "robust_known": (g == C_) & (n_correct >= 7),
                  "consistent_wrong": (g == W_) & (same_as_greedy >= 6),
                  "latent_known": (g != C_) & (n_correct >= 2)}
        labels["unstable"] = ~labels["robust_known"] & ~labels["consistent_wrong"]
        return samples, labels

    def gold_score(self, split, items):
        """Teacher-forced mean per-token log-prob of a gold alias as the assistant response; max
        over the first listed alias plus the two shortest distinct others."""
        c = self.ctx
        prompts, conts, owner = [], [], []
        for i, it in enumerate(items):
            first = it.aliases[0]
            others = sorted({a for a in it.aliases[1:] if a != first}, key=lambda a: (len(a), a))[:2]
            for a in [first] + others:
                ids = c.tok(a, add_special_tokens=False).input_ids
                if ids:
                    prompts.append(c.prompts[split][i])
                    conts.append(ids)
                    owner.append(i)
        lps = mu.teacher_force(c.model, c.fmt, prompts, conts)
        out = np.full(len(items), -np.inf)
        for i, lp in zip(owner, lps):
            out[i] = max(out[i], float(lp.mean()))
        return out

    @staticmethod
    def assign_strata(states, gold, tau_store):
        ks = np.where(states == C_, "known", np.where(gold >= tau_store, "stored_outcompeted", "missing"))
        return ks.astype(object)


# ----------------------------------------------------------------------------- confidence prompt
class ConfidencePromptQA:
    """Unsteered model told the scoring rule; answers only above t = lam / (1 + lam). No free
    parameter, no hook: only the instruction text changes."""

    def __init__(self, ctx, instance):
        self.ctx = ctx
        self.instance = instance
        self.fill = CONFIDENCE_PROMPT_INSTANCES[instance]

    def build_prompt(self, item):
        return UnsteeredQA(self.ctx).build_prompt(item, USER_TEMPLATE_CONFIDENCE, **self.fill)[0]

    def answer(self, items):
        prompts = [self.build_prompt(it) for it in items]
        return mu.generate(self.ctx.model, self.ctx.fmt, prompts, QA_MAX_NEW_TOKENS)

    def classify_output(self, gen, items):
        return make_output(self.instance, gen, items, extra={"lambda_stated": self.fill["lam"]})

    def score(self, out):
        m = summary_metrics(out.states, out.degenerate, out.truncated, out.n_tokens)
        m["S_matching_lambda"] = m[f"S_lam{float(self.fill['lam']):g}"]
        return m


# ----------------------------------------------------------------------------- threshold gates
class ConfidenceThreshold:
    """Post-hoc gate on unsteered outputs: abstain if unsteered abstained or signal < tau.
    Cannot change an answer, only withdraw it."""

    SIGNALS = ("margin", "ku_projection", "wq_answer_projection", "refusal_projection")

    def __init__(self, signal="margin"):
        assert signal in self.SIGNALS
        self.signal = signal
        self.taus = {}

    def compute_signal(self, ctx, split):
        return ctx.signal(self.signal, split)

    def fit_threshold_on_dev(self, base_states_dev, signal_dev):
        self.taus = {lam: fit_threshold(base_states_dev, signal_dev, lam) for lam in LAMBDAS}
        return self.taus

    @staticmethod
    def apply_threshold(base_states, signal, tau):
        return apply_threshold(base_states, signal, tau)

    @staticmethod
    def match_coverage(base_states, signal, tiebreak, k):
        return match_coverage(base_states, signal, tiebreak, k)

    @staticmethod
    def risk_coverage_curve(base_states, signal, tiebreak):
        return risk_coverage(base_states, signal, tiebreak)

    @staticmethod
    def aurc(base_states, signal, tiebreak):
        return aurc(base_states, signal, tiebreak)


class PopularityGate(ConfidenceThreshold):
    """Frequency-only gate on log10(s_pop + 1); no model internals. PopQA only."""

    def __init__(self):
        self.signal = "popularity"
        self.taus = {}

    def compute_signal(self, ctx, split):
        return np.array([it.log_pop for it in ctx.items[split]])

    @staticmethod
    def fit_popularity_surprisal_logreg(log_pop, surprisal, y):
        X = PopularityGate.features(log_pop, surprisal, np.nanmean(surprisal))
        model = fit_l2_logreg(X, y)
        model.surprisal_fill_ = float(np.nanmean(surprisal))
        return model

    @staticmethod
    def features(log_pop, surprisal, fill):
        s = np.where(np.isfinite(surprisal), surprisal, fill)
        return np.stack([log_pop, s], 1)


# ----------------------------------------------------------------------------- random placebo
class RandomDirectionPlacebo(SteeredQA):
    """Norm-matched isotropic random directions at the shared site (layer L_star, last prompt token
    + every generated token, added norm c_star * rbar_{L_star})."""

    def __init__(self, ctx, n_draws):
        super().__init__(ctx)
        self.directions = self.sample_directions(n_draws)

    def sample_directions(self, n):
        return random_unit_directions(self.ctx.spec["d_model"], n, RANDOM_SEED_BASE)

    def intervention(self, k, multiplier=1.0, layer=None, positions="both"):
        f = self.ctx.frozen
        L = f["L_star"] if layer is None else layer
        return steer(f"random_dir_placebo_s{k}", L, self.directions[k],
                     multiplier * f["c_star"] * float(self.ctx.rbar[L]), positions)

    def run(self, k, prompts, items, multiplier=1.0, max_new_tokens=QA_MAX_NEW_TOKENS):
        gen = self.answer(self.intervention(k, multiplier), prompts, max_new_tokens)
        return make_output(f"random_dir_placebo_s{k}", gen, items,
                           extra={"seed": RANDOM_SEED_BASE + k, "multiplier": multiplier})

    @staticmethod
    def score(out):
        return summary_metrics(out.states, out.degenerate, out.truncated, out.n_tokens)

    @staticmethod
    def summarize_distribution(values, learned=None):
        v = np.asarray(values, dtype=np.float64)
        out = {"n_draws": int(len(v)), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
               "median": float(np.median(v)), "p05": float(np.percentile(v, 5)),
               "p95": float(np.percentile(v, 95)), "min": float(v.min()), "max": float(v.max())}
        if learned is not None and math.isfinite(learned):
            n_ge = int((v >= learned).sum())
            out.update({"learned": float(learned), "rank_from_top": n_ge + 1,
                        "empirical_p": (1 + n_ge) / (len(v) + 1),
                        "z": float((learned - out["mean"]) / out["std"]) if out["std"] > 0 else float("nan")})
        return out

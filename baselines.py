"""Baselines and controls: unsteered labelling, confidence prompt, margin threshold, popularity
gate, norm-matched random-direction placebo, majority-vote and best-of-k sampling baselines
(amendment A9), plus the shared L2 logistic-regression helper."""
import math
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import model_utils as mu
from config import (CONFIDENCE_PROMPT_INSTANCES, LAMBDAS, N_LABEL_SAMPLES, PROBE_C_GRID,
                    QA_MAX_NEW_TOKENS, RANDOM_SEED_BASE, SAMPLE_SEED, SEED_SAMPLE_STRIDE, USER_TEMPLATE_BASE,
                    USER_TEMPLATE_CONFIDENCE)
from scoring import (A_, C_, W_, CondOutput, apply_threshold, aurc, auroc, classify, fit_threshold,
                     make_output, match_coverage, normalize_answer, risk_coverage, summary_metrics)
from steering import SteeredQA, random_unit_directions, steer

SAMPLE_SPLIT_OFFSET = {"train": 0, "test": 1, "train_reserve": 2, "dev": 3}


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


def sample_cluster_key(text, state, degenerate, slot):
    """Cluster key for semantic entropy and majority vote. Answers cluster by normalised text.
    Amendment A5: an abstained or degenerate sample asserts no answer, so it is its own singleton
    cluster (`slot` makes it unique). In the first run all abstentions shared one cluster, so a model
    that abstained on all 8 samples got entropy 0 ("certain"); for the abstention-heavy Qwen this
    made low entropy predict *wrong* and gave the semantic-entropy probe test AUROC 0.41."""
    if degenerate:
        return f"<degenerate:{slot}>"
    if state == A_:
        return f"<abstain:{slot}>"
    return normalize_answer(text)


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
        """Residual caches, entity surprisal and log P(abstain) = log sum over the IDK written forms
        of P(form + end-of-turn) (amendment A5)."""
        c = self.ctx
        r_last, r_idk, lp_canon, surpr = mu.probe_pass(c.model, c.fmt, c.prompts[split], c.subj_spans[split],
                                                       c.cache_layers)
        lps = [lp_canon]
        for ids in c.fmt.idk_variant_ids[1:]:
            tf = mu.teacher_force(c.model, c.fmt, c.prompts[split], [ids + [c.fmt.eot_id]] * len(c.prompts[split]))
            lps.append(np.array([float(x.sum()) for x in tf]))
        return r_last, r_idk, np.logaddexp.reduce(np.stack(lps, 1), axis=1), surpr

    def answer(self, prompts):
        return mu.generate(self.ctx.model, self.ctx.fmt, prompts, QA_MAX_NEW_TOKENS)

    def classify_output(self, name, gen, items, confidence=None, kind="answer_logprob"):
        return make_output(name, gen, items, confidence, kind)

    def score(self, out):
        return summary_metrics(out.states, out.degenerate, out.truncated, out.n_tokens)

    def sample_labels(self, split, items, greedy_out, seed=0):
        """8 temperature-1.0 samples per question -> per-sample states and consistency labels.
        Sampling seed = SAMPLE_SEED + split offset + 1000 * seed (seed 0 = the registered seeds)."""
        c = self.ctx
        sseed = SAMPLE_SEED + SAMPLE_SPLIT_OFFSET[split] + SEED_SAMPLE_STRIDE * seed
        samples = mu.sample(c.model, c.fmt, c.prompts[split], N_LABEL_SAMPLES, QA_MAX_NEW_TOKENS, sseed)
        N = len(items)
        st = np.zeros((N, N_LABEL_SAMPLES), dtype=np.int8)
        dg = np.zeros((N, N_LABEL_SAMPLES), dtype=bool)
        entropy = np.zeros(N)
        n_abst = np.zeros(N, dtype=int)
        same_as_greedy = np.zeros(N, dtype=int)
        for i, (it, ss) in enumerate(zip(items, samples)):
            keys = []
            g_norm = normalize_answer(greedy_out.texts[i])
            for k, (text, _, _) in enumerate(ss):
                s, d, _, _ = classify(text, it.aliases)
                st[i, k], dg[i, k] = s, d
                keys.append(sample_cluster_key(text, s, d, k))
                n_abst[i] += int(s == A_ and not d)
                same_as_greedy[i] += int(not d and s != A_ and normalize_answer(text) == g_norm)
            p = np.array(list(Counter(keys).values()), dtype=np.float64) / len(keys)
            entropy[i] = float(-(p * np.log(p)).sum())
        n_correct = (st == C_).sum(1)
        g = greedy_out.states
        labels = {"sample_states": st, "sample_degenerate": dg, "n_correct_samples": n_correct,
                  "semantic_entropy": entropy, "n_abstained_samples": n_abst, "sample_seed": sseed,
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

    SIGNALS = ("margin", "ku_projection", "read_projection", "wq_answer_projection", "refusal_projection")

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


# ----------------------------------------------------------------------------- sampling baselines (A9)
class SampleVoteQA:
    """Majority vote and best-of-k over the 8 saved temperature-1 samples (no extra generation;
    best-of-k needs one teacher-forced pass per sample). Each is gated like the margin threshold:
    abstain if the chosen sample abstains or its confidence < tau_dev(lambda)."""

    def __init__(self, ctx):
        self.ctx = ctx

    @staticmethod
    def majority(samples_i, states_i, degen_i):
        keys = [sample_cluster_key(t, s, d, k) for k, ((t, _, _), s, d) in enumerate(zip(samples_i, states_i, degen_i))]
        cnt = Counter(keys)
        best = max(cnt, key=lambda k: (cnt[k], -keys.index(k)))
        j = keys.index(best)
        return j, cnt[best] / len(keys)

    def majority_outputs(self, split):
        c = self.ctx
        samples, lab = c.samples[split], c.labels[split]
        pick, conf = [], []
        for i in range(len(samples)):
            j, share = self.majority(samples[i], lab["sample_states"][i], lab["sample_degenerate"][i])
            pick.append(j)
            conf.append(share)
        return self._compose("majority_vote_k8", split, pick, np.array(conf), "vote_share")

    def best_of_k_outputs(self, split):
        c = self.ctx
        samples = c.samples[split]
        n, k = len(samples), len(samples[0])
        P, conts = [], []
        for i in range(n):
            for s in samples[i]:
                P.append(c.prompts[split][i])
                conts.append(list(s[1]) + [c.fmt.eot_id])
        lps = mu.teacher_force(c.model, c.fmt, P, conts)
        lp = np.array([float(x.sum()) for x in lps]).reshape(n, k)
        pick = lp.argmax(1)
        conf = lp[np.arange(n), pick] - c.cache[split]["logp_idk"]
        return self._compose("best_of_k8", split, list(pick), conf, "best_sample_margin"), lp

    def _compose(self, name, split, pick, conf, kind):
        c = self.ctx
        items, samples = c.items[split], c.samples[split]
        texts = [samples[i][j][0] for i, j in enumerate(pick)]
        toks = [samples[i][j][1] for i, j in enumerate(pick)]
        trunc = [samples[i][j][2] for i, j in enumerate(pick)]
        cls = [classify(t, it.aliases) for t, it in zip(texts, items)]
        return CondOutput(name, [x[0] for x in cls], [x[1] for x in cls], trunc, [len(t) for t in toks],
                          [x[2] for x in cls], [x[3] for x in cls], texts, conf, kind, None, toks,
                          extra={"picked_sample": np.array(pick)})


# ----------------------------------------------------------------------------- random placebo
class RandomDirectionPlacebo(SteeredQA):
    """Norm-matched isotropic random directions at the shared site (layer L_star, last prompt token
    + every generated token, added norm alpha* = the knowledge direction's frozen norm)."""

    def __init__(self, ctx, n_draws, seed_base=RANDOM_SEED_BASE):
        super().__init__(ctx)
        self.seed_base = seed_base
        self.directions = self.sample_directions(n_draws)

    def sample_directions(self, n):
        return random_unit_directions(self.ctx.spec["d_model"], n, self.seed_base)

    def intervention(self, k, multiplier=1.0, layer=None, positions="both", alpha=None):
        f = self.ctx.frozen
        L = f["L_star"] if layer is None else layer
        a = f["alpha"] if alpha is None else alpha
        return steer(f"random_dir_placebo_s{k}", L, self.directions[k], multiplier * a, positions)

    def run(self, k, prompts, items, multiplier=1.0, max_new_tokens=QA_MAX_NEW_TOKENS):
        gen = self.answer(self.intervention(k, multiplier), prompts, max_new_tokens)
        return make_output(f"random_dir_placebo_s{k}", gen, items,
                           extra={"seed": self.seed_base + k, "multiplier": multiplier})

    @staticmethod
    def score(out):
        return summary_metrics(out.states, out.degenerate, out.truncated, out.n_tokens)

    @staticmethod
    def summarize_distribution(values, learned=None):
        v = np.asarray(values, dtype=np.float64)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            return {"n_draws": 0}
        out = {"n_draws": int(len(v)), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
               "median": float(np.median(v)), "p05": float(np.percentile(v, 5)),
               "p95": float(np.percentile(v, 95)), "min": float(v.min()), "max": float(v.max())}
        if learned is not None and math.isfinite(learned):
            n_ge = int((v >= learned).sum())
            out.update({"learned": float(learned), "rank_from_top": n_ge + 1,
                        "empirical_p": (1 + n_ge) / (len(v) + 1),
                        "z": float((learned - out["mean"]) / out["std"]) if out["std"] > 0 else float("nan")})
        return out

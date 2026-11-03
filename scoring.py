"""Answer normalisation, alias matching, abstention / degeneration detection, metrics and
statistics (paired bootstrap, exact McNemar, Holm, TOST, AUROC, trend test)."""
import re
import unicodedata
from collections import Counter

import numpy as np
from scipy.stats import binomtest, norm, rankdata, spearmanr

from config import BOOTSTRAP_SEED, N_BOOTSTRAP

C_, A_, W_ = 1, 0, -1   # outcome codes: correct / abstained / wrong (degenerate counts as wrong)
STATE_NAMES = {C_: "correct", A_: "abstained", W_: "wrong"}

ABSTENTION_PHRASES = ["i don't know", "i do not know", "i'm not sure", "i am not sure", "not sure",
                      "i cannot", "i can't", "i'm unable", "i am unable", "unable to",
                      "i don't have", "no information", "cannot determine", "i'm sorry", "sorry",
                      "unknown"]
REFUSAL_PHRASING = ["i'm sorry", "sorry", "i cannot", "i can't", "i'm unable", "unable to"]
SHORT_IDK_PHRASES = ["i don't know", "i do not know", "unknown"]
INSTRUCTION_REFUSAL_PHRASES = ["i'm sorry", "i am sorry", "i apologize", "as an ai",
                               "as a language model", "as an assistant", "i cannot", "i can't",
                               "i am unable to", "i'm unable to", "i am not able to",
                               "i'm not able to"]
ARTICLES = {"a", "an", "the"}


# ----------------------------------------------------------------------------- normalisation
def _strip_accents(s):
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def normalize_answer(s):
    """lowercase, NFKD (accents dropped), punctuation -> space, articles removed, whitespace collapsed."""
    s = _strip_accents(s).lower()
    s = "".join(ch if ch.isalnum() else " " for ch in s)
    return " ".join(w for w in s.split() if w not in ARTICLES)


def light_normalize(s):
    """For phrase detection: lowercase, NFKD, unified apostrophes, leading quotes/markup removed."""
    s = _strip_accents(s).lower().replace("’", "'").replace("‘", "'").replace("`", "'")
    s = " ".join(s.split())
    return s.lstrip(" \"'*_-:>#.([")


def _starts_with_phrase(text, phrase):
    return text.startswith(phrase) and (len(text) == len(phrase) or not text[len(phrase)].isalnum())


def is_degenerate(text):
    t = text.strip()
    if not t:
        return True
    chars = [c for c in t if not c.isspace()]
    if sum(c.isalnum() for c in chars) / len(chars) < 0.30:
        return True
    words = re.findall(r"[\w']+", t.lower())
    if len(words) >= 6 and Counter(words).most_common(1)[0][1] / len(words) >= 0.5:
        return True
    tris = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return len(tris) != len(set(tris))


def abstention_type(text):
    """None if not an abstention, else 'refusal_phrasing' or 'idk_phrasing'."""
    t = light_normalize(text)
    starts = any(_starts_with_phrase(t, p) for p in ABSTENTION_PHRASES)
    short_idk = len(t.split()) <= 6 and any(p in t for p in SHORT_IDK_PHRASES)
    if not (starts or short_idk):
        return None
    return "refusal_phrasing" if any(_starts_with_phrase(t, p) for p in REFUSAL_PHRASING) else "idk_phrasing"


def alias_match(text, aliases, strict=False):
    out = normalize_answer(text)
    padded = f" {out} "
    for a in aliases:
        na = normalize_answer(a)
        if not na:
            continue
        if (out == na) if strict else (f" {na} " in padded):
            return True
    return False


def classify(text, aliases):
    """Order: degenerate -> abstained -> answer (correct / wrong).
    Returns (state, degenerate, abstention subtype, strict-exact correct)."""
    if is_degenerate(text):
        return W_, True, None, False
    at = abstention_type(text)
    if at is not None:
        return A_, False, at, False
    if alias_match(text, aliases):
        return C_, False, None, alias_match(text, aliases, strict=True)
    return W_, False, None, False


def is_instruction_refusal(text):
    head = light_normalize(text)[:64]
    return any(p in head for p in INSTRUCTION_REFUSAL_PHRASES)


# ----------------------------------------------------------------------------- per-question scores
def score_vec(states, lam=1.0):
    return np.where(states == C_, 1.0, np.where(states == A_, 0.0, -lam))


def abstention_score(states, lam=1.0):
    """S_lambda in points (x100)."""
    return float(100.0 * score_vec(states, lam).mean()) if len(states) else float("nan")


def _safe_div(a, b):
    return float(a / b) if b else float("nan")


def summary_metrics(states, degenerate=None, truncated=None, n_tokens=None):
    n = len(states)
    c, a, w = (states == C_).sum(), (states == A_).sum(), (states == W_).sum()
    out = {"n": int(n), "abstention_score": abstention_score(states),
           "accuracy_answered": _safe_div(c, c + w), "abstention_rate": _safe_div(a, n),
           "wrong_rate": _safe_div(w, n), "accuracy": _safe_div(c, n)}
    for lam in (0.5, 1.0, 2.0, 3.0):
        out[f"S_lam{lam:g}"] = abstention_score(states, lam)
    if degenerate is not None:
        out["degenerate_rate"] = float(np.mean(degenerate)) if n else float("nan")
    if truncated is not None:
        out["truncation_rate"] = float(np.mean(truncated)) if n else float("nan")
    if n_tokens is not None:
        ans = states != A_
        out["mean_answer_tokens"] = float(np.mean(n_tokens[ans])) if ans.any() else float("nan")
    return out


def transitions(base, cond):
    """3x3 counts, rows = unsteered state (C, A, W), cols = condition state (C, A, W)."""
    order = [C_, A_, W_]
    return [[int(((base == r) & (cond == c)).sum()) for c in order] for r in order]


def transition_metrics(base, cond, robust_known=None):
    nC, nA, nW = (base == C_).sum(), (base == A_).sum(), (base == W_).sum()
    wc = ((base == W_) & (cond == C_)).sum()
    wa = ((base == W_) & (cond == A_)).sum()
    ac = ((base == A_) & (cond == C_)).sum()
    cw = ((base == C_) & (cond == W_)).sum()
    ca = ((base == C_) & (cond == A_)).sum()
    out = {"transitions_3x3": transitions(base, cond),
           "recovered_rate": _safe_div(wc, nW), "W_to_A_rate": _safe_div(wa, nW),
           "A_to_C_rate": _safe_div(ac, nA), "harmed_rate": _safe_div(cw + ca, nC),
           "over_abstention_rate": _safe_div(ca, nC),
           "selectivity_lift": float(((wa + 0.5) / (nW + 1)) / ((ca + 0.5) / (nC + 1)))}
    if robust_known is not None and robust_known.any():
        rk = robust_known
        out["collateral"] = _safe_div(((base[rk] == C_) & (cond[rk] != C_)).sum(), rk.sum())
        out["over_abstention_robust_known"] = _safe_div(((base[rk] == C_) & (cond[rk] == A_)).sum(), rk.sum())
    return out


# ----------------------------------------------------------------------------- thresholds / coverage
def order_by_signal(signal, tiebreak):
    """Indices sorted by signal descending, ties by question-id hash ascending."""
    return np.lexsort((tiebreak, -signal))


def apply_threshold(base, signal, tau):
    """Abstain if the base output abstained or signal < tau; otherwise keep the base outcome."""
    out = base.copy()
    out[(base != A_) & (signal < tau)] = A_
    return out


def fit_threshold(base, signal, lam):
    """tau maximising S_lambda over all unique signal values of answered items plus -inf
    (ties -> the smallest tau, i.e. the highest coverage)."""
    ans = base != A_
    s = score_vec(base, lam)[ans]
    sig = signal[ans]
    if not ans.any():
        return float("-inf")
    order = np.argsort(-sig, kind="stable")
    sig_sorted, cum = sig[order], np.cumsum(s[order])
    # threshold at sig_sorted[j] keeps every item with signal >= that value
    uniq = np.unique(sig_sorted)
    # count of items with signal >= uniq[k] (sig_sorted[::-1] is ascending)
    n_keep = len(sig_sorted) - np.searchsorted(sig_sorted[::-1], uniq, side="left")
    totals = np.concatenate([[cum[-1]], cum[n_keep - 1]])
    taus = np.concatenate([[-np.inf], uniq])
    best = np.max(totals)
    cand = taus[np.isclose(totals, best)]
    return float(cand.min())


def match_coverage(base, signal, tiebreak, k):
    """Answer the k base-answered items with the highest signal; abstain elsewhere. None if
    k exceeds the base answered count (unmatched)."""
    ans_idx = np.flatnonzero(base != A_)
    if k > len(ans_idx):
        return None
    keep = ans_idx[order_by_signal(signal[ans_idx], tiebreak[ans_idx])[:k]]
    out = np.full_like(base, A_)
    out[keep] = base[keep]
    return out


def matched_score(base, signal, tiebreak, k, lam=1.0):
    m = match_coverage(base, signal, tiebreak, k)
    return float("nan") if m is None else abstention_score(m, lam)


def surplus_vs_threshold(cond, base, signal, tiebreak, lam=1.0):
    k = int((cond != A_).sum())
    return abstention_score(cond, lam) - matched_score(base, signal, tiebreak, k, lam)


def risk_coverage(states, conf, tiebreak):
    """Selective risk W/(C+W) on a 1% coverage grid, answering the most confident answered items."""
    n = len(states)
    ans_idx = np.flatnonzero(states != A_)
    ranked = states[ans_idx[order_by_signal(conf[ans_idx], tiebreak[ans_idx])]]
    max_cov = len(ans_idx) / n
    grid = np.round(np.arange(0.01, max_cov + 1e-9, 0.01), 2)
    risks = []
    cum_w = np.cumsum(ranked == W_)
    for c in grid:
        k = max(1, int(round(c * n)))
        k = min(k, len(ranked))
        risks.append(cum_w[k - 1] / k)
    return grid, np.array(risks)


def aurc(states, conf, tiebreak):
    grid, risks = risk_coverage(states, conf, tiebreak)
    if len(grid) < 2:
        return float(risks[0]) if len(grid) else float("nan")
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(risks, grid) / (grid[-1] - grid[0]))


def selective_accuracy_at(states, conf, tiebreak, coverage):
    n = len(states)
    ans_idx = np.flatnonzero(states != A_)
    k = int(round(coverage * n))
    if k > len(ans_idx) or k == 0:
        return float("nan")
    top = states[ans_idx[order_by_signal(conf[ans_idx], tiebreak[ans_idx])[:k]]]
    return float((top == C_).mean())


# ----------------------------------------------------------------------------- statistics
_BOOT_CACHE = {}


def bootstrap_indices(n, B=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    key = (n, B, seed)
    if key not in _BOOT_CACHE:
        _BOOT_CACHE[key] = np.random.default_rng(seed).integers(0, n, size=(B, n))
    return _BOOT_CACHE[key]


def bootstrap(stat_fn, n, alpha=0.05):
    """Paired bootstrap over questions: stat_fn(idx) is recomputed on each resample (so coverage
    matching / curve construction are redone inside the resample). Returns mean, sd, CI."""
    idx = bootstrap_indices(n)
    vals = np.array([stat_fn(r) for r in idx], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"boot_mean": float("nan"), "sd": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "ci90_low": float("nan"), "ci90_high": float("nan"), "n_valid": 0}
    return {"boot_mean": float(vals.mean()), "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "ci_low": float(np.percentile(vals, 100 * alpha / 2)),
            "ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "ci90_low": float(np.percentile(vals, 5)), "ci90_high": float(np.percentile(vals, 95)),
            "n_valid": int(len(vals))}


def mcnemar_exact(x_correct, y_correct):
    """Exact McNemar: two-sided binomial test on the discordant pairs."""
    b = int((x_correct & ~y_correct).sum())
    c = int((~x_correct & y_correct).sum())
    if b + c == 0:
        return 1.0, b, c
    return float(binomtest(b, b + c, 0.5).pvalue), b, c


def holm(pvals):
    p = np.asarray(pvals, dtype=np.float64)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def tost_equivalent(ci90_low, ci90_high, margin):
    return bool(ci90_low > -margin and ci90_high < margin)


def auroc(labels, scores):
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(scores)
    labels, scores = labels[ok], scores[ok]
    n1, n0 = labels.sum(), (~labels).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).correlation)


def cochran_armitage(successes, totals, doses):
    """Two-sided Cochran-Armitage trend test; returns (z, p)."""
    s, n, x = (np.asarray(v, dtype=np.float64) for v in (successes, totals, doses))
    N, p_bar = n.sum(), s.sum() / n.sum()
    if p_bar in (0.0, 1.0):
        return 0.0, 1.0
    t = np.sum(x * (s - n * p_bar))
    var = p_bar * (1 - p_bar) * (np.sum(n * x ** 2) - np.sum(n * x) ** 2 / N)
    if var <= 0:
        return 0.0, 1.0
    z = t / np.sqrt(var)
    return float(z), float(2 * norm.sf(abs(z)))


def empirical_percentile(value, draws):
    """Share of draws strictly below value, in percent (ties count half)."""
    d = np.asarray(draws, dtype=np.float64)
    return float(100.0 * ((d < value).sum() + 0.5 * (d == value).sum()) / len(d))


# ----------------------------------------------------------------------------- per-condition outputs
class CondOutput:
    """Per-question outcomes of one condition on one question set (fixed question order)."""

    def __init__(self, name, states, degenerate, truncated, n_tokens, abst_type, strict_correct,
                 texts, confidence, confidence_kind, ans_logprob=None, token_ids=None, extra=None):
        self.name = name
        self.states = np.asarray(states, dtype=np.int8)
        self.degenerate = np.asarray(degenerate, dtype=bool)
        self.truncated = np.asarray(truncated, dtype=bool)
        self.n_tokens = np.asarray(n_tokens)
        self.abst_type = list(abst_type)
        self.strict_correct = np.asarray(strict_correct, dtype=bool)
        self.texts = list(texts)
        self.confidence = np.asarray(confidence, dtype=np.float64)
        self.confidence_kind = confidence_kind
        self.ans_logprob = (np.asarray(ans_logprob, dtype=np.float64) if ans_logprob is not None
                            else np.full(len(self.states), np.nan))
        self.token_ids = token_ids
        self.extra = extra or {}

    def __len__(self):
        return len(self.states)

    def subset(self, idx, name=None):
        idx = np.asarray(idx)
        return CondOutput(name or self.name, self.states[idx], self.degenerate[idx],
                          self.truncated[idx], self.n_tokens[idx], [self.abst_type[i] for i in idx],
                          self.strict_correct[idx], [self.texts[i] for i in idx],
                          self.confidence[idx], self.confidence_kind, self.ans_logprob[idx],
                          None if self.token_ids is None else [self.token_ids[i] for i in idx],
                          dict(self.extra))

    def strict_states(self):
        """Sensitivity analysis: strict exact match instead of word-boundary containment."""
        s = self.states.copy()
        s[(s == C_) & ~self.strict_correct] = W_
        return s


def make_output(name, gen, items, confidence=None, confidence_kind="answer_logprob", extra=None):
    cls = [classify(t, it.aliases) for t, it in zip(gen.texts, items)]
    conf = gen.ans_logprob if confidence is None else confidence
    return CondOutput(name, [c[0] for c in cls], [c[1] for c in cls], gen.truncated, gen.n_tokens,
                      [c[2] for c in cls], [c[3] for c in cls], gen.texts, conf, confidence_kind,
                      gen.ans_logprob, gen.token_ids, extra)


def policy_output(name, base, states, confidence=None, extra=None):
    """Post-hoc policy on a base output (threshold gate): withdrawn answers become abstentions."""
    states = np.asarray(states, dtype=np.int8)
    withdrawn = (base.states != A_) & (states == A_)
    texts = [("<withdrawn by gate> " + t) if w else t for t, w in zip(base.texts, withdrawn)]
    abst = ["gate" if w else a for a, w in zip(base.abst_type, withdrawn)]
    return CondOutput(name, states, base.degenerate & ~withdrawn, base.truncated & ~withdrawn,
                      base.n_tokens, abst, base.strict_correct & (states == C_), texts,
                      base.confidence if confidence is None else confidence, base.confidence_kind,
                      base.ans_logprob, base.token_ids, extra)

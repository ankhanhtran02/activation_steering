"""Method outputs, computed offline from the per-question records and thresholds (spec 7, appendix A), and the
dev tuning that produces the frozen configuration.

Every thresholded method uses its own dev-tuned threshold(s), chosen separately for U_1 (primary) and U_3
(sensitivity). Single thresholds come from the dev quantiles of the method's score over answered drafts at
5% steps, plus -inf and +inf. Ties go to the smaller threshold (higher coverage, as SQ2's fit_threshold),
then to the earlier configuration in grid order.
"""
from dataclasses import dataclass

import numpy as np

from scoring import A_, C_, W_
from vesecs.search import STOPS_ON_INVALID, replay

INF = float("inf")


@dataclass
class Out:
    label: np.ndarray          # strict label per question (C_ / A_ / W_)
    label_sq2: np.ndarray
    fpe: np.ndarray
    text: list
    changed: np.ndarray        # output differs from A1


def utility(labels, lam):
    labels = np.asarray(labels)
    return float(100.0 * np.mean(np.where(labels == C_, 1.0, np.where(labels == W_, -float(lam), 0.0)))) if len(labels) else float("nan")


def counts(labels):
    labels = np.asarray(labels)
    n = max(len(labels), 1)
    return {"right": 100.0 * float(np.sum(labels == C_)) / n, "wrong": 100.0 * float(np.sum(labels == W_)) / n,
            "idk": 100.0 * float(np.sum(labels == A_)) / n}


def tau_grid(scores, step=0.05):
    s = np.asarray([x for x in scores if x is not None and np.isfinite(x)], dtype=np.float64)
    qs = np.quantile(s, np.round(np.arange(0, 1 + 1e-9, step), 10)) if len(s) else np.zeros(0)
    return [-INF] + sorted(set(float(q) for q in qs)) + [INF]


class SplitRecords:
    """Per-question arrays of one model on one split (loaded by vesecs.pipeline.load_records)."""

    def __init__(self, items, drafts, scores, tries=None, b5=None, searches=None, ecs=None, judges=None, t_idk=None,
                 n_sel=10):
        self.items = items
        self.N = len(items)
        self.qids = [it["qid"] for it in items]
        self.a1_text = drafts["texts"]
        self.a1_label = np.asarray(drafts["label_strict"])
        self.a1_label2 = np.asarray(drafts["label_sq2"])
        self.a1_idk = np.asarray(drafts["is_idk"], dtype=bool)
        self.first_token = drafts["first_token"]
        self.sD = np.asarray(scores["sD"], dtype=np.float64)
        self.sD_before = np.asarray(scores["sD_before"], dtype=np.float64)
        self.sD_after = np.asarray(scores["sD_after"], dtype=np.float64)
        self.m = np.asarray(scores["margin"], dtype=np.float64)
        self.tries = tries
        self.b5 = b5
        self.searches = searches or {}
        self.ecs = ecs
        self.judges = judges or {}
        self.t_idk = t_idk
        self.n_sel = n_sel

    @property
    def answered(self):
        return ~self.a1_idk

    def subset(self, keep_qids):
        keep = set(keep_qids)
        idx = [i for i, q in enumerate(self.qids) if q in keep]
        return self.take(idx)

    def take(self, idx):
        idx = list(idx)
        pick = lambda L: None if L is None else [L[i] for i in idx]
        sr = SplitRecords.__new__(SplitRecords)
        sr.items = pick(self.items)
        sr.N = len(idx)
        sr.qids = pick(self.qids)
        sr.a1_text = pick(self.a1_text)
        for k in ("a1_label", "a1_label2", "a1_idk", "sD", "sD_before", "sD_after", "m"):
            setattr(sr, k, getattr(self, k)[idx])
        sr.first_token = pick(self.first_token)
        sr.tries, sr.b5, sr.ecs = pick(self.tries), pick(self.b5), pick(self.ecs)
        sr.searches = {k: pick(v) for k, v in self.searches.items()}
        sr.judges = {k: (np.asarray(v)[idx] if isinstance(v, np.ndarray) else pick(v)) for k, v in self.judges.items()}
        sr.t_idk, sr.n_sel = self.t_idk, self.n_sel
        sr.ecs_beta, sr.ecs_variants = getattr(self, "ecs_beta", None), getattr(self, "ecs_variants", [])
        return sr


# ----------------------------------------------------------------------------- outputs
def _a1(sr, fpe=1.0):
    return Out(sr.a1_label.copy(), sr.a1_label2.copy(), np.full(sr.N, float(fpe)), list(sr.a1_text), np.zeros(sr.N, bool))


def gate(sr, score, tau, fpe):
    """A1 if answered and score >= tau, else I don't know (the draft's own I don't know stays)."""
    score = np.asarray(score, dtype=np.float64)
    keep = sr.a1_idk | (np.isfinite(score) & (score >= tau))
    lab = np.where(keep, sr.a1_label, A_)
    lab2 = np.where(keep, sr.a1_label2, A_)
    text = [t if k else "I don't know. [gated]" for t, k in zip(sr.a1_text, keep)]
    return Out(lab, lab2, np.full(sr.N, float(fpe)), text, ~keep)


def search_out(sr, rounds_list, tau, method, max_rounds=None, key="sD_clean"):
    stop = STOPS_ON_INVALID[method]
    lab, lab2, fpe, text, changed = [], [], [], [], []
    for i in range(sr.N):
        out, cost = replay(sr.sD[i], sr.a1_idk[i], rounds_list[i], tau, stop, max_rounds, key)
        if out == "A1":
            lab.append(sr.a1_label[i]); lab2.append(sr.a1_label2[i]); text.append(sr.a1_text[i]); changed.append(False)
        elif out == "IDK":
            lab.append(A_); lab2.append(A_); text.append("I don't know. [search]"); changed.append(True)
        else:
            lab.append(out["label_strict"]); lab2.append(out["label_sq2"]); text.append(out["text"]); changed.append(True)
        fpe.append(cost)
    return Out(np.array(lab), np.array(lab2), np.array(fpe, dtype=np.float64), text, np.array(changed))


def ecs_arrays(sr, variant):
    """[N, k] arrays over the candidate slots C (first k slots) for one echo variant."""
    k = len(sr.ecs[0]["C"]) if sr.N else 3
    E = np.full((sr.N, k), np.nan)
    lab = np.zeros((sr.N, k), int)
    lab2 = np.zeros((sr.N, k), int)
    sd = np.full((sr.N, k), np.nan)
    idk = np.zeros((sr.N, k), bool)
    texts = [[None] * k for _ in range(sr.N)]
    top_is_idk = np.zeros(sr.N, bool)
    for i, r in enumerate(sr.ecs):
        top_is_idk[i] = r["top_a"] == sr.t_idk
        for j, c in enumerate(r["C"][:k]):
            E[i, j] = r["E"][variant][j]
            comp = r["completion"][str(j)]
            lab[i, j], lab2[i, j], idk[i, j], texts[i][j] = comp["label_strict"], comp["label_sq2"], comp["is_idk"], comp["text"]
            sd[i, j] = np.nan if comp["sD"] is None else comp["sD"]
    return {"E": E, "lab": lab, "lab2": lab2, "sD": sd, "idk": idk, "texts": texts, "top_is_idk": top_is_idk}


def ecs_choice(arr, tk, ts):
    """Index of the output per question: -1 = keep A1, j = switch to candidate j (spec 6.1 step 5)."""
    E = arr["E"]
    keep_first = (~arr["top_is_idk"]) & (np.nan_to_num(E[:, 0], nan=-INF) >= tk)
    j = np.argmax(np.nan_to_num(E, nan=-INF), axis=1)
    best = np.nan_to_num(E[np.arange(len(j)), j], nan=-INF)
    switch = (~keep_first) & (best >= ts)
    return np.where(switch, j, -1)


def ecs_out(sr, arr, tk, ts, tD, fpe=1.0):
    ch = ecs_choice(arr, tk, ts)
    rows = np.arange(sr.N)
    jj = np.maximum(ch, 0)
    sw = ch >= 0
    lab = np.where(sw, arr["lab"][rows, jj], sr.a1_label)
    lab2 = np.where(sw, arr["lab2"][rows, jj], sr.a1_label2)
    idk = np.where(sw, arr["idk"][rows, jj], sr.a1_idk)
    sd = np.where(sw, arr["sD"][rows, jj], sr.sD)
    gate_off = (~idk) & ~(np.nan_to_num(sd, nan=-INF) >= tD)
    lab = np.where(gate_off, A_, lab)
    lab2 = np.where(gate_off, A_, lab2)
    text = []
    for i in range(sr.N):
        t = arr["texts"][i][ch[i]] if sw[i] else sr.a1_text[i]
        text.append("I don't know. [after-gate]" if gate_off[i] else t)
    changed = sw | gate_off
    return Out(lab, lab2, np.full(sr.N, float(fpe)), text, changed), ch


def r3_out(sr, fpe=9.0):
    j = sr.judges["R3"]
    return Out(np.asarray(j["label_strict"]), np.asarray(j["label_sq2"]), np.full(sr.N, float(fpe)), list(j["text"]),
               np.array([t != a for t, a in zip(j["text"], sr.a1_text)]))


# ----------------------------------------------------------------------------- tuning
def _best(cands, lam):
    """cands: [(params, Out)] in grid order (tau ascending first). First maximum of U_lam wins."""
    best = None
    for params, out in cands:
        u = utility(out.label, lam)
        if best is None or u > best[0] + 1e-12:
            best = (u, params, out)
    return {**best[1], "U_dev": best[0], "fpe_dev": float(np.mean(best[2].fpe))}


def tune_gate(sr, score, fpe, lams, step):
    grid = tau_grid(np.asarray(score)[sr.answered], step)
    cands = [({"tau": t}, gate(sr, score, t, fpe)) for t in grid]
    return {f"lam{l}": _best(cands, l) for l in lams}


def tune_search(sr, configs, method, lams, step, max_rounds=None, key="sD_clean"):
    """configs: [(params, rounds_list)] in grid order. Candidates are ordered by tau (ascending), then by
    config, so ties go to the smaller tau, then to the earlier config."""
    grid = tau_grid(sr.sD[sr.answered], step)
    cands = []
    for t in grid:
        for params, rounds in configs:
            cands.append(({**params, "tau": t}, search_out(sr, rounds, t, method, max_rounds, key)))
    return {f"lam{l}": _best(cands, l) for l in lams}


def tune_ecs(sr, variant, lams, step):
    arr = ecs_arrays(sr, variant)
    Eall = arr["E"][np.isfinite(arr["E"])]
    gk = tau_grid(Eall, step)
    gD = tau_grid(sr.sD[sr.answered], step)
    best = {l: None for l in lams}
    tD = np.asarray(gD)[:, None]
    rows = np.arange(sr.N)
    for tk in gk:
        for ts in gk:
            ch = ecs_choice(arr, tk, ts)
            sw = ch >= 0
            jj = np.maximum(ch, 0)
            lab = np.where(sw, arr["lab"][rows, jj], sr.a1_label)
            idk = np.where(sw, arr["idk"][rows, jj], sr.a1_idk)
            sd = np.nan_to_num(np.where(sw, arr["sD"][rows, jj], sr.sD), nan=-INF)
            off = (~idk)[None, :] & ~(sd[None, :] >= tD)
            L = np.where(off, A_, lab[None, :])                         # [n_tD, N]
            for l in lams:
                U = 100.0 * np.mean(np.where(L == C_, 1.0, np.where(L == W_, -float(l), 0.0)), axis=1)
                k = int(np.argmax(U))                                   # first maximum = smallest tau_D
                if best[l] is None or U[k] > best[l][0] + 1e-12:
                    best[l] = (float(U[k]), {"tau_keep": tk, "tau_sw": ts, "tau_D": float(gD[k])})
    return {f"lam{l}": {**best[l][1], "U_dev": best[l][0], "fpe_dev": 1.0} for l in lams}


def b5_matched_N(sr, b5_tuned, ves_fpe, n_max):
    """Smallest N whose mean dev FPE (at its own tuned tau) is at least VES's."""
    for N in range(1, n_max + 1):
        if b5_tuned[N]["lam1"]["fpe_dev"] >= ves_fpe - 1e-9:
            return N, True
    return n_max, False

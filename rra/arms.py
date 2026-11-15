"""Arms (spec section 6), computed offline from the per-question records, and the tuning-set threshold search.

Every arm uses the same stored A1 and A2 per question. If A1 is "I don't know", every arm outputs it (B0 too:
A1 is an abstention). Per question (A1 answered):
  B0        A1
  B3        A1 if s1 >= t1, else IDK                                        (primary bar)
  R-all     A1 if s1 >= t1, else A2 (IDK if null)
  R-band    A1 if s1 >= t1; A2 if tb <= s1 < t1; else IDK
  R-check   A1 if s1 >= t1; else A2 if not null and D(A2) >= t3; else IDK  (= B7, one round)
  RRA       A1 if s1 >= t1; else A2 if s2 >= t2 (IDK if null); else IDK
  RRA+check RRA, and a routed A2 must also pass D(A2) >= t3
  RRA-EU    argmax {(1+lam) p1 - lam, (1+lam) p2 - lam, 0} -> keep / retry / IDK (first maximum; Platt p1, p2)
  RRA-peer  A1 if s1 >= t1; else the other model's A1 if its own s1 >= tp and it is not null; else IDK
  Oracle    A1 if right; else A2 if right; else IDK      (Oracle-noswitch: A1 if right, else IDK)

FPE: 1 per question; + 1 generation per routed retry; + 0.5 when a routed, non-null A2 is scored (R-check,
RRA+check); RRA-peer: + 1 (the other model's pass, which also gives its detector score) per flagged question.

Tuning: thresholds come from the tuning-set quantiles of the relevant score at 5% steps plus -inf / +inf
(vesecs.methods.tau_grid; answered questions; t3 over non-null A2). Pairs and triples are searched jointly.
Utilities are integer sums (lambda in {1, 3}), so ties are exact; the first maximum in grid order wins (each
threshold ascending, outer to inner = smaller thresholds first, as SQ2 / VES-ECS).
"""
from dataclasses import dataclass

import numpy as np

from pilot.metrics import same_answer
from scoring import A_, C_, W_, is_degenerate, normalize_answer
from vesecs.grader import hedged_form, is_idk
from vesecs.methods import tau_grid

INF = float("inf")
ARMS_MAIN = ["B0", "B3", "R-all", "R-band", "R-check", "RRA", "RRA-EU", "RRA+check", "RRA-peer", "Oracle",
             "Oracle-noswitch"]
TUNED = ["B3", "R-all", "R-band", "R-check", "RRA", "RRA+check", "RRA-peer"]


def null_reason(a1_text, a2_text):
    """None if A2 is a usable fallback, else why it is null (spec 2.1; gold-free: the grader's syntactic
    checks garbled -> I don't know -> hedged, then equal to A1 after normalisation or one containing the other
    as whole words)."""
    if a2_text is None:
        return "not_generated"
    if is_degenerate(a2_text):
        return "garbled"
    if is_idk(a2_text):
        return "idk"
    if hedged_form(a2_text):
        return "hedged"
    if normalize_answer(a1_text) == normalize_answer(a2_text):
        return "same_as_A1"
    if same_answer(a1_text, a2_text):
        return "contains_A1"
    return None


def uvec(labels, lam):
    labels = np.asarray(labels)
    return np.where(labels == C_, 1.0, np.where(labels == W_, -float(lam), 0.0))


class Recs:
    """Per-question arrays of one model on one question set (rra.pipeline builds them)."""

    FIELDS = ("a1_lab", "a1_lab2", "a1_idk", "s1", "s2", "a2_lab", "a2_lab2", "a2_null", "dA2",
              "pr_lab", "pr_lab2", "pr_null", "pr_s1", "p1", "p2")

    def __init__(self, qids, items, a1_text, a1_lab, a1_lab2, a1_idk, s1, s2, a2_text, a2_lab, a2_lab2, a2_null,
                 a2_null_reason, dA2, a1_first=None):
        self.qids, self.items = list(qids), list(items)
        self.N = len(self.qids)
        self.a1_text, self.a2_text = list(a1_text), list(a2_text)
        self.a1_lab, self.a1_lab2 = np.asarray(a1_lab, int), np.asarray(a1_lab2, int)
        self.a1_idk = np.asarray(a1_idk, bool)
        self.s1, self.s2 = np.asarray(s1, np.float64), np.asarray(s2, np.float64)
        self.a2_lab, self.a2_lab2 = np.asarray(a2_lab, int), np.asarray(a2_lab2, int)
        self.a2_null = np.asarray(a2_null, bool)
        self.a2_null_reason = list(a2_null_reason)
        self.dA2 = np.asarray(dA2, np.float64)
        self.a1_first = list(a1_first) if a1_first is not None else [None] * self.N
        self.pr_text = None
        self.pr_lab = self.pr_lab2 = self.pr_null = self.pr_s1 = None
        self.p1 = self.p2 = None

    @property
    def answered(self):
        return ~self.a1_idk

    def set_peer(self, text, lab, lab2, idk, s1):
        """The other model's A1 on the same questions (RRA-peer). Null by the same rule as A2."""
        self.pr_text = list(text)
        self.pr_lab, self.pr_lab2 = np.asarray(lab, int), np.asarray(lab2, int)
        self.pr_s1 = np.asarray(s1, np.float64)
        pidk = np.asarray(idk, bool)
        self.pr_null = np.array([bool(pidk[i]) or null_reason(self.a1_text[i], self.pr_text[i]) is not None
                                 for i in range(self.N)])

    def set_platt(self, p1, p2):
        self.p1, self.p2 = np.asarray(p1, np.float64), np.asarray(p2, np.float64)

    def take(self, idx):
        idx = np.asarray(idx, int)
        r = Recs.__new__(Recs)
        r.qids = [self.qids[i] for i in idx]
        r.items = [self.items[i] for i in idx]
        r.N = len(idx)
        r.a1_text = [self.a1_text[i] for i in idx]
        r.a2_text = [self.a2_text[i] for i in idx]
        r.a2_null_reason = [self.a2_null_reason[i] for i in idx]
        r.a1_first = [self.a1_first[i] for i in idx]
        r.pr_text = None if self.pr_text is None else [self.pr_text[i] for i in idx]
        for k in self.FIELDS:
            v = getattr(self, k, None)
            setattr(r, k, None if v is None else v[idx])
        return r

    @staticmethod
    def concat(parts):
        r = Recs.__new__(Recs)
        for k in ("qids", "items", "a1_text", "a2_text", "a2_null_reason", "a1_first"):
            setattr(r, k, sum((list(getattr(p, k)) for p in parts), []))
        r.pr_text = None if any(p.pr_text is None for p in parts) else sum((p.pr_text for p in parts), [])
        r.N = len(r.qids)
        for k in Recs.FIELDS:
            vs = [getattr(p, k, None) for p in parts]
            setattr(r, k, None if any(v is None for v in vs) else np.concatenate(vs))
        return r


@dataclass
class Out:
    label: np.ndarray          # strict label per question (C_ / A_ / W_)
    label2: np.ndarray         # SQ2-grader label
    fpe: np.ndarray
    kind: np.ndarray           # "A1", "A2", "PEER", "IDK" (not routed), "IDK_null", "IDK_check", "IDK_A1"
    routed: np.ndarray         # a retry was generated (cost)


def _route(r, keep, retry, check=None, fpe_cfg=None, fallback="A2", oracle=False):
    """keep / retry: bool masks over answered questions (keep wins); check: bool mask a routed non-null fallback
    must pass (None = no check)."""
    f = fpe_cfg or {"base": 1.0, "generation": 1.0, "scoring": 0.5}
    ans = r.answered
    keep = ans & keep
    retry = ans & ~keep & retry
    if fallback == "A2":
        null, lab, lab2 = r.a2_null, r.a2_lab, r.a2_lab2
    else:
        null, lab, lab2 = r.pr_null, r.pr_lab, r.pr_lab2
    usable = retry & ~null
    passed = usable if check is None else usable & check
    out_lab = np.where(keep, r.a1_lab, np.where(passed, lab, A_))
    out_lab2 = np.where(keep, r.a1_lab2, np.where(passed, lab2, A_))
    out_lab = np.where(ans, out_lab, r.a1_lab)            # A1 "I don't know" stays
    out_lab2 = np.where(ans, out_lab2, r.a1_lab2)
    kind = np.full(r.N, "IDK", dtype=object)
    kind[~ans] = "IDK_A1"
    kind[keep] = "A1"
    kind[retry & null] = "IDK_null"
    kind[usable & ~passed] = "IDK_check"
    kind[passed] = "A2" if fallback == "A2" else "PEER"
    fpe = np.full(r.N, f["base"])
    fpe = fpe + np.where(retry, f["generation"], 0.0)
    if check is not None and not oracle:
        fpe = fpe + np.where(usable, f["scoring"], 0.0)
    return Out(out_lab.astype(int), out_lab2.astype(int), fpe, kind, retry)


def ge(x, t):
    """x >= t with NaN -> False, except that -inf accepts everything (incl. NaN) and +inf nothing."""
    x = np.asarray(x, np.float64)
    if t == -INF:
        return np.ones(len(x), bool)
    if t == INF:
        return np.zeros(len(x), bool)
    return np.isfinite(x) & (x >= t)


def outputs(r, name, th, lam=1, fpe_cfg=None):
    """Out of arm `name` with thresholds th (dict) on records r."""
    ans = r.answered
    if name == "B0":
        return _route(r, ans, np.zeros(r.N, bool), fpe_cfg=fpe_cfg)
    if name == "B3":
        return _route(r, ge(r.s1, th["t1"]), np.zeros(r.N, bool), fpe_cfg=fpe_cfg)
    if name == "R-all":
        return _route(r, ge(r.s1, th["t1"]), np.ones(r.N, bool), fpe_cfg=fpe_cfg)
    if name == "R-band":
        return _route(r, ge(r.s1, th["t1"]), ge(r.s1, th["tb"]), fpe_cfg=fpe_cfg)
    if name == "R-check":
        return _route(r, ge(r.s1, th["t1"]), np.ones(r.N, bool), check=ge(r.dA2, th["t3"]), fpe_cfg=fpe_cfg)
    if name == "RRA":
        return _route(r, ge(r.s1, th["t1"]), ge(r.s2, th["t2"]), fpe_cfg=fpe_cfg)
    if name == "RRA+check":
        return _route(r, ge(r.s1, th["t1"]), ge(r.s2, th["t2"]), check=ge(r.dA2, th["t3"]), fpe_cfg=fpe_cfg)
    if name == "RRA-EU":
        lam = float(lam)
        eu = np.stack([(1 + lam) * r.p1 - lam, (1 + lam) * r.p2 - lam, np.zeros(r.N)])
        act = np.argmax(np.nan_to_num(eu, nan=-INF), axis=0)       # first maximum: keep, retry, IDK
        return _route(r, act == 0, act == 1, fpe_cfg=fpe_cfg)
    if name == "RRA-peer":
        o = _route(r, ge(r.s1, th["t1"]), ge(r.pr_s1, th["tp"]), fpe_cfg=fpe_cfg, fallback="PEER")
        f = fpe_cfg or {"base": 1.0, "generation": 1.0}
        o.fpe = np.full(r.N, f["base"]) + np.where(ans & ~ge(r.s1, th["t1"]), f["generation"], 0.0)
        return o
    if name == "Oracle":
        return _route(r, r.a1_lab == C_, np.ones(r.N, bool), check=r.a2_lab == C_, fpe_cfg=fpe_cfg, oracle=True)
    if name == "Oracle-noswitch":
        return _route(r, r.a1_lab == C_, np.zeros(r.N, bool), fpe_cfg=fpe_cfg)
    raise ValueError(name)


def utility(labels, lam):
    labels = np.asarray(labels)
    return float(100.0 * uvec(labels, lam).mean()) if len(labels) else float("nan")


def arm_metrics(o, lam_for_u3=None, o3=None):
    lab = o.label
    n = max(len(lab), 1)
    a2 = np.isin(o.kind, ["A2", "PEER"])
    right_r, wrong_r = int(np.sum(a2 & (lab == C_))), int(np.sum(a2 & (lab == W_)))
    return {"right": 100.0 * float(np.sum(lab == C_)) / n, "wrong": 100.0 * float(np.sum(lab == W_)) / n,
            "idk": 100.0 * float(np.sum(lab == A_)) / n, "U1": utility(lab, 1),
            "U3": utility(o3.label, 3) if o3 is not None else None, "U1_sq2_grader": utility(o.label2, 1),
            "retry_rate": 100.0 * float(np.mean(o.routed)) if len(lab) else None,
            "n_routed": int(o.routed.sum()), "n_fallback_output": int(a2.sum()),
            "n_routed_null": int(np.sum(o.kind == "IDK_null")), "n_check_rejected": int(np.sum(o.kind == "IDK_check")),
            "routed_right": right_r, "routed_wrong": wrong_r,
            "routed_precision": (right_r / (right_r + wrong_r)) if (right_r + wrong_r) else None,
            "fpe": float(np.mean(o.fpe)) if len(lab) else None}


# ----------------------------------------------------------------------------- tuning
def grids(r, step):
    ans = r.answered
    g = {"t1": tau_grid(r.s1[ans], step), "t2": tau_grid(r.s2[ans], step), "tb": tau_grid(r.s1[ans], step),
         "t3": tau_grid(r.dA2[ans & ~r.a2_null], step)}
    if r.pr_s1 is not None:
        g["tp"] = tau_grid(r.pr_s1[ans & ~r.pr_null], step)
    return g


def _mask_matrix(x, grid):
    return np.stack([ge(x, t) for t in grid]).astype(np.float64)        # [G, N]


def tune(r, lams, step):
    """Thresholds of every tuned arm for each lambda, plus the tuning-set utility. Returns
    {arm: {"lam1": {t1, ..., U}, "lam3": {...}}}."""
    ans = r.answered
    g = grids(r, step)
    G1 = g["t1"]
    K = _mask_matrix(r.s1, G1).astype(bool) & ans[None, :]               # keep A1 [G1, N]
    F = (~K) & ans[None, :]                                              # flagged answered
    M2 = _mask_matrix(r.s2, g["t2"])
    MB = _mask_matrix(r.s1, g["tb"])
    M3 = _mask_matrix(r.dA2, g["t3"]) * (~r.a2_null)[None, :]
    MP = _mask_matrix(r.pr_s1, g["tp"]) * (~r.pr_null)[None, :] if "tp" in g else None
    out = {a: {} for a in TUNED}
    for lam in lams:
        uk = uvec(r.a1_lab, lam)
        ua2 = np.where(r.a2_null, 0.0, uvec(r.a2_lab, lam))
        base = (K * uk[None, :]).sum(1)                                  # [G1]
        Fu = F * ua2[None, :]                                            # flagged x A2 utility [G1, N]
        res = {}
        res["B3"] = _pick({"t1": G1}, base)
        res["R-all"] = _pick({"t1": G1}, base + Fu.sum(1))
        res["R-band"] = _pick({"t1": G1, "tb": g["tb"]}, base[:, None] + Fu @ MB.T)
        res["R-check"] = _pick({"t1": G1, "t3": g["t3"]}, base[:, None] + Fu @ M3.T)
        res["RRA"] = _pick({"t1": G1, "t2": g["t2"]}, base[:, None] + Fu @ M2.T)
        U = np.empty((len(G1), len(g["t2"]), len(g["t3"])))
        for a in range(len(G1)):                                         # RRA+check: [t1, t2, t3]
            U[a] = base[a] + M2 @ (Fu[a][:, None] * M3.T)
        res["RRA+check"] = _pick({"t1": G1, "t2": g["t2"], "t3": g["t3"]}, U)
        if MP is not None:
            upr = np.where(r.pr_null, 0.0, uvec(r.pr_lab, lam))
            res["RRA-peer"] = _pick({"t1": G1, "tp": g["tp"]}, base[:, None] + (F * upr[None, :]) @ MP.T)
        for a, v in res.items():
            v["U_tune"] = 100.0 * v.pop("_sum") / max(r.N, 1)
            out[a][f"lam{lam}"] = v
    return {a: v for a, v in out.items() if v}


def _pick(axes, U):
    """First maximum in C order (grid ascending, outer axis first). U holds exact integer sums."""
    U = np.asarray(U, np.float64)
    k = int(np.argmax(U))
    idx = np.unravel_index(k, U.shape)
    return {**{name: float(grid[i]) for (name, grid), i in zip(axes.items(), idx)}, "_sum": float(U.flat[k])}


def all_outputs(r, frozen_arms, lam, fpe_cfg=None):
    """{arm: Out} for every arm with the frozen thresholds of lambda `lam`."""
    k = f"lam{lam}"
    out = {"B0": outputs(r, "B0", {}, fpe_cfg=fpe_cfg)}
    for a in TUNED:
        if a in frozen_arms and k in frozen_arms[a]:
            if a == "RRA-peer" and r.pr_s1 is None:
                continue
            out[a] = outputs(r, a, frozen_arms[a][k], lam, fpe_cfg)
    if r.p1 is not None and r.p2 is not None:
        out["RRA-EU"] = outputs(r, "RRA-EU", {}, lam, fpe_cfg)
    out["Oracle"] = outputs(r, "Oracle", {}, fpe_cfg=fpe_cfg)
    out["Oracle-noswitch"] = outputs(r, "Oracle-noswitch", {}, fpe_cfg=fpe_cfg)
    return out

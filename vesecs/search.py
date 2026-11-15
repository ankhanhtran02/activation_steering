"""The common search loop (spec 5.1) and the VES proposals (5.2, 5.3).

Live runs use "replay mode": for every answered draft the loop runs K rounds with step 4(c)'s acceptance
disabled, recording each candidate and its clean score. `replay` then reproduces the method exactly for any
threshold tau: proposal k depends only on the earlier candidates, and in the replay those are exactly the
rejected ones. VES, VES-free, VES-placebo and B7 are deterministic and stop at the first IDK / garbled /
hedged / duplicate proposal; B5 skips it and takes its next try.

Round record: {k, text, token_ids, truncated, label_strict, label_sq2, status, sD_clean, sD_steered, fpe}.
FPE per round: VES 2.0 (pull vector of the newest rejected answer) + 1.0 (steered generation) + 0.5 (clean
scoring, valid candidates only); VES-free, VES-placebo and B7 1.0 + 0.5; B5 1.0 per try + 0.5 if scored.
"""
import numpy as np

from vesecs.grader import candidate_status, grade_both, normalize_answer
from vesecs.io import stable_int

STOPS_ON_INVALID = {"VES": True, "VES_free": True, "VES_placebo": True, "B7": True, "B5": False}


class Scorer:
    """Clean scoring: an unsteered teacher-forced pass over prompt + A + end (SQ2's extraction), returning
    s_D and the after-states at `keep_layers` (VES-free). Cached by (question index, continuation)."""

    def __init__(self, eng, D, keep_layers=()):
        self.eng, self.D = eng, D
        self.layers = sorted(set([D.layer]) | set(keep_layers))
        self.li = self.layers.index(D.layer)
        self.cache = {}
        self.n_passes = 0

    def score(self, qidx, prompts, conts):
        """qidx: question indices (cache keys); prompts / conts aligned. Returns [(sD, {L: after f32})]."""
        keys = [(int(q), tuple(int(t) for t in c)) for q, c in zip(qidx, conts)]
        todo, seen = [], set()
        for j, k in enumerate(keys):
            if k not in self.cache and k not in seen:
                todo.append(j)
                seen.add(k)
        if todo:
            o = self.eng.tf([prompts[j] for j in todo], [list(conts[j]) for j in todo], layers=self.layers)
            self.n_passes += len(todo)
            rp, ra = o["res_prompt"], o["res_ans"]
            sd = self.D.score(before=rp[:, self.li] if self.D.moment in ("prompt", "both") else None,
                              after=ra[:, self.li] if self.D.moment in ("answer", "both") else None)
            for n, j in enumerate(todo):
                self.cache[keys[j]] = (float(sd[n]), {L: ra[n, i].float().numpy() for i, L in enumerate(self.layers)})
        return [self.cache[k] for k in keys]


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v * 0.0


def remove_generic(g, ubar):
    """Steps 1-2: g_hat = g / |g|; u = g_hat - (g_hat . ubar) ubar, normalised."""
    gh = unit(g)
    return unit(gh - (gh @ ubar) * ubar)


def gram_schmidt(us, tol=1e-6):
    ws = []
    for u in us:
        r = np.asarray(u, dtype=np.float64).copy()
        for w in ws:
            r -= (r @ w) * w
        n = np.linalg.norm(r)
        if n > tol:
            ws.append(r / n)
    return ws


class _Entry:
    __slots__ = ("text", "ids", "cont", "norm", "u", "after")

    def __init__(self, text, ids, cont, after=None):
        self.text, self.ids, self.cont = text, list(ids), list(cont)
        self.norm = normalize_answer(text)
        self.u, self.after = None, after or {}


def run_search(eng, ctx, method, layer=None, alpha=None, pull_layers=None, log=print):
    """Replay-mode run of one method on one split. ctx: SplitCtx (vesecs.pipeline). Returns a list with one
    entry per question: None for unanswered drafts, else the list of round records."""
    K = int(ctx.cfg["ves"]["K"])
    fpe = ctx.cfg["fpe"]
    N = len(ctx.items)
    answered = [i for i in range(N) if not ctx.a1_idk[i]]
    R = {i: [_Entry(ctx.a1_text[i], ctx.a1_ids[i], ctx.a1_cont[i], ctx.a1_after(i))] for i in answered}
    rec = {i: [] for i in answered}
    active = list(answered)
    pull_layers = sorted(set(pull_layers or ([layer] if layer is not None else [])))
    for k in range(1, K + 1):
        if not active:
            break
        if method in ("VES", "VES_free", "VES_placebo"):
            _ensure_u(eng, ctx, method, layer, pull_layers,
                      [(i, j, e) for i in active for j, e in enumerate(R[i]) if e.u is None or layer not in e.u])
            V = np.zeros((len(active), eng.d), np.float32)
            for r, i in enumerate(active):
                ws = gram_schmidt([e.u[layer] for e in R[i]])
                if ws:
                    V[r] = (-float(alpha) * ctx.rho[layer] * np.sum(ws, axis=0)).astype(np.float32)
            gen = eng.generate([ctx.prompts[i] for i in active], add=(layer, V))
        elif method == "B7":
            V = None
            ban = [sorted({e.ids[0] for e in R[i] if e.ids}) for i in active]
            gen = eng.generate([ctx.prompts[i] for i in active], ban=ban)
        else:
            raise ValueError(method)
        status = [candidate_status(gen.texts[r], {e.norm for e in R[i]}) for r, i in enumerate(active)]
        conts = [eng.cont(gen.token_ids[r], gen.truncated[r]) for r in range(len(active))]
        valid = [r for r, s in enumerate(status) if s == "valid"]
        scores = ctx.scorer.score([active[r] for r in valid], [ctx.prompts[active[r]] for r in valid], [conts[r] for r in valid])
        clean = dict(zip(valid, scores))
        steered = {}
        if method == "VES" and valid:
            st = eng.steered_states([ctx.prompts[active[r]] for r in valid], [conts[r] for r in valid], layer,
                                    V[valid], [ctx.D.layer])
            b, a = st[ctx.D.layer]
            sv = ctx.D.score(before=b if ctx.D.moment in ("prompt", "both") else None,
                             after=a if ctx.D.moment in ("answer", "both") else None)
            steered = {r: float(sv[n]) for n, r in enumerate(valid)}
        nxt = []
        for r, i in enumerate(active):
            g = grade_both(gen.texts[r], ctx.items[i]["aliases"])
            ok = status[r] == "valid"
            cost = (fpe["backward"] if method == "VES" else 0.0) + fpe["generation"] + (fpe["scoring"] if ok else 0.0)
            rec[i].append({"k": k, "text": gen.texts[r], "token_ids": list(gen.token_ids[r]),
                           "truncated": bool(gen.truncated[r]), "label_strict": g["label_strict"],
                           "label_sq2": g["label_sq2"], "status": status[r],
                           "sD_clean": clean[r][0] if ok else None, "sD_steered": steered.get(r), "fpe": cost})
            if ok:
                R[i].append(_Entry(gen.texts[r], gen.token_ids[r], conts[r], clean[r][1]))
                nxt.append(i)
        log(f"SEARCH: {ctx.model} {ctx.split} {method} layer={layer} alpha={alpha} round={k} active={len(active)} "
            f"valid={len(valid)}")
        active = nxt
    return [rec[i] if i in rec else None for i in range(N)]


def _ensure_u(eng, ctx, method, layer, pull_layers, todo):
    """Unit push directions u_A for rejected answers. todo: [(question index, rejection index j, entry)]."""
    if not todo:
        return
    if method == "VES":
        keys = [(i, tuple(e.cont)) for i, _, e in todo]
        uniq, seen = [], set()
        for n, key in enumerate(keys):
            if key not in ctx.pull_cache and key not in seen:
                seen.add(key)
                uniq.append(n)
        if uniq:
            g = eng.pull_vectors([ctx.prompts[todo[n][0]] for n in uniq], [todo[n][2].cont for n in uniq], pull_layers)
            for m, n in enumerate(uniq):
                ctx.pull_cache[keys[n]] = {L: g[L][m] for L in pull_layers}
            ctx.n_backward += len(uniq)
        for (i, _, e), key in zip(todo, keys):
            e.u = e.u or {}
            e.u[layer] = remove_generic(ctx.pull_cache[key][layer], ctx.ubar[layer])
    elif method == "VES_free":
        for i, _, e in todo:
            e.u = e.u or {}
            e.u[layer] = unit(np.asarray(e.after[layer], np.float64) - ctx.mu[layer])
    elif method == "VES_placebo":
        # the j-th rejected answer of a question gets the pull vector of a randomly chosen answered pool
        # question (seeded mapping: model, qid, j), processed through steps 1-2
        for i, j, e in todo:
            p = stable_int(ctx.model, ctx.items[i]["qid"], "ves_placebo", j) % len(ctx.ghat_pool[layer])
            e.u = e.u or {}
            e.u[layer] = remove_generic(ctx.ghat_pool[layer][p], ctx.ubar[layer])


def b5_rounds(ctx, i, scorer):
    """B5 proposals for question i: tries 11-18 in order, statuses against the rejected set (A1 + earlier valid
    tries), clean scores of valid ones."""
    fpe = ctx.cfg["fpe"]
    tries = ctx.tries[i][ctx.n_sel:]
    rejected = {normalize_answer(ctx.a1_text[i])}
    rounds, to_score = [], []
    for j, t in enumerate(tries):
        st = candidate_status(t["text"], rejected)
        if st == "valid":
            rejected.add(normalize_answer(t["text"]))
            to_score.append(j)
        rounds.append({"k": j + 1, "try_idx": ctx.n_sel + j + 1, "text": t["text"], "token_ids": t["token_ids"],
                       "truncated": t["truncated"], "label_strict": t["label_strict"], "label_sq2": t["label_sq2"],
                       "status": st, "sD_clean": None, "sD_steered": None,
                       "fpe": fpe["generation"] + (fpe["scoring"] if st == "valid" else 0.0)})
    return rounds, to_score


# ----------------------------------------------------------------------------- offline replay
def replay(a1_sD, a1_idk, rounds, tau, stop_on_invalid, max_rounds=None, key="sD_clean"):
    """Returns (output, fpe): output = "A1", "IDK" or the accepted round dict."""
    if a1_idk:
        return "A1", 1.0
    if a1_sD is not None and np.isfinite(a1_sD) and a1_sD >= tau:
        return "A1", 1.0
    cost = 1.0
    for r in (rounds or [])[:max_rounds]:
        cost += r["fpe"]
        if r["status"] != "valid":
            if stop_on_invalid:
                return "IDK", cost
            continue
        s = r.get(key)
        if s is not None and np.isfinite(s) and s >= tau:
            return r, cost
    return "IDK", cost

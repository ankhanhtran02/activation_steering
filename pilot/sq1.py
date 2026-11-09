"""SQ1: do steering methods know anything beyond confidence? (a cheap test, then a check of it)

Directions (all at 0.4 depth unless noted), fitted on folds 0-1 only:
  caa          CAA (Rimsky et al. 2024): mean difference of the residual at the answer letter between
               A/B prompts completed with the "I don't know." option vs the model's own wrong answer
               (option order randomised), on questions the model gets wrong;
  iti          ITI (Li et al. 2023): top-48 attention heads by probe accuracy (correct vs wrong main
               answers, last answer token), mass-mean shift scaled by each head's SD, added to the
               attention output of those layers;
  push         plain "say I don't know" push: mean last-prompt residual under an instruction to decline
               minus under the normal instruction (no right/wrong contrast);
  random_k     20 isotropic random directions (should show no extra knowledge);
  smart_probe  the push applied only where the cross-fitted internal mistake detector flags a likely
               error (should show extra knowledge if the detector has any);
  smart_oracle planted control: the push applied where the true label says "wrong", with 25% of labels
               flipped (checks that the test can see a signal that is there by construction).
Each direction takes the sign that lowers confidence and a dose that lowers the mean confidence on
128 train-side questions by the same amount (see config.SQ1_DOSE_FLIP_FRAC).

Cheap test (no text generation): on held-out answered questions, delta = steered - unsteered
confidence (teacher-forced margin). Selectivity = AUROC of delta for right vs wrong main answers using
only pairs in the same decile of unsteered confidence (0.5 = no knowledge beyond confidence).
Robustness (adjusted selectivity): the same AUROC after delta is regressed on the starting confidence
(cubic + deciles), the answer's length and the dataset / relation, so that a direction cannot look
selective because wrong answers sit lower inside a decile, are longer, or come from another dataset.

Check: on 300 of those questions, real steered answers are generated; the real selectivity is the same
within-decile AUROC with "still answers" (vs declines) as the score.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config import IDK_STRING, USER_TEMPLATE_FORCE_IDK
from data import qid_hash
from pilot import config as PC
from pilot.metrics import decile_bins, residualize, safe_auroc, spearman, stratified_auroc, transition_counts
from pilot.session import resid_add, unit
from scoring import A_, C_, W_, abstention_score, classify, is_degenerate, match_coverage
from steering import SteeringHook

CAA_TEMPLATE = "{question}\n\nChoices:\n(A) {a}\n(B) {b}\n\nAnswer with the letter of your choice only."


def fit_iti(sess, sh, states, rng):
    hr = sh["head_rows"]
    H = sh["heads"]
    y = states[hr] == C_
    if min(y.sum(), (~y).sum()) < PC.ITI_MIN_CLASS:
        return None, {"skipped": f"fewer than {PC.ITI_MIN_CLASS} right or wrong answers on the training folds"}
    nh, hd = sess.spec["n_heads"], sess.spec["head_dim"]
    val, trn = [], []                                   # stratified split: both classes on both sides
    for cls in (True, False):
        idx = np.flatnonzero(y == cls)[rng.permutation(int((y == cls).sum()))]
        nv = min(max(int(round(len(idx) * PC.ITI_VAL_FRAC)), 1), len(idx) - 1)
        val += idx[:nv].tolist()
        trn += idx[nv:].tolist()
    val, trn = np.array(val), np.array(trn)
    accs = np.zeros((sess.n_layers, nh))
    for l in range(sess.n_layers):
        Xl = H[:, l].float().view(len(hr), nh, hd).numpy()
        for h in range(nh):
            Z = Xl[:, h]
            sc = StandardScaler().fit(Z[trn])
            clf = LogisticRegression(max_iter=500).fit(sc.transform(Z[trn]), y[trn])
            accs[l, h] = clf.score(sc.transform(Z[val]), y[val])
    k = min(PC.ITI_TOP_K, accs.size)
    flat = np.argsort(-accs, axis=None, kind="stable")[:k]
    sel = [(int(i // nh), int(i % nh)) for i in flat]
    u = {}
    for l, h in sel:
        Z = H[:, l, h * hd:(h + 1) * hd].float()
        th = unit(Z[torch.as_tensor(y)].mean(0) - Z[torch.as_tensor(~y)].mean(0))
        sigma = float((Z @ th).std())
        W = sess.layers[l].self_attn.o_proj.weight[:, h * hd:(h + 1) * hd].float().cpu()
        u[l] = u.get(l, 0) + sigma * (W @ th)
    return u, {"heads": sel, "val_acc_top": [float(accs[l, h]) for l, h in sel],
               "val_acc_median_all": float(np.median(accs)), "n_rows": int(len(hr))}


def sq1_covariates(pool, ev, m0, bins, sh):
    """What a confidence change could track without any knowledge beyond confidence: the starting
    confidence (cubic in the standardised margin, the decile bins, and a piecewise-linear spline with 19
    knots at its 5th..95th percentiles), the main answer's length in tokens (the margin is a sum over
    tokens) and the dataset / PopQA relation. In simulations the spline removes curved confidence
    artifacts that a cubic alone leaves behind, without weakening a real right/wrong signal."""
    z = (m0 - m0.mean()) / (m0.std() + 1e-9)
    cols = [z, z ** 2, z ** 3]
    cols += [(bins == b).astype(float) for b in np.unique(bins)[1:]]
    cols += [np.maximum(0.0, z - k) for k in np.quantile(z, np.linspace(0.05, 0.95, 19))]
    ntok = np.array([len(sh["token_ids"][i]) for i in ev], dtype=float)
    cols += [ntok, np.log1p(ntok)]
    rel = np.array([pool[i].prop for i in ev])
    cols += [(rel == r).astype(float) for r in sorted(set(rel))[1:]]
    return np.column_stack(cols)


def fit_caa(sess, pool, sh, states, train_side, L, rng):
    wrong = np.flatnonzero(train_side & (states == W_))[:PC.SQ1_CAA_MAX_PAIRS]
    if len(wrong) < PC.SQ1_CAA_MIN_PAIRS:
        return None, {"skipped": f"only {len(wrong)} wrong answers on the training folds"}
    P, Cp, Cn = [], [], []
    tok = lambda s: sess.tok(s, add_special_tokens=False).input_ids
    for i in wrong:
        idk_first = bool(rng.random() < 0.5)
        a, b = (IDK_STRING, sh["texts"][i]) if idk_first else (sh["texts"][i], IDK_STRING)
        P.append(sess.user_prompt(CAA_TEMPLATE.format(question=pool[i].question, a=a, b=b)))
        pos, neg = ("A", "B") if idk_first else ("B", "A")
        Cp.append(tok("(" + pos))
        Cn.append(tok("(" + neg))
    o = sess.tf(P + P, Cp + Cn, layers=[L])
    acts = o["res_ans"][:, 0].float()
    n = len(P)
    return unit(acts[:n].mean(0) - acts[n:].mean(0)), {"n_pairs": n}


def run_sq1(sess, pool, sh, probe_oof, n_random, n_eval):
    L = sess.plan["steer_layer"]
    li = sess.plan["record_layers"].index(L)
    folds = np.array([it.extra["fold"] for it in pool])
    states = sh["states"]
    answered = states != A_
    train_side = np.isin(folds, PC.SQ1_TRAIN_FOLDS)
    prompts = [sess.prompt(it.question) for it in pool]
    conts = sh["conts"]
    tiebreak = np.array([qid_hash(it.qid) for it in pool])
    tune = np.flatnonzero(train_side & answered)[:PC.SQ1_N_TUNE]
    eval_q = np.flatnonzero(~train_side)[:n_eval]
    ev = eval_q[answered[eval_q]]
    gen = ev[:PC.SQ1_N_GEN]
    rng = np.random.default_rng(PC.PILOT_SEED + 11)
    out = {"layer": L, "n_tune": int(len(tune)), "n_eval_questions": int(len(eval_q)), "n_eval_answered": int(len(ev)),
           "n_gen": int(len(gen)), "n_random": n_random}

    R_tr = sh["res_prompt"][np.flatnonzero(train_side), li].float()
    rnorm = float(R_tr.norm(dim=1).mean())
    sd_proj = lambda v: float((R_tr @ unit(v)).std())

    # ------------------------------------------------------------ directions
    dirs, fit_info = {}, {}
    fidx = np.flatnonzero(train_side)[:PC.SQ1_FORCE_N]
    Rf = sess.last_prompt_resid([sess.prompt(pool[i].question, USER_TEMPLATE_FORCE_IDK) for i in fidx], [L])[:, 0]
    dirs["push"] = ("resid", unit((Rf - sh["res_prompt"][fidx, li].float()).mean(0)))
    fit_info["push"] = {"n": int(len(fidx))}
    v, fit_info["caa"] = fit_caa(sess, pool, sh, states, train_side, L, rng)
    if v is not None:
        dirs["caa"] = ("resid", v)
    u, fit_info["iti"] = fit_iti(sess, sh, states, rng)
    if u is not None:
        dirs["iti"] = ("attn", u)
    for k in range(n_random):
        g = torch.Generator().manual_seed(PC.SQ1_RANDOM_SEED_BASE + k)
        dirs[f"random_{k}"] = ("resid", unit(torch.randn(sess.spec["d_model"], generator=g)))
    out["fit"] = fit_info
    res_dirs = {k: v for k, (kind, v) in dirs.items() if kind == "resid" and not k.startswith("random")}
    out["cosines"] = {f"{a}~{b}": float(torch.dot(res_dirs[a], res_dirs[b]))
                      for i, a in enumerate(res_dirs) for b in list(res_dirs)[i + 1:]}

    def specs(kind, vec, alpha):
        if kind == "resid":
            return resid_add(sess.layers[L], alpha * vec)
        return [(sess.layers[l].self_attn, SteeringHook("add", alpha * ul, "both")) for l, ul in vec.items()]

    # ------------------------------------------------------------ doses (tune items only)
    P_t, C_t = [prompts[i] for i in tune], [conts[i] for i in tune]
    m0_t = sess.margin(P_t, C_t)
    pos = m0_t[m0_t > 0]
    target = float(np.quantile(pos, PC.SQ1_DOSE_FLIP_FRAC)) if len(pos) >= 5 else float(np.std(m0_t))
    target = max(target, 1e-3)
    out["dose_target_mean_drop"] = target

    def dose_search(kind, vec, alpha0, alpha_max):
        """Scale |alpha| until the mean confidence drop on the tune items is within 15% of the target:
        multiplicative steps until the target is bracketed, then geometric interpolation."""
        sign, alpha, hist, reached = 1.0, min(alpha0, alpha_max), [], False
        lo, hi = None, None                             # (|alpha|, drop) below / above the target
        for it in range(PC.SQ1_DOSE_MAX_ITERS):
            m1 = sess.margin(P_t, C_t, specs(kind, vec, sign * alpha))
            f = float(-(m1 - m0_t).mean())
            hist.append({"alpha": sign * alpha, "mean_drop": f})
            flipped = False
            if it == 0 and f < 0:
                sign, f, flipped = -sign, -f, True     # the opposite sign lowers confidence (verified next step)
            if not flipped and abs(f - target) <= PC.SQ1_DOSE_TOL * target:
                reached = True
                break
            if not flipped:
                if f < target and (lo is None or alpha > lo[0]):
                    lo = (alpha, f)
                if f > target and (hi is None or alpha < hi[0]):
                    hi = (alpha, f)
            if lo is not None and hi is not None and lo[1] > 0:
                w = (np.log(target) - np.log(lo[1])) / (np.log(hi[1]) - np.log(lo[1]))
                new = float(np.exp(np.log(lo[0]) + np.clip(w, 0.05, 0.95) * (np.log(hi[0]) - np.log(lo[0]))))
            else:
                factor = 4.0 if f <= 0 else float(np.clip(target / f, 0.25, 4.0))
                new = min(alpha * factor, alpha_max)
            if new == alpha and not flipped:
                break
            alpha = new
        good = [h for h in hist if h["mean_drop"] > 0]
        best = min(good, key=lambda h: abs(h["mean_drop"] - target)) if good else hist[-1]
        return best["alpha"], best["mean_drop"], hist, reached

    doses = {}
    for name, (kind, vec) in dirs.items():
        if kind == "resid":
            a0, amax = PC.SQ1_ALPHA_INIT_SIGMA * sd_proj(vec), PC.SQ1_ALPHA_MAX_RNORM * rnorm
        else:
            a0, amax = PC.ITI_ALPHA_INIT, PC.ITI_ALPHA_MAX
        a, f, hist, reached = dose_search(kind, vec, a0, amax)
        doses[name] = {"alpha": a, "tune_mean_drop": f, "reached": reached, "search": hist,
                       "alpha_over_rnorm": (abs(a) / rnorm) if kind == "resid" else None}
    out["doses"] = doses

    # ------------------------------------------------------------ cheap test on held-out answered items
    P_e, C_e = [prompts[i] for i in ev], [conts[i] for i in ev]
    m0_e = sess.margin(P_e, C_e)
    right = states[ev] == C_
    bins = decile_bins(m0_e)
    delta = {name: sess.margin(P_e, C_e, specs(kind, vec, doses[name]["alpha"])) - m0_e
             for name, (kind, vec) in dirs.items()}
    q_flag = float(np.mean(states[train_side & answered] == W_)) if (train_side & answered).any() else 0.5
    p = probe_oof[ev]
    flag = np.zeros(len(ev), dtype=bool)
    fin = np.flatnonzero(np.isfinite(p))
    if len(fin):
        flag[fin[np.argsort(p[fin], kind="stable")[:int(round(q_flag * len(fin)))]]] = True
    noise = np.random.default_rng(PC.PILOT_SEED + 13).random(len(ev)) < PC.SQ1_ORACLE_NOISE
    gate_oracle = (states[ev] == W_) ^ noise
    delta["smart_probe"] = delta["push"] * flag
    delta["smart_oracle"] = delta["push"] * gate_oracle
    out["smart"] = {"flag_rate": float(flag.mean()) if len(flag) else float("nan"), "target_flag_rate": q_flag,
                    "flag_precision_wrong": float(np.mean(states[ev][flag] == W_)) if flag.any() else float("nan"),
                    "oracle_gate_rate": float(gate_oracle.mean()) if len(ev) else float("nan")}

    def drop_gap(d):
        gaps, w = [], []
        for b in np.unique(bins):
            m = bins == b
            if (m & right).any() and (m & ~right).any():
                gaps.append(float(-d[m & ~right].mean() + d[m & right].mean()))
                w.append(int(m.sum()))
        md = float(-d.mean())
        g = float(np.average(gaps, weights=w)) if gaps else float("nan")
        return g, (g / md if md > 0 else float("nan"))

    cheap = {}
    for name, d in delta.items():
        g, gn = drop_gap(d)
        cheap[name] = {"c_auroc": stratified_auroc(d, right, bins), "mean_drop": float(-d.mean()),
                       "drop_gap_wrong_minus_right": g, "drop_gap_normalised": gn,
                       "plain_auroc": safe_auroc(right, d)}
    # robustness: the same statistic after regressing the confidence change on the covariates above
    cov = sq1_covariates(pool, ev, m0_e, bins, sh) if len(ev) else np.zeros((0, 0))
    for name, d in delta.items():
        cheap[name]["c_auroc_adjusted"] = (stratified_auroc(residualize(d, cov), right, bins)
                                           if len(ev) >= PC.SQ1_ADJ_MIN_ROWS_PER_COV * cov.shape[1] else float("nan"))

    def null_summary(key):
        v = np.array([cheap[f"random_{k}"][key] for k in range(n_random)])
        v = v[np.isfinite(v)]
        for name in cheap:
            x = cheap[name][key]
            cheap[name][f"pct_vs_random{key[len('c_auroc'):]}"] = (
                float(100 * ((v < x).sum() + 0.5 * (v == x).sum()) / len(v)) if len(v) and np.isfinite(x) else float("nan"))
        return {"values": v.tolist(), "max": float(v.max()) if len(v) else float("nan"),
                "mean": float(v.mean()) if len(v) else float("nan"),
                "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan")}

    out["random_c_auroc"] = null_summary("c_auroc")
    out["random_c_auroc_adjusted"] = null_summary("c_auroc_adjusted")
    out["adjustment_covariates"] = int(cov.shape[1]) if cov.ndim == 2 else 0
    out["cheap"] = cheap
    out["eval_balance"] = {"n_right": int(right.sum()), "n_wrong": int((~right).sum())}

    # ------------------------------------------------------------ real steered answers on 300 questions
    G = len(gen)
    P_g = [prompts[i] for i in gen]
    base = sess.greedy(P_g)
    items_g = [pool[i] for i in gen]
    base_st = np.array([classify(t, it.aliases)[0] for t, it in zip(base.texts, items_g)], dtype=np.int8)
    ba = base_st != A_
    right_g = base_st == C_
    m0_g = m0_e[:G]
    bins_g = decile_bins(m0_g)
    gen_conds = [c for c in ("caa", "iti", "push", "smart_probe", "smart_oracle") if c in delta]
    gen_conds += [f"random_{k}" for k in range(min(n_random, PC.SQ1_N_RANDOM_GEN))]
    gates = {"smart_probe": flag[:G], "smart_oracle": gate_oracle[:G]}
    real, texts_out = {}, {"unsteered": base.texts}
    for c in gen_conds:
        if c in gates:
            rows = np.flatnonzero(gates[c])
            kind, vec = dirs["push"]
            g = sess.greedy([P_g[j] for j in rows], specs(kind, vec, doses["push"]["alpha"]))
            texts = list(base.texts)
            for j, t in zip(rows, g.texts):
                texts[j] = t
        else:
            kind, vec = dirs[c]
            texts = sess.greedy(P_g, specs(kind, vec, doses[c]["alpha"])).texts
        st = np.array([classify(t, it.aliases)[0] for t, it in zip(texts, items_g)], dtype=np.int8)
        texts_out[c] = texts
        flip = (st == A_) & ba
        d_g = delta[c][:G]
        nflip = int(flip.sum())
        k = int((st != A_).sum())
        rule = match_coverage(base_st, m0_g, tiebreak[gen], k)
        real[c] = {"abstention_rate": float(np.mean(st == A_)), "n_flips_to_idk": nflip,
                   "degenerate_rate": float(np.mean([is_degenerate(t) for t in texts])),
                   "real_c_auroc": stratified_auroc((~flip[ba]).astype(float), right_g[ba], bins_g[ba]),
                   "cheap_c_auroc_same_items": stratified_auroc(d_g[ba], right_g[ba], bins_g[ba]),
                   "item_auroc_drop_predicts_flip": (safe_auroc(flip[ba], -d_g[ba])
                                                     if min(nflip, int(ba.sum()) - nflip) >= PC.SQ1_MIN_FLIPS else float("nan")),
                   "score_S": abstention_score(st), "score_S_unsteered": abstention_score(base_st),
                   "surplus_vs_confidence_rule": (abstention_score(st) - abstention_score(rule)) if rule is not None else float("nan"),
                   "transitions": transition_counts(base_st, st)}
    out["real"] = real
    item = [real[c]["item_auroc_drop_predicts_flip"] for c in gen_conds]
    item = [x for x in item if np.isfinite(x)]
    rand_real = [real[c]["real_c_auroc"] for c in gen_conds if c.startswith("random")]
    rand_real = [x for x in rand_real if np.isfinite(x)]
    sp = real.get("smart_probe", {}).get("real_c_auroc", float("nan"))
    out["agreement"] = {
        "median_item_auroc": float(np.median(item)) if item else float("nan"), "n_conditions_item_auroc": len(item),
        "real_smart_probe_c_auroc": sp, "real_random_c_auroc_max": max(rand_real) if rand_real else float("nan"),
        "real_smart_above_all_random": bool(np.isfinite(sp) and rand_real and sp > max(rand_real)),
        "spearman_cheap_vs_real": spearman([real[c]["cheap_c_auroc_same_items"] for c in gen_conds],
                                           [real[c]["real_c_auroc"] for c in gen_conds])}
    per_item = {"eval_idx": ev, "gen_idx": gen, "m0_eval": m0_e, "delta": delta, "flag": flag,
                "gate_oracle": gate_oracle, "texts_gen": texts_out, "base_states_gen": base_st}
    return out, per_item

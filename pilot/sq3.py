"""SQ3: can combining internal states across rewordings fix wrong answers?

500 PopQA questions, 5 wordings each (the original + 4 templates per relation; pilot/data.py).
1. Room for improvement (no extra compute beyond the greedy answers per wording): among the wrong
   answers to the original wording, the share answered correctly under at least one other wording.
   For context, the same share using 4 of the T=1 tries on the original wording instead.
2. Internal averaging: at the last prompt token of the original wording, the residual at layer L becomes
   (1 - beta) h + beta * mean of the 5 wordings' residuals. Tuned on 200 questions over 3 layers x 2
   strengths (most net fixes); the best setting is tested on the other 300 against
   - voting across wordings (majority of the 5 greedy answers),
   - voting across repeated tries (original greedy answer + 4 T=1 tries),
   - averaging answer probabilities across wordings (each distinct answer scored by its mean
     probability over the 5 wordings; "I don't know." only if every wording declined),
   - control: the same averaging with the 4 rewordings of a different question (same relation,
     different subject), which keeps the smoothing but removes the fact.
"""
import numpy as np
import torch

from pilot import config as PC
from pilot.metrics import cluster_answers, is_idk, majority, transition_counts
from scoring import C_, W_, classify


def _states(texts, items):
    return np.array([classify(t, it.aliases)[0] for t, it in zip(texts, items)], dtype=np.int8)


def run_sq3(sess, pool, sh):
    idx = [i for i, it in enumerate(pool) if it.extra.get("sq3")]
    items = [pool[i] for i in idx]
    n = len(items)
    W = [it.extra["sq3"]["wordings"] for it in items]
    is_tune = np.array([it.extra["sq3"]["split"] == "tune" for it in items])
    layers = sess.plan["sq3_layers"]
    out = {"n": n, "n_tune": int(is_tune.sum()), "n_test": int((~is_tune).sum()), "layers": layers}

    P = [[sess.prompt(W[j][k]) for j in range(n)] for k in range(5)]
    gens = [sess.greedy(P[k]) for k in range(5)]
    st = [_states(g.texts, items) for g in gens]
    base = st[0]

    # 1. room for improvement
    wrong0 = base == W_
    other_right = np.any(np.stack([st[k] == C_ for k in range(1, 5)]), axis=0)
    tries = [sh["dis_texts"][i][:PC.SQ3_N_VOTE_TRIES] for i in idx]
    try_right = np.array([any(classify(t, it.aliases)[0] == C_ for t in ts) for ts, it in zip(tries, items)])
    out["room"] = {"n_wrong_original": int(wrong0.sum()),
                   "n_fixable_by_rewording": int((wrong0 & other_right).sum()),
                   "fixable_frac": float((wrong0 & other_right).sum() / wrong0.sum()) if wrong0.any() else float("nan"),
                   "fixable_frac_not_correct": (float(((base != C_) & other_right).sum() / (base != C_).sum())
                                                if (base != C_).any() else float("nan")),
                   "fixable_frac_by_4_tries": float((wrong0 & try_right).sum() / wrong0.sum()) if wrong0.any() else float("nan"),
                   "accuracy_per_wording": [float(np.mean(s == C_)) for s in st]}

    # 2. internal states of every wording
    R = [sess.last_prompt_resid(P[k], layers) for k in range(5)]          # each [n, 3, d]
    target = torch.stack(R).mean(0)
    rng = np.random.default_rng(PC.PILOT_SEED + 31)
    partner = np.arange(n)
    for rel in sorted({it.prop for it in items}):
        g = np.flatnonzero([it.prop == rel for it in items])
        if len(g) < 2:
            continue
        for _ in range(100):
            perm = g[rng.permutation(len(g))]
            if all(items[a].subj != items[b].subj for a, b in zip(g, perm)):
                break
        partner[g] = perm
    ctrl = (R[0] + sum(R[k][partner] for k in range(1, 5))) / 5
    out["control_partner_same_subject"] = int(sum(items[a].subj == items[b].subj for a, b in enumerate(partner)))

    tune, test = np.flatnonzero(is_tune), np.flatnonzero(~is_tune)
    grid = []
    for li, L in enumerate(layers):
        for beta in PC.SQ3_STRENGTHS:
            g = sess.greedy_blend([P[0][j] for j in tune], L, target[tune, li], beta)
            s = _states(g.texts, [items[j] for j in tune])
            tc = transition_counts(base[tune], s)
            grid.append({"layer": L, "layer_index": li, "beta": beta, **tc})
    best = sorted(grid, key=lambda r: (-r["net"], r["beta"], r["layer_index"]))[0]
    out["tuning"] = {"grid": grid, "best": {"layer": best["layer"], "beta": best["beta"], "net": best["net"]}}

    # 3. test
    items_t = [items[j] for j in test]
    base_t = base[test]
    methods, texts = {}, {}
    li, L, beta = best["layer_index"], best["layer"], best["beta"]
    g = sess.greedy_blend([P[0][j] for j in test], L, target[test, li], beta)
    texts["internal_avg"] = g.texts
    g = sess.greedy_blend([P[0][j] for j in test], L, ctrl[test, li], beta)
    texts["control_other_subject"] = g.texts
    texts["vote_wordings"] = []
    texts["vote_tries"] = []
    for j in test:
        ws = [gens[k].texts[j] for k in range(5)]
        texts["vote_wordings"].append(ws[majority(ws)])
        vs = [gens[0].texts[j]] + tries[j]
        texts["vote_tries"].append(vs[majority(vs)])
    texts["prob_avg"] = _prob_average(sess, P, gens, test)
    for m, tx in texts.items():
        methods[m] = transition_counts(base_t, _states(tx, items_t))
    out["test"] = methods
    out["test_n"] = int(len(test))
    return out, {"texts_test": texts, "base_states": base, "states_by_wording": st, "partner": partner}


def _prob_average(sess, P, gens, rows):
    """For each question: candidates = one representative per distinct answer across the 5 wordings;
    pick the candidate with the highest mean probability over the wordings. "I don't know." is the
    output only if every wording declined: its short, formulaic sequence would otherwise outscore most
    answers and turn this baseline into an abstainer, which would make it a weak fixer and bias the
    comparison toward internal averaging."""
    eot = sess.fmt.eot_id
    jobs, owners = [], []
    cand_text = {}
    for j in rows:
        texts = [gens[k].texts[j] for k in range(5)]
        ids_k = [sess.answer_cont(gens[k].token_ids[j], gens[k].truncated[j]) for k in range(5)]
        cid, reps = cluster_answers(texts)
        cands = []
        for c in range(len(reps)):
            k = cid.index(c)
            if is_idk(texts[k]):
                continue
            cont = ids_k[k] if ids_k[k] and ids_k[k][-1] == eot else ids_k[k] + [eot]
            cands.append((texts[k], cont))
        if not cands:
            cands.append(("I don't know.", sess.idk_cont))
        cand_text[j] = [c[0] for c in cands]
        if len(cands) == 1:
            continue                                   # nothing to choose: no forward passes needed
        for ci, (_, cont) in enumerate(cands):
            for k in range(5):
                jobs.append((P[k][j], cont))
                owners.append((j, ci, k))
    lp = sess.tf([a for a, _ in jobs], [b for _, b in jobs])["lp"] if jobs else np.zeros(0)
    score = {}
    for (j, ci, k), v in zip(owners, lp):
        score.setdefault((j, ci), []).append(v)
    out = []
    for j in rows:
        if len(cand_text[j]) == 1:
            out.append(cand_text[j][0])
            continue
        vals = [np.logaddexp.reduce(score[(j, ci)]) - np.log(5) for ci in range(len(cand_text[j]))]
        out.append(cand_text[j][int(np.argmax(vals))])
    return out

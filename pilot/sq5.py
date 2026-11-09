"""SQ5: can the model give a vaguer but correct answer? (starts with no steering at all)

200 questions with a natural coarser answer: 100 TriviaQA year questions (coarse: the decade) and 100
PopQA place-of-birth questions (coarse: the city's country, from Wikidata P17).
Score per answer: fine-correct +1, coarse-correct +0.5, "I don't know" 0, wrong -1 (x100 = points).

Two routers decide, per question, whether to keep the fine answer or ask the coarse question instead.
Both are a threshold on one score, tuned on one half of the questions and applied to the other; both
also tune whether to ask the coarse question when the fine answer was "I don't know":
  simple prompt   : the score is the confidence (margin);
  attainable pick : the score is the cross-fitted P(fine answer right) from confidence + the internal
                    mistake detector (trained on the main pool, out-of-fold; no SQ5 labels are used).
The decision uses gap = attainable - simple (points), with a bootstrap CI over questions. The oracle
pick (the better of the model's own fine and coarse answers, which needs the labels) is reported as an
upper bound only: it is always far ahead, so a go against it would be automatic.
If the gap is not "stop", one steering direction is tried: the mean last-prompt residual under the
coarse question minus under the fine question (fitted on one half, used on the other), added to the
fine prompt on the questions the simple prompt sends to the coarse question.
"""
import numpy as np

from data import qid_hash
from pilot import config as PC
from pilot.metrics import bootstrap_ci
from pilot.probes import crossfit_lr, output_features
from pilot.session import resid_add, unit
from scoring import A_, C_, alias_match, classify


def grade_two_level(text, fine_aliases, coarse_aliases):
    """+1 fine answer correct, +COARSE_CREDIT coarse answer correct, 0 declined, -1 wrong or garbled."""
    st, degenerate, _, _ = classify(text, fine_aliases)
    if st == C_:
        return 1.0
    if st == A_:
        return 0.0
    if not degenerate and alias_match(text, coarse_aliases):
        return PC.COARSE_CREDIT
    return -1.0


def _rule(score, declined, tau, coarse_if_declined):
    return np.where(declined, coarse_if_declined, score < tau)


def _tune(score, declined, s_fine, s_coarse):
    """Threshold (and the declined-case choice) maximising the mean score on these questions."""
    ans = ~declined
    cands = np.concatenate([[-np.inf], np.unique(score[ans & np.isfinite(score)]), [np.inf]])
    best = None
    for cid in (False, True):
        for t in cands:
            v = float(np.mean(np.where(_rule(score, declined, t, cid), s_coarse, s_fine)))
            if best is None or v > best[0] + 1e-12:
                best = (v, t, cid)
    return best


def cross_route(score, declined, s_fine, s_coarse, half):
    """Cross-fitted routing: the rule tuned on half h is applied to the other half. Returns the per-item
    score, the per-item decision (True = coarse question) and the tuned rules."""
    pol, coarse, rules = np.zeros(len(score)), np.zeros(len(score), bool), {}
    for h in (0, 1):
        t = half == h
        _, tau, cid = _tune(score[t], declined[t], s_fine[t], s_coarse[t])
        o = ~t
        c = _rule(score[o], declined[o], tau, cid)
        coarse[o] = c
        pol[o] = np.where(c, s_coarse[o], s_fine[o])
        rules[h] = {"tau": tau, "coarse_if_declined": bool(cid)}
    return pol, coarse, rules


def route_scores(pool, sh, probe_oof):
    """Cross-fitted P(main answer right) on the main pool: confidence + detector (attainable pick), and
    two descriptive variants (outputs only; outputs + detector)."""
    folds = np.array([it.extra["fold"] for it in pool])
    y = sh["states"] == C_
    pop = (sh["states"] != A_) & np.isfinite(probe_oof)
    F_out = output_features(sh)
    return {"confidence_plus_detector": crossfit_lr(np.column_stack([sh["margin"], probe_oof]), y, pop, folds),
            "outputs_only": crossfit_lr(F_out, y, pop, folds),
            "outputs_plus_detector": crossfit_lr(np.column_stack([F_out, probe_oof]), y, pop, folds)}


def run_sq5(sess, pool, sh, probe_oof, steer):
    idx = np.array([i for i, it in enumerate(pool) if it.extra.get("sq5")])
    items = [pool[i] for i in idx]
    n = len(items)
    out = {"n": n, "n_year": sum(it.extra["sq5"]["kind"] == "year" for it in items),
           "n_pob": sum(it.extra["sq5"]["kind"] == "pob" for it in items)}
    if n < 4:
        out["error"] = "too few SQ5 questions"
        return out, {}
    m = sh["margin"][idx]
    declined = sh["states"][idx] == A_
    fine_texts = [sh["texts"][i] for i in idx]
    P_coarse = [sess.prompt(it.extra["sq5"]["coarse_question"]) for it in items]
    coarse_texts = sess.greedy(P_coarse).texts
    grade = lambda t, it: grade_two_level(t, it.aliases, it.extra["sq5"]["coarse_aliases"])
    s_fine = np.array([grade(t, it) for t, it in zip(fine_texts, items)])
    s_coarse = np.array([grade(t, it) for t, it in zip(coarse_texts, items)])
    half = np.array([qid_hash("sq5" + it.qid) % 2 for it in items])
    pts = lambda x: float(100 * np.mean(x))

    simple, coarse_simple, rules_simple = cross_route(m, declined, s_fine, s_coarse, half)
    rs = route_scores(pool, sh, probe_oof)
    routers = {}
    for name, sc in rs.items():
        s_items = sc[idx]
        if not np.isfinite(s_items[~declined]).any():
            routers[name] = None
            continue
        s_items = np.where(np.isfinite(s_items), s_items, -np.inf)    # unscored answered items: treated as least confident
        pol, _, rules = cross_route(s_items, declined, s_fine, s_coarse, half)
        routers[name] = (pol, rules)
    ideal = np.maximum(s_fine, s_coarse)
    out.update({"score_fine_only": pts(s_fine), "score_coarse_only": pts(s_coarse), "score_simple_prompt": pts(simple),
                "rules_simple": rules_simple, "share_sent_to_coarse": float(coarse_simple.mean()),
                "score_oracle_pick": pts(ideal), "oracle_gap_points_upper_bound": pts(ideal) - pts(simple),
                "score_oracle_with_idk": pts(np.maximum(ideal, 0.0))})
    for name, r in routers.items():
        out[f"score_router_{name}"] = pts(r[0]) if r else float("nan")
    att = routers.get("confidence_plus_detector")
    if att is None:
        out["error"] = "no detector scores: the attainable pick is undefined"
        return out, {}
    gap_items = 100 * (att[0] - simple)
    out["gap_points"] = float(gap_items.mean())
    out["gap_ci"] = bootstrap_ci(lambda q: float(gap_items[q].mean()), n)
    out["gap_items"] = gap_items.tolist()          # for the pooled CI over models (signals.py)
    out["rules_attainable"] = att[1]
    out["by_kind"] = {k: {"fine": pts(s_fine[msk]), "coarse": pts(s_coarse[msk]), "simple": pts(simple[msk]),
                          "attainable": pts(att[0][msk]), "oracle": pts(ideal[msk])}
                      for k in ("year", "pob") for msk in [np.array([it.extra["sq5"]["kind"] == k for it in items])]
                      if msk.any()}
    per_item = {"idx": idx, "coarse_texts": coarse_texts, "s_fine": s_fine, "s_coarse": s_coarse, "simple": simple,
                "attainable": att[0], "coarse_simple": coarse_simple}

    if steer and out["gap_points"] > PC.SQ5_STOP_GAP:
        out["steering"] = _steer(sess, items, coarse_simple, s_fine, half, P_coarse, grade)
        out["steering"]["gain_over_simple_prompt_points"] = out["steering"]["score_policy_steer"] - pts(simple)
    return out, per_item


def _steer(sess, items, coarse_simple, s_fine, half, P_coarse, grade):
    """On the questions the simple prompt sends to the coarse question, add a 'be coarser' direction to the
    fine prompt instead of switching prompts; strength tuned on one half, applied to the other."""
    L = sess.plan["steer_layer"]
    P_fine = [sess.prompt(it.question) for it in items]
    Rf = sess.last_prompt_resid(P_fine, [L])[:, 0]
    Rc = sess.last_prompt_resid(P_coarse, [L])[:, 0]
    pol = s_fine.copy()
    info = {}
    for h in (0, 1):
        own, other = np.flatnonzero(half == h), np.flatnonzero(half != h)
        v = unit((Rc[own] - Rf[own]).mean(0))
        sd = float((Rf[own] @ v).std())
        best = None
        for c in PC.SQ5_STEER_C:
            g = sess.greedy([P_fine[j] for j in own], resid_add(sess.layers[L], c * sd * v))
            s_st = np.array([grade(t, items[j]) for t, j in zip(g.texts, own)])
            score = float(np.mean(np.where(coarse_simple[own], s_st, s_fine[own])))
            if best is None or score > best[0]:
                best = (score, c)
        rows = other[coarse_simple[other]]
        if len(rows):
            g = sess.greedy([P_fine[j] for j in rows], resid_add(sess.layers[L], best[1] * sd * v))
            pol[rows] = [grade(t, items[j]) for t, j in zip(g.texts, rows)]
        info[f"half{h}"] = {"c": best[1], "tune_score": float(100 * best[0])}
    info["score_policy_steer"] = float(100 * pol.mean())
    return info

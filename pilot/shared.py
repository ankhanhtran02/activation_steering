"""Shared setup, run once per model on the whole pool:

- the main answer (greedy) and its automatic grade;
- 10 T=1 tries (disagreement questions, SQ2) and a separate 8 T=1 tries (answer consistency). All 18
  are independent draws from one prompt; the first 10 and the last 8 never overlap;
- the confidence in the main answer: m = log P(answer + eot) - log P("I don't know." + eot);
- the internal state (residual) at 5 layers, at the last prompt token and at the last answer token,
  and the attention-head outputs at the last answer token for the SQ1 training folds (ITI).
"""
import csv

import numpy as np

from pilot import config as PC
from pilot.metrics import consistency_features
from scoring import A_, C_, STATE_NAMES, classify


def run_screen(sess, items):
    """SQ2 screening extension: only the 10 T=1 tries per question (they define disagreement)."""
    prompts = [sess.prompt(it.question) for it in items]
    tries = sess.sample(prompts, PC.N_DIS_TRIES, PC.EXT_SAMPLE_SEED)
    texts = [[t[0] for t in row] for row in tries]
    return {"qids": [it.qid for it in items], "dis_texts": texts,
            "dis_correct": np.array([sum(classify(t, it.aliases)[0] == C_ for t in row) for row, it in zip(texts, items)])}


def run_shared(sess, pool, heads=True, seed=None):
    """heads=False and a different seed for the follow-up of extension disagreement questions; their
    disagreement status comes from the screening tries, never from the tries drawn here."""
    N = len(pool)
    plan = sess.plan
    prompts = [sess.prompt(it.question) for it in pool]
    gen = sess.greedy(prompts)
    cls = [classify(t, it.aliases) for t, it in zip(gen.texts, pool)]
    states = np.array([c[0] for c in cls], dtype=np.int8)
    degenerate = np.array([c[1] for c in cls], dtype=bool)
    conts = [sess.answer_cont(gen.token_ids[i], gen.truncated[i]) for i in range(N)]

    tfa = sess.tf(prompts, conts, layers=plan["record_layers"], heads=heads)
    tfi = sess.tf(prompts, [sess.idk_cont] * N)
    margin = tfa["lp"] - tfi["lp"]
    n_tok = np.array([max(len(c), 1) for c in conts])

    tries = sess.sample(prompts, PC.N_DIS_TRIES + PC.N_CON_TRIES, PC.SAMPLE_SEED if seed is None else seed)
    dis = [[t[0] for t in row[:PC.N_DIS_TRIES]] for row in tries]
    con = [[t[0] for t in row[PC.N_DIS_TRIES:]] for row in tries]
    dis_correct = np.array([sum(classify(t, it.aliases)[0] == C_ for t in row) for row, it in zip(dis, pool)])
    con_correct = np.array([sum(classify(t, it.aliases)[0] == C_ for t in row) for row, it in zip(con, pool)])
    agree, distinct, idk_frac = consistency_features(gen.texts, con)

    folds = np.array([it.extra["fold"] for it in pool])
    head_rows = np.flatnonzero(np.isin(folds, PC.SQ1_TRAIN_FOLDS) & (states != A_)) if heads else np.zeros(0, int)
    return {
        "qids": [it.qid for it in pool], "texts": gen.texts, "token_ids": gen.token_ids, "truncated": gen.truncated,
        "states": states, "degenerate": degenerate, "conts": conts,
        "lp_ans": tfa["lp"], "lp_idk": tfi["lp"], "margin": margin, "mean_tok_lp": tfa["lp"] / n_tok,
        "res_prompt": tfa["res_prompt"], "res_ans": tfa["res_ans"], "record_layers": plan["record_layers"],
        "heads": tfa["heads"][head_rows].clone() if heads else None, "head_rows": head_rows,
        "dis_texts": dis, "con_texts": con, "dis_correct": dis_correct, "con_correct": con_correct,
        "agree": agree, "distinct": distinct, "idk_frac": idk_frac,
    }


def shared_summary(sh, pool):
    st = sh["states"]
    ds = np.array([it.dataset for it in pool])
    out = {"n": int(len(st)), "accuracy": float(np.mean(st == C_)), "abstention_rate": float(np.mean(st == A_)),
           "accuracy_answered": float(np.mean(st[st != A_] == C_)) if (st != A_).any() else float("nan"),
           "degenerate_rate": float(np.mean(sh["degenerate"])),
           "mean_dis_correct": float(np.mean(sh["dis_correct"])), "mean_margin": float(np.mean(sh["margin"]))}
    for d in np.unique(ds):
        m = ds == d
        out[f"accuracy_{d}"] = float(np.mean(st[m] == C_))
        out[f"abstention_rate_{d}"] = float(np.mean(st[m] == A_))
    return out


def write_grading_check(sh, pool, model_name, path, seed):
    """Random answered main answers for the two-stage hand check against the automatic grader: rows with
    stage 1 (first GRADING_STAGE_N) are always labelled, stage 2 rows only if pilot.analyze asks for them.
    Fill `human_label` with correct / wrong / abstained."""
    rng = np.random.default_rng(seed)
    answered = np.flatnonzero(sh["states"] != A_)
    pick = answered[rng.permutation(len(answered))[:2 * PC.GRADING_STAGE_N]]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "model", "qid", "dataset", "question", "gold_aliases", "output", "auto_label", "human_label"])
        for j, i in enumerate(pick):
            it = pool[i]
            w.writerow([1 if j < PC.GRADING_STAGE_N else 2, model_name, it.qid, it.dataset, it.question,
                        " | ".join(it.aliases), sh["texts"][i], STATE_NAMES[int(sh["states"][i])], ""])
    return len(pick)

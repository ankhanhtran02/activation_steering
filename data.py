"""Dataset loading, subject-disjoint splits, popularity bins and gold aliases."""
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from config import (DATA_DIR, DEV_SUBSET_PER_DECILE, EQ_PER_RELATION, EQ_RELATIONS, N_ADV_HELDOUT,
                    N_ADV_TRAIN, N_ALPACA_BENIGN, N_ALPACA_PPL, N_ALPACA_TRAIN, N_MMLU,
                    N_PER_DECILE, SPLIT_FRACTIONS, SPLIT_SALT)


@dataclass
class QAItem:
    qid: str
    question: str
    aliases: list
    dataset: str                     # "popqa" or "eq"
    subj: str = ""
    subj_id: int = -1
    prop: str = ""                   # PopQA property or EntityQuestions relation
    log_pop: float = float("nan")
    decile: int = -1                 # 0 = least popular
    tercile: int = -1                # 0 = least popular
    extra: dict = field(default_factory=dict)


def qid_hash(qid):
    return int(hashlib.sha1(qid.encode()).hexdigest()[:12], 16)


def read_jsonl(name):
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run setup.py first (main.py is offline)")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def subject_pool(subj_id):
    h = int(hashlib.sha1((SPLIT_SALT + str(subj_id)).encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < SPLIT_FRACTIONS[0]:
        return "train"
    if h < SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]:
        return "dev"
    return "test"


def _norm_words(s):
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in s).split())


def load_popqa(test_n_per_decile):
    rows = read_jsonl("popqa.jsonl")
    n_raw = len(rows)
    # a few PopQA rows have no subject ("Who is the author of ?"): unanswerable, dropped
    rows = [r for r in rows if r["subj"] and r["subj"].strip()]
    log_pop = np.log10(np.array([r["s_pop"] for r in rows], dtype=np.float64) + 1.0)
    dec_edges = np.quantile(log_pop, np.linspace(0.1, 0.9, 9))
    ter_edges = np.quantile(log_pop, [1 / 3, 2 / 3])
    items_by_pool = {"train": [], "dev": [], "test": []}
    for r, lp in zip(rows, log_pop):
        it = QAItem(qid=f"popqa_{r['id']}", question=r["question"],
                    aliases=[a for a in r["possible_answers"] if a.strip()], dataset="popqa",
                    subj=r["subj"], subj_id=r["subj_id"], prop=r["prop"], log_pop=float(lp),
                    decile=int(np.searchsorted(dec_edges, lp, side="right")),
                    tercile=int(np.searchsorted(ter_edges, lp, side="right")))
        items_by_pool[subject_pool(r["subj_id"])].append(it)

    pools_subj = {k: {it.subj_id for it in v} for k, v in items_by_pool.items()}
    overlap = (len(pools_subj["train"] & pools_subj["dev"]) + len(pools_subj["train"] & pools_subj["test"])
               + len(pools_subj["dev"] & pools_subj["test"]))
    assert overlap == 0, "subject leakage across splits"

    rng = np.random.default_rng(0)
    splits = {"train": [], "train_reserve": [], "dev": [], "test": []}
    shortfall = {}
    for pool in ["train", "dev", "test"]:
        for d in range(10):
            cand = sorted([it for it in items_by_pool[pool] if it.decile == d], key=lambda x: x.qid)
            perm = [cand[i] for i in rng.permutation(len(cand))]
            for r, it in enumerate(perm):
                it.extra["decile_rank"] = r
            if pool == "train":
                n1, n2 = N_PER_DECILE["train"], N_PER_DECILE["train_reserve"]
                splits["train"] += perm[:n1]
                splits["train_reserve"] += perm[n1:n1 + n2]
                if len(perm) < n1 + n2:
                    shortfall[f"train_d{d}"] = len(perm)
            else:
                n = N_PER_DECILE["dev"] if pool == "dev" else test_n_per_decile
                splits[pool] += perm[:n]
                if len(perm) < n:
                    shortfall[f"{pool}_d{d}"] = len(perm)
    for k in splits:  # fixed, popularity-mixed question order (batches are formed in this order)
        splits[k].sort(key=lambda it: qid_hash(it.qid))

    rng_sub = np.random.default_rng(0)
    dev_subset = []
    for d in range(10):
        idx = [i for i, it in enumerate(splits["dev"]) if it.decile == d]
        dev_subset += [idx[i] for i in rng_sub.permutation(len(idx))[:DEV_SUBSET_PER_DECILE]]
    pool_subjects = sorted({it.subj for k in ("train", "dev") for it in items_by_pool[k]})
    return splits, sorted(dev_subset), pool_subjects, {"shortfall": shortfall,
                                                         "decile_edges": dec_edges.tolist(),
                                                         "tercile_edges": ter_edges.tolist(),
                                                         "pool_sizes": {k: len(v) for k, v in items_by_pool.items()},
                                                         "n_dropped_no_subject": n_raw - len(rows)}


def eq_template_subject(questions):
    """EntityQuestions uses one fixed template per relation ("Which country is X located in?").
    The subject is what remains after removing the relation's common prefix and suffix."""
    prefix = os.path.commonprefix(questions)
    suffix = os.path.commonprefix([q[::-1] for q in questions])[::-1]
    # never cut into a word: back off to the nearest space
    prefix = prefix[:prefix.rfind(" ") + 1] if " " in prefix else ""
    cut = suffix.find(" ")
    if suffix and not suffix[0].isspace() and not suffix.startswith("'s") and any(c.isalnum() for c in suffix):
        suffix = suffix[cut:] if cut >= 0 else ""
    return [q[len(prefix):len(q) - len(suffix)].strip() for q in questions]


def load_entity_questions(excluded_subjects):
    """The leakage filter matches PopQA train/dev subject strings (>= 4 chars, word boundaries)
    against each question's subject slot, not the template wording: PopQA contains entities
    literally named "Music", "Country", "Born" or "Capital" that would otherwise remove whole
    relations whose template contains those words."""
    rows = read_jsonl("entity_questions_test.jsonl")
    excl = {_norm_words(s) for s in excluded_subjects if len(s.strip()) >= 4}
    excl.discard("")
    max_len = max(len(s.split()) for s in excl)

    def contains_excluded(text):
        w = _norm_words(text).split()
        for n in range(1, max_len + 1):
            for i in range(len(w) - n + 1):
                if " ".join(w[i:i + n]) in excl:
                    return True
        return False

    rng = np.random.default_rng(0)
    items, n_excluded = [], 0
    for rel in EQ_RELATIONS:
        rel_rows = [r for r in rows if r["relation"] == rel]
        subjects = eq_template_subject([r["question"] for r in rel_rows])
        kept = []
        for r, subj in zip(rel_rows, subjects):
            if contains_excluded(subj):
                n_excluded += 1
            else:
                kept.append((r, subj))
        for i in rng.permutation(len(kept))[:EQ_PER_RELATION]:
            r, subj = kept[i]
            items.append(QAItem(qid=f"eq_{rel}_{r['idx']}", question=r["question"], subj=subj,
                                aliases=[a for a in r["answers"] if a.strip()], dataset="eq", prop=rel))
    return items, {"n_excluded_by_subject_filter": n_excluded}


def eq_subset_indices(items, n_total):
    """First k items of each relation (in sampled order); k differs by one across relations."""
    base, extra = divmod(n_total, len(EQ_RELATIONS))
    out = []
    for j, rel in enumerate(EQ_RELATIONS):
        idx = [i for i, it in enumerate(items) if it.prop == rel]
        out += idx[:base + (1 if j < extra else 0)]
    return sorted(out)


def load_instruction_sets():
    adv = read_jsonl("advbench.jsonl")
    perm = np.random.default_rng(0).permutation(len(adv))
    adv_train = [adv[i]["goal"] for i in perm[:N_ADV_TRAIN]]
    adv_heldout = [adv[i]["goal"] for i in perm[N_ADV_TRAIN:N_ADV_TRAIN + N_ADV_HELDOUT]]
    alp = read_jsonl("alpaca_noinput.jsonl")
    perm = np.random.default_rng(0).permutation(len(alp))
    a, b, c = N_ALPACA_TRAIN, N_ALPACA_TRAIN + N_ALPACA_BENIGN, N_ALPACA_TRAIN + N_ALPACA_BENIGN + N_ALPACA_PPL
    return {"adv_train": adv_train, "adv_heldout": adv_heldout,
            "alpaca_train": [alp[i]["instruction"] for i in perm[:a]],
            "alpaca_benign": [alp[i]["instruction"] for i in perm[a:b]],
            "alpaca_ppl": [(alp[i]["instruction"], alp[i]["output"]) for i in perm[b:c]]}


def load_mmlu():
    rows = read_jsonl("mmlu_test.jsonl")
    subjects = sorted({r["subject"] for r in rows})
    rng = np.random.default_rng(0)
    base, extra = divmod(N_MMLU, len(subjects))
    bonus = set(rng.permutation(len(subjects))[:extra].tolist())
    out = []
    for j, s in enumerate(subjects):
        cand = [r for r in rows if r["subject"] == s]
        k = base + (1 if j in bonus else 0)
        out += [cand[i] for i in rng.permutation(len(cand))[:k]]
    return out


def trim_test(items, n_per_decile):
    """Degradation-ladder step: keep the first n items per decile in sampling order (a prefix of
    the original draw, so the smaller test set is nested in the larger one)."""
    return [it for it in items if it.extra["decile_rank"] < n_per_decile]


def load_all(test_n_per_decile=N_PER_DECILE["test"]):
    splits, dev_subset, pool_subjects, popqa_info = load_popqa(test_n_per_decile)
    eq_items, eq_info = load_entity_questions(pool_subjects)
    return {"popqa": splits, "dev_subset": dev_subset, "eq": eq_items,
            "instr": load_instruction_sets(), "mmlu": load_mmlu(),
            "info": {"popqa": popqa_info, "eq": eq_info,
                     "sizes": {k: len(v) for k, v in splits.items()} | {"eq": len(eq_items)}}}

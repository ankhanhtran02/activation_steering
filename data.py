"""Dataset loading, subject-disjoint splits, popularity bins, gold aliases, seed redraws and data
checksums."""
import hashlib
import json
import os
from dataclasses import dataclass, field

import numpy as np

from config import (DATA_DIR, DATA_FILES, DEV_SUBSET_PER_DECILE, EQ_PER_RELATION, EQ_RELATIONS, N_ADV_HELDOUT,
                    N_ADV_TRAIN, N_ALPACA_BENIGN, N_ALPACA_TRAIN, N_MMLU, N_MMLU_DEV, N_PER_DECILE, N_REF_DEV,
                    N_REF_TEST, PREVIOUSLY_SCORED_PATH, REF_MIN_CHARS, SEED_SPLIT_BASE, SPLIT_FRACTIONS, SPLIT_SALT,
                    TEST_MAX_PER_DECILE, TQA_DEV_N, TQA_TEST_N)


@dataclass
class QAItem:
    qid: str
    question: str
    aliases: list
    dataset: str                     # "popqa", "eq" or "tqa"
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


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_data_checksums(manifest):
    """Compares every data_cache file with the sha256 setup.py recorded (amendment A14). Returns
    {key: (status, recorded, actual)}; status in ok / mismatch / unrecorded / missing."""
    recorded = manifest.get("datasets", {})
    out = {}
    for key, fname in DATA_FILES.items():
        path = DATA_DIR / fname
        rec = recorded.get(key, {}).get("sha256")
        if not path.exists():
            out[key] = ("missing", rec, None)
            continue
        actual = file_sha256(path)
        out[key] = ("unrecorded" if rec is None else ("ok" if rec == actual else "mismatch"), rec, actual)
    return out


def subject_pool(subj_id):
    h = int(hashlib.sha1((SPLIT_SALT + str(subj_id)).encode()).hexdigest(), 16) % 10_000 / 10_000
    if h < SPLIT_FRACTIONS[0]:
        return "train"
    if h < SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]:
        return "dev"
    return "test"


def _norm_words(s):
    return " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in s).split())


def _dev_subset(dev_items, seed):
    rng_sub = np.random.default_rng(seed)
    out = []
    for d in range(10):
        idx = [i for i, it in enumerate(dev_items) if it.decile == d]
        out += [idx[i] for i in rng_sub.permutation(len(idx))[:DEV_SUBSET_PER_DECILE]]
    return sorted(out)


def load_previously_scored():
    """Final F1: question ids scored on a test split by run 1 or run 2 (previously_scored_qids.json)."""
    keys = ("popqa_test", "entity_questions", "triviaqa")
    if not PREVIOUSLY_SCORED_PATH.exists():
        raise FileNotFoundError(f"{PREVIOUSLY_SCORED_PATH} missing: the final run needs it to exclude every "
                                "question an earlier run scored")
    d = json.loads(PREVIOUSLY_SCORED_PATH.read_text())
    return {k: set(d.get(k, [])) for k in keys}


def load_popqa():
    """Train / train_reserve / dev are the registered seed-0 draws (unchanged). Final F1: test is every
    test-pool question not scored by an earlier run, in the seed-0 permutation order per decile."""
    prev = load_previously_scored()["popqa_test"]
    rows = read_jsonl(DATA_FILES["popqa"])
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
    shortfall, fresh_info = {}, {"per_decile_fresh": [], "per_decile_excluded": [], "excluded_max_rank": -1}
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
            elif pool == "dev":
                splits["dev"] += perm[:N_PER_DECILE["dev"]]
                if len(perm) < N_PER_DECILE["dev"]:
                    shortfall[f"dev_d{d}"] = len(perm)
            else:
                excluded = [it for it in perm if it.qid in prev]
                fresh = [it for it in perm if it.qid not in prev]
                if TEST_MAX_PER_DECILE is not None:
                    fresh = fresh[:TEST_MAX_PER_DECILE]
                splits["test"] += fresh
                fresh_info["per_decile_fresh"].append(len(fresh))
                fresh_info["per_decile_excluded"].append(len(excluded))
                if excluded:
                    fresh_info["excluded_max_rank"] = max(fresh_info["excluded_max_rank"],
                                                          max(it.extra["decile_rank"] for it in excluded))
    assert not any(it.qid in prev for it in splits["test"]), "a previously scored question entered the test set"
    fresh_info["n_test"] = len(splits["test"])
    fresh_info["n_previously_scored_in_pool"] = int(sum(fresh_info["per_decile_excluded"]))
    for k in splits:  # fixed, popularity-mixed question order (batches are formed in this order)
        splits[k].sort(key=lambda it: qid_hash(it.qid))

    dev_subset = _dev_subset(splits["dev"], 0)
    pool_subjects = sorted({it.subj for k in ("train", "dev") for it in items_by_pool[k]})
    return splits, dev_subset, pool_subjects, items_by_pool, {
        "shortfall": shortfall, "decile_edges": dec_edges.tolist(), "tercile_edges": ter_edges.tolist(),
        "pool_sizes": {k: len(v) for k, v in items_by_pool.items()}, "n_dropped_no_subject": n_raw - len(rows),
        "fresh_test": fresh_info}


def draw_seed_splits(pools, seed):
    """Amendment A8: seed s >= 1 redraws train (100/decile), train_reserve (100/decile) and dev
    (40/decile) from the same subject-disjoint pools with rng(SEED_SPLIT_BASE + s); test never
    changes. Seed 0 is the registered draw returned by load_popqa (never produced here)."""
    assert seed >= 1
    rng = np.random.default_rng(SEED_SPLIT_BASE + seed)
    out = {"train": [], "train_reserve": [], "dev": []}
    for pool in ("train", "dev"):
        for d in range(10):
            cand = sorted([it for it in pools[pool] if it.decile == d], key=lambda x: x.qid)
            perm = [cand[i] for i in rng.permutation(len(cand))]
            if pool == "train":
                n1, n2 = N_PER_DECILE["train"], N_PER_DECILE["train_reserve"]
                out["train"] += perm[:n1]
                out["train_reserve"] += perm[n1:n1 + n2]
            else:
                out["dev"] += perm[:N_PER_DECILE["dev"]]
    for k in out:
        out[k].sort(key=lambda it: qid_hash(it.qid))
    return out, _dev_subset(out["dev"], SEED_SPLIT_BASE + seed)


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
    rows = read_jsonl(DATA_FILES["entity_questions"])
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

    # final F1: questions run 1 / run 2 scored are skipped; the per-relation permutation (and so the rng
    # stream) is the registered one, so the new items are the next ones in the registered order
    prev = load_previously_scored()["entity_questions"]
    rng = np.random.default_rng(0)
    items, n_excluded, n_prev = [], 0, 0
    for rel in EQ_RELATIONS:
        rel_rows = [r for r in rows if r["relation"] == rel]
        subjects = eq_template_subject([r["question"] for r in rel_rows])
        kept = []
        for r, subj in zip(rel_rows, subjects):
            if contains_excluded(subj):
                n_excluded += 1
            else:
                kept.append((r, subj))
        taken = 0
        for i in rng.permutation(len(kept)):
            if taken == EQ_PER_RELATION:
                break
            r, subj = kept[i]
            qid = f"eq_{rel}_{r['idx']}"
            if qid in prev:
                n_prev += 1
                continue
            items.append(QAItem(qid=qid, question=r["question"], subj=subj,
                                aliases=[a for a in r["answers"] if a.strip()], dataset="eq", prop=rel))
            taken += 1
    return items, {"n_excluded_by_subject_filter": n_excluded, "n_previously_scored_skipped": n_prev}


def load_triviaqa():
    """Amendment A10: TriviaQA rc.nocontext validation; dev 300 (threshold only, as in run 2) / test
    1,000. Final F1: test skips every question run 2 scored and repeated question ids (the
    rc.nocontext file lists some questions more than once), and never overlaps dev. Directions are
    transferred from PopQA, not refit; dev is used only for the threshold tau."""
    prev = load_previously_scored()["triviaqa"]
    rows = read_jsonl(DATA_FILES["triviaqa"])
    rows = sorted(rows, key=lambda r: r["question_id"])
    perm = np.random.default_rng(0).permutation(len(rows))

    def item(r):
        al = [a for a in dict.fromkeys(r["aliases"] + [r["value"]]) if a and a.strip()]
        return QAItem(qid=f"tqa_{r['question_id']}", question=r["question"], aliases=al, dataset="tqa")
    dev = [item(rows[i]) for i in perm[:TQA_DEV_N]]
    seen = {it.qid for it in dev} | set(prev)
    test = []
    for i in perm[TQA_DEV_N:]:
        it = item(rows[i])
        if it.qid in seen:
            continue
        seen.add(it.qid)
        test.append(it)
        if len(test) == TQA_TEST_N:
            break
    for k in (dev, test):
        k.sort(key=lambda it: qid_hash(it.qid))
    return {"dev": dev, "test": test}


def eq_subset_indices(items, n_total, key="prop", groups=EQ_RELATIONS):
    """First k items of each relation (in sampled order); k differs by one across relations."""
    base, extra = divmod(n_total, len(groups))
    out = []
    for j, rel in enumerate(groups):
        idx = [i for i, it in enumerate(items) if getattr(it, key) == rel]
        out += idx[:base + (1 if j < extra else 0)]
    return sorted(out)


def tqa_subset_indices(items, n_total):
    """Fixed random-draw subset: the n_total items with the smallest question-id hash."""
    order = sorted(range(len(items)), key=lambda i: qid_hash(items[i].qid))
    return sorted(order[:n_total])


def load_reference_text():
    """Amendment A11: BBC News articles from May 2025 (after both models' release, so they cannot be
    memorised) replace the Alpaca outputs as perplexity text. 200 dev (dev-selection cap) / 200 test
    (reported side effect), disjoint, seed 0; first 128 tokens of the article body are scored."""
    rows = read_jsonl(DATA_FILES["bbc_news"])
    rows = [r for r in rows if r.get("content") and len(r["content"]) >= REF_MIN_CHARS and r.get("title")]
    rows = sorted(rows, key=lambda r: r["link"])
    perm = np.random.default_rng(0).permutation(len(rows))
    pick = lambda i: {"title": rows[i]["title"].replace(" - BBC News", "").strip(), "text": rows[i]["content"].strip()}
    return {"dev": [pick(i) for i in perm[:N_REF_DEV]], "test": [pick(i) for i in perm[N_REF_DEV:N_REF_DEV + N_REF_TEST]]}


def load_instruction_sets():
    adv = read_jsonl(DATA_FILES["advbench"])
    perm = np.random.default_rng(0).permutation(len(adv))
    adv_train = [adv[i]["goal"] for i in perm[:N_ADV_TRAIN]]
    adv_heldout = [adv[i]["goal"] for i in perm[N_ADV_TRAIN:N_ADV_TRAIN + N_ADV_HELDOUT]]
    alp = read_jsonl(DATA_FILES["alpaca"])
    perm = np.random.default_rng(0).permutation(len(alp))
    a, b = N_ALPACA_TRAIN, N_ALPACA_TRAIN + N_ALPACA_BENIGN
    return {"adv_train": adv_train, "adv_heldout": adv_heldout,
            "alpaca_train": [alp[i]["instruction"] for i in perm[:a]],
            "alpaca_benign": [alp[i]["instruction"] for i in perm[a:b]]}


def _stratified_mmlu(rows, n, seed=0):
    subjects = sorted({r["subject"] for r in rows})
    rng = np.random.default_rng(seed)
    base, extra = divmod(n, len(subjects))
    bonus = set(rng.permutation(len(subjects))[:extra].tolist())
    out = []
    for j, s in enumerate(subjects):
        cand = [r for r in rows if r["subject"] == s]
        k = base + (1 if j in bonus else 0)
        out += [cand[i] for i in rng.permutation(len(cand))[:k]]
    return out


def load_mmlu():
    return _stratified_mmlu(read_jsonl(DATA_FILES["mmlu"]), N_MMLU)


def load_mmlu_dev():
    """Amendment A11: 300 questions from the MMLU validation split (disjoint from the test 500)."""
    return _stratified_mmlu(read_jsonl(DATA_FILES["mmlu_dev"]), N_MMLU_DEV)


def load_all():
    """Final F1: the PopQA test size is fixed by the fresh pool (never cut by the budget ladder)."""
    splits, dev_subset, pool_subjects, pools, popqa_info = load_popqa()
    eq_items, eq_info = load_entity_questions(pool_subjects)
    tqa = load_triviaqa()
    return {"popqa": splits, "dev_subset": dev_subset, "pools": pools, "eq": eq_items, "tqa": tqa,
            "instr": load_instruction_sets(), "mmlu": load_mmlu(), "mmlu_dev": load_mmlu_dev(),
            "ref": load_reference_text(),
            "info": {"popqa": popqa_info, "eq": eq_info,
                     "sizes": {k: len(v) for k, v in splits.items()} | {"eq": len(eq_items), "tqa_dev": len(tqa["dev"]),
                                                                       "tqa_test": len(tqa["test"])}}}

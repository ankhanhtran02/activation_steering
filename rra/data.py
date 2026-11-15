"""Question sets (spec section 5).

  Pool         the SQ2 pool (3,000; pool.jsonl + <model>/shared.pt of the SQ2 run). Trains R.
  Tuning       VES/ECS dev (600; stored drafts, states, B7 rounds) + fresh dev (1,000).
  Replay       VES/ECS test (2,000). Diagnostic only.
  Fresh test   4,000 new questions. Confirmatory.

Fresh dev and test questions:
  - Excluded ids: the SQ2 pool and extension (pilot_qids.json), VES/ECS dev and test (questions.json of the
    VES/ECS run), the earlier study's scored test questions (previously_scored_qids.json) and its TriviaQA dev
    questions (the exclusion lists SQ2 and VES/ECS used).
  - TriviaQA (rc.nocontext validation): 700 dev and 2,800 test, seeded order.
  - PopQA: 6 relations not used before (occupation, director, producer, screenwriter, composer, author), 50 dev
    and 200 test per relation.
  - Isolation: no PopQA subject and no unordered {subject, object} pair is shared with any other set (pool,
    extension, VES/ECS dev, VES/ECS test, the other fresh split). Pairs are compared by PopQA entity id (the
    object id comes from data_cache/ves_popqa_ids.jsonl) and, as a conservative backup, by normalised entity
    names. Fresh PopQA items linked by a subject or a pair are kept in one split (union-find).
  - Duplicate question texts (whitespace-collapsed, case-folded) are dropped: a PopQA question whose text occurs
    more than once in PopQA is ambiguous and never used; any question whose text is already used by another set
    is dropped. isolation_report asserts all of this.
"""
from collections import Counter

import numpy as np

from data import qid_hash
from rra import cfg as C
from vesecs.data import _UF, _clean_aliases, _pair_keys, _pilot_ids
from vesecs.io import read_json, read_jsonl

TQA_RELATION = "triviaqa"


def text_key(q):
    return " ".join(str(q).split()).casefold()


# ----------------------------------------------------------------------------- previous runs
def sq2_dir(cfg):
    p = C.resolve(cfg["paths"]["sq2_run_dir"])
    for f in ("pool.jsonl", "extension_pool.jsonl", "pilot_qids.json"):
        if not (p / f).exists():
            raise SystemExit(f"SQ2 run folder {p} has no {f}: set RRA_SQ2_RUN_DIR (or paths.sq2_run_dir)")
    return p


def ves_dir(cfg):
    p = C.resolve(cfg["paths"]["ves_run_dir"])
    if not (p / "configs" / "frozen_config.json").exists() or not (p / "runs" / "questions.json").exists():
        raise SystemExit(f"VES/ECS run folder {p} has no configs/frozen_config.json or runs/questions.json: "
                         "set RRA_VES_RUN_DIR (or paths.ves_run_dir)")
    return p


def load_pool(cfg):
    """SQ2 pool rows in SQ2 order (the row order of every array in shared.pt)."""
    return read_jsonl(sq2_dir(cfg) / "pool.jsonl")


def load_sq2_shared(cfg, model):
    import torch
    p = sq2_dir(cfg) / model / "shared.pt"
    if not p.exists():
        raise SystemExit(f"missing {p}: the SQ2 pool record of {model} is needed")
    return torch.load(p, weights_only=False, map_location="cpu")


def ves_questions(cfg):
    """VES/ECS dev and test items ({"dev": [...], "test": [...]}), in the VES/ECS question order."""
    return read_json(ves_dir(cfg) / "runs" / "questions.json")["splits"]


def ves_split_items(cfg, model, split):
    """VES/ECS items of one model in its drafts' order (split: 'dev' / 'test')."""
    d = ves_dir(cfg) / "runs" / model / split
    dr = read_json(d / "drafts.json")
    by = {it["qid"]: it for it in ves_questions(cfg)[split]}
    return [by[q] for q in dr["qids"]]


# ----------------------------------------------------------------------------- fresh questions
def _popqa_ids_table():
    from pilot import config as PC
    p = PC.DATA_DIR / "ves_popqa_ids.jsonl"
    if not p.exists():
        return None
    return {int(r["id"]): r for r in read_jsonl(p)}


def _context(cfg):
    """Everything the construction and the isolation check need about the other sets."""
    from pilot import config as PC
    from pilot import data as pdata
    sq2 = sq2_dir(cfg)
    pool = read_jsonl(sq2 / "pool.jsonl")
    ext = read_jsonl(sq2 / "extension_pool.jsonl")
    vq = ves_questions(cfg)
    prev = pdata.load_previously_scored()
    tqa_rows = pdata.read_jsonl(PC.DATA_FILES["triviaqa"])
    earlier_dev = set() if PC.SMOKE else pdata.tqa_earlier_dev_ids(tqa_rows)
    popqa = pdata.read_jsonl(PC.DATA_FILES["popqa_full"])
    by_id = {int(r["id"]): r for r in popqa}
    ids_tab = _popqa_ids_table()

    def obj_id_of(r):
        if ids_tab is None:
            return None
        e = ids_tab.get(int(r["id"]))
        return None if e is None else e.get("obj_id")

    def keys_of_qid(qid):
        r = by_id.get(int(qid.split("_", 1)[1]))
        if r is None:
            raise SystemExit(f"{qid} not found in the PopQA file")
        return int(r["subj_id"]), _pair_keys(r["subj_id"], obj_id_of(r), r["subj"], r["obj"])

    others = {"pool+extension": [it["qid"] for it in pool + ext], "ves_dev": [it["qid"] for it in vq["dev"]],
              "ves_test": [it["qid"] for it in vq["test"]]}
    texts = {"pool+extension": [it["question"] for it in pool + ext], "ves_dev": [it["question"] for it in vq["dev"]],
             "ves_test": [it["question"] for it in vq["test"]]}
    excluded = {"pilot (pool + extension)": _pilot_ids(cfg) | set(others["pool+extension"]),
                "ves_dev": set(others["ves_dev"]), "ves_test": set(others["ves_test"]),
                "earlier popqa test": prev["popqa_test"], "earlier triviaqa test": prev["triviaqa"],
                "earlier entity questions": prev["entity_questions"], "earlier triviaqa dev": earlier_dev}
    return {"pool": pool, "ext": ext, "vq": vq, "tqa_rows": tqa_rows, "popqa": popqa, "by_id": by_id,
            "ids_tab": ids_tab, "obj_id_of": obj_id_of, "keys_of_qid": keys_of_qid, "others": others,
            "texts": texts, "excluded": excluded}


def build_fresh(cfg):
    """Returns ({"dev_fresh": [items], "test_fresh": [items]}, info). Items sorted by question-id hash."""
    d = cfg["data"]
    seed = int(d["shuffle_seed"])
    rels = list(d["popqa_relations"])
    X = _context(cfg)
    excluded = set().union(*X["excluded"].values())
    used_subj, used_pairs = set(), set()
    for g, qids in X["others"].items():
        for q in qids:
            if q.startswith("popqa_"):
                s, pk = X["keys_of_qid"](q)
                used_subj.add(s)
                used_pairs |= pk
    used_text = {text_key(t) for ts in X["texts"].values() for t in ts}
    pop_text_count = Counter(text_key(r["question"]) for r in X["popqa"])

    # ---------------------------------------------------------------- PopQA
    elig, dropped = [], Counter()
    for r in sorted(X["popqa"], key=lambda r: int(r["id"])):
        if r["prop"] not in rels:
            continue
        qid = f"popqa_{r['id']}"
        aliases = _clean_aliases(r["possible_answers"])
        if not (r["subj"] or "").strip() or not aliases:
            dropped["no subject or answer"] += 1
            continue
        if qid in excluded:
            dropped["excluded id"] += 1
            continue
        tk = text_key(r["question"])
        if pop_text_count[tk] > 1 or tk in used_text:
            dropped["duplicate text"] += 1
            continue
        pk = _pair_keys(r["subj_id"], X["obj_id_of"](r), r["subj"], r["obj"])
        if int(r["subj_id"]) in used_subj or (pk & used_pairs):
            dropped["shared subject or pair"] += 1
            continue
        elig.append({"qid": qid, "source": "popqa", "relation": r["prop"], "question": r["question"],
                     "aliases": aliases, "subj": r["subj"], "subj_id": int(r["subj_id"]), "obj": r.get("obj") or "",
                     "obj_id": X["obj_id_of"](r), "_pairs": pk})
    rng = np.random.default_rng([seed, 21])
    elig = [elig[i] for i in rng.permutation(len(elig))]
    uf = _UF()
    for j, it in enumerate(elig):
        uf.union(("item", j), ("subj", it["subj_id"]))
        for k in it["_pairs"]:
            uf.union(("item", j), ("pair",) + k)
    comps, order = {}, []
    for j in range(len(elig)):
        r = uf.find(("item", j))
        if r not in comps:
            comps[r] = []
            order.append(r)
        comps[r].append(j)
    n_dev, n_test = int(d["popqa_dev_per_relation"]), int(d["popqa_test_per_relation"])
    dev_have = Counter()
    split_of = {}
    for r in order:                       # components in seeded order: to dev while a relation still needs dev items
        rc = Counter(elig[j]["relation"] for j in comps[r])
        to_dev = any(dev_have[rel] < n_dev for rel in rc)
        for j in comps[r]:
            split_of[j] = "dev_fresh" if to_dev else "test_fresh"
        if to_dev:
            dev_have.update(rc)
    picked = {"dev_fresh": [], "test_fresh": []}
    taken = Counter()
    for j, it in enumerate(elig):         # seeded order within each relation
        sp = split_of[j]
        cap = n_dev if sp == "dev_fresh" else n_test
        if taken[(sp, it["relation"])] < cap:
            taken[(sp, it["relation"])] += 1
            picked[sp].append(it)
    short = {f"{sp}/{rel}": taken[(sp, rel)] for sp in picked for rel in rels
             if taken[(sp, rel)] < (n_dev if sp == "dev_fresh" else n_test)}
    if short:
        raise SystemExit(f"not enough isolated PopQA questions: {short}")
    for sp in picked:
        for it in picked[sp]:
            used_text.add(text_key(it["question"]))

    # ---------------------------------------------------------------- TriviaQA
    seen, tqa = set(), []
    for r in sorted(X["tqa_rows"], key=lambda r: r["question_id"]):
        qid = f"tqa_{r['question_id']}"
        if qid in excluded or qid in seen:
            continue
        seen.add(qid)
        aliases = _clean_aliases(dict.fromkeys(list(r["aliases"] or []) + [r["value"]]))
        if not aliases:
            continue
        tqa.append({"qid": qid, "source": "tqa", "relation": TQA_RELATION, "question": r["question"],
                    "aliases": aliases, "subj": "", "subj_id": None, "obj": r["value"], "obj_id": None})
    trng = np.random.default_rng([seed, 22])
    tqa = [tqa[i] for i in trng.permutation(len(tqa))]
    need = {"dev_fresh": int(d["tqa_dev"]), "test_fresh": int(d["tqa_test"])}
    tq = {"dev_fresh": [], "test_fresh": []}
    for it in tqa:
        sp = "dev_fresh" if len(tq["dev_fresh"]) < need["dev_fresh"] else (
            "test_fresh" if len(tq["test_fresh"]) < need["test_fresh"] else None)
        if sp is None:
            break
        k = text_key(it["question"])
        if k in used_text:
            dropped["tqa duplicate text"] += 1
            continue
        used_text.add(k)
        tq[sp].append(it)
    if any(len(tq[s]) < need[s] for s in need):
        raise SystemExit(f"not enough TriviaQA questions: {need} needed, {[len(v) for v in tq.values()]} found")

    splits = {s: picked[s] + tq[s] for s in ("dev_fresh", "test_fresh")}
    for name, items in splits.items():
        for it in items:
            it.pop("_pairs", None)
            it["split"] = name
        items.sort(key=lambda it: qid_hash(it["qid"]))
    info = {"n_popqa_eligible": len(elig), "n_popqa_components": len(order), "dropped": dict(dropped),
            "popqa_ids_file": X["ids_tab"] is not None, "composition": composition(splits),
            "n_excluded_ids": len(excluded)}
    return splits, info


def composition(splits):
    out = {}
    for name, items in splits.items():
        out[name] = {"n": len(items), "by_source": dict(Counter(it["source"] for it in items)),
                     "by_relation": dict(sorted(Counter(it["relation"] for it in items).items()))}
    return out


def isolation_report(cfg, fresh):
    """Integrity check 1. Returns (ok, details). Recomputes every rule from the raw files."""
    X = _context(cfg)
    groups = dict(X["others"])
    groups.update({s: [it["qid"] for it in fresh[s]] for s in ("dev_fresh", "test_fresh")})
    texts = dict(X["texts"])
    texts.update({s: [it["question"] for it in fresh[s]] for s in ("dev_fresh", "test_fresh")})
    subj = {g: set() for g in groups}
    pairs = {g: set() for g in groups}
    for g, qids in groups.items():
        for q in qids:
            if q.startswith("popqa_"):
                s, pk = X["keys_of_qid"](q)
                subj[g].add(s)
                pairs[g] |= pk
    det = {"id_overlap": {}, "subject_overlap": {}, "pair_overlap": {}, "text_overlap": {}}
    for s in ("dev_fresh", "test_fresh"):
        ids = set(groups[s])
        for k, v in X["excluded"].items():
            det["id_overlap"][f"{s} vs {k}"] = len(ids & v)
        tk = Counter(text_key(t) for t in texts[s])
        det["text_overlap"][f"{s} within"] = sum(c - 1 for c in tk.values() if c > 1)
        for g in groups:
            if g == s or (s == "test_fresh" and g == "dev_fresh"):
                continue
            det["subject_overlap"][f"{s} vs {g}"] = len(subj[s] & subj[g])
            det["pair_overlap"][f"{s} vs {g}"] = len(pairs[s] & pairs[g])
            det["text_overlap"][f"{s} vs {g}"] = len(set(tk) & {text_key(t) for t in texts[g]})
        det["id_overlap"][f"{s} vs other fresh split"] = len(ids & set(groups["test_fresh" if s == "dev_fresh" else "dev_fresh"]))
    # an ambiguous PopQA text (it occurs more than once in PopQA) must not be used
    pop_count = Counter(text_key(r["question"]) for r in X["popqa"])
    det["ambiguous_popqa_text"] = sum(1 for s in ("dev_fresh", "test_fresh") for it in fresh[s]
                                      if it["source"] == "popqa" and pop_count[text_key(it["question"])] > 1)
    det["duplicate_ids_within_split"] = {s: len(groups[s]) - len(set(groups[s])) for s in ("dev_fresh", "test_fresh")}
    det["pair_ids_available"] = X["ids_tab"] is not None
    ok = all(v == 0 for k in ("id_overlap", "subject_overlap", "pair_overlap", "text_overlap", "duplicate_ids_within_split")
             for v in det[k].values()) and det["ambiguous_popqa_text"] == 0
    return ok, det

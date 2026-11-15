"""Questions (spec section 3).

Training data: the SQ2 pool (3,000 questions of the SQ2 run: pool.jsonl + <model>/shared.pt). It is used only
to fit detectors and compute steering constants. Extension questions are never used; their ids are excluded.

Dev and test: new questions.
  - Excluded: every id in pilot_qids.json (pool and extension; the run folder's copy and the repository's),
    the earlier study's scored test questions (previously_scored_qids.json) and its TriviaQA dev questions
    (the same exclusion lists SQ2 used).
  - PopQA: the 8 SQ2 relations, all remaining eligible items, shuffled (seed 0), 25% dev / 75% test, capped at
    200 dev and 600 test.
  - Isolation (PopQA): a dev or test subject is not a subject in the pool, the extension or the other split;
    the unordered {subject, object} pair does not occur there either. Pairs are compared by PopQA entity id
    (subject and object ids share one id space; the object id comes from vesecs.setup_data) and, as a
    conservative backup, by normalised entity names. Items connected by a subject or a pair are kept in the
    same split (union-find), so the dev/test assignment cannot break isolation.
  - TriviaQA (rc.nocontext validation): fills each split to 600 dev / 2,000 test questions; a question whose
    text (whitespace-collapsed, case-folded) duplicates another used question is dropped.
"""
from collections import Counter

import numpy as np

from data import qid_hash
from vesecs import cfg as C
from vesecs.io import read_json, read_jsonl

TQA_RELATION = "triviaqa"


def _text_key(q):
    return " ".join(str(q).split()).casefold()


def _clean_aliases(seq):
    """Non-empty alias strings. Raw dataset fields can be null (three PopQA rows have subj = null; the same
    fix as pilot/data.py, commit b7d9bc8): a missing list or value counts as empty instead of raising."""
    return [a for a in seq or [] if a and a.strip()]


def _popqa_keep(r, rels):
    """Gold aliases if PopQA row r is in one of the relations and has a subject, else None."""
    if r["prop"] not in rels or not (r["subj"] or "").strip():
        return None
    return _clean_aliases(r["possible_answers"]) or None


def _name_key(s):
    from scoring import normalize_answer
    return normalize_answer(s or "")


def sq2_dir(cfg):
    p = C.resolve(cfg["paths"]["sq2_run_dir"])
    for f in ("pool.jsonl", "extension_pool.jsonl", "pilot_qids.json"):
        if not (p / f).exists():
            raise SystemExit(f"SQ2 run folder {p} has no {f}: set paths.sq2_run_dir in configs/config.yaml "
                             "or VES_SQ2_RUN_DIR")
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


def load_sq2_probe(cfg, model):
    import torch
    p = sq2_dir(cfg) / model / "probe.pt"
    return torch.load(p, weights_only=False, map_location="cpu") if p.exists() else None


def _pilot_ids(cfg):
    from pilot import config as PC
    ids = set()
    for p in (sq2_dir(cfg) / "pilot_qids.json", PC.PILOT_QIDS_PATH):
        if p.exists():
            d = read_json(p)
            for k, v in d.items():
                if isinstance(v, list):
                    ids |= set(v)
    return ids


def _popqa_ids_table(cfg):
    from pilot import config as PC
    p = PC.DATA_DIR / cfg["paths"]["popqa_ids_file"]
    if not p.exists():
        return None
    return {int(r["id"]): r for r in read_jsonl(p)}


class _UF:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _pair_keys(subj_id, obj_id, subj, obj):
    keys = set()
    if subj_id is not None and obj_id is not None:
        keys.add(("id",) + tuple(sorted((int(subj_id), int(obj_id)))))
    a, b = _name_key(subj), _name_key(obj)
    if a and b:
        keys.add(("name",) + tuple(sorted((a, b))))
    return keys


def build_splits(cfg):
    """Returns ({"dev": [items], "test": [items]}, info). Items are dicts sorted by question-id hash."""
    from pilot import config as PC
    from pilot import data as pdata
    d = cfg["data"]
    seed = int(d["shuffle_seed"])
    rels = list(d["popqa_relations"])
    sq2 = sq2_dir(cfg)
    pool = read_jsonl(sq2 / "pool.jsonl")
    ext = read_jsonl(sq2 / "extension_pool.jsonl")
    pilot_ids = _pilot_ids(cfg)
    missing = [it["qid"] for it in pool + ext if it["qid"] not in pilot_ids]
    if missing:
        raise SystemExit(f"{len(missing)} pool / extension ids are missing from pilot_qids.json")
    prev = pdata.load_previously_scored()
    tqa_rows = pdata.read_jsonl(PC.DATA_FILES["triviaqa"])
    earlier_dev = set() if PC.SMOKE else pdata.tqa_earlier_dev_ids(tqa_rows)
    excluded = pilot_ids | prev["popqa_test"] | prev["triviaqa"] | prev["entity_questions"] | earlier_dev

    popqa = pdata.read_jsonl(PC.DATA_FILES["popqa_full"])
    by_id = {int(r["id"]): r for r in popqa}
    ids_tab = _popqa_ids_table(cfg)

    def obj_id_of(r):
        if ids_tab is None:
            return None
        e = ids_tab.get(int(r["id"]))
        return None if e is None else e.get("obj_id")

    used_subj, used_pairs = set(), set()
    for it in pool + ext:
        if it["dataset"] != "popqa":
            continue
        r = by_id.get(int(it["qid"].split("_", 1)[1]))
        if r is None:
            raise SystemExit(f"{it['qid']} not found in the PopQA file")
        used_subj.add(int(r["subj_id"]))
        used_pairs |= _pair_keys(r["subj_id"], obj_id_of(r), r["subj"], r["obj"])

    elig = []
    for r in sorted(popqa, key=lambda r: int(r["id"])):
        qid = f"popqa_{r['id']}"
        aliases = _popqa_keep(r, rels)
        if aliases is None or qid in excluded:
            continue
        pk = _pair_keys(r["subj_id"], obj_id_of(r), r["subj"], r["obj"])
        if int(r["subj_id"]) in used_subj or (pk & used_pairs):
            continue
        elig.append({"qid": qid, "source": "popqa", "relation": r["prop"], "question": r["question"],
                     "aliases": aliases, "subj": r["subj"], "subj_id": int(r["subj_id"]), "obj": r.get("obj") or "",
                     "obj_id": obj_id_of(r), "_pairs": pk})
    rng = np.random.default_rng(seed)
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
    n_dev_target = int(round(float(d["popqa_dev_frac"]) * len(elig)))
    dev_idx, test_idx = [], []
    for r in order:
        (dev_idx if len(dev_idx) < n_dev_target else test_idx).extend(comps[r])
    dev_pop = [elig[j] for j in sorted(dev_idx)][:int(d["popqa_dev_cap"])]
    test_pop = [elig[j] for j in sorted(test_idx)][:int(d["popqa_test_cap"])]

    used_text = {_text_key(it["question"]) for it in pool + ext + dev_pop + test_pop}
    seen, tqa = set(), []
    for r in sorted(tqa_rows, key=lambda r: r["question_id"]):
        qid = f"tqa_{r['question_id']}"
        if qid in excluded or qid in seen:
            continue
        seen.add(qid)
        aliases = _clean_aliases(dict.fromkeys(list(r["aliases"] or []) + [r["value"]]))
        if not aliases:
            continue
        tqa.append({"qid": qid, "source": "tqa", "relation": TQA_RELATION, "question": r["question"],
                    "aliases": aliases, "subj": "", "subj_id": None, "obj": r["value"], "obj_id": None})
    trng = np.random.default_rng([seed, 1])
    tqa = [tqa[i] for i in trng.permutation(len(tqa))]
    need = {"dev": int(d["dev_size"]) - len(dev_pop), "test": int(d["test_size"]) - len(test_pop)}
    picked = {"dev": [], "test": []}
    dropped_dup = 0
    for it in tqa:
        split = "dev" if len(picked["dev"]) < need["dev"] else ("test" if len(picked["test"]) < need["test"] else None)
        if split is None:
            break
        k = _text_key(it["question"])
        if k in used_text:
            dropped_dup += 1
            continue
        used_text.add(k)
        picked[split].append(it)
    if len(picked["dev"]) < need["dev"] or len(picked["test"]) < need["test"]:
        raise SystemExit(f"not enough TriviaQA questions: {need} needed, {[len(v) for v in picked.values()]} found")

    splits = {"dev": dev_pop + picked["dev"], "test": test_pop + picked["test"]}
    for name, items in splits.items():
        for it in items:
            it.pop("_pairs", None)
            it["split"] = name
        items.sort(key=lambda it: qid_hash(it["qid"]))
    info = {"n_popqa_eligible": len(elig), "n_popqa_dev_target": n_dev_target, "popqa_ids_file": ids_tab is not None,
            "tqa_dropped_duplicate_text": dropped_dup, "composition": composition(splits),
            "n_excluded_ids": len(excluded), "n_pool": len(pool), "n_extension": len(ext)}
    return splits, info


def composition(splits):
    out = {}
    for name, items in splits.items():
        out[name] = {"n": len(items), "by_source": dict(Counter(it["source"] for it in items)),
                     "by_relation": dict(sorted(Counter(it["relation"] for it in items).items()))}
    return out


def isolation_report(cfg, splits):
    """Integrity check 1. Returns (ok, details)."""
    from pilot import config as PC
    from pilot import data as pdata
    sq2 = sq2_dir(cfg)
    pool = read_jsonl(sq2 / "pool.jsonl")
    ext = read_jsonl(sq2 / "extension_pool.jsonl")
    pilot_ids = _pilot_ids(cfg)
    prev = pdata.load_previously_scored()
    tqa_rows = pdata.read_jsonl(PC.DATA_FILES["triviaqa"])
    earlier_dev = set() if PC.SMOKE else pdata.tqa_earlier_dev_ids(tqa_rows)
    popqa = {int(r["id"]): r for r in pdata.read_jsonl(PC.DATA_FILES["popqa_full"])}
    ids_tab = _popqa_ids_table(cfg)

    def keys(qid):
        r = popqa[int(qid.split("_", 1)[1])]
        oid = None if ids_tab is None else ids_tab.get(int(r["id"]), {}).get("obj_id")
        return int(r["subj_id"]), _pair_keys(r["subj_id"], oid, r["subj"], r["obj"])

    groups = {"pool+extension": [it["qid"] for it in pool + ext], "dev": [it["qid"] for it in splits["dev"]],
              "test": [it["qid"] for it in splits["test"]]}
    subj = {g: set() for g in groups}
    pairs = {g: set() for g in groups}
    for g, qids in groups.items():
        for q in qids:
            if q.startswith("popqa_"):
                s, pk = keys(q)
                subj[g].add(s)
                pairs[g] |= pk
    dev, test = set(groups["dev"]), set(groups["test"])
    excl = {"pilot (pool + extension)": pilot_ids, "earlier popqa test": prev["popqa_test"],
            "earlier triviaqa test": prev["triviaqa"], "earlier entity questions": prev["entity_questions"],
            "earlier triviaqa dev": earlier_dev}
    det = {"id_overlap": {f"{s} vs {k}": len(ids & v) for s, ids in (("dev", dev), ("test", test)) for k, v in excl.items()}}
    det["id_overlap"]["dev vs test"] = len(dev & test)
    det["subject_overlap"] = {"dev vs pool+extension": len(subj["dev"] & subj["pool+extension"]),
                              "test vs pool+extension": len(subj["test"] & subj["pool+extension"]),
                              "dev vs test": len(subj["dev"] & subj["test"])}
    det["pair_overlap"] = {"dev vs pool+extension": len(pairs["dev"] & pairs["pool+extension"]),
                           "test vs pool+extension": len(pairs["test"] & pairs["pool+extension"]),
                           "dev vs test": len(pairs["dev"] & pairs["test"])}
    det["pair_ids_available"] = ids_tab is not None
    det["duplicate_ids_within_split"] = {s: len(v) - len(set(v)) for s, v in (("dev", groups["dev"]), ("test", groups["test"]))}
    ok = (all(v == 0 for v in det["id_overlap"].values()) and all(v == 0 for v in det["subject_overlap"].values())
          and all(v == 0 for v in det["pair_overlap"].values())
          and all(v == 0 for v in det["duplicate_ids_within_split"].values()))
    return ok, det


def shrink_test(items, target, seed):
    """Budget cut 5: drop TriviaQA test questions (seeded) until `target` remain. Returns kept qids."""
    n_drop = max(0, len(items) - int(target))
    tqa = sorted(it["qid"] for it in items if it["source"] == "tqa")
    rng = np.random.default_rng([int(seed), 5])
    drop = set(tqa[i] for i in rng.permutation(len(tqa))[:n_drop])
    return [it["qid"] for it in items if it["qid"] not in drop]

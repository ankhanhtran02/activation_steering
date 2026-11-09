"""The pilot's question pool (built once, used by every sub-question).

~3,000 short factual questions: TriviaQA (rc.nocontext validation) and PopQA (clean relations), with
no PopQA relation above 10% of the pool. Questions that the earlier study scored on a test split
(previously_scored_qids.json) and its TriviaQA dev questions are excluded. Every pilot question id is
written to pilot_qids.json so a full study can exclude them (one-time use).

Per item `extra` holds: fold (0..4, grouped by PopQA subject), group, and, where applicable,
sq3 = {"wordings": [5 strings], "split": "tune"|"test"} and
sq5 = {"kind": "pob"|"year", "coarse_question": str, "coarse_aliases": [...]}.
"""
import csv
import json
import re
from collections import Counter

import numpy as np

from data import QAItem, qid_hash
from pilot import config as PC

# ----------------------------------------------------------------------------- SQ3 reword templates
# Wording 0 is always the original PopQA question (PopQA questions come from one template per relation).
SQ3_TEMPLATES = {
    "place of birth": ["What city was {s} born in?", "{s} was born in which city?",
                       "Where was {s} born? Name the city.", "Name the city where {s} was born."],
    "country": ["Which country is {s} in?", "{s} is located in which country?",
                "What country is {s} located in?", "Name the country where {s} is located."],
    "capital": ["Which city is the capital of {s}?", "What city is the capital of {s}?",
                "{s} has which city as its capital?", "Name the capital of {s}."],
    "capital of": ["{s} is the capital of what?", "Of what is {s} the capital?",
                   "{s} serves as the capital of which place?", "Which place has {s} as its capital?"],
    "sport": ["Which sport does {s} play?", "{s} plays which sport?",
              "What sport is played by {s}?", "Name the sport that {s} plays."],
    "religion": ["What religion does {s} belong to?", "{s} is affiliated with which religion?",
                 "What is the religion associated with {s}?", "Name the religion of {s}."],
    "father": ["Who was the father of {s}?", "Who is {s}'s father?",
               "Name the father of {s}.", "Which person is the father of {s}?"],
    "mother": ["Who was the mother of {s}?", "Who is {s}'s mother?",
               "Name the mother of {s}.", "Which person is the mother of {s}?"],
}
assert set(SQ3_TEMPLATES) == set(PC.POPQA_QUOTAS) or PC.SMOKE

YEAR_RE = re.compile(r"^(1[0-9]{3}|20[0-2][0-9])$")
YEAR_Q_RE = re.compile(r"\b(year|when)\b", re.IGNORECASE)
DECADE_WORDS = {1900: "nineteen hundreds", 1910: "nineteen tens", 1920: "twenties", 1930: "thirties",
                1940: "forties", 1950: "fifties", 1960: "sixties", 1970: "seventies", 1980: "eighties",
                1990: "nineties"}
# Wikidata gives one current country (P17). Common short forms are added; for the UK its constituent
# countries are accepted (lenient: a city in England answered "Scotland" would count; documented).
COUNTRY_EXTRA_ALIASES = {
    "United States of America": ["United States", "USA", "U.S.", "U.S.A.", "US", "America"],
    "United States": ["United States of America", "USA", "U.S.", "U.S.A.", "US", "America"],
    "United Kingdom": ["UK", "U.K.", "Britain", "Great Britain", "England", "Scotland", "Wales", "Northern Ireland"],
    "People's Republic of China": ["China"], "Kingdom of the Netherlands": ["Netherlands", "Holland"],
    "Russia": ["Russian Federation"], "Germany": ["Federal Republic of Germany"],
}
SQ5_YEAR_SUFFIX = " Give only the decade (for example: the 1960s), not the exact year."


def read_jsonl(name):
    path = PC.DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `bash run_pilot.sh setup` first (the experiment is offline)")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_previously_scored():
    d = json.loads(PC.PREVIOUSLY_SCORED_PATH.read_text()) if PC.PREVIOUSLY_SCORED_PATH.exists() else {}
    return {k: set(d.get(k, [])) for k in ("popqa_test", "entity_questions", "triviaqa")}


def tqa_earlier_dev_ids(rows):
    """The earlier study's TriviaQA dev questions (data.load_triviaqa: sorted by id, rng(0), first 300)."""
    rows = sorted(rows, key=lambda r: r["question_id"])
    perm = np.random.default_rng(0).permutation(len(rows))
    return {f"tqa_{rows[i]['question_id']}" for i in perm[:300]}


def decade_aliases(year):
    d = int(year) // 10 * 10
    out = [f"{d}s", f"{d}'s", f"the {d}s"]
    if d in DECADE_WORDS:
        out.append(DECADE_WORDS[d])
    return out


def load_wikidata_countries():
    """{city Wikidata id: [country names and aliases]} from setup_pilot.py's SPARQL fetch."""
    from scoring import normalize_answer
    out = {}
    for r in read_jsonl(PC.DATA_FILES["wikidata_country"]):
        # Wikidata aliases include ISO codes ("se", "ao") that whole-word matching could hit by accident
        aliases = [a for a in r.get("aliases", []) if len(normalize_answer(a)) > 2 or normalize_answer(a) in ("us", "uk")]
        names = [r["country_label"]] + aliases
        names += COUNTRY_EXTRA_ALIASES.get(r["country_label"], [])
        out.setdefault(r["city_qid"], [])
        out[r["city_qid"]] += [n for n in names if n and n not in out[r["city_qid"]]]
    return out


def qid_from_uri(uri):
    return uri.rstrip("/").rsplit("/", 1)[-1] if uri else ""


def _fold_of(group):
    return qid_hash(group) % PC.N_FOLDS


def build_pool():
    """Returns (items sorted by question-id hash, info dict)."""
    prev = load_previously_scored()
    rng = np.random.default_rng(PC.PILOT_SEED)
    info = {}

    # PopQA: clean relations, quota per relation, earlier test questions excluded
    by_rel = {rel: [] for rel in PC.POPQA_QUOTAS}
    for r in read_jsonl(PC.DATA_FILES["popqa_full"]):
        qid = f"popqa_{r['id']}"
        if r["prop"] in by_rel and r["subj"].strip() and qid not in prev["popqa_test"]:
            by_rel[r["prop"]].append(r)
    pop_items = []
    for rel in sorted(PC.POPQA_QUOTAS):
        cand = sorted(by_rel[rel], key=lambda r: int(r["id"]))
        if len(cand) < PC.POPQA_QUOTAS[rel]:
            raise RuntimeError(f"PopQA relation {rel}: {len(cand)} questions < quota {PC.POPQA_QUOTAS[rel]}")
        for i in rng.permutation(len(cand))[:PC.POPQA_QUOTAS[rel]]:
            r = cand[i]
            pop_items.append(QAItem(qid=f"popqa_{r['id']}", question=r["question"],
                                    aliases=[a for a in r["possible_answers"] if a.strip()], dataset="popqa",
                                    subj=r["subj"], subj_id=int(r["subj_id"]), prop=rel,
                                    log_pop=float(np.log10(int(r["s_pop"]) + 1)),
                                    extra={"group": f"popqa_subj_{r['subj_id']}", "obj": r.get("obj", ""),
                                           "obj_qid": qid_from_uri(r.get("o_uri", ""))}))

    # TriviaQA: 100 year-answer questions (SQ5) + random others; earlier test and dev questions excluded
    tqa_rows = read_jsonl(PC.DATA_FILES["triviaqa"])
    excluded = prev["triviaqa"] | (set() if PC.SMOKE else tqa_earlier_dev_ids(tqa_rows))
    seen, years, others = set(), [], []
    for r in sorted(tqa_rows, key=lambda r: r["question_id"]):
        qid = f"tqa_{r['question_id']}"
        if qid in excluded or qid in seen:
            continue
        seen.add(qid)
        aliases = [a for a in dict.fromkeys(list(r["aliases"]) + [r["value"]]) if a and a.strip()]
        it = QAItem(qid=qid, question=r["question"], aliases=aliases, dataset="tqa", prop="triviaqa",
                    extra={"group": qid, "value": r["value"]})
        is_year = bool(YEAR_RE.match(r["value"].strip())) and bool(YEAR_Q_RE.search(r["question"]))
        (years if is_year else others).append(it)
    if len(years) < PC.N_SQ5_YEAR:
        raise RuntimeError(f"only {len(years)} TriviaQA year questions < {PC.N_SQ5_YEAR}")
    tqa_items = [years[i] for i in rng.permutation(len(years))[:PC.N_SQ5_YEAR]]
    tqa_items += [others[i] for i in rng.permutation(len(others))[:PC.N_TQA - PC.N_SQ5_YEAR]]
    for it in tqa_items[:PC.N_SQ5_YEAR]:
        y = it.extra["value"].strip()
        it.extra["sq5"] = {"kind": "year", "coarse_question": it.question + SQ5_YEAR_SUFFIX,
                           "coarse_aliases": decade_aliases(y)}

    # SQ5: place-of-birth questions whose city has a Wikidata country
    countries = load_wikidata_countries()
    pob = [it for it in pop_items if it.prop == "place of birth" and countries.get(it.extra["obj_qid"])]
    pob = [pob[i] for i in rng.permutation(len(pob))]
    for it in pob[:PC.N_SQ5_POB]:
        it.extra["sq5"] = {"kind": "pob", "coarse_question": f"In what country was {it.subj} born?",
                           "coarse_aliases": countries[it.extra["obj_qid"]]}
    info["n_sq5"] = {"year": PC.N_SQ5_YEAR, "pob": min(len(pob), PC.N_SQ5_POB), "pob_with_country": len(pob)}

    # SQ3: 500 PopQA questions, a third of each relation's quota, reworded with templates
    n_sq3 = {rel: int(round(PC.N_SQ3 * q / sum(PC.POPQA_QUOTAS.values()))) for rel, q in PC.POPQA_QUOTAS.items()}
    diff = PC.N_SQ3 - sum(n_sq3.values())
    for rel in sorted(n_sq3, key=lambda r: -PC.POPQA_QUOTAS[r])[:abs(diff)]:
        n_sq3[rel] += 1 if diff > 0 else -1
    sq3 = []
    for rel in sorted(n_sq3):
        cand = sorted([it for it in pop_items if it.prop == rel], key=lambda it: qid_hash(it.qid))
        sq3 += cand[:n_sq3[rel]]
    sq3.sort(key=lambda it: qid_hash("sq3" + it.qid))
    for j, it in enumerate(sq3):
        it.extra["sq3"] = {"wordings": [it.question] + [t.format(s=it.subj) for t in SQ3_TEMPLATES[it.prop]],
                           "split": "tune" if j < PC.N_SQ3_TUNE else "test"}

    pool = pop_items + tqa_items
    for it in pool:
        it.extra["fold"] = _fold_of(it.extra["group"])
    pool.sort(key=lambda it: qid_hash(it.qid))   # fixed, mixed order (batches are formed in this order)
    assert len({it.qid for it in pool}) == len(pool), "duplicate question in the pool"

    topics = Counter(it.prop for it in pool)
    info["n_pool"] = len(pool)
    info["topic_counts"] = dict(topics)
    info["max_popqa_relation_frac"] = max(v for k, v in topics.items() if k != "triviaqa") / len(pool)
    info["topic_cap_ok"] = info["max_popqa_relation_frac"] <= PC.TOPIC_CAP_FRAC + 1e-9
    info["tqa_topics"] = "TriviaQA has no topic labels; treated as mixed-topic (not checked against the cap)"
    info["fold_sizes"] = dict(Counter(it.extra["fold"] for it in pool))
    info["n_sq3"] = len(sq3)
    return pool, info


def build_ext_pool(pool):
    """SQ2 screening extension: questions disjoint from the main pool, PopQA subjects disjoint too (so the
    detector, trained on the main pool, never saw a fact about the same entity). Same sources and
    exclusions; PopQA quotas are filled up to what is available. Every item gets fold EXT_FOLD."""
    prev = load_previously_scored()
    rng = np.random.default_rng(PC.PILOT_SEED + 101)
    used = {it.qid for it in pool}
    used_subj = {it.subj_id for it in pool if it.dataset == "popqa"}
    by_rel = {rel: [] for rel in PC.EXT_POPQA_QUOTAS}
    for r in read_jsonl(PC.DATA_FILES["popqa_full"]):
        qid = f"popqa_{r['id']}"
        if (r["prop"] in by_rel and r["subj"].strip() and qid not in prev["popqa_test"] and qid not in used
                and int(r["subj_id"]) not in used_subj):
            by_rel[r["prop"]].append(r)
    items, shortfall = [], {}
    for rel in sorted(PC.EXT_POPQA_QUOTAS):
        cand = sorted(by_rel[rel], key=lambda r: int(r["id"]))
        if len(cand) < PC.EXT_POPQA_QUOTAS[rel]:
            shortfall[rel] = len(cand)
        seen_subj = set()
        for i in rng.permutation(len(cand)):
            r = cand[i]
            if sum(it.prop == rel for it in items) >= PC.EXT_POPQA_QUOTAS[rel]:
                break
            if int(r["subj_id"]) in seen_subj:
                continue
            seen_subj.add(int(r["subj_id"]))
            items.append(QAItem(qid=f"popqa_{r['id']}", question=r["question"],
                                aliases=[a for a in r["possible_answers"] if a.strip()], dataset="popqa",
                                subj=r["subj"], subj_id=int(r["subj_id"]), prop=rel,
                                extra={"group": f"popqa_subj_{r['subj_id']}", "fold": PC.EXT_FOLD}))
    tqa_rows = read_jsonl(PC.DATA_FILES["triviaqa"])
    excluded = prev["triviaqa"] | (set() if PC.SMOKE else tqa_earlier_dev_ids(tqa_rows)) | used
    cand, seen = [], set()
    for r in sorted(tqa_rows, key=lambda r: r["question_id"]):
        qid = f"tqa_{r['question_id']}"
        if qid in excluded or qid in seen:
            continue
        seen.add(qid)
        cand.append(QAItem(qid=qid, question=r["question"], dataset="tqa", prop="triviaqa",
                           aliases=[a for a in dict.fromkeys(list(r["aliases"]) + [r["value"]]) if a and a.strip()],
                           extra={"group": qid, "fold": PC.EXT_FOLD}))
    items += [cand[i] for i in rng.permutation(len(cand))[:PC.N_EXT_TQA]]
    items.sort(key=lambda it: qid_hash(it.qid))
    assert not ({it.qid for it in items} & used), "extension overlaps the main pool"
    return items, {"n_ext": len(items), "topic_counts": dict(Counter(it.prop for it in items)),
                   "popqa_shortfall": shortfall}


def write_pilot_qids(pool, path):
    by_ds = {}
    for it in pool:
        by_ds.setdefault(it.dataset, []).append(it.qid)
    payload = {"description": "Every question the pilot used, including the SQ2 screening extension. A full "
                              "study must exclude these ids (the pilot's questions are for one-time use).",
               "n": len(pool), **{k: sorted(v) for k, v in by_ds.items()}}
    path.write_text(json.dumps(payload, indent=1))


def write_pool(pool, path):
    with open(path, "w", encoding="utf-8") as f:
        for it in pool:
            f.write(json.dumps({"qid": it.qid, "dataset": it.dataset, "prop": it.prop, "question": it.question,
                                "aliases": it.aliases, "subj": it.subj, "extra": it.extra}, ensure_ascii=False) + "\n")


def write_paraphrase_check(pool, path, n=None):
    """Hand check of the rewordings: one row per template (each template once) plus random extras.
    Fill `meaning_preserved` with yes / no."""
    n = n or PC.SQ3_N_PARA_CHECK
    rng = np.random.default_rng(PC.PILOT_SEED + 3)
    items = [it for it in pool if it.extra.get("sq3")]
    rows, used = [], set()
    for rel in sorted({it.prop for it in items}):
        rel_items = [it for it in items if it.prop == rel]
        for k in range(1, 5):
            it = rel_items[rng.integers(len(rel_items))]
            rows.append((it, k))
            used.add((it.qid, k))
    while len(rows) < n and len(used) < 4 * len(items):
        it = items[rng.integers(len(items))]
        k = int(rng.integers(1, 5))
        if (it.qid, k) not in used:
            used.add((it.qid, k))
            rows.append((it, k))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "relation", "template", "original", "reworded", "meaning_preserved"])
        for it, k in rows:
            w.writerow([it.qid, it.prop, k, it.question, it.extra["sq3"]["wordings"][k], ""])
    return len(rows)


if __name__ == "__main__":
    pool, info = build_pool()
    write_pilot_qids(pool, PC.PILOT_QIDS_PATH)
    print(json.dumps(info, indent=1))
    print(f"wrote {PC.PILOT_QIDS_PATH}")

"""Downloads for the pilot (needs network; run once before pilot.main, which is offline).

    python -m pilot.setup_pilot

- both models at their pinned revisions (~31 GB in bf16; Qwen2.5-7B-Instruct is usually cached already);
- PopQA test at the pinned revision, keeping the object's Wikidata id (o_uri) for SQ5;
- TriviaQA rc.nocontext validation (reuses data_cache/triviaqa_validation.jsonl from the earlier study);
- the country (Wikidata P17, best-ranked statements, with English aliases) of every PopQA
  place-of-birth city, via the Wikidata entity API (wbgetentities). Wikidata is live and cannot be
  pinned: the fetched file's sha256 and fetch time are recorded, and pilot.main refuses to run if the
  file changes afterwards.
Writes pilot_manifest.json.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from pilot import config as PC


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def download_models():
    from huggingface_hub import snapshot_download
    out = {}
    for spec in PC.MODELS:
        path = snapshot_download(spec["hf_id"], revision=spec["revision"], allow_patterns=PC.MODEL_FILE_PATTERNS,
                                 token=os.environ.get("HF_TOKEN"))
        got = os.path.basename(path)
        if got != spec["revision"]:
            raise RuntimeError(f"{spec['hf_id']}: snapshot {got} != pinned {spec['revision']}")
        print(f"model {spec['hf_id']} @ {got}")
        out[spec["name"]] = {"hf_id": spec["hf_id"], "revision": got}
    return out


def download_popqa_full():
    from datasets import load_dataset
    ds = load_dataset(PC.POPQA_PIN["repo"], split=PC.POPQA_PIN["split"], revision=PC.POPQA_PIN["revision"])
    rows = []
    for r in ds:
        ans = r["possible_answers"]
        rows.append({"id": int(r["id"]), "question": r["question"], "subj": r["subj"], "subj_id": int(r["subj_id"]),
                     "prop": r["prop"], "obj": r["obj"], "o_uri": r["o_uri"], "s_pop": int(r["s_pop"]),
                     "possible_answers": json.loads(ans) if isinstance(ans, str) else list(ans)})
    write_jsonl(PC.DATA_DIR / PC.DATA_FILES["popqa_full"], rows)
    return rows


def ensure_triviaqa():
    path = PC.DATA_DIR / PC.DATA_FILES["triviaqa"]
    if path.exists():
        print(f"{path.name} already present (earlier study's pinned download); reused")
        return
    from datasets import load_dataset
    p = PC.TRIVIAQA_PIN
    rows = [{"question_id": r["question_id"], "question": r["question"], "value": r["answer"]["value"],
             "aliases": list(r["answer"]["aliases"])}
            for r in load_dataset(p["repo"], p["config"], split=p["split"], revision=p["revision"])]
    write_jsonl(path, rows)


def _wbgetentities(ids, props):
    """Wikidata entity API (not the SPARQL query service, which rate-limits hard during outages).
    Honours HTTP 429 / Retry-After."""
    url = PC.WIKIDATA_API + "?" + urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(ids),
                                                          "props": props, "languages": "en", "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": PC.HTTP_USER_AGENT})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)["entities"]
        except urllib.error.HTTPError as e:
            if attempt == 5 or e.code not in (429, 500, 502, 503, 504):
                raise
            wait = int(e.headers.get("Retry-After", "0") or 0) or 30 * (attempt + 1)
            print(f"wikidata HTTP {e.code}; waiting {wait} s")
            time.sleep(wait)
        except urllib.error.URLError:
            if attempt == 5:
                raise
            time.sleep(30 * (attempt + 1))


def _best_rank_values(claims, prop):
    """Item ids of a property's best-ranked statements (preferred if any, else normal; never deprecated),
    i.e. Wikidata's 'truthy' values."""
    st = [c for c in claims.get(prop, []) if c.get("rank") != "deprecated"
          and c.get("mainsnak", {}).get("snaktype") == "value"]
    pref = [c for c in st if c.get("rank") == "preferred"]
    return [c["mainsnak"]["datavalue"]["value"]["id"] for c in (pref or st)]


def fetch_countries(popqa_rows, chunk=50):
    cities = sorted({r["o_uri"].rstrip("/").rsplit("/", 1)[-1] for r in popqa_rows
                     if r["prop"] == "place of birth" and r.get("o_uri")})
    cities = [c for c in cities if c.startswith("Q") and c[1:].isdigit()]
    city_country = {}
    for i in range(0, len(cities), chunk):
        for qid, ent in _wbgetentities(cities[i:i + chunk], "claims").items():
            city_country[qid] = _best_rank_values(ent.get("claims", {}), "P17")
        time.sleep(1.0)
    countries = sorted({c for v in city_country.values() for c in v})
    names = {}
    for i in range(0, len(countries), chunk):
        for qid, ent in _wbgetentities(countries[i:i + chunk], "labels|aliases").items():
            names[qid] = (ent.get("labels", {}).get("en", {}).get("value", ""),
                          [a["value"] for a in ent.get("aliases", {}).get("en", [])])
        time.sleep(1.0)
    out = [{"city_qid": city, "country_qid": c, "country_label": names.get(c, ("", []))[0],
            "aliases": names.get(c, ("", []))[1]}
           for city, cs in sorted(city_country.items()) for c in cs if names.get(c, ("", []))[0]]
    print(f"wikidata: {len({r['city_qid'] for r in out})} of {len(cities)} cities have a country")
    write_jsonl(PC.DATA_DIR / PC.DATA_FILES["wikidata_country"], out)


def main():
    PC.DATA_DIR.mkdir(parents=True, exist_ok=True)
    old = json.loads(PC.MANIFEST_PATH.read_text()) if PC.MANIFEST_PATH.exists() else {}
    pop_path = PC.DATA_DIR / PC.DATA_FILES["popqa_full"]
    rec = old.get("datasets", {}).get("popqa_full", {})
    if pop_path.exists() and rec.get("sha256") == sha256_file(pop_path):
        print(f"{pop_path.name} already downloaded")
        popqa_rows = [json.loads(l) for l in open(pop_path, encoding="utf-8")]
    else:
        popqa_rows = download_popqa_full()
    ensure_triviaqa()
    wd_path = PC.DATA_DIR / PC.DATA_FILES["wikidata_country"]
    wd_rec = old.get("datasets", {}).get("wikidata_country", {})
    fetched = wd_rec.get("fetched_utc")
    if wd_path.exists() and wd_rec.get("sha256") == sha256_file(wd_path) and os.environ.get("PILOT_REFETCH_WIKIDATA") != "1":
        print(f"{wd_path.name} already fetched ({fetched}); set PILOT_REFETCH_WIKIDATA=1 to fetch again")
    else:
        fetch_countries(popqa_rows)
        fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    sources = {"popqa_full": PC.POPQA_PIN, "triviaqa": PC.TRIVIAQA_PIN,
               "wikidata_country": {"endpoint": PC.WIKIDATA_API, "action": "wbgetentities", "property": "P17",
                                    "rank": "best (preferred, else normal)"}}
    datasets = {}
    for key, fname in PC.DATA_FILES.items():
        p = PC.DATA_DIR / fname
        with open(p, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        datasets[key] = {"file": fname, "sha256": sha256_file(p), "n_rows": n, "source": sources[key]}
    datasets["wikidata_country"]["fetched_utc"] = fetched
    manifest = {"models": download_models(), "datasets": datasets}
    PC.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {PC.MANIFEST_PATH}")


if __name__ == "__main__":
    main()

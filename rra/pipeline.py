"""Run plan (spec section 10):

  A  environment, artifact loading (VES/ECS run, SQ2 run), question sets, isolation asserts, unit tests
  B  pool, per model: A2 for answered questions (A1's first token banned), strict grading; R (cross-fitted +
     full), shuffled-label check, Platt fits. Integrity checks 2-4 run here with the model loaded.
  C  fresh dev, per model: A1 + clean states, A2, clean D(A2), grading; VES/ECS dev and test A1 are
     teacher-forced again (R may choose a depth the VES/ECS run did not store; this also gives check 2)
  D  tuning (offline) on VES/ECS dev + fresh dev; comparator C; freeze (frozen_config.json, SHA-256 logged)
  E  fresh test, per model: A1 + clean states, A2, clean D(A2), grading
  F  bootstrap, verdict, report, audit file (rra.analysis)
  E2 optional: tries 1-10 on the fresh test (confident-error stratum); G optional: Gemma-2-9B-it end to end.
     Both run only after F has written the verdict; F is then re-run (the verdict cannot change).

Every stage writes its outputs atomically and is skipped when they exist, so re-running with the same
RRA_OUT_DIR resumes a crashed run. A failing required stage stops the run (the report says what is missing); a
failing optional stage is reported and the run continues.
"""
import contextlib
import os
import platform
import time
import traceback
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from rra import arms as AR
from rra import cfg as C
from rra import data as RD
from rra import probes as PR
from rra.budget import Budget
from rra.checks import Integrity
from rra.engine import RunStopped, free_memory, is_oom, load_engine, with_oom_retry
from scoring import A_, C_, auroc
from vesecs.detectors import MOMENTS, Probe, crossfit_family, fit_family, score_family
from vesecs.grader import grade_both
from vesecs.io import load_npz, read_json, save_npy, save_npz, sha256_file, sha256_json, write_json


class StageFailed(Exception):
    pass


class HardStop(Exception):
    pass


def depth_of(layer, n_layers, depths):
    for d in depths:
        if C.depth_layer(d, n_layers) == int(layer):
            return float(d)
    return None


class Pipeline:
    def __init__(self, out_dir=None):
        self.cfg = C.load_config()
        self.run = C.RunDir(self.cfg, out_dir)
        self.specs = C.model_specs(self.cfg)
        self.names = [s["name"] for s in self.specs]
        self.gspec = C.gemma_spec(self.cfg)
        self.budget = Budget(self.cfg, self.run)
        self.integ = Integrity(self.run.results / "integrity.json")
        self.missing = []
        self.done = set()
        self._eng = None
        self._shared = {}
        self._recs = {}
        self.seed = int(self.cfg["seed"])
        self.retries = int(self.cfg["checks"]["oom_retries"])
        self.ves = RD.ves_dir(self.cfg)
        self.fresh = None
        print(f"RUN: {self.run.root} models={self.names} smoke={self.cfg['smoke']} ves_run={self.ves} "
              f"sq2_run={RD.sq2_dir(self.cfg)}")

    # ================================================================ helpers
    def pdir(self, m, split="pool"):
        return self.run.split_dir(m, split)

    def vdir(self, m, vs):
        return self.ves / "runs" / m / vs

    def spec(self, m):
        if m == self.gspec["name"]:
            return self.gspec
        return next(s for s in self.specs if s["name"] == m)

    def stage(self, name, model, split, path, fn, optional=False):
        """Runs fn unless `path` exists (resume). Required stage failure -> StageFailed (the run stops)."""
        key = (name, model, split)
        path = Path(path)
        if path.exists():
            self.done.add(key)
            return True
        if self.budget.hard_stop():
            print(f"HARD_STOP: {name}[{model}/{split}] not started")
            self.missing.append(f"{name}[{model}/{split}] (hard stop)")
            raise HardStop(f"{name}[{model}/{split}]")
        t = time.time()
        try:
            fn()
            if not path.exists():
                raise RuntimeError(f"stage did not write {path}")
            self.done.add(key)
            return True
        except (RunStopped, HardStop):
            raise
        except Exception as e:
            traceback.print_exc()
            print(f"STAGE_FAILED: {name}[{model}/{split}] {type(e).__name__}: {e}")
            self.missing.append(f"{name}[{model}/{split}] (failed: {type(e).__name__}: {e})")
            if not optional:
                raise StageFailed(f"{name}[{model}/{split}]: {type(e).__name__}: {e}") from e
            return False
        finally:
            try:
                req, opt = self.rest()
                self.budget.log_stage(name, model, split, time.time() - t, req, opt)
                self.budget.dump()
            except Exception as e:                              # bookkeeping never stops the experiment
                print(f"BUDGET_ERROR: {type(e).__name__}: {e}")

    @contextlib.contextmanager
    def batch(self, eng, bs):
        """Run with the batch size of the run being reproduced (same batch composition = same numbers)."""
        old = eng.bs
        eng.bs = int(bs)
        try:
            yield
        finally:
            eng.bs = old

    def gpu(self, eng, unit, n, fn, name):
        """A timed GPU step with the memory-error policy; records the measured rate."""
        t = time.time()
        res, bs = with_oom_retry(eng, fn, name, self.retries)
        self.budget.record(eng.name, unit, n, time.time() - t)
        return res, bs

    def engine(self, m):
        if self._eng is not None and self._eng.name == m:
            return self._eng
        self.release()
        spec = self.spec(m)
        print(f"LOADING: {m} {spec['hf_id']}@{spec['revision'][:12]}")
        t = time.time()
        self._eng = load_engine(spec, self.cfg)
        self.budget.record(m, "load", 1, time.time() - t)
        if m not in self.budget.probe:
            try:
                self.timing_probe(self._eng)
            except RunStopped:
                raise
            except Exception as e:
                traceback.print_exc()
                print(f"BUDGET_ERROR: timing probe failed ({type(e).__name__}: {e}); completed stages give the rates")
        return self._eng

    def release(self):
        if self._eng is not None:
            self._eng.release()
            self._eng = None
        free_memory()

    def shared(self, m):
        if m not in self._shared:
            self._shared = {m: RD.load_sq2_shared(self.cfg, m)}      # one model's pool record in memory
        return self._shared[m]

    def pool_items(self):
        if not hasattr(self, "_pool"):
            self._pool = RD.load_pool(self.cfg)
        return self._pool

    def layers_of(self, m, n_layers=None):
        if n_layers is None:
            n_layers = int(self.spec(m)["n_layers"])
        return C.record_layers(self.cfg, n_layers)

    def D(self, m):
        """The frozen detector D of the VES/ECS run (not refitted); Gemma's own D in stage G."""
        if m == self.gspec["name"]:
            return Probe.from_npz(load_npz(self.pdir(m) / "D.npz"))
        return Probe.from_npz(load_npz(self.vdir(m, "pool") / "det_D.npz"))

    def R(self, m):
        return Probe.from_npz(load_npz(self.pdir(m) / "R.npz"))

    def register(self, m, split, what, sha):
        p = self.run.logs / "records_registry.json"
        r = read_json(p) if p.exists() else {}
        r[f"{m}/{split}/{what}"] = sha
        write_json(p, r, indent=1)

    # ---------------------------------------------------------------- timing probe
    def timing_probe(self, eng):
        n = min(int(self.cfg["budget"]["timing_probe_n"]), len(self.pool_items()))
        items = self.pool_items()[:n]
        P = eng.prompts([it["question"] for it in items])
        eng.generate(P[:min(8, n)])
        rates = {}
        t = time.time()
        g = eng.generate(P)
        rates["gen"] = n / (time.time() - t)
        conts = [eng.cont(ids, tr) for ids, tr in zip(g.token_ids, g.truncated)]
        t = time.time()
        eng.tf(P, conts, layers=[eng.layer(0.5)])
        rates["tf"] = n / (time.time() - t)
        if self.cfg["optional"]["E2"] and eng.name in self.names:
            k = min(int(self.cfg["decoding"]["tries_prompts_per_batch"]), n)
            nt = int(self.cfg["decoding"]["n_tries"])
            t = time.time()
            eng.tries(P[:k], [it["qid"] for it in items[:k]], nt, try_offset=1000)
            rates["samp"] = k * nt / (time.time() - t)
        rates = {k: round(float(v), 3) for k, v in rates.items()}
        write_json(self.run.logs / f"timing_probe_{eng.name}.json", {"n": n, "rates": rates,
                                                                     "load_s": eng.load_seconds}, indent=1)
        self.budget.set_probe(eng.name, rates)
        print(f"TIMING_PROBE: {eng.name} " + " ".join(f"{k}={v}/s" for k, v in rates.items()))

    # ---------------------------------------------------------------- remaining work (budget projection)
    def rest(self):
        """(required work not done, {optional stage: work}) as [(model, unit, n)] for the projection."""
        n_pool = len(self.pool_items())
        nd = len(self.fresh["dev_fresh"]) if self.fresh else 0
        nt = len(self.fresh["test_fresh"]) if self.fresh else 0
        req = []

        def todo(name, m, split):
            return (name, m, split) not in self.done

        for m in self.names:
            af = self.answered_frac(m)
            nv = sum(len(read_json(self.vdir(m, vs) / "drafts.json")["qids"]) for vs in ("dev", "test")) \
                if todo("ves_states", m, "test_ves") else 0
            for name, split, w in (("state_reproduction", "pool", [("tf", 64)]), ("pool_a2", "pool", [("gen", af * n_pool)]),
                                   ("ves_states", "test_ves", [("tf", nv)]), ("fallback_reproduction", "test_ves", [("gen", 64)]),
                                   ("a1", "dev_fresh", [("gen", nd)]), ("states", "dev_fresh", [("tf", nd)]),
                                   ("a2", "dev_fresh", [("gen", af * nd), ("tf", af * nd)]), ("fit_R", "pool", [("fit", 2)]),
                                   ("a1", "test_fresh", [("gen", nt), ("load", 1)]), ("states", "test_fresh", [("tf", nt)]),
                                   ("a2", "test_fresh", [("gen", af * nt), ("tf", af * nt)])):
                if todo(name, m, split):
                    req += [(m, u, k) for u, k in w]
        if todo("analysis", None, "all"):
            req.append(("all", "analysis", 1))
        opt = {}
        if self.cfg["optional"]["E2"]:
            w = []
            for m in self.names:
                if todo("e2_tries", m, "test_fresh"):
                    w += [(m, "load", 1), (m, "samp", nt * int(self.cfg["decoding"]["n_tries"]))]
            if w:
                opt["E2"] = w + [("all", "analysis", 1)]
        if self.cfg["optional"]["G"]:
            g = self.gspec["name"]
            af = np.mean([self.answered_frac(m) for m in self.names])
            w = []
            if todo("g_pool", g, "pool"):
                w += [(g, "load", 1), (g, "gen", n_pool * (1 + af)), (g, "tf", n_pool)]
            for sp, n in (("dev_fresh", nd), ("test_fresh", nt)):
                if todo("g_fresh", g, sp):
                    w += [(g, "gen", n * (1 + af)), (g, "tf", n * (1 + af))]
            if todo("g_fit", g, "pool"):
                w += [(g, "fit", 2)]
            if w:
                opt["G"] = w + [("all", "analysis", 1)]
        return [(m, u, k) for m, u, k in req], opt

    def answered_frac(self, m):
        p = self.pdir(m) / "labels.json"
        if p.exists():
            return float(np.mean(read_json(p)["answered"]))
        return 0.9

    # ================================================================ phase A
    def phase_a(self):
        print("==== PHASE A (setup)")
        vers = {}
        for pkg in ["torch", "transformers", "tokenizers", "numpy", "scipy", "scikit-learn", "pyyaml"]:
            try:
                vers[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                vers[pkg] = None
        vers["python"] = platform.python_version()
        vers["cuda"] = torch.version.cuda
        vers["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        write_json(self.run.logs / "versions.json", vers, indent=1)
        print("VERSIONS: " + " ".join(f"{k}={v}" for k, v in vers.items()))
        self.load_artifacts()
        qpath = self.run.root / "runs" / "questions.json"
        if not qpath.exists():
            fresh, info = RD.build_fresh(self.cfg)
            write_json(qpath, {"splits": fresh, "info": info})
        q = read_json(qpath)
        self.fresh = q["splits"]
        comp = q["info"]["composition"]
        print("QUESTIONS: " + " ".join(f"{s}={c['n']} {c['by_source']}" for s, c in comp.items()))
        if not self.integ.has("isolation"):
            ok, det = RD.isolation_report(self.cfg, self.fresh)
            self.integ.record("isolation", None, {"pass": bool(ok), **det})
            print(f"CHECK: isolation pass={ok}")
        if not self.integ.d["isolation"]["all"]["pass"]:
            raise RunStopped("isolation check failed; see results/integrity.json")
        if not self.integ.has("verdict_code") or not self.integ.has("budget_guard"):
            from rra.tests import run_all
            res = run_all(verbose=False)
            self.integ.record("verdict_code", None, {"pass": res["verdict_tests_pass"], **res})
            self.integ.record("budget_guard", None, {"pass": res["budget_tests_pass"], **res})
            print(f"CHECK: unit tests {res['n_pass']}/{res['n_tests']} passed; verdict tests pass="
                  f"{res['verdict_tests_pass']}; budget guard tests pass={res['budget_tests_pass']}")
            if res["failures"]:
                print("UNIT_TEST_FAILURES: " + "; ".join(res["failures"]))
        shap = self.run.logs / "model_sha256.json"
        if not shap.exists():
            from vesecs.pipeline import model_weight_hashes
            write_json(shap, model_weight_hashes(self.specs), indent=1)
            print(f"MODEL_SHA256: written to {shap}")

    def load_artifacts(self):
        """The VES/ECS artifacts reused unchanged; their A1 / A2 hashes go into the records registry (pairing)."""
        need = ["configs/frozen_config.json", "runs/questions.json"]
        for m in self.names:
            need += [f"runs/{m}/pool/{f}" for f in ("det_D.npz", "pool_D_oof.npy", "detectors.json")]
            need += [f"runs/{m}/{s}/{f}" for s in ("dev", "test") for f in ("drafts.json", "draft_states.npz", "search_B7.json")]
        miss = [p for p in need if not (self.ves / p).exists()]
        if miss:
            raise RunStopped(f"VES/ECS run folder {self.ves} lacks {miss}")
        self.ves_frozen = read_json(self.ves / "configs" / "frozen_config.json")
        info = {"ves_run": str(self.ves), "files": {}}
        for m in self.names:
            for f in ("det_D.npz", "pool_D_oof.npy"):
                info["files"][f"{m}/pool/{f}"] = sha256_file(self.vdir(m, "pool") / f)
            for vs, split in (("dev", "dev_ves"), ("test", "test_ves")):
                dr = read_json(self.vdir(m, vs) / "drafts.json")
                b7 = read_json(self.vdir(m, vs) / "search_B7.json")
                a1 = sha256_json({"texts": dr["texts"], "token_ids": dr["token_ids"]})
                a2 = sha256_json([[r[0]["text"], r[0]["token_ids"]] if r else None for r in b7])
                key = f"{m}/{split}"
                reg = self.run.logs / "records_registry.json"
                r = read_json(reg) if reg.exists() else {}
                if f"{key}/A1" not in r:
                    self.register(m, split, "A1", a1)
                    self.register(m, split, "A2", a2)
                info["files"][f"{m}/{vs}/drafts.json"] = sha256_file(self.vdir(m, vs) / "drafts.json")
                info["files"][f"{m}/{vs}/search_B7.json"] = sha256_file(self.vdir(m, vs) / "search_B7.json")
        write_json(self.run.logs / "ves_artifacts.json", info, indent=1)
        print(f"ARTIFACTS: VES/ECS run {self.ves}; frozen D and dev/test records of {self.names}")

    # ================================================================ phase B / C (per model)
    def model_block(self, m):
        print(f"==== PHASE B/C (pool, VES/ECS sets, fresh dev) {m}")
        self.pool_labels(m)
        self.check_state_reproduction(m)
        self.pool_a2(m)
        for split in C.VES:
            self.ves_states(m, split)
        self.check_frozen_D(m)
        self.check_fallback(m)
        self.fresh_a1(m, "dev_fresh")
        self.fresh_states(m, "dev_fresh")
        self.fresh_a2(m, "dev_fresh")
        self.release()
        self.fit_R(m)

    def pool_labels(self, m):
        p = self.pdir(m) / "labels.json"
        if p.exists():
            return read_json(p)
        sh = self.shared(m)
        pool = self.pool_items()
        g = [grade_both(t, it["aliases"]) for t, it in zip(sh["texts"], pool)]
        out = {"label_strict": [x["label_strict"] for x in g], "label_sq2": [x["label_sq2"] for x in g],
               "answered": [x["label_strict"] != A_ for x in g],
               "first_token": [int(t[0]) if len(t) else None for t in sh["token_ids"]]}
        vl = self.vdir(m, "pool") / "labels.json"
        if vl.exists():
            out["matches_ves_labels"] = read_json(vl)["label_strict"] == out["label_strict"]
        write_json(p, out)
        ls = np.array(out["label_strict"])
        print(f"POOL_LABELS: {m} n={len(ls)} right={np.mean(ls == C_):.3f} idk={np.mean(ls == A_):.3f} "
              f"matches VES/ECS strict labels={out.get('matches_ves_labels')}")
        return out

    def check_state_reproduction(self, m):
        path = self.pdir(m) / "check_state_reproduction.json"

        def fn():
            eng = self.engine(m)
            sh = self.shared(m)
            pool = self.pool_items()
            n = min(int(self.cfg["checks"]["state_verify_n"]), len(pool))
            B = int(self.cfg["checks"]["sq2_batch_size"])
            k = min(len(pool), max(n, B))                                    # SQ2's first batch
            P = eng.prompts([it["question"] for it in pool[:k]])
            conts = [list(c) for c in sh["conts"][:k]]
            layers = [int(x) for x in sh["record_layers"]]
            with self.batch(eng, B):
                o, bs = with_oom_retry(eng, lambda: eng.tf(P, conts, layers=layers), f"state_reproduction[{m}]",
                                       self.retries)
            per, mx = {}, 0.0
            for li, L in enumerate(layers):
                for nm, a, b in (("before", o["res_prompt"][:n, li], sh["res_prompt"][:n, li]),
                                 ("after", o["res_ans"][:n, li], sh["res_ans"][:n, li])):
                    a, b = a.float(), b.float()
                    d = float((a - b).abs().max())
                    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
                    per[f"{L}_{nm}"] = {"max_abs": d, "min_cos": float(cos.min())}
                    mx = max(mx, d)
            tol = float(self.cfg["checks"]["state_verify_tol"])
            res = {"pass": bool(mx < tol), "max_abs_diff": mx, "tol": tol, "n": n, "batch_size": bs,
                   "sq2_batch_composition": bs == B, "per_layer": per}
            self.integ.record("state_reproduction", m, res)
            write_json(path, res, indent=1)
            print(f"CHECK: state_reproduction {m} max abs diff {mx:.3g} (tol {tol}) pass={res['pass']}")
        self.stage("state_reproduction", m, "pool", path, fn)

    def pool_a2(self, m):
        path = self.pdir(m) / "a2.json"

        def fn():
            eng = self.engine(m)
            lab = self.pool_labels(m)
            sh = self.shared(m)
            pool = self.pool_items()
            ans = [i for i in range(len(pool)) if lab["answered"][i]]
            P = eng.prompts([pool[i]["question"] for i in ans])
            ids = [list(sh["token_ids"][i]) for i in ans]
            g, bs = self.gpu(eng, "gen", len(ans), lambda: eng.first_token_ban(P, ids), f"pool_a2[{m}]")
            rows = [None] * len(pool)
            for k, i in enumerate(ans):
                gr = grade_both(g.texts[k], pool[i]["aliases"])
                rows[i] = {"text": g.texts[k], "token_ids": list(g.token_ids[k]), "truncated": bool(g.truncated[k]),
                           "label_strict": gr["label_strict"], "label_sq2": gr["label_sq2"],
                           "null_reason": AR.null_reason(sh["texts"][i], g.texts[k])}
            write_json(path, {"rows": rows, "batch_size": bs})
            y2 = [r is not None and r["null_reason"] is None and r["label_strict"] == C_ for r in rows]
            print(f"POOL_A2: {m} answered={len(ans)} null={sum(r is not None and r['null_reason'] is not None for r in rows)} "
                  f"y2=1: {sum(y2)}")
        self.stage("pool_a2", m, "pool", path, fn)

    def ves_states(self, m, split):
        vs = "dev" if split == "dev_ves" else "test"
        path = self.pdir(m, split) / "states.npz"

        def fn():
            eng = self.engine(m)
            items = RD.ves_split_items(self.cfg, m, vs)
            dr = read_json(self.vdir(m, vs) / "drafts.json")
            P = eng.prompts([it["question"] for it in items])
            conts = [eng.cont(t, tr) for t, tr in zip(dr["token_ids"], dr["truncated"])]
            layers = self.layers_of(m, eng.n_layers)
            with self.batch(eng, self.cfg["checks"]["ves_batch_size"]):      # the VES/ECS run's batches
                o, bs = self.gpu(eng, "tf", len(P), lambda: eng.tf(P, conts, layers), f"ves_states[{m}/{split}]")
            s1 = score_family(self.D(m), o["res_prompt"], o["res_ans"], layers)
            stored = load_npz(self.vdir(m, vs) / "draft_states.npz")["sD"]
            save_npz(path, res_prompt=o["res_prompt"].numpy(), res_ans=o["res_ans"].numpy(), layers=np.array(layers),
                     s1=s1, s1_stored=stored, batch_size=np.array(bs))
            print(f"VES_STATES: {m} {split} n={len(P)} max |s1 - stored| = {np.max(np.abs(s1 - stored)):.3g}")
        self.stage("ves_states", m, split, path, fn)

    def check_frozen_D(self, m):
        path = self.pdir(m, "test_ves") / "check_frozen_D.json"

        def fn():
            tol = float(self.cfg["checks"]["frozen_D_tol"])
            res = {"tol": tol}
            for split in C.VES:
                z = load_npz(self.pdir(m, split) / "states.npz")
                d = np.abs(z["s1"] - z["s1_stored"])
                res[split] = {"max_abs_diff": float(d.max()) if len(d) else 0.0, "n": int(len(d)),
                              "batch_size": int(z["batch_size"])}
            res["pass"] = bool(res["test_ves"]["max_abs_diff"] < tol)
            self.integ.record("frozen_D", m, res)
            write_json(path, res, indent=1)
            print(f"CHECK: frozen_D {m} replay max |s1 - stored| {res['test_ves']['max_abs_diff']:.3g} pass={res['pass']}")
        self.stage("frozen_D", m, "test_ves", path, fn)

    def check_fallback(self, m):
        path = self.pdir(m, "test_ves") / "check_fallback.json"

        def fn():
            eng = self.engine(m)
            items = RD.ves_split_items(self.cfg, m, "test")
            dr = read_json(self.vdir(m, "test") / "drafts.json")
            b7 = read_json(self.vdir(m, "test") / "search_B7.json")
            ans = [i for i in range(len(items)) if not dr["is_idk"][i]]
            n = min(int(self.cfg["checks"]["fallback_verify_n"]), len(ans))
            B = int(self.cfg["checks"]["ves_batch_size"])
            first = ans[:min(len(ans), max(n, B))]                           # B7's first batch
            P = eng.prompts([items[i]["question"] for i in first])
            ids = [dr["token_ids"][i] for i in first]
            with self.batch(eng, B):
                g, bs = with_oom_retry(eng, lambda: eng.first_token_ban(P, ids), f"fallback_reproduction[{m}]",
                                       self.retries)
            diff = [i for j, i in enumerate(first[:n]) if list(g.token_ids[j]) != list(b7[i][0]["token_ids"])]
            rate = len(diff) / max(n, 1)
            mx = float(self.cfg["checks"]["fallback_max_diff_rate"])
            res = {"pass": bool(rate <= mx), "informational": True, "n": n, "n_differ": len(diff), "rate": rate,
                   "max_rate": mx, "batch_size": bs,
                   "examples": [{"qid": items[i]["qid"], "stored": b7[i][0]["text"],
                                 "regenerated": g.texts[first.index(i)]} for i in diff[:5]],
                   "policy": "stored A2 is used for replay and VES/ECS dev; regenerated A2 elsewhere"}
            self.integ.record("fallback_reproduction", m, res)
            write_json(path, res, indent=1)
            print(f"CHECK: fallback_reproduction {m} {len(diff)}/{n} differ (rate {rate:.3f}; reported if > {mx})")
        self.stage("fallback_reproduction", m, "test_ves", path, fn)

    # ---------------------------------------------------------------- fresh sets (also used by stage G)
    def fresh_a1(self, m, split, optional=False, name="a1"):
        path = self.pdir(m, split) / "a1.json"

        def fn():
            eng = self.engine(m)
            items = self.fresh[split]
            P = eng.prompts([it["question"] for it in items])
            g, bs = self.gpu(eng, "gen", len(P), lambda: eng.generate(P), f"a1[{m}/{split}]")
            gr = [grade_both(t, it["aliases"]) for t, it in zip(g.texts, items)]
            out = {"qids": [it["qid"] for it in items], "texts": g.texts, "token_ids": [list(t) for t in g.token_ids],
                   "truncated": [bool(x) for x in g.truncated], "label_strict": [x["label_strict"] for x in gr],
                   "label_sq2": [x["label_sq2"] for x in gr], "garbled": [x["garbled"] for x in gr],
                   "hedged": [x["hedged"] for x in gr], "is_idk": [x["label_strict"] == A_ for x in gr],
                   "first_token": [int(t[0]) if len(t) else None for t in g.token_ids], "batch_size": bs}
            out["sha256"] = sha256_json({"texts": out["texts"], "token_ids": out["token_ids"]})
            self.register(m, split, "A1", out["sha256"])
            write_json(path, out)
            ls = np.array(out["label_strict"])
            print(f"A1: {m} {split} n={len(ls)} right={np.mean(ls == C_):.3f} idk={np.mean(ls == A_):.3f}")
        return self.stage(name, m, split, path, fn, optional)

    def fresh_states(self, m, split, optional=False, name="states"):
        path = self.pdir(m, split) / "states.npz"

        def fn():
            eng = self.engine(m)
            items = self.fresh[split]
            a1 = read_json(self.pdir(m, split) / "a1.json")
            P = eng.prompts([it["question"] for it in items])
            conts = [eng.cont(t, tr) for t, tr in zip(a1["token_ids"], a1["truncated"])]
            layers = self.layers_of(m, eng.n_layers)
            o, bs = self.gpu(eng, "tf", len(P), lambda: eng.tf(P, conts, layers), f"states[{m}/{split}]")
            save_npz(path, res_prompt=o["res_prompt"].numpy(), res_ans=o["res_ans"].numpy(), layers=np.array(layers),
                     batch_size=np.array(bs))
        return self.stage(name, m, split, path, fn, optional)

    def fresh_a2(self, m, split, optional=False, name="a2", keep_states=False):
        """A2 for answered questions, strict grading, null check, then the clean teacher-forced pass of every
        non-null A2: D(A2) with the frozen D, or (stage G, D not fitted yet) the A2 states at all depths."""
        path = self.pdir(m, split) / "a2.json"

        def fn():
            eng = self.engine(m)
            items = self.fresh[split]
            a1 = read_json(self.pdir(m, split) / "a1.json")
            ans = [i for i in range(len(items)) if not a1["is_idk"][i]]
            P = eng.prompts([it["question"] for it in items])
            g, bs = self.gpu(eng, "gen", len(ans), lambda: eng.first_token_ban([P[i] for i in ans],
                                                                              [a1["token_ids"][i] for i in ans]),
                             f"a2[{m}/{split}]")
            rows = [None] * len(items)
            for k, i in enumerate(ans):
                gr = grade_both(g.texts[k], items[i]["aliases"])
                rows[i] = {"text": g.texts[k], "token_ids": list(g.token_ids[k]), "truncated": bool(g.truncated[k]),
                           "label_strict": gr["label_strict"], "label_sq2": gr["label_sq2"],
                           "null_reason": AR.null_reason(a1["texts"][i], g.texts[k]), "dA2": None}
            nn = [i for i in ans if rows[i]["null_reason"] is None]
            conts = [eng.cont(rows[i]["token_ids"], rows[i]["truncated"]) for i in nn]
            if keep_states:
                layers = self.layers_of(m, eng.n_layers)
                o, _ = self.gpu(eng, "tf", len(nn), lambda: eng.tf([P[i] for i in nn], conts, layers),
                                f"a2_states[{m}/{split}]")
                save_npz(self.pdir(m, split) / "a2_states.npz", res_prompt=o["res_prompt"].numpy(),
                         res_ans=o["res_ans"].numpy(), layers=np.array(layers), rows=np.array(nn, dtype=np.int64))
            else:
                D = self.D(m)
                o, _ = self.gpu(eng, "tf", len(nn), lambda: eng.tf([P[i] for i in nn], conts, [D.layer]),
                                f"dA2[{m}/{split}]")
                s = score_family(D, o["res_prompt"], o["res_ans"], [D.layer])
                for k, i in enumerate(nn):
                    rows[i]["dA2"] = float(s[k])
            sha = sha256_json([[r["text"], r["token_ids"]] if r else None for r in rows])
            self.register(m, split, "A2", sha)
            write_json(path, {"rows": rows, "batch_size": bs, "sha256": sha})
            y2 = sum(1 for r in rows if r and r["null_reason"] is None and r["label_strict"] == C_)
            print(f"A2: {m} {split} answered={len(ans)} non-null={len(nn)} right={y2}")
        return self.stage(name, m, split, path, fn, optional)

    # ---------------------------------------------------------------- R
    def fit_R(self, m):
        path = self.pdir(m) / "R.json"

        def fn():
            lab = self.pool_labels(m)
            rows = read_json(self.pdir(m) / "a2.json")["rows"]
            sh = self.shared(m)
            Rp, Ra, layers = sh["res_prompt"], sh["res_ans"], [int(x) for x in sh["record_layers"]]
            d_oof = np.load(self.vdir(m, "pool") / "pool_D_oof.npy")
            self._fit_R_common(m, lab, rows, Rp, Ra, layers, d_oof, shuffled=True)
        self.stage("fit_R", m, "pool", path, fn)

    def _fit_R_common(self, m, lab, rows, Rp, Ra, layers, d_oof, shuffled):
        pool = self.pool_items()
        answered = np.array(lab["answered"], bool)
        y1 = np.array(lab["label_strict"]) == C_
        y2 = np.array([r is not None and r["null_reason"] is None and r["label_strict"] == C_ for r in rows])
        folds = np.array([it["extra"]["fold"] for it in pool])
        t = time.time()
        probe, oof, infos = PR.fit_R(Rp, Ra, y2, answered, folds, layers)
        units = 1
        out = {"layers": layers, "n_train": int(answered.sum()), "n_pos": int(y2[answered].sum()),
               "selected": {**probe.info, "layer": probe.layer, "moment": probe.moment,
                            "depth": depth_of(probe.layer, int(self.spec(m)["n_layers"]), self.cfg["depths"])},
               "fold_choices": infos}
        ok = answered & np.isfinite(oof)
        out["crossfit_auroc"] = float(auroc(y2[ok], oof[ok]))
        if shuffled:
            a_sh = PR.shuffled_R_auroc(Rp, Ra, y2, answered, folds, layers, self.seed)
            units = 2
            lo, hi = self.cfg["checks"]["shuffled_range"]
            out["shuffled_auroc"] = a_sh
            self.integ.record("shuffled_labels_R", m, {"pass": bool(lo <= a_sh <= hi), "auroc": a_sh, "range": [lo, hi]})
        self.budget.record(m, "fit", units, time.time() - t)
        save_npz(self.pdir(m) / "R.npz", **probe.to_npz())
        save_npy(self.pdir(m) / "R_oof.npy", oof)
        okd = answered & np.isfinite(d_oof)
        p1 = PR.platt_fit(d_oof[okd], y1[okd])
        p2 = PR.platt_fit(oof[ok], y2[ok])
        out["platt"] = {"p1": p1, "p2": p2}
        out["calibration_pool_crossfit"] = {"p1": PR.calibration_curve(PR.platt_apply(p1, d_oof[okd]), y1[okd]),
                                            "p2": PR.calibration_curve(PR.platt_apply(p2, oof[ok]), y2[ok])}
        out["pool_auroc_y2_from_D_oof"] = float(auroc(y2[ok & okd], -d_oof[ok & okd])) if (ok & okd).any() else None
        out["R_sha256"] = sha256_file(self.pdir(m) / "R.npz")
        write_json(self.pdir(m) / "R.json", out, indent=1)
        print(f"R: {m} layer {probe.layer} ({out['selected']['depth']}) {probe.moment} lambda_rel {probe.lam_rel} "
              f"cross-fitted AUROC {out['crossfit_auroc']:.3f} (n={out['n_train']}, y2=1: {out['n_pos']})"
              + (f" | shuffled {out['shuffled_auroc']:.3f}" if shuffled else ""))
        return out

    def check_b3_replay(self):
        if self.integ.has("b3_replay"):
            return
        fzv = self.ves_frozen["models"]
        vals_new, vals_old = [], []
        per = {}
        for m in self.names:
            dr = read_json(self.vdir(m, "test") / "drafts.json")
            z = load_npz(self.pdir(m, "test_ves") / "states.npz")
            tau = float(fzv[m]["methods"]["B3"]["lam1"]["tau"])
            lab = np.array(dr["label_strict"])
            idk = np.array(dr["is_idk"], bool)
            for s, acc in ((z["s1"], vals_new), (z["s1_stored"], vals_old)):
                keep = idk | (np.isfinite(s) & (s >= tau))
                acc.append(AR.utility(np.where(keep, lab, A_), 1))
            per[m] = {"tau": tau, "U1_recomputed": vals_new[-1], "U1_stored_scores": vals_old[-1]}
        new, old = float(np.mean(vals_new)), float(np.mean(vals_old))
        summ = self.ves / "results" / "summary.json"
        exp, src = old, "B3 from the stored VES/ECS scores"
        if summ.exists():
            try:
                exp = float(read_json(summ)["main_table"]["B3"]["pooled"]["U1"])
                src = "VES/ECS results/summary.json"
            except Exception:
                pass
        tol = float(self.cfg["checks"]["b3_replay_tol"])
        res = {"pass": bool(abs(new - exp) <= tol), "U1_recomputed": new, "U1_expected": exp, "expected_source": src,
               "spec_value": float(self.cfg["checks"]["b3_replay_expected"]), "tol": tol, "per_model": per}
        self.integ.record("b3_replay", None, res)
        print(f"CHECK: b3_replay pooled U1 {new:.2f} vs {exp:.2f} ({src}) pass={res['pass']}")

    # ================================================================ records
    def recs(self, m, split, peer=True):
        key = (m, split, peer)
        if key in self._recs:
            return self._recs[key]
        r = self._recs_one(m, split)
        if peer and m in self.names and len(self.names) > 1:
            o = self._recs_one([x for x in self.names if x != m][0], split)
            pos = {q: i for i, q in enumerate(o.qids)}
            idx = [pos.get(q) for q in r.qids]
            if all(i is not None for i in idx):
                r.set_peer([o.a1_text[i] for i in idx], o.a1_lab[idx], o.a1_lab2[idx], o.a1_idk[idx], o.s1[idx])
        self._recs[key] = r
        return r

    def _recs_one(self, m, split):
        k1 = (m, split, "one")
        if k1 in self._recs:
            return self._recs[k1]
        if split in C.VES:
            vs = "dev" if split == "dev_ves" else "test"
            items = RD.ves_split_items(self.cfg, m, vs)
            dr = read_json(self.vdir(m, vs) / "drafts.json")
            b7 = read_json(self.vdir(m, vs) / "search_B7.json")
            a2 = [(r[0] if r else None) for r in b7]
            a2_rows = []
            for i, x in enumerate(a2):
                if x is None or dr["is_idk"][i]:
                    a2_rows.append(None)
                    continue
                reason = AR.null_reason(dr["texts"][i], x["text"])
                a2_rows.append({"text": x["text"], "label_strict": x["label_strict"], "label_sq2": x["label_sq2"],
                                "null_reason": reason, "dA2": x.get("sD_clean") if reason is None else None})
        else:
            items = self.fresh[split]
            dr = read_json(self.pdir(m, split) / "a1.json")
            a2_rows = read_json(self.pdir(m, split) / "a2.json")["rows"]
            if m == self.gspec["name"]:
                z2 = load_npz(self.pdir(m, split) / "a2_states.npz")
                if len(z2["rows"]):
                    s = score_family(self.D(m), z2["res_prompt"], z2["res_ans"], [int(x) for x in z2["layers"]])
                    for k, i in enumerate(z2["rows"]):
                        a2_rows[int(i)]["dA2"] = float(s[k])
        z = load_npz(self.pdir(m, split) / "states.npz")
        layers = [int(x) for x in z["layers"]]
        s1 = score_family(self.D(m), z["res_prompt"], z["res_ans"], layers)
        s2 = score_family(self.R(m), z["res_prompt"], z["res_ans"], layers)
        N = len(items)
        g = lambda r, k, d: d if r is None else r[k]
        r = AR.Recs([it["qid"] for it in items], items, dr["texts"], dr["label_strict"], dr["label_sq2"], dr["is_idk"],
                    s1, s2, [g(x, "text", None) for x in a2_rows], [g(x, "label_strict", A_) for x in a2_rows],
                    [g(x, "label_sq2", A_) for x in a2_rows],
                    [x is None or x["null_reason"] is not None for x in a2_rows],
                    [("not_generated" if x is None else x["null_reason"]) for x in a2_rows],
                    [np.nan if (x is None or x.get("dA2") is None) else x["dA2"] for x in a2_rows], dr["first_token"])
        assert r.N == N and len(r.a2_text) == N
        pl = read_json(self.pdir(m) / "R.json")["platt"]
        r.set_platt(PR.platt_apply(pl["p1"], s1), PR.platt_apply(pl["p2"], s2))
        self._recs[k1] = r
        return r

    def tuning_recs(self, m):
        return AR.Recs.concat([self.recs(m, s) for s in C.TUNING])

    # ================================================================ phase D
    def phase_d(self):
        print("==== PHASE D (tuning, freeze)")
        if self.run.frozen_path.exists():
            fz = read_json(self.run.frozen_path)
        else:
            lams, step = self.cfg["lambdas"], float(self.cfg["thresholds"]["quantile_step"])
            fz = {"models": {}, "ves_run": str(self.ves), "sq2_run": str(RD.sq2_dir(self.cfg))}
            for m in self.names:
                r = self.tuning_recs(m)
                T = AR.tune(r, lams, step)
                Rj = read_json(self.pdir(m) / "R.json")
                fz["models"][m] = {
                    "arms": T, "n_tune": r.N, "n_tune_answered": int(r.answered.sum()),
                    "R": {"file": str(self.pdir(m) / "R.npz"), "sha256": sha256_file(self.pdir(m) / "R.npz"),
                          "selected": Rj["selected"], "crossfit_auroc": Rj["crossfit_auroc"]},
                    "D": {"file": str(self.vdir(m, "pool") / "det_D.npz"),
                          "sha256": sha256_file(self.vdir(m, "pool") / "det_D.npz")},
                    "platt": Rj["platt"],
                    "ves_B3_tau": {k: v["tau"] for k, v in self.ves_frozen["models"][m]["methods"]["B3"].items()}}
                print(f"TUNING: {m} U_tune(lam1): " + " ".join(f"{a}={v['lam1']['U_tune']:.1f}" for a, v in T.items()))
            fz["comparator"] = {}
            for lam in lams:
                k = f"lam{lam}"
                u = {a: float(np.mean([fz["models"][m]["arms"][a][k]["U_tune"] for m in self.names]))
                     for a in ("R-band", "R-check")}
                choice = "R-band" if u["R-band"] > u["R-check"] + 1e-12 else "R-check"   # exact tie: R-check
                fz["comparator"][k] = {"arm": choice, "U_tune_pooled": u}
            print(f"COMPARATOR: {fz['comparator']['lam1']}")
            write_json(self.run.frozen_path, fz, indent=1)
        sha = sha256_file(self.run.frozen_path)
        write_json(self.run.logs / "frozen_config_sha256.json", {"sha256": sha}, indent=1)
        print(f"FROZEN_CONFIG: {self.run.frozen_path} sha256={sha}")
        self.pairing_check(list(C.TUNING) + ["test_ves"])
        return fz

    def pairing_check(self, splits):
        reg = self.run.logs / "records_registry.json"
        r = read_json(reg) if reg.exists() else {}
        for m in self.names:
            prev = self.integ.d.get("pairing", {}).get(m, {})
            res = dict(prev)
            ok = prev.get("pass", True)
            for split in splits:
                if split in C.VES:
                    vs = "dev" if split == "dev_ves" else "test"
                    dr = read_json(self.vdir(m, vs) / "drafts.json")
                    b7 = read_json(self.vdir(m, vs) / "search_B7.json")
                    h = {"A1": sha256_json({"texts": dr["texts"], "token_ids": dr["token_ids"]}),
                         "A2": sha256_json([[x[0]["text"], x[0]["token_ids"]] if x else None for x in b7])}
                else:
                    p1, p2 = self.pdir(m, split) / "a1.json", self.pdir(m, split) / "a2.json"
                    if not (p1.exists() and p2.exists()):
                        continue
                    a1, a2 = read_json(p1), read_json(p2)
                    h = {"A1": sha256_json({"texts": a1["texts"], "token_ids": a1["token_ids"]}),
                         "A2": sha256_json([[x["text"], x["token_ids"]] if x else None for x in a2["rows"]])}
                good = all(h[w] == r.get(f"{m}/{split}/{w}") for w in ("A1", "A2"))
                res[split] = {"matches_registry": bool(good)}
                ok = ok and good
            res["pass"] = bool(ok)
            self.integ.record("pairing", m, res)

    # ================================================================ phase E
    def test_gate(self):
        if not self.run.frozen_path.exists():
            return False, "configs/frozen_config.json does not exist"
        ok, bad = self.integ.status(self.names)
        if ok:
            return True, ""
        if os.environ.get("RRA_ALLOW_FAILED_CHECKS") == "1":
            print("WARNING: integrity checks failed but RRA_ALLOW_FAILED_CHECKS=1: " + "; ".join(bad))
            self.missing.append("integrity checks failed; fresh-test phase forced by RRA_ALLOW_FAILED_CHECKS=1: "
                                + "; ".join(bad))
            return True, ""
        return False, ("integrity checks did not pass: " + "; ".join(bad) + " (results/integrity.json; "
                       "RRA_ALLOW_FAILED_CHECKS=1 overrides and the report then says so)")

    def phase_e(self, m, fz):
        print(f"==== PHASE E (fresh test) {m}")
        for what, p in (("R", self.pdir(m) / "R.npz"), ("D", self.vdir(m, "pool") / "det_D.npz")):
            want, got = fz["models"][m][what]["sha256"], sha256_file(p)
            if got != want:
                raise RunStopped(f"{what} of {m} differs from the frozen configuration (sha256 {got[:12]} != {want[:12]})")
        self.fresh_a1(m, "test_fresh")
        self.fresh_states(m, "test_fresh")
        self.fresh_a2(m, "test_fresh")
        self.release()

    # ================================================================ phase F + optional stages
    def analysis(self, final=False):
        from rra.analysis import analyze
        t = time.time()
        res = analyze(self)
        self.budget.record("all", "analysis", 1, time.time() - t)
        self.done.add(("analysis", None, "all"))
        self.budget.dump()
        return res

    def optional_stages(self):
        _, opt = self.rest()
        if "E2" in opt:
            if self.budget.decide("E2", [], opt):
                print("==== OPTIONAL STAGE E2 (tries 1-10 on the fresh test)")
                for m in self.names:
                    self.e2_tries(m)
                    self.release()
            else:
                self.missing.append("optional stage E2 cut by the budget guard (logs/timing.jsonl)")
        _, opt = self.rest()
        if "G" in opt:
            if self.budget.decide("G", [], opt):
                print("==== OPTIONAL STAGE G (Gemma-2-9B-it end to end)")
                self.stage_g()
            else:
                self.missing.append("optional stage G cut by the budget guard (logs/timing.jsonl)")

    def e2_tries(self, m):
        path = self.pdir(m, "test_fresh") / "tries.json"

        def fn():
            eng = self.engine(m)
            items = self.fresh["test_fresh"]
            P = eng.prompts([it["question"] for it in items])
            qids = [it["qid"] for it in items]
            n = int(self.cfg["decoding"]["n_tries"])
            dec = self.cfg["decoding"]
            k0 = int(dec["tries_prompts_per_batch"])
            t = time.time()
            try:
                for attempt in range(self.retries + 1):
                    try:
                        out = eng.tries(P, qids, n, try_offset=1)
                        break
                    except Exception as e:
                        if not is_oom(e) or attempt == self.retries:
                            raise
                        free_memory()
                        dec["tries_prompts_per_batch"] = max(1, int(dec["tries_prompts_per_batch"]) // 2)
                        print(f"OOM_RETRY: e2_tries[{m}] prompts per batch -> {dec['tries_prompts_per_batch']}")
            finally:
                dec["tries_prompts_per_batch"] = k0
            self.budget.record(m, "samp", len(P) * n, time.time() - t)
            rec = [[{"text": tx, "label_strict": grade_both(tx, it["aliases"])["label_strict"]} for tx, _, _ in row]
                   for row, it in zip(out, items)]
            write_json(path, rec)
            print(f"E2: {m} {len(rec)} questions x {n} tries")
        self.stage("e2_tries", m, "test_fresh", path, fn, optional=True)

    # ---------------------------------------------------------------- stage G
    def stage_g(self):
        g = self.gspec["name"]
        try:
            ok = self.g_pool(g)
            for split in C.FRESH:
                ok = ok and self.fresh_a1(g, split, optional=True, name="g_fresh_a1")
                ok = ok and self.fresh_states(g, split, optional=True, name="g_fresh_states")
                ok = ok and self.fresh_a2(g, split, optional=True, name="g_fresh", keep_states=True)
            self.release()
            if ok:
                ok = self.g_fit(g)
            if ok:
                from rra.analysis import generality
                p = self.run.results / "generality.json"
                if not p.exists():
                    write_json(p, generality(self, g), indent=1)
                    print(f"GENERALITY: {p}")
        except HardStop:
            self.missing.append("optional stage G stopped at the hard stop")
        except RunStopped as e:
            print(f"STAGE_FAILED: G stopped: {e}")
            self.missing.append(f"optional stage G: {e}")
        finally:
            self.release()

    def g_pool(self, g):
        """Gemma on the pool: A1, strict labels, states at the SQ2 depths, A2 for answered questions."""
        pdir = self.pdir(g)
        path = pdir / "a2.json"

        def fn():
            eng = self.engine(g)
            pool = self.pool_items()
            P = eng.prompts([it["question"] for it in pool])
            if not (pdir / "a1.json").exists():
                a, _ = self.gpu(eng, "gen", len(P), lambda: eng.generate(P), f"g_pool_a1[{g}]")
                gr = [grade_both(t, it["aliases"]) for t, it in zip(a.texts, pool)]
                write_json(pdir / "a1.json", {"texts": a.texts, "token_ids": [list(t) for t in a.token_ids],
                                              "truncated": [bool(x) for x in a.truncated]})
                write_json(pdir / "labels.json", {"label_strict": [x["label_strict"] for x in gr],
                                                  "label_sq2": [x["label_sq2"] for x in gr],
                                                  "answered": [x["label_strict"] != A_ for x in gr]})
            a1 = read_json(pdir / "a1.json")
            lab = read_json(pdir / "labels.json")
            if not (pdir / "states.npz").exists():
                conts = [eng.cont(t, tr) for t, tr in zip(a1["token_ids"], a1["truncated"])]
                layers = self.layers_of(g, eng.n_layers)
                o, _ = self.gpu(eng, "tf", len(P), lambda: eng.tf(P, conts, layers), f"g_pool_states[{g}]")
                save_npz(pdir / "states.npz", res_prompt=o["res_prompt"].numpy(), res_ans=o["res_ans"].numpy(),
                         layers=np.array(layers))
            ans = [i for i in range(len(pool)) if lab["answered"][i]]
            a, bs = self.gpu(eng, "gen", len(ans), lambda: eng.first_token_ban([P[i] for i in ans],
                                                                              [a1["token_ids"][i] for i in ans]),
                             f"g_pool_a2[{g}]")
            rows = [None] * len(pool)
            for k, i in enumerate(ans):
                gr = grade_both(a.texts[k], pool[i]["aliases"])
                rows[i] = {"text": a.texts[k], "token_ids": list(a.token_ids[k]), "truncated": bool(a.truncated[k]),
                           "label_strict": gr["label_strict"], "label_sq2": gr["label_sq2"],
                           "null_reason": AR.null_reason(a1["texts"][i], a.texts[k])}
            write_json(path, {"rows": rows, "batch_size": bs})
            ls = np.array(lab["label_strict"])
            print(f"G_POOL: {g} n={len(ls)} right={np.mean(ls == C_):.3f} idk={np.mean(ls == A_):.3f}")
        return self.stage("g_pool", g, "pool", path, fn, optional=True)

    def g_fit(self, g):
        """Gemma's own D (SQ2 recipe on its pool states and strict labels; cross-fitted for Platt) and R."""
        pdir = self.pdir(g)
        path = pdir / "R.json"

        def fn():
            lab = read_json(pdir / "labels.json")
            rows = read_json(pdir / "a2.json")["rows"]
            z = load_npz(pdir / "states.npz")
            Rp, Ra = torch.from_numpy(z["res_prompt"]), torch.from_numpy(z["res_ans"])
            layers = [int(x) for x in z["layers"]]
            ls = np.array(lab["label_strict"])
            y, pop = ls == C_, ls != A_
            folds = np.array([it["extra"]["fold"] for it in self.pool_items()])
            t = time.time()
            fam = fit_family(Rp, Ra, y, np.flatnonzero(pop), layers, restrict={"D": MOMENTS})
            oof, _ = crossfit_family(Rp, Ra, y, pop, folds, layers, restrict={"D": MOMENTS})
            self.budget.record(g, "fit", 1, time.time() - t)
            save_npz(pdir / "D.npz", **fam["D"].to_npz())
            save_npy(pdir / "D_oof.npy", oof["D"])
            ok = pop & np.isfinite(oof["D"])
            write_json(pdir / "D.json", {"selected": {**fam["D"].info, "layer": fam["D"].layer, "moment": fam["D"].moment},
                                         "crossfit_auroc": float(auroc(y[ok], oof["D"][ok]))}, indent=1)
            self._fit_R_common(g, lab, rows, Rp, Ra, layers, oof["D"], shuffled=False)
        return self.stage("g_fit", g, "pool", path, fn, optional=True)

    # ================================================================ main
    def main(self):
        status = 0
        try:
            self.phase_a()
            for m in self.names:
                self.model_block(m)
            self.check_b3_replay()
            fz = self.phase_d()
            go, why = self.test_gate()
            if not go:
                print(f"TEST_PHASE_REFUSED: {why}")
                self.missing.append(f"fresh-test phase refused: {why}")
            else:
                for m in self.names:
                    self.phase_e(m, fz)
            self.pairing_check(list(C.TUNING) + ["test_ves", "test_fresh"])
            res = self.analysis()
            if go and res.get("verdicts", {}).get("RRA", {}).get("verdict") not in (None, "NOT_RUN"):
                v0 = res["verdicts"]["RRA"]["verdict"]
                self.optional_stages()
                res = self.analysis()
                if res["verdicts"]["RRA"]["verdict"] != v0:
                    raise RuntimeError("the verdict changed after the optional stages")
        except (RunStopped, StageFailed, HardStop) as e:
            kind = {RunStopped: "RUN_STOPPED", StageFailed: "RUN_STOPPED", HardStop: "HARD_STOP"}[type(e)]
            print(f"{kind}: {e}")
            self.missing.append(f"{kind}: {e}")
            status = 2 if isinstance(e, HardStop) else 1
            self.release()
            try:
                self.analysis()
            except Exception:
                traceback.print_exc()
                print("REPORT_FAILED")
        finally:
            self.budget.dump()
        print(f"DONE: {self.run.root} (status {status})")
        return status


def main():
    raise SystemExit(Pipeline().main())


if __name__ == "__main__":
    main()

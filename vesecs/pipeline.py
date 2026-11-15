"""Run plan (spec section 11): A (setup) -> B pool + C dev for model 1, then model 2 -> D freeze -> E test for
model 1, then model 2 -> F analysis (vesecs.analysis). Every stage writes its outputs atomically and is
skipped when they exist, so re-running with the same VES_OUT_DIR resumes a crashed run. A failing stage is
reported (STAGE_FAILED) and the stages that do not depend on it still run; the report marks what is missing.
"""
import gc
import os
import platform
import traceback
from collections import Counter
from importlib import metadata
from pathlib import Path

import numpy as np
import torch

from scoring import A_, C_
from vesecs import cfg as C
from vesecs import checks as CK
from vesecs import data as VD
from vesecs import methods as M
from vesecs.budget import Budget, vllm_available
from vesecs.detectors import MOMENTS, Probe, crossfit_family, crossfit_vector, fit_family, fit_vector, score_family
from vesecs.grader import grade_both
from vesecs.io import (load_npz, read_json, save_npy, save_npz, sha256_file, sha256_json, stable_int, write_json)
from vesecs.search import Scorer, b5_rounds, run_search, unit

DET_NAMES = ("D", "D_before", "D_after")


def vname(v):
    kind, beta = v
    return kind if beta is None else f"{kind}_b{float(beta):g}"


class SplitCtx:
    """What the search loop needs about one model on one split."""


class Pipeline:
    def __init__(self, out_dir=None):
        self.cfg = C.load_config()
        self.run = C.RunDir(self.cfg, out_dir)
        self.specs = C.model_specs(self.cfg)
        self.names = [s["name"] for s in self.specs]
        self.budget = Budget(self.cfg, self.run, len(self.specs))
        self.budget.models = self.names
        self.integ = CK.Integrity(self.run.results / "integrity.json")
        self.missing = []
        self._eng = None
        self._shared, self._scorers, self._pull = {}, {}, {}
        self.seed = int(self.cfg["seed"])
        print(f"RUN: {self.run.root} models={self.names} smoke={self.cfg['smoke']}")

    # ================================================================ helpers
    def stage(self, name, model, split, path, fn, load=None):
        path = Path(path)
        if path.exists():
            self.budget.done.add((name, model, split))
            return load(path) if load else (read_json(path) if path.suffix == ".json" else True)
        if self.budget.hard_stop():
            print(f"HARD_STOP: {name}[{model}/{split}] not started")
            self.budget.skip(name, model, split, "hard stop")
            self.missing.append(f"{name}[{model}/{split}] (hard stop)")
            return None
        self.budget.start(name, model, split)
        ok = False
        try:
            res = fn()
            ok = True
            return res
        except Exception as e:
            traceback.print_exc()
            print(f"STAGE_FAILED: {name}[{model}/{split}] {type(e).__name__}: {e}")
            self.missing.append(f"{name}[{model}/{split}] (failed: {type(e).__name__}: {e})")
            return None
        finally:
            self.budget_safe(self.budget.finish, name, model, split, ok)
            self.budget_safe(self.budget.dump)

    def budget_safe(self, fn, *args):
        """Budget bookkeeping must never stop the experiment."""
        try:
            return fn(*args)
        except Exception as e:
            print(f"BUDGET_ERROR: {fn.__name__}: {type(e).__name__}: {e}")
            return None

    def spec(self, m):
        return next(s for s in self.specs if s["name"] == m)

    def engine(self, m):
        if self._eng is not None and self._eng.name == m:
            return self._eng
        self.release()
        from vesecs.engine import Engine
        spec = self.spec(m)
        print(f"LOADING: {m} {spec['hf_id']}@{spec['revision'][:12]}")
        self._eng = Engine(spec, self.cfg)
        if m not in self.budget.rates:
            try:
                rates = self.timing_probe(self._eng)
            except Exception as e:
                traceback.print_exc()
                print(f"BUDGET_ERROR: timing probe failed ({type(e).__name__}: {e}); using SQ2's measured GB10 rates")
                rates = None
            if rates:
                self.budget_safe(self.budget.set_rates, m, rates)
        return self._eng

    def release(self):
        if self._eng is not None:
            self._eng.release()
            self._eng = None
        self._scorers.clear()
        self._pull.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def pdir(self, m, split="pool"):
        return self.run.split_dir(m, split)

    def shared(self, m):
        if m not in self._shared:
            self._shared = {m: VD.load_sq2_shared(self.cfg, m)}       # keep one model's pool record in memory
        return self._shared[m]

    def pool_items(self):
        if not hasattr(self, "_pool"):
            self._pool = VD.load_pool(self.cfg)
        return self._pool

    # ================================================================ phase A
    def phase_a(self):
        vers = {}
        for pkg in ["torch", "transformers", "tokenizers", "numpy", "scipy", "scikit-learn", "pyyaml", "vllm"]:
            try:
                vers[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                vers[pkg] = None
        vers["python"] = platform.python_version()
        vers["cuda"] = torch.version.cuda
        vers["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        write_json(self.run.logs / "versions.json", vers, indent=1)
        print("VERSIONS: " + " ".join(f"{k}={v}" for k, v in vers.items()))

        qpath = self.run.root / "runs" / "questions.json"
        if not qpath.exists():
            splits, info = VD.build_splits(self.cfg)
            write_json(qpath, {"splits": splits, "info": info})
        q = read_json(qpath)
        self.splits = q["splits"]
        self.budget.sizes = {"pool": len(self.pool_items()), "dev": len(self.splits["dev"]),
                             "test": len(self.splits["test"])}
        comp = q["info"]["composition"]
        print("QUESTIONS: " + " ".join(f"{s}={c['n']} {c['by_source']}" for s, c in comp.items()))
        if "isolation" not in self.integ.d:
            ok, det = VD.isolation_report(self.cfg, self.splits)
            self.integ.record("isolation", None, {"pass": bool(ok), **det})
            print(f"CHECK: isolation pass={ok}")
        if not self.integ.d["isolation"]["all"]["pass"]:
            raise SystemExit("isolation check failed; see results/integrity.json")

        if "verdict_code" not in self.integ.d:
            from vesecs.tests import run_all
            res = run_all(verbose=False)
            self.integ.record("verdict_code", None, {"pass": res["verdict_tests_pass"], **res})
            print(f"CHECK: unit tests {res['n_pass']}/{res['n_tests']} passed; verdict tests pass="
                  f"{res['verdict_tests_pass']}")
            if res["failures"]:
                print("UNIT_TEST_FAILURES: " + "; ".join(res["failures"]))
        shap = self.run.logs / "model_sha256.json"
        if not shap.exists():
            write_json(shap, model_weight_hashes(self.specs), indent=1)
            print(f"MODEL_SHA256: written to {shap}")

    def timing_probe(self, eng):
        """Phase A's timing probe: measured rates on this machine (pool questions)."""
        import time
        n = int(self.cfg["budget"]["timing_probe_n"])
        pool = self.pool_items()[:n]
        sh = self.shared(eng.name)
        P = eng.prompts([it["question"] for it in pool])
        conts = [list(c) for c in sh["conts"][:n]]
        eng.generate(P[:min(8, n)])
        rates = {}
        t = time.time()
        eng.generate(P)
        rates["gen"] = n / (time.time() - t)
        t = time.time()
        eng.tf(P, conts, layers=[eng.layer(0.6)])
        rates["tf"] = n / (time.time() - t)
        k = min(int(self.cfg["decoding"]["tries_prompts_per_batch"]), n)
        nt = int(self.cfg["decoding"]["n_tries"])
        t = time.time()
        eng.tries(P[:k], [it["qid"] for it in pool[:k]], nt, try_offset=1000)
        rates["samp"] = k * nt / (time.time() - t)
        kb = min(32, n)
        t = time.time()
        eng.pull_vectors(P[:kb], conts[:kb], [eng.layer(0.5)])
        rates["bwd"] = kb / (time.time() - t)
        ke = min(32, n)
        allowed = np.ones(eng.model.lm_head.out_features, bool)
        t = time.time()
        eng.echo(P[:ke], [c[0] if c else None for c in conts[:ke]], eng.layer(0.6), eng.layer(0.75), allowed, -1, 3, 1.0,
                 [("exact", None), ("1p", 1.0), ("placebo", 1.0)], placebo_token=lambda i, j: 100)
        rates["echo"] = ke / (time.time() - t)
        rates = {k: round(float(v), 3) for k, v in rates.items()}
        write_json(self.run.logs / f"timing_probe_{eng.name}.json", {"n": n, "rates": rates, "load_s": eng.load_seconds},
                   indent=1)
        print(f"TIMING_PROBE: {eng.name} " + " ".join(f"{k}={v}/s" for k, v in rates.items()))
        return rates

    # ================================================================ phase B (pool)
    def pool_labels(self, m):
        p = self.pdir(m) / "labels.json"
        if p.exists():
            return read_json(p)
        sh = self.shared(m)
        pool = self.pool_items()
        g = [grade_both(t, it["aliases"]) for t, it in zip(sh["texts"], pool)]
        out = {"label_strict": [x["label_strict"] for x in g], "label_sq2": [x["label_sq2"] for x in g],
               "label_sq2_stored": [int(s) for s in sh["states"]], "garbled": [x["garbled"] for x in g],
               "hedged": [x["hedged"] for x in g], "first_token": [int(t[0]) if len(t) else None for t in sh["token_ids"]]}
        ls, l2 = np.array(out["label_strict"]), np.array(out["label_sq2_stored"])
        out["n_changed_vs_sq2"] = int((ls != l2).sum())
        out["n_sq2_regrade_mismatch"] = int((np.array(out["label_sq2"]) != l2).sum())
        out["n_hedged"] = int(np.sum(out["hedged"]))
        write_json(p, out)
        print(f"POOL_LABELS: {m} strict vs SQ2 changed={out['n_changed_vs_sq2']} hedged={out['n_hedged']} "
              f"sq2 regrade mismatch={out['n_sq2_regrade_mismatch']}")
        return out

    def pool_states(self, m):
        """(Rp, Ra, layers): SQ2's stored states, or the recomputed ones if a needed depth was missing."""
        rp = self.pdir(m) / "pool_states.npz"
        if rp.exists():
            z = load_npz(rp)
            return torch.from_numpy(z["res_prompt"]), torch.from_numpy(z["res_ans"]), [int(x) for x in z["layers"]]
        sh = self.shared(m)
        return sh["res_prompt"], sh["res_ans"], [int(x) for x in sh["record_layers"]]

    def needed_pool_layers(self, n_layers):
        return sorted({C.depth_layer(d, n_layers) for d in self.cfg["depths"]["record"]})

    def b_verify(self, m):
        def fn():
            eng = self.engine(m)
            sh = self.shared(m)
            stored = [int(x) for x in sh["record_layers"]]
            need = self.needed_pool_layers(eng.n_layers)
            n = min(int(self.cfg["checks"]["state_verify_n"]), len(sh["conts"]))
            pool = self.pool_items()
            P = eng.prompts([it["question"] for it in pool])
            conts = [list(c) for c in sh["conts"]]
            o = eng.tf(P[:n], conts[:n], layers=stored)
            res = {"stored_layers": stored, "needed_layers": need, "n": n, "per_layer": {}}
            for li, L in enumerate(stored):
                for nm, a, b in (("before", o["res_prompt"][:, li], sh["res_prompt"][:n, li]),
                                 ("after", o["res_ans"][:, li], sh["res_ans"][:n, li])):
                    a, b = a.float(), b.float()
                    cos = torch.nn.functional.cosine_similarity(a, b, dim=1)
                    res["per_layer"][f"{L}_{nm}"] = {"max_abs": float((a - b).abs().max()), "min_cos": float(cos.min()),
                                                     "mean_cos": float(cos.mean())}
            missing = [L for L in need if L not in stored]
            res["recomputed"] = bool(missing)
            if missing:
                full = eng.tf(P, conts, layers=need)
                save_npz(self.pdir(m) / "pool_states.npz", res_prompt=full["res_prompt"].numpy(),
                         res_ans=full["res_ans"].numpy(), layers=np.array(need))
            write_json(self.pdir(m) / "verify.json", res, indent=1)
            print(f"VERIFY_STATES: {m} min cos " + f"{min(v['min_cos'] for v in res['per_layer'].values()):.5f}"
                  + (f"; recomputed layers {missing}" if missing else ""))
            return res
        return self.stage("verify", m, "pool", self.pdir(m) / "verify.json", fn)

    def b_detectors(self, m):
        def fn():
            lab = self.pool_labels(m)
            Rp, Ra, layers = self.pool_states(m)
            ls = np.array(lab["label_strict"])
            y, pop = ls == C_, ls != A_
            folds = np.array([it["extra"]["fold"] for it in self.pool_items()])
            fam = fit_family(Rp, Ra, y, np.flatnonzero(pop), layers)
            oof, infos = crossfit_family(Rp, Ra, y, pop, folds, layers)
            from scoring import auroc
            out = {"layers": layers, "n_train": int(pop.sum()), "selected": {}, "crossfit_auroc": {}, "fold_choices": infos}
            for k in DET_NAMES:
                save_npz(self.pdir(m) / f"det_{k}.npz", **fam[k].to_npz())
                out["selected"][k] = fam[k].info
                out["crossfit_auroc"][k] = auroc(y[pop], oof[k][pop])
            save_npy(self.pdir(m) / "pool_D_oof.npy", oof["D"])
            rng = np.random.default_rng(self.seed)
            ysh = y.copy()
            ysh[pop] = y[pop][rng.permutation(int(pop.sum()))]
            oofs, _ = crossfit_family(Rp, Ra, ysh, pop, folds, layers, restrict={"D": MOMENTS})
            a_sh = auroc(ysh[pop], oofs["D"][pop])
            lo, hi = self.cfg["checks"]["shuffled_range"]
            out["shuffled_D_auroc"] = a_sh
            self.integ.record("shuffled_labels_D", m, {"pass": bool(lo <= a_sh <= hi), "auroc": a_sh, "range": [lo, hi]})
            pr = VD.load_sq2_probe(self.cfg, m)
            if pr is not None:
                st = np.array(lab["label_sq2_stored"])
                ok = (st != A_) & np.isfinite(pr["oof"])
                out["sq2_stored_crossfit_auroc"] = auroc(st[ok] == C_, pr["oof"][ok])
                if (ls == st).all():
                    both = ok & np.isfinite(oof["D"])
                    out["max_abs_diff_vs_sq2_oof"] = float(np.abs(oof["D"][both] - pr["oof"][both]).max())
            write_json(self.pdir(m) / "detectors.json", out, indent=1)
            print(f"DETECTORS: {m} " + " ".join(f"{k}: layer {fam[k].layer} {fam[k].moment} auroc_cf "
                                                f"{out['crossfit_auroc'][k]:.3f}" for k in DET_NAMES)
                  + f" | SQ2 stored {out.get('sq2_stored_crossfit_auroc', float('nan')):.3f} | shuffled {a_sh:.3f}")
            return out
        return self.stage("detectors", m, "pool", self.pdir(m) / "detectors.json", fn)

    def detectors(self, m):
        info = read_json(self.pdir(m) / "detectors.json")
        return {k: Probe.from_npz(load_npz(self.pdir(m) / f"det_{k}.npz"), info["selected"].get(k)) for k in DET_NAMES}

    def b_ubar(self, m):
        path = self.pdir(m) / "ves_consts.npz"

        def fn():
            eng = self.engine(m)
            sh = self.shared(m)
            lab = np.array(self.pool_labels(m)["label_strict"])
            ans = np.flatnonzero(lab != A_)
            rng = np.random.default_rng(self.seed)
            pick = np.sort(rng.choice(ans, size=min(int(self.cfg["ves"]["n_ubar"]), len(ans)), replace=False))
            steer = sorted({eng.layer(d) for d in self.cfg["ves"]["steer_depths"]})
            pool = self.pool_items()
            P = eng.prompts([pool[i]["question"] for i in pick])
            g = eng.pull_vectors(P, [list(sh["conts"][i]) for i in pick], steer)
            Rp, Ra, layers = self.pool_states(m)
            ghat = np.stack([g[L] / np.maximum(np.linalg.norm(g[L], axis=1, keepdims=True), 1e-12) for L in steer], 1)
            ubar = np.stack([unit(ghat[:, k].mean(0)) for k in range(len(steer))])
            rho = np.array([float(np.median(Rp[:, li].float().norm(dim=1).numpy())) for li in range(len(layers))])
            mu = np.stack([Ra[torch.as_tensor(ans), layers.index(L)].float().mean(0).numpy() for L in steer])
            save_npz(path, steer_layers=np.array(steer), ubar=ubar, mu=mu, ghat=ghat.astype(np.float32),
                     rho_layers=np.array(layers), rho=rho, ubar_rows=pick)
            print(f"VES_CONSTS: {m} steer layers {steer} rho " + " ".join(f"{L}:{r:.1f}" for L, r in zip(layers, rho)))
            return True
        return self.stage("ubar", m, "pool", path, fn, load=lambda p: True)

    def ves_consts(self, m):
        z = load_npz(self.pdir(m) / "ves_consts.npz")
        steer = [int(x) for x in z["steer_layers"]]
        return {"steer": steer, "ubar": {L: z["ubar"][k] for k, L in enumerate(steer)},
                "mu": {L: z["mu"][k].astype(np.float64) for k, L in enumerate(steer)},
                "ghat": {L: z["ghat"][:, k] for k, L in enumerate(steer)},
                "rho": {int(L): float(r) for L, r in zip(z["rho_layers"], z["rho"])}}

    def b_lens(self, m):
        path = self.pdir(m) / "lens.json"

        def fn():
            eng = self.engine(m)
            sh = self.shared(m)
            lab = self.pool_labels(m)
            ls = np.array(lab["label_strict"])
            first = lab["first_token"]
            idk_first = [first[i] for i in range(len(ls)) if ls[i] == A_ and first[i] is not None]
            t_idk = int(Counter(idk_first).most_common(1)[0][0]) if idk_first else int(eng.fmt.idk_ids[0])
            allowed, dec = allowed_mask(eng, t_idk)
            save_npy(self.pdir(m) / "allowed.npy", allowed)
            Rp, _, layers = self.pool_states(m)
            ec = self.cfg["ecs"]
            has = np.array([f is not None for f in first])
            fa = np.array([f if f is not None else -1 for f in first])
            agree = {}
            for dep in ec["lens_depths"]:
                L = eng.layer(dep)
                top, _ = eng.lens(Rp[:, layers.index(L)], allowed, t_idk, int(ec["k"]))
                agree[str(dep)] = float(np.mean(top[has] == fa[has]))
            ok = [d for d in sorted(ec["lens_depths"]) if agree[str(d)] >= float(ec["agreement_min"])]
            a_dep = float(ok[0]) if ok else 0.75
            b_dep = 0.75 if a_dep < 0.75 else 0.90
            # placebo tokens: the most frequent alphabetic tokens (>= 2 letters) in the pool's answers and tries
            cnt = Counter()
            texts = list(sh["texts"]) + [t for row in sh["dis_texts"] for t in row] + [t for row in sh["con_texts"] for t in row]
            for ids in eng.tok(texts, add_special_tokens=False).input_ids:
                cnt.update(ids)
            alpha_ok = [i for i in range(len(dec)) if allowed[i] and dec[i].strip().isalpha() and len(dec[i].strip()) >= 2]
            alpha_set = set(alpha_ok)
            ranked = [t for t, _ in sorted(cnt.items(), key=lambda x: (-x[1], x[0])) if t in alpha_set]
            n_pl = int(ec["n_placebo_tokens"])
            ranked_set = set(ranked)
            placebo = ranked[:n_pl] + [t for t in alpha_ok if t not in ranked_set][:max(0, n_pl - len(ranked))]
            out = {"t_idk": t_idk, "t_idk_text": eng.tok.decode([t_idk]), "agreement": agree, "a_depth": a_dep,
                   "b_depth": b_dep, "a_layer": eng.layer(a_dep), "b_layer": eng.layer(b_dep),
                   "n_allowed": int(allowed.sum()), "placebo_tokens": placebo,
                   "placebo_examples": [eng.tok.decode([t]) for t in placebo[:20]]}
            write_json(path, out, indent=1)
            print(f"LENS: {m} agreement {agree} -> a={a_dep} (layer {out['a_layer']}), b={b_dep} (layer {out['b_layer']}); "
                  f"t_IDK={t_idk} {out['t_idk_text']!r}")
            return out
        return self.stage("lens", m, "pool", path, fn)

    def lens_info(self, m):
        info = read_json(self.pdir(m) / "lens.json")
        info["allowed"] = np.load(self.pdir(m) / "allowed.npy")
        return info

    # ---------------------------------------------------------------- echo (pool, dev, test)
    def echo_variants(self, split, beta_star=None):
        betas = self.cfg["ecs"]["betas"] if beta_star is None else [beta_star]
        v = [("exact", None)] + [("1p", float(b)) for b in betas]
        if "drop_placebos" not in self.budget.cuts:
            v += [("placebo", float(b)) for b in betas]
        return v

    def placebo_fn(self, m, qids, lens):
        toks = lens["placebo_tokens"]
        return lambda i, j: toks[stable_int(m, qids[i], "ecs_placebo", j) % len(toks)]

    def echo_stage(self, m, split, items, first, beta_star=None):
        d = self.pdir(m, split)
        path = d / "ecs_raw.json"

        def fn():
            eng = self.engine(m)
            lens = self.lens_info(m)
            consts = self.ves_consts(m)
            P = eng.prompts([it["question"] for it in items])
            qids = [it["qid"] for it in items]
            variants = self.echo_variants(split, beta_star)
            out_q, states = eng.echo(P, first, lens["a_layer"], lens["b_layer"], lens["allowed"], lens["t_idk"],
                                     int(self.cfg["ecs"]["k"]), consts["rho"][lens["a_layer"]], variants,
                                     placebo_token=self.placebo_fn(m, qids, lens))
            for v in variants:
                save_npy(d / f"echo_{vname(v)}.npy", states[v])
            for r, f in zip(out_q, first):
                r["first_token"] = f
            write_json(path, {"variants": [vname(v) for v in variants], "records": out_q})
            print(f"ECHO: {m} {split} n={len(out_q)} variants={[vname(v) for v in variants]} "
                  f"lens agreement={np.mean([r['top_a'] == f for r, f in zip(out_q, first) if f is not None]):.3f}")
            return read_json(path)
        return self.stage("echo", m, split, path, fn)

    def completions_stage(self, m, split, items, a1):
        """completion(q, c) for every candidate slot (c = f(A1) -> A1); dev/test also get clean s_D."""
        d = self.pdir(m, split)
        path = d / "ecs_completions.json"

        def fn():
            eng = self.engine(m)
            raw = read_json(d / "ecs_raw.json")["records"]
            P = eng.prompts([it["question"] for it in items])
            todo = [(i, j, c) for i, r in enumerate(raw) for j, c in enumerate(r["slots"]) if c != a1["first_token"][i]]
            g = eng.forced([P[i] for i, _, _ in todo], [c for _, _, c in todo])
            comp = [dict() for _ in raw]
            for n, (i, j, c) in enumerate(todo):
                gr = grade_both(g.texts[n], items[i]["aliases"])
                comp[i][str(j)] = {"text": g.texts[n], "token_ids": list(g.token_ids[n]), "truncated": bool(g.truncated[n]),
                                   "label_strict": gr["label_strict"], "label_sq2": gr["label_sq2"],
                                   "is_idk": gr["label_strict"] == A_, "sD": None, "is_A1": False}
            for i, r in enumerate(raw):
                for j, c in enumerate(r["slots"]):
                    if c == a1["first_token"][i]:
                        comp[i][str(j)] = {"text": a1["texts"][i], "token_ids": list(a1["token_ids"][i]),
                                           "truncated": bool(a1["truncated"][i]), "label_strict": a1["label_strict"][i],
                                           "label_sq2": a1["label_sq2"][i], "is_idk": a1["is_idk"][i],
                                           "sD": a1.get("sD", [None] * len(raw))[i], "is_A1": True}
            if split != "pool":
                ctx = self.ctx(m, split)
                sc = [(i, j) for (i, j, c) in todo if not comp[i][str(j)]["is_idk"]]
                conts = [eng.cont(comp[i][str(j)]["token_ids"], comp[i][str(j)]["truncated"]) for i, j in sc]
                res = ctx.scorer.score([i for i, _ in sc], [ctx.prompts[i] for i, _ in sc], conts)
                for (i, j), (s, _) in zip(sc, res):
                    comp[i][str(j)]["sD"] = s
            write_json(path, comp)
            n_c = sum(1 for (i, j, c) in todo if comp[i][str(j)]["label_strict"] == C_)
            print(f"COMPLETIONS: {m} {split} {len(todo)} forced completions, {n_c} correct")
            return comp
        return self.stage("completions", m, split, path, fn)

    def echo_rows(self, raw, comp):
        rows = []
        for i, r in enumerate(raw):
            for j, c in enumerate(r["slots"]):
                rows.append((i, j, int(comp[i][str(j)]["label_strict"] == C_)))
        return rows

    def b_echo_detectors(self, m):
        path = self.pdir(m) / "echo_detectors.json"

        def fn():
            d = self.pdir(m)
            raw = read_json(d / "ecs_raw.json")["records"]
            comp = read_json(d / "ecs_completions.json")
            rows = self.echo_rows(raw, comp)
            y = np.array([r[2] for r in rows], bool)
            lens = self.lens_info(m)
            out = {"n_rows": len(rows), "n_pos": int(y.sum()), "fits": {}}
            for v in [("exact", None)] + [("1p", float(b)) for b in self.cfg["ecs"]["betas"]]:
                X = np.load(d / f"echo_{vname(v)}.npy")
                p = fit_vector(X, y, np.arange(len(y)), lens["b_layer"])
                save_npz(d / f"det_E_{vname(v)}.npz", **p.to_npz())
                out["fits"][vname(v)] = p.info
            write_json(path, out, indent=1)
            print(f"ECHO_DETECTORS: {m} rows={len(rows)} pos={int(y.sum())} " +
                  " ".join(f"{k}: loo {v['loo_auroc_train']:.3f}" for k, v in out["fits"].items()))
            return out
        return self.stage("echo_detectors", m, "pool", path, fn)

    def echo_probe(self, m, name):
        d = self.pdir(m)
        return Probe.from_npz(load_npz(d / f"det_E_{name}.npz"))

    def phase_b(self, m):
        print(f"==== PHASE B (pool) {m}")
        self.pool_labels(m)
        self.b_verify(m)
        self.b_detectors(m)
        self.b_ubar(m)
        if self.b_lens(m) is None:
            return
        lab = self.pool_labels(m)
        sh = self.shared(m)
        pool = self.pool_items()
        a1 = {"first_token": lab["first_token"], "texts": list(sh["texts"]), "token_ids": [list(t) for t in sh["token_ids"]],
              "truncated": [bool(x) for x in sh["truncated"]], "label_strict": lab["label_strict"],
              "label_sq2": lab["label_sq2"], "is_idk": [x == A_ for x in lab["label_strict"]]}
        if self.echo_stage(m, "pool", pool, lab["first_token"]) is None:
            return
        if self.completions_stage(m, "pool", pool, a1) is None:
            return
        self.b_echo_detectors(m)

    # ================================================================ dev / test
    def split_items(self, m, split):
        items = self.splits[split]
        if split != "test":
            return items
        p = self.pdir(m, split) / "questions_active.json"
        if p.exists():
            keep = set(read_json(p))
        else:
            keep = set(VD.shrink_test(items, self.budget.n_test(), self.seed)) if "shrink_test" in self.budget.cuts \
                else {it["qid"] for it in items}
            write_json(p, sorted(keep))
        return [it for it in items if it["qid"] in keep]

    def drafts(self, m, split):
        return read_json(self.pdir(m, split) / "drafts.json")

    def c_drafts(self, m, split):
        d = self.pdir(m, split)
        path = d / "drafts.json"

        def fn():
            eng = self.engine(m)
            items = self.split_items(m, split)
            g = eng.generate(eng.prompts([it["question"] for it in items]))
            gr = [grade_both(t, it["aliases"]) for t, it in zip(g.texts, items)]
            out = {"qids": [it["qid"] for it in items], "texts": g.texts, "token_ids": [list(t) for t in g.token_ids],
                   "truncated": [bool(x) for x in g.truncated], "label_strict": [x["label_strict"] for x in gr],
                   "label_sq2": [x["label_sq2"] for x in gr], "garbled": [x["garbled"] for x in gr],
                   "hedged": [x["hedged"] for x in gr], "is_idk": [x["label_strict"] == A_ for x in gr],
                   "first_token": [int(t[0]) if len(t) else None for t in g.token_ids]}
            out["sha256"] = sha256_json({"texts": out["texts"], "token_ids": out["token_ids"]})
            reg = self.run.logs / "drafts_registry.json"
            r = read_json(reg) if reg.exists() else {}
            r[f"{m}/{split}"] = out["sha256"]
            write_json(reg, r, indent=1)
            write_json(path, out)
            ls = np.array(out["label_strict"])
            print(f"DRAFTS: {m} {split} n={len(ls)} right={np.mean(ls == C_):.3f} idk={np.mean(ls == A_):.3f}")
            return out
        return self.stage("drafts", m, split, path, fn)

    def c_states(self, m, split):
        d = self.pdir(m, split)
        path = d / "draft_states.npz"

        def fn():
            eng = self.engine(m)
            items = self.split_items(m, split)
            dr = self.drafts(m, split)
            dets = self.detectors(m)
            lens = read_json(self.pdir(m) / "lens.json")
            layers = sorted({p.layer for p in dets.values()} | {eng.layer(x) for x in self.cfg["ves"]["steer_depths"]}
                            | {lens["a_layer"]})
            P = eng.prompts([it["question"] for it in items])
            conts = [eng.cont(t, tr) for t, tr in zip(dr["token_ids"], dr["truncated"])]
            tfa = eng.tf(P, conts, layers)
            tfi = eng.tf(P, [eng.idk_cont] * len(P))
            n_tok = np.array([max(len(c), 1) for c in conts])
            sc = {k: score_family(dets[k], tfa["res_prompt"], tfa["res_ans"], layers) for k in DET_NAMES}
            save_npz(path, layers=np.array(layers), res_prompt=tfa["res_prompt"].numpy(), res_ans=tfa["res_ans"].numpy(),
                     lp_ans=tfa["lp"], lp_idk=tfi["lp"], margin=tfa["lp"] - tfi["lp"], mean_tok_lp=tfa["lp"] / n_tok,
                     sD=sc["D"], sD_before=sc["D_before"], sD_after=sc["D_after"])
            return True
        return self.stage("states", m, split, path, fn, load=lambda p: True)

    def scores(self, m, split):
        z = load_npz(self.pdir(m, split) / "draft_states.npz")
        return z

    def c_tries(self, m, split):
        d = self.pdir(m, split)
        path = d / "tries.json"

        def fn():
            eng = self.engine(m)
            items = self.split_items(m, split)
            P = eng.prompts([it["question"] for it in items])
            qids = [it["qid"] for it in items]
            n = int(self.cfg["decoding"]["n_tries"])
            out = None
            if "vllm_tries" in self.budget.cuts and vllm_available():
                try:
                    from vesecs.vllm_tries import vllm_tries
                    out = vllm_tries(self, eng, P, qids, n)
                    print(f"TRIES: {m} {split} via vLLM")
                except Exception as e:
                    traceback.print_exc()
                    print(f"VLLM_FALLBACK: {type(e).__name__}: {e}; using Hugging Face generation")
                    out = None
            if out is None:
                out = eng.tries(P, qids, n)
            rec = [[{"text": t, "token_ids": list(ids), "truncated": bool(tr), **grade_both(t, it["aliases"])}
                    for t, ids, tr in row] for row, it in zip(out, items)]
            write_json(path, rec)
            return rec
        return self.stage("tries", m, split, path, fn)

    def ctx(self, m, split):
        eng = self.engine(m)
        c = SplitCtx()
        c.cfg, c.model, c.split = self.cfg, m, split
        c.items = self.split_items(m, split)
        c.prompts = eng.prompts([it["question"] for it in c.items])
        dr = self.drafts(m, split)
        c.a1_text, c.a1_ids = dr["texts"], dr["token_ids"]
        c.a1_idk = np.array(dr["is_idk"], bool)
        c.a1_cont = [eng.cont(t, tr) for t, tr in zip(dr["token_ids"], dr["truncated"])]
        z = self.scores(m, split)
        layers = [int(x) for x in z["layers"]]
        consts = self.ves_consts(m)
        Ra = z["res_ans"]
        steer = [L for L in consts["steer"] if L in layers]
        c.a1_after = lambda i: {L: Ra[i, layers.index(L)].astype(np.float32) for L in steer}
        c.D = self.detectors(m)["D"]
        key = (m, split)
        if key not in self._scorers:
            self._scorers[key] = Scorer(eng, c.D, consts["steer"])
            self._pull[key] = {}
        c.scorer, c.pull_cache = self._scorers[key], self._pull[key]
        c.rho, c.ubar, c.mu, c.ghat_pool = consts["rho"], consts["ubar"], consts["mu"], consts["ghat"]
        c.n_backward = 0
        c.n_sel = int(self.cfg["decoding"]["n_select_tries"])
        tp = self.pdir(m, split) / "tries.json"
        c.tries = read_json(tp) if tp.exists() else None
        return c

    def c_b5(self, m, split):
        path = self.pdir(m, split) / "b5.json"

        def fn():
            c = self.ctx(m, split)
            out, todo = [None] * len(c.items), []
            for i in range(len(c.items)):
                if c.a1_idk[i]:
                    continue
                rounds, to_score = b5_rounds(c, i, c.scorer)
                out[i] = rounds
                todo += [(i, j) for j in to_score]
            conts = [self._eng.cont(out[i][j]["token_ids"], out[i][j]["truncated"]) for i, j in todo]
            res = c.scorer.score([i for i, _ in todo], [c.prompts[i] for i, _ in todo], conts)
            for (i, j), (s, _) in zip(todo, res):
                out[i][j]["sD_clean"] = s
            write_json(path, out)
            print(f"B5: {m} {split} scored {len(todo)} valid tries")
            return out
        return self.stage("b5", m, split, path, fn)

    def echo_scores_stage(self, m, split, beta_star=None):
        """E scores of every echo row. On dev it first picks beta (dev AUROC of E_1P(f(A1)) for A1's correctness),
        fits E_placebo at that beta, and runs the pool diagnostics (cross-fitted AUROC; shuffled-label check)."""
        d = self.pdir(m, split)
        path = d / "ecs.json"

        def fn():
            from scoring import auroc
            raw = read_json(d / "ecs_raw.json")
            recs, variants = raw["records"], raw["variants"]
            comp = read_json(d / "ecs_completions.json")
            dr = self.drafts(m, split)
            pdir = self.pdir(m)
            lens = self.lens_info(m)
            rows = [(i, j) for i, r in enumerate(recs) for j in range(len(r["slots"]))]
            Xs = {v: np.load(d / f"echo_{v}.npy") for v in variants}
            info = {}
            ans = ~np.array(dr["is_idk"], bool)
            yA1 = np.array(dr["label_strict"]) == C_
            jf = [r["slots"].index(r["first_token"]) if r["first_token"] in r["slots"] else None for r in recs]
            fidx = {(i, j): n for n, (i, j) in enumerate(rows)}

            def score(probe, v):
                return probe.score(vector=Xs[v])

            bstar = beta_star
            if split == "dev":
                au = {}
                for b in self.cfg["ecs"]["betas"]:
                    v = vname(("1p", float(b)))
                    s = score(self.echo_probe(m, v), v)
                    q = [i for i in range(len(recs)) if ans[i] and jf[i] is not None]
                    au[str(b)] = auroc(yA1[q], np.array([s[fidx[(i, jf[i])]] for i in q]))
                bstar = float(max(self.cfg["ecs"]["betas"], key=lambda b: (au[str(b)], -self.cfg["ecs"]["betas"].index(b))))
                info["beta_auroc_dev"], info["beta_star"] = au, bstar
                # pool: placebo detector at beta*, diagnostics
                praw = read_json(pdir / "ecs_raw.json")
                pcomp = read_json(pdir / "ecs_completions.json")
                prow = self.echo_rows(praw["records"], pcomp)
                py = np.array([r[2] for r in prow], bool)
                pfold = np.array([self.pool_items()[i]["extra"]["fold"] for i, _, _ in prow])
                allr = np.ones(len(py), bool)
                cf = {}
                names = ["exact", vname(("1p", bstar))]
                if "drop_placebos" not in self.budget.cuts and (pdir / f"echo_{vname(('placebo', bstar))}.npy").exists():
                    vp = vname(("placebo", bstar))
                    Xp = np.load(pdir / f"echo_{vp}.npy")
                    p = fit_vector(Xp, py, np.arange(len(py)), lens["b_layer"])
                    save_npz(pdir / "det_E_placebo.npz", **p.to_npz())
                    info["placebo_fit"] = p.info
                    names.append(vp)
                for v in names:
                    Xv = np.load(pdir / f"echo_{v}.npy")
                    oof = crossfit_vector(Xv, py, allr, pfold, lens["b_layer"])
                    cf[v] = auroc(py, oof)
                info["pool_crossfit_auroc"] = cf
                rng = np.random.default_rng(self.seed)
                ysh = py[rng.permutation(len(py))]
                X1 = np.load(pdir / f"echo_{vname(('1p', bstar))}.npy")
                a_sh = auroc(ysh, crossfit_vector(X1, ysh, allr, pfold, lens["b_layer"]))
                lo, hi = self.cfg["checks"]["shuffled_range"]
                self.integ.record("shuffled_labels_E1P", m, {"pass": bool(lo <= a_sh <= hi), "auroc": a_sh,
                                                            "beta": bstar, "range": [lo, hi]})
                info["shuffled_E1P_auroc"] = a_sh
                print(f"ECS_BETA: {m} dev AUROC by beta {au} -> beta*={bstar}; pool cross-fitted AUROC {cf}; "
                      f"shuffled {a_sh:.3f}")
            use = {"exact": ("exact", self.echo_probe(m, "exact")),
                   "1p": (vname(("1p", bstar)), self.echo_probe(m, vname(("1p", bstar))))}
            vpl = vname(("placebo", bstar))
            if vpl in variants and (pdir / "det_E_placebo.npz").exists():
                use["placebo"] = (vpl, Probe.from_npz(load_npz(pdir / "det_E_placebo.npz")))
            if split == "dev":
                for b in self.cfg["ecs"]["betas"]:
                    v = vname(("1p", float(b)))
                    use[v] = (v, self.echo_probe(m, v))
            out = []
            sc = {k: score(p, v) for k, (v, p) in use.items()}
            for i, r in enumerate(recs):
                E = {k: [float(sc[k][fidx[(i, j)]]) for j in range(len(r["slots"]))] for k in sc}
                out.append({**r, "E": E, "slot_A1": jf[i], "completion": comp[i]})
            write_json(path, {"beta_star": bstar, "variants_scored": {k: v for k, (v, _) in use.items()},
                              "info": info, "records": out})
            return read_json(path)
        return self.stage("echo_scores", m, split, path, fn)

    def ves_tag(self, method, depth=None, alpha=None):
        return method if depth is None else f"{method}_d{float(depth):g}_a{float(alpha):g}"

    def search_stage(self, m, split, stage_name, configs, method, pull_depths=None):
        """configs: [(depth, alpha)] (None for B7). Each config is written to search_<tag>.json; the stage marker
        lists them."""
        d = self.pdir(m, split)
        tag_all = stage_name if split == "dev" else f"{stage_name}_" + "_".join(self.ves_tag(method, *c) for c in configs)
        marker = d / f"stage_{tag_all}.json"

        def fn():
            eng = self.engine(m)
            c = self.ctx(m, split)
            pl = sorted({eng.layer(x) for x in (pull_depths or [])})
            done = []
            for dep, al in configs:
                tag = self.ves_tag(method, dep, al)
                f = d / f"search_{tag}.json"
                if not f.exists():
                    if self.budget.hard_stop():
                        print(f"HARD_STOP: search {tag} not started")
                        break
                    L = eng.layer(dep) if dep is not None else None
                    rec = run_search(eng, c, method, layer=L, alpha=al, pull_layers=pl or ([L] if L is not None else None))
                    write_json(f, rec)
                done.append(tag)
            if len(done) < len(configs):
                raise RuntimeError("hard stop inside the search stage")
            write_json(marker, {"configs": done, "n_backward": c.n_backward})
            return True
        return self.stage(stage_name, m, split, marker, fn, load=lambda p: True)

    def c_checks(self, m):
        path = self.pdir(m, "dev") / "checks.json"

        def fn():
            eng = self.engine(m)
            ch = self.cfg["checks"]
            c = self.ctx(m, "dev")
            dr = self.drafts(m, "dev")
            consts = self.ves_consts(m)
            Ls = eng.layer(self.cfg["ves"]["steer_depths"][0])
            res = {}
            res["zero_steering"] = CK.zero_steering(eng, c.prompts, dr, Ls, int(ch["zero_steer_n"]))
            z = self.scores(m, "dev")
            layers = [int(x) for x in z["layers"]]
            res["hook_reset"] = CK.hook_reset(eng, c.prompts, c.a1_cont, torch.from_numpy(z["res_prompt"]), layers,
                                              float(ch["hook_reset_tol"]))
            ans = [i for i in range(len(c.items)) if not c.a1_idk[i]][:int(ch["grad_n_questions"])]
            res["gradient"] = CK.gradient(eng, [c.prompts[i] for i in ans], [c.a1_cont[i] for i in ans], Ls,
                                          consts["rho"][Ls], float(ch["grad_eps_rel"]), float(ch["grad_tol"]), self.seed)
            lens = self.lens_info(m)
            n_e = int(ch["echo_equivalence_n"])
            res["identity_echo"] = CK.identity_echo(eng, c.prompts[:max(n_e, 4)], dr["first_token"][:max(n_e, 4)],
                                                    lens["a_layer"], lens["b_layer"], lens["allowed"], lens["t_idk"],
                                                    int(self.cfg["ecs"]["k"]), consts["rho"][lens["a_layer"]])
            res["echo_equivalence"] = CK.echo_equivalence(eng, c.prompts[:n_e], dr["first_token"][:n_e], lens["a_layer"],
                                                          lens["b_layer"], lens["allowed"], lens["t_idk"],
                                                          int(self.cfg["ecs"]["k"]), consts["rho"][lens["a_layer"]], 1.0,
                                                          float(ch["echo_equivalence_min_cos"]))
            for k, v in res.items():
                self.integ.record(k, m, v)
            write_json(path, res, indent=1)
            print("CHECKS: " + m + " " + " ".join(f"{k}={v['pass']}" for k, v in res.items()))
            return res
        return self.stage("checks", m, "dev", path, fn)

    def judges_stage(self, m, split):
        """R1 (5 outside clues) and R2 (clues + D) gates, fitted on the pool with strict labels (SQ2's logistic
        regression); R3 majority vote among tries 11-18."""
        path = self.pdir(m, split) / "judges.json"

        def fn():
            from pilot.metrics import consistency_features, majority
            from pilot.probes import fit_lr_score, output_features
            sh = self.shared(m)
            ls = np.array(self.pool_labels(m)["label_strict"])
            y, pop = ls == C_, ls != A_
            oof = np.load(self.pdir(m) / "pool_D_oof.npy")
            Fp = output_features(sh)
            dr = self.drafts(m, split)
            z = self.scores(m, split)
            tries = read_json(self.pdir(m, split) / "tries.json")
            n_sel = int(self.cfg["decoding"]["n_select_tries"])
            con = [[t["text"] for t in row[n_sel:]] for row in tries]
            agree, distinct, idk = consistency_features(dr["texts"], con)
            Fn = np.column_stack([z["margin"], z["mean_tok_lp"], agree, distinct, idk])
            r1 = fit_lr_score(Fp, y, pop, Fn)
            r2 = fit_lr_score(np.column_stack([Fp, oof]), y, pop & np.isfinite(oof), np.column_stack([Fn, z["sD"]]))
            items = self.split_items(m, split)
            r3t = [row[majority(row)] if row else "" for row in con]
            g = [grade_both(t, it["aliases"]) for t, it in zip(r3t, items)]
            out = {"R1": r1, "R2": r2, "R3": {"text": r3t, "label_strict": [x["label_strict"] for x in g],
                                              "label_sq2": [x["label_sq2"] for x in g]}}
            write_json(path, out)
            return out
        return self.stage("judges", m, split, path, fn)

    def phase_c(self, m):
        print(f"==== PHASE C (dev) {m}")
        if self.c_drafts(m, "dev") is None or self.c_states(m, "dev") is None:
            return
        items = self.split_items(m, "dev")
        dr = self.drafts(m, "dev")
        self.c_tries(m, "dev")
        if (self.pdir(m, "dev") / "tries.json").exists():
            self.c_b5(m, "dev")
        if (self.pdir(m) / "echo_detectors.json").exists():
            if self.echo_stage(m, "dev", items, dr["first_token"]) is not None:
                a1 = {**dr, "sD": list(self.scores(m, "dev")["sD"])}
                if self.completions_stage(m, "dev", items, a1) is not None:
                    self.echo_scores_stage(m, "dev")
        grid = self.budget.ves_grid()
        self.search_stage(m, "dev", "ves", grid, "VES", pull_depths=sorted({g[0] for g in grid}))
        vt = self.ves_tune(m)
        if vt is not None:
            dep, al = vt["lam1"]["depth"], vt["lam1"]["alpha"]
            self.search_stage(m, "dev", "ves_free", [(dep, a) for a in self.cfg["ves_free"]["alphas"]], "VES_free")
            if "drop_placebos" not in self.budget.cuts:
                self.search_stage(m, "dev", "ves_placebo", [(dep, al)], "VES_placebo")
        self.search_stage(m, "dev", "b7", [(None, None)], "B7")
        self.c_checks(m)
        if self.cfg["reference_rows"] and "drop_reference_rows" not in self.budget.cuts \
                and (self.pdir(m, "dev") / "tries.json").exists():
            self.judges_stage(m, "dev")
        self.tuning_stage(m)

    def ves_tune(self, m):
        """VES's dev choice of (depth, alpha, tau), needed before VES-free / VES-placebo run."""
        path = self.pdir(m, "dev") / "ves_tuned.json"
        if path.exists():
            return read_json(path)
        d = self.pdir(m, "dev")
        try:
            sr = self.records(m, "dev")
            cfgs = []
            for dep, al in self.budget.ves_grid():
                f = d / f"search_{self.ves_tag('VES', dep, al)}.json"
                if f.exists():
                    cfgs.append(({"depth": dep, "alpha": al}, read_json(f)))
            if not cfgs:
                return None
            t = M.tune_search(sr, cfgs, "VES", self.cfg["lambdas"], self.cfg["thresholds"]["quantile_step"])
            write_json(path, t, indent=1)
            print(f"VES_TUNED: {m} lam1 {t['lam1']}")
            return t
        except Exception as e:
            traceback.print_exc()
            print(f"STAGE_FAILED: ves_tune[{m}] {e}")
            return None

    # ---------------------------------------------------------------- records + tuning
    def records(self, m, split):
        d = self.pdir(m, split)
        items = self.split_items(m, split)
        dr = self.drafts(m, split)
        z = self.scores(m, split)
        lens = read_json(self.pdir(m) / "lens.json")
        ld = lambda name: read_json(d / name) if (d / name).exists() else None
        searches = {}
        for f in sorted(d.glob("search_*.json")):
            searches[f.stem[len("search_"):]] = read_json(f)
        ecs = ld("ecs.json")
        judges = ld("judges.json")
        if judges:
            judges = {"R1": np.asarray(judges["R1"], dtype=np.float64), "R2": np.asarray(judges["R2"], dtype=np.float64),
                      "R3": judges["R3"]}
        sr = M.SplitRecords(items, dr, {k: z[k] for k in ("sD", "sD_before", "sD_after", "margin")}, tries=ld("tries.json"),
                            b5=ld("b5.json"), searches=searches, ecs=ecs["records"] if ecs else None, judges=judges,
                            t_idk=lens["t_idk"], n_sel=int(self.cfg["decoding"]["n_select_tries"]))
        sr.ecs_beta = ecs["beta_star"] if ecs else None
        sr.ecs_variants = list(ecs["records"][0]["E"]) if ecs and ecs["records"] else []
        return sr

    def tuning_stage(self, m):
        path = self.pdir(m, "dev") / "tuning.json"

        def fn():
            sr = self.records(m, "dev")
            out = tune_all(self, sr, m)
            write_json(path, out, indent=1)
            print(f"TUNING: {m} dev U1: " + " ".join(f"{k}={v['lam1']['U_dev']:.1f}" for k, v in out["methods"].items()
                                                    if isinstance(v, dict) and "lam1" in v))
            return out
        return self.stage("tuning", m, "dev", path, fn)

    # ================================================================ phase D
    def phase_d(self):
        print("==== PHASE D (freeze)")
        if self.run.frozen_path.exists():
            fz = read_json(self.run.frozen_path)
        else:
            fz = {"models": {}}
            for m in self.names:
                p = self.pdir(m, "dev") / "tuning.json"
                if not p.exists():
                    print(f"FREEZE: no dev tuning for {m}")
                    continue
                t = read_json(p)
                t["lens"] = {k: v for k, v in read_json(self.pdir(m) / "lens.json").items()
                             if k in ("a_depth", "b_depth", "a_layer", "b_layer", "t_idk")}
                t["detectors"] = read_json(self.pdir(m) / "detectors.json")["selected"]
                fz["models"][m] = t
            fz["budget_cuts_at_freeze"] = list(self.budget.cuts)
            write_json(self.run.frozen_path, fz, indent=1)
        sha = sha256_file(self.run.frozen_path)
        write_json(self.run.logs / "frozen_config_sha256.json", {"sha256": sha}, indent=1)
        print(f"FROZEN_CONFIG: {self.run.frozen_path} sha256={sha}")
        self.pairing_check("dev")
        return fz

    def pairing_check(self, split):
        reg = self.run.logs / "drafts_registry.json"
        r = read_json(reg) if reg.exists() else {}
        for m in self.names:
            p = self.pdir(m, split) / "drafts.json"
            if not p.exists():
                continue
            dr = read_json(p)
            sha = sha256_json({"texts": dr["texts"], "token_ids": dr["token_ids"]})
            ok = sha == dr["sha256"] == r.get(f"{m}/{split}")
            prev = self.integ.d.get("pairing", {}).get(m, {})
            self.integ.record("pairing", m, {"pass": bool(ok and prev.get("pass", True)),
                                             **{**prev, split: {"sha256": sha, "matches_registry": bool(ok)}}})

    # ================================================================ phase E
    def test_gate(self):
        """The test phase starts only with frozen_config.json and every integrity check passed."""
        if not self.run.frozen_path.exists():
            return False, "configs/frozen_config.json does not exist"
        ok, bad = self.integ.status(self.names)
        if ok:
            return True, ""
        if os.environ.get("VES_ALLOW_FAILED_CHECKS") == "1":
            print("WARNING: integrity checks failed but VES_ALLOW_FAILED_CHECKS=1: " + "; ".join(bad))
            self.missing.append("integrity checks failed; test phase forced by VES_ALLOW_FAILED_CHECKS=1: " + "; ".join(bad))
            return True, ""
        return False, ("integrity checks did not pass: " + "; ".join(bad) + " (see results/integrity.json; "
                       "VES_ALLOW_FAILED_CHECKS=1 overrides and the report then says so)")

    def phase_e(self, m, fz):
        print(f"==== PHASE E (test) {m}")
        t = fz["models"].get(m)
        if t is None:
            print(f"PHASE E: no frozen config for {m}")
            return
        if self.c_drafts(m, "test") is None or self.c_states(m, "test") is None:
            return
        items = self.split_items(m, "test")
        dr = self.drafts(m, "test")
        self.c_tries(m, "test")
        if (self.pdir(m, "test") / "tries.json").exists():
            self.c_b5(m, "test")
        beta = t["methods"]["ECS_1p"]["beta"] if "ECS_1p" in t["methods"] else None
        if beta is not None and self.echo_stage(m, "test", items, dr["first_token"], beta_star=beta) is not None:
            a1 = {**dr, "sD": list(self.scores(m, "test")["sD"])}
            if self.completions_stage(m, "test", items, a1) is not None:
                self.echo_scores_stage(m, "test", beta_star=beta)
        mt = t["methods"]
        if "VES" in mt:
            ves_cfgs = sorted({(mt["VES"][k]["depth"], mt["VES"][k]["alpha"]) for k in ("lam1", "lam3")})
            self.search_stage(m, "test", "ves", ves_cfgs, "VES", pull_depths=sorted({c[0] for c in ves_cfgs}))
            dep = mt["VES"]["lam1"]["depth"]
            if "VES_free" in mt:
                self.search_stage(m, "test", "ves_free", sorted({(dep, mt["VES_free"][k]["alpha"]) for k in ("lam1", "lam3")}),
                                  "VES_free")
            if "VES_placebo" in mt and "drop_placebos" not in self.budget.cuts:
                self.search_stage(m, "test", "ves_placebo", [(dep, mt["VES"]["lam1"]["alpha"])], "VES_placebo")
        self.search_stage(m, "test", "b7", [(None, None)], "B7")
        if "R1" in mt and "drop_reference_rows" not in self.budget.cuts and (self.pdir(m, "test") / "tries.json").exists():
            self.judges_stage(m, "test")

    # ================================================================ main
    def main(self):
        self.phase_a()
        for m in self.names:
            if self.budget.hard_stop():
                break
            self.phase_b(m)
            if (self.pdir(m) / "detectors.json").exists() and (self.pdir(m) / "ves_consts.npz").exists():
                self.phase_c(m)
            self.release()
        fz = self.phase_d()
        go, why = self.test_gate()
        if not go:
            print(f"TEST_PHASE_REFUSED: {why}")
            self.missing.append(f"test phase refused: {why}")
        for m in (self.names if go else []):
            if self.budget.hard_stop():
                print(f"HARD_STOP: test phase of {m} not started")
                self.missing.append(f"test phase of {m} (hard stop)")
                continue
            self.phase_e(m, fz)
            self.release()
        self.pairing_check("test")
        from vesecs.analysis import analyze
        self.budget_safe(self.budget.start, "analysis", None, "all")
        try:
            analyze(self)
        finally:
            self.budget_safe(self.budget.finish, "analysis", None, "all")
            self.budget_safe(self.budget.dump)
        print(f"DONE: {self.run.root}")


def tune_all(pl, sr, m):
    """Dev tuning of every method for one model (the frozen configuration)."""
    cfg = pl.cfg
    lams, step = cfg["lambdas"], cfg["thresholds"]["quantile_step"]
    fx = cfg["fpe_fixed"]
    out = {"methods": {}, "n_dev": sr.N, "n_answered": int(sr.answered.sum())}
    T = out["methods"]
    T["B1"] = M.tune_gate(sr, sr.m, fx["B1"], lams, step)
    T["B3"] = M.tune_gate(sr, sr.sD, fx["B3"], lams, step)
    T["B3b"] = M.tune_gate(sr, sr.sD_before, fx["B3b"], lams, step)
    if sr.b5 is not None:
        b5 = {N: M.tune_search(sr, [({"N": N}, sr.b5)], "B5", lams, step, max_rounds=N)
              for N in range(1, int(cfg["b5"]["N_max"]) + 1)}
        T["B5"] = {f"N{N}": v for N, v in b5.items()}
    if "B7" in sr.searches:
        T["B7"] = M.tune_search(sr, [({}, sr.searches["B7"])], "B7", lams, step)
    vt = pl.ves_tune(m)
    if vt is not None:
        T["VES"] = vt
        dep, al = vt["lam1"]["depth"], vt["lam1"]["alpha"]
        free = [({"depth": dep, "alpha": a}, sr.searches[pl.ves_tag("VES_free", dep, a)]) for a in cfg["ves_free"]["alphas"]
                if pl.ves_tag("VES_free", dep, a) in sr.searches]
        if free:
            T["VES_free"] = M.tune_search(sr, free, "VES_free", lams, step)
        tag = pl.ves_tag("VES_placebo", dep, al)
        if tag in sr.searches:
            T["VES_placebo"] = M.tune_search(sr, [({"depth": dep, "alpha": al}, sr.searches[tag])], "VES_placebo", lams, step)
        tag = pl.ves_tag("VES", dep, al)
        T["VES_unclean"] = M.tune_search(sr, [({"depth": dep, "alpha": al}, sr.searches[tag])], "VES", lams, step,
                                         key="sD_steered")
        if "B5" in T:
            nm, matched = M.b5_matched_N(sr, {int(k[1:]): v for k, v in T["B5"].items()}, vt["lam1"]["fpe_dev"],
                                         int(cfg["b5"]["N_max"]))
            T["B5"]["N_matched"] = {"N": nm, "matched": matched, "ves_fpe_dev": vt["lam1"]["fpe_dev"]}
    if sr.ecs is not None:
        for name in ("1p", "exact", "placebo"):
            if name in sr.ecs_variants:
                T[f"ECS_{name}"] = {"beta": sr.ecs_beta, **M.tune_ecs(sr, name, lams, step)}
    if sr.judges and "R1" in sr.judges:
        T["R1"] = M.tune_gate(sr, sr.judges["R1"], fx["R1"], lams, step)
        T["R2"] = M.tune_gate(sr, sr.judges["R2"], fx["R2"], lams, step)
    # dev utilities of untuned rows, for the record
    out["dev_B0_U1"] = M.utility(sr.a1_label, 1)
    return out


def allowed_mask(eng, t_idk):
    """True for tokens that may be ECS candidates: not special / added, at least one alphanumeric character, not
    t_IDK. Returns (mask over the logit vocabulary, decoded token strings)."""
    tok = eng.tok
    V = int(eng.model.lm_head.out_features)
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
    n = min(len(tok), V)
    dec = tok.batch_decode([[i] for i in range(n)])
    allowed = np.zeros(V, bool)
    for i, s in enumerate(dec):
        allowed[i] = (i not in special) and (i != t_idk) and any(ch.isalnum() for ch in s)
    return allowed, dec


def model_weight_hashes(specs):
    from huggingface_hub import snapshot_download
    from pilot import config as PC
    out = {}
    for s in specs:
        try:
            snap = Path(snapshot_download(s["hf_id"], revision=s["revision"], local_files_only=True,
                                          allow_patterns=PC.MODEL_FILE_PATTERNS))
            out[s["name"]] = {"hf_id": s["hf_id"], "revision": s["revision"],
                              "files": {p.name: sha256_file(p) for p in sorted(snap.glob("*.safetensors"))}}
        except Exception as e:
            out[s["name"]] = {"error": f"{type(e).__name__}: {e}"}
    return out


def main():
    Pipeline().main()


if __name__ == "__main__":
    main()

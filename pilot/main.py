"""Pilot run (GPU, offline): python -m pilot.main   (run_pilot.sh sets the run folder and logs)

Order: pool -> per model [shared setup, cross-fitted detector, SQ1, SQ3, SQ5] -> SQ2 (CPU, both
models) -> SQ4 only if the provisional SQ2 signal is GO or BORDERLINE -> results.json and the
provisional go / stop signals. Each phase is saved to <run>/<model>/<phase>.pt as soon as it ends;
PILOT_RESUME=1 with the same PILOT_OUT_DIR reuses finished phases after a crash. A failing phase is
reported (PHASE_FAILED) and the others still run.

Signals become final only after the hand checks: python -m pilot.analyze (see pilot/README.md).
"""
import gc
import hashlib
import json
import platform
import traceback
from importlib import metadata

import numpy as np
import torch

from pilot import config as PC
from pilot import data as pdata
from pilot import signals as SG
from pilot.budget import Budget
from pilot.metrics import to_py
from pilot.probes import crossfit_probe
from pilot.shared import run_screen, run_shared, shared_summary, write_grading_check
from pilot.sq1 import run_sq1
from pilot.sq2 import analyze as sq2_analyze, extension_disagreements
from pilot.sq3 import run_sq3
from pilot.sq4 import run_sq4
from pilot.sq5 import run_sq5
from scoring import A_, C_


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_hash():
    h = hashlib.sha256()
    for p in sorted(PC.ROOT.glob("pilot/*.py")) + [PC.ROOT / "pilot" / "PREREGISTRATION.md", PC.ROOT / "model_utils.py",
                                                   PC.ROOT / "scoring.py", PC.ROOT / "steering.py"]:
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def versions():
    out = {}
    for pkg in ["torch", "transformers", "tokenizers", "numpy", "scikit-learn", "scipy"]:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    out["python"] = platform.python_version()
    return out


class Runner:
    def __init__(self):
        self.out = PC.OUT_DIR
        if (self.out / "results.json").exists() and not PC.RESUME:
            raise SystemExit(f"{self.out} already holds results; use a new folder (or PILOT_RESUME=1 to continue it)")
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads(PC.MANIFEST_PATH.read_text())
        self.check_data()
        self.pool, info = pdata.build_pool()
        print(f"POOL: n={info['n_pool']} topic_cap_ok={info['topic_cap_ok']} "
              f"max_popqa_relation_frac={info['max_popqa_relation_frac']:.3f} sq3={info['n_sq3']} sq5={info['n_sq5']}")
        self.ext, ext_info = pdata.build_ext_pool(self.pool)
        info["extension"] = ext_info
        print(f"EXTENSION: n={ext_info['n_ext']} popqa_shortfall={ext_info['popqa_shortfall']}")
        pdata.write_pool(self.pool, self.out / "pool.jsonl")
        pdata.write_pool(self.ext, self.out / "extension_pool.jsonl")
        pdata.write_pilot_qids(self.pool + self.ext, self.out / "pilot_qids.json")
        if not PC.SMOKE:
            pdata.write_pilot_qids(self.pool + self.ext, PC.PILOT_QIDS_PATH)
        n_para = pdata.write_paraphrase_check(self.pool, self.out / "paraphrase_check.csv")
        print(f"REWORDING_CHECK: wrote {n_para} rows to paraphrase_check.csv (fill meaning_preserved)")
        self.budget = Budget([m["name"] for m in PC.MODELS])
        self.results = {"meta": {"run_id": PC.RUN_ID, "out_dir": str(self.out), "smoke": PC.SMOKE,
                                 "models": {m["name"]: {"hf_id": m["hf_id"], "revision": self.revision(m)} for m in PC.MODELS},
                                 "code_sha256": code_hash(), "versions": versions(),
                                 "layer_plans": {m["name"]: PC.layer_plan(m) for m in PC.MODELS}},
                        "pool": info, "per_model": {}, "models_run": []}
        self.shared, self.probes, self.screens, self.followups = {}, {}, {}, {}
        self.names = [m["name"] for m in PC.MODELS]

    def revision(self, spec):
        m = self.manifest.get("models", {}).get(spec["name"], {})
        rev = m.get("revision") or spec["revision"]
        if spec["revision"] and rev != spec["revision"]:
            raise SystemExit(f"{spec['name']}: manifest revision {rev} != pinned {spec['revision']} (rerun setup)")
        return rev

    def check_data(self):
        for key, fname in PC.DATA_FILES.items():
            p = PC.DATA_DIR / fname
            rec = self.manifest.get("datasets", {}).get(key, {}).get("sha256")
            if not p.exists():
                raise SystemExit(f"missing {p}: run `bash run_pilot.sh setup`")
            got = sha256_file(p)
            status = "ok" if rec == got else ("unrecorded" if rec is None else "MISMATCH")
            print(f"DATA_CHECKSUM: {key} {status}")
            if status == "MISMATCH":
                raise SystemExit(f"{p} changed since setup (sha256 {got[:12]} != {rec[:12]})")

    def save_results(self):
        (self.out / "results.json").write_text(json.dumps(to_py(self.results), indent=1))
        self.budget.dump(self.out / "budget.json")

    def phase(self, key, model, path, fn):
        if PC.RESUME and path is not None and path.exists():
            print(f"RESUME: {key}[{model}] loaded from {path.name}")
            self.budget.done.add((key, model))
            return torch.load(path, weights_only=False)
        self.budget.start(key, model)
        try:
            res = fn()
            if path is not None:
                torch.save(res, path)
            return res
        except Exception as e:
            traceback.print_exc()
            print(f"PHASE_FAILED: {key}[{model}] {type(e).__name__}: {e}")
            return None
        finally:
            self.budget.finish(key, model)

    def run_model(self, spec):
        from pilot.session import Session
        name = spec["name"]
        mdir = self.out / name
        mdir.mkdir(exist_ok=True)
        holder = {}

        def S():
            if "s" not in holder:
                print(f"LOADING: {name} {spec['hf_id']}@{self.revision(spec)[:12]}")
                holder["s"] = Session(spec, self.revision(spec))
            return holder["s"]

        def shared_fn():
            s = S()
            rates = s.calibrate([s.prompt(it.question) for it in self.pool[:256]])
            self.budget.set_rates(rates, name)
            return run_shared(s, self.pool)

        res = self.results["per_model"].setdefault(name, {})
        self.results["models_run"].append(name)
        sh = self.phase("shared", name, mdir / "shared.pt", shared_fn)
        if sh is None:
            self._free(holder)
            return
        self.shared[name] = sh
        res["shared"] = shared_summary(sh, self.pool)
        print("SHARED: model=" + name + " " + " ".join(f"{k}={v:.3f}" for k, v in res["shared"].items()
                                                         if isinstance(v, float)))
        gpath = self.out / f"grading_check_{name}.csv"
        if not gpath.exists():
            write_grading_check(sh, self.pool, name, gpath, PC.PILOT_SEED + len(self.shared))
        self.save_results()

        folds = np.array([it.extra["fold"] for it in self.pool])
        pr = self.phase("probe", name, mdir / "probe.pt", lambda: dict(zip(("oof", "chosen"), crossfit_probe(
            sh["res_prompt"], sh["res_ans"], sh["states"] == C_, sh["states"] != A_, folds))))
        if pr is not None:
            self.probes[name] = pr
            res["probe"] = {"folds": pr["chosen"]}

        for key, fn in (("sq1", lambda: run_sq1(S(), self.pool, sh, pr["oof"], self.budget.n_random(), self.budget.n_eval())),
                        ("sq3", lambda: run_sq3(S(), self.pool, sh)),
                        ("sq5", lambda: run_sq5(S(), self.pool, sh, pr["oof"], steer=True))):
            if self.budget.hard_stop():
                print(f"HARD_STOP: {key}[{name}] not started")
                res[key] = None
                continue
            if not self.budget.runs(key, name):
                print(f"SKIPPED_BY_BUDGET: {key}[{name}] (ladder {self.budget.flags})")
                res[key] = {"skipped": f"budget ladder {self.budget.flags}"}
                continue
            if key in ("sq1", "sq5") and pr is None:
                res[key] = None
                continue
            r = self.phase(key, name, mdir / f"{key}.pt", fn)
            res[key] = r[0] if r is not None else None
            self.save_results()

        # SQ2 screening extension: the 10 tries on the extra questions, then (once every model has
        # screened) the full record of the disagreement questions found there, while this model is loaded
        if not self.budget.hard_stop():
            n_scr = self.budget.n_ext(len(self.ext))
            scr = self.phase("screen", name, mdir / "screen.pt", lambda: run_screen(S(), self.ext[:n_scr]))
            if scr is not None:
                self.screens[name] = scr
        self.maybe_followup(name, S, mdir)
        self._free(holder)

    def ext_disagreements(self):
        if not all(m in self.screens for m in self.names[:2]):
            return None, 0
        return extension_disagreements(self.screens, self.names)

    def maybe_followup(self, name, S, mdir):
        idx, n = self.ext_disagreements()
        if idx is None or name in self.followups or self.budget.hard_stop():
            return
        items = [self.ext[i] for i in idx]
        print(f"EXTENSION_DISAGREEMENT: {len(items)} of {n} screened questions")
        if not items:
            return
        fu = self.phase("followup", name, mdir / "followup.pt",
                        lambda: run_shared(S(), items, heads=False, seed=PC.FOLLOWUP_SAMPLE_SEED))
        if fu is not None:
            self.followups[name] = fu

    def _free(self, holder):
        holder.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self):
        for spec in PC.MODELS:
            if self.budget.hard_stop():
                print(f"HARD_STOP: model {spec['name']} not started")
                break
            self.run_model(spec)

        # follow-up of the extension disagreement questions for models that screened before the last
        # model did (reloaded once; the SQ4 model goes last and stays loaded for SQ4)
        from pilot.session import Session
        kept = {}
        todo = [m for m in self.names if m in self.screens and m not in self.followups and m in self.shared]
        for name in sorted(todo, key=lambda m: m == PC.SQ4_MODEL):
            spec = next(m for m in PC.MODELS if m["name"] == name)
            holder = {}

            def S(spec=spec, holder=holder):
                if "s" not in holder:
                    print(f"LOADING: {spec['name']} (extension follow-up)")
                    holder["s"] = Session(spec, self.revision(spec))
                return holder["s"]

            self.maybe_followup(name, S, self.out / name)
            if name == PC.SQ4_MODEL and "s" in holder:
                kept = holder
            else:
                self._free(holder)

        idx, n_screened = self.ext_disagreements()
        ext = ({"items": [self.ext[i] for i in idx], "n_screened": n_screened, "followup": self.followups}
               if idx is not None else None)
        sq2 = self.phase("sq2", None, self.out / "sq2.pt",
                         lambda: sq2_analyze(self.pool, self.shared, self.probes, ext)) if len(self.probes) >= 2 else None
        self.results["sq2"] = sq2 or {"error": "SQ2 needs the shared setup and detector of both models"}
        # The SQ4 gate needs SQ2's provisional signal (automatic labels). Only the gate decision is
        # printed; the signals themselves wait for the grading check (pilot.analyze).
        s2 = SG.sq2(self.results["sq2"])
        self.results["sq4_gate"] = {"sq2_provisional_signal": s2["signal"], "run_sq4": s2["signal"] in PC.SQ4_RUN_ON}
        print(f"SQ4_GATE: run_sq4={s2['signal'] in PC.SQ4_RUN_ON}")
        self.save_results()

        spec = next((m for m in PC.MODELS if m["name"] == PC.SQ4_MODEL), None)
        if s2["signal"] not in PC.SQ4_RUN_ON:
            self.results["sq4"] = {"skipped": f"provisional SQ2 signal {s2['signal']} (SQ4 runs only on GO / BORDERLINE)"}
        elif not self.budget.runs("sq4", None) or self.budget.hard_stop():
            self.results["sq4"] = {"skipped": f"budget (ladder {self.budget.flags})"}
        elif spec is None or spec["name"] not in self.shared:
            self.results["sq4"] = {"skipped": "SQ4 model has no shared setup"}
        else:
            holder = kept

            def fn():
                if "s" not in holder:
                    holder["s"] = Session(spec, self.revision(spec))
                return run_sq4(holder["s"], self.pool, self.shared[spec["name"]], self.budget.lora_epochs())

            r = self.phase("sq4", None, self.out / "sq4.pt", fn)
            self.results["sq4"] = r[0] if r is not None else {"error": "SQ4 failed"}
        self._free(kept)

        self.results["budget"] = self.budget.summary()
        self.save_results()
        print(f"DONE: results in {self.out}. Go / stop signals are not printed before the hand checks: fill "
              f"grading_check_<model>.csv and paraphrase_check.csv, then run: python -m pilot.analyze --run-dir {self.out}")


if __name__ == "__main__":
    Runner().run()

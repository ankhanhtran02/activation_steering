"""Compute budget for the 12-hour GB10 limit.

The work of every phase is counted in four units (greedy answers, teacher-forced sequences, sampled
sequences, LoRA training sequences) and converted to seconds with throughput measured on the first
model (re-measured on the second). The projection is corrected by the ratio of actual to planned time
of finished phases. If it exceeds the cap (11 h), ladder steps (config.DEGRADATION_LADDER) are applied
in the pre-registered order, only to phases that have not started. No phase starts after 11.5 h.
"""
import json
import time

from pilot import config as PC

# expected teacher-forced passes per direction in the dose search (bracketing + secant steps)
DOSE_EVALS_PER_DIRECTION = 5


class Budget:
    def __init__(self, model_names, answered_frac=0.7):
        self.t0 = time.time()
        self.models = list(model_names)
        self.rates = None
        self.flags = []
        self.applied_to = {}
        self.done, self.started = set(), set()
        self.planned_done, self.actual_done = 0.0, 0.0
        self._t = {}
        self.answered_frac = answered_frac
        self.log = []

    def elapsed(self):
        return time.time() - self.t0

    # ------------------------------------------------------------ work plan
    def n_random(self):
        return 10 if "sq1_random_10" in self.flags else PC.SQ1_N_RANDOM

    def n_eval(self):
        return 500 if "sq1_eval_500" in self.flags else PC.SQ1_N_EVAL

    def lora_epochs(self):
        return 1 if "sq4_one_epoch" in self.flags else PC.LORA_EPOCHS

    def n_ext(self, n_full=None):
        n_full = n_full if n_full is not None else PC.N_EXT_TQA + sum(PC.EXT_POPQA_QUOTAS.values())
        return n_full // 2 if "sq2_extension_half" in self.flags else n_full

    def runs(self, phase, model):
        first = model == self.models[0]
        if phase == "sq3" and "sq3_single_model" in self.flags and not first and ("sq3", model) not in self.started:
            return False
        if phase == "sq1" and "sq1_single_model" in self.flags and not first and ("sq1", model) not in self.started:
            return False
        if phase == "sq4" and "sq4_skip" in self.flags and ("sq4", None) not in self.started:
            return False
        return True

    def work(self, phase):
        n = sum(PC.POPQA_QUOTAS.values()) + PC.N_TQA
        a = self.answered_frac
        if phase == "shared":
            return {"gen": n, "fwd": 2 * n, "samp": n * (PC.N_DIS_TRIES + PC.N_CON_TRIES), "load": 1}
        if phase == "sq1":
            nd = 3 + self.n_random()
            ev = self.n_eval() * a
            return {"fwd": PC.SQ1_FORCE_N + 2 * PC.SQ1_CAA_MAX_PAIRS + 2 * PC.SQ1_N_TUNE * (1 + nd * DOSE_EVALS_PER_DIRECTION)
                    + 2 * ev * (1 + nd), "gen": PC.SQ1_N_GEN * (4 + min(self.n_random(), PC.SQ1_N_RANDOM_GEN) + 1)}
        if phase == "sq3":
            nt = PC.N_SQ3 - PC.N_SQ3_TUNE
            return {"gen": 5 * PC.N_SQ3 + 6 * PC.N_SQ3_TUNE + 2 * nt, "fwd": 5 * PC.N_SQ3 * 0.5 + nt * 15}
        if phase == "sq5":
            return {"gen": PC.N_SQ5_POB + PC.N_SQ5_YEAR + 600, "fwd": 400}
        if phase == "screen":
            return {"samp": PC.N_DIS_TRIES * self.n_ext()}
        if phase == "followup":
            nd = PC.EXT_EXPECTED_DISAGREE
            return {"gen": nd, "fwd": 2 * nd, "samp": (PC.N_DIS_TRIES + PC.N_CON_TRIES) * nd, "load": 0.5}
        if phase == "sq4":
            n_tr = n * len(PC.SQ4_TRAIN_FOLDS) / PC.N_FOLDS
            return {"train": 2 * self.lora_epochs() * n_tr, "gen": 2 * (n - n_tr), "load": 1}
        return {}

    def cost(self, w):
        r = self.rates or {"gen": 10.0, "fwd": 16.0, "samp": 60.0}
        return (w.get("gen", 0) / r["gen"] + w.get("fwd", 0) / r["fwd"] + w.get("samp", 0) / r["samp"]
                + w.get("train", 0) / (r["fwd"] / 3.5) + w.get("load", 0) * 150)

    def pending(self):
        keys = []
        for m in self.models:
            for ph in ("shared", "sq1", "sq3", "sq5", "screen", "followup"):
                if (ph, m) not in self.done and self.runs(ph, m):
                    keys.append((ph, m))
        if ("sq4", None) not in self.done and self.runs("sq4", None):
            keys.append(("sq4", None))
        return keys

    def projected_total(self):
        scale = self.actual_done / self.planned_done if self.planned_done > 0 else 1.0
        rest = sum(self.cost(self.work(ph)) for ph, _ in self.pending())
        return self.elapsed() + scale * rest + PC.ANALYSIS_RESERVE_SECONDS, scale

    # ------------------------------------------------------------ events
    def set_rates(self, rates, model):
        self.rates = rates
        self._log("rates", model=model, **{k: round(v, 2) for k, v in rates.items()})
        self.degrade(f"rates[{model}]")

    def start(self, phase, model):
        self.started.add((phase, model))
        self._t[(phase, model)] = time.time()

    def finish(self, phase, model):
        key = (phase, model)
        self.done.add(key)
        if key in self._t:
            took = time.time() - self._t.pop(key)
            planned = self.cost(self.work(phase))
            if planned > 0:          # CPU-only phases (probe, SQ2) have no planned GPU work
                self.actual_done += took
                self.planned_done += planned
        print(f"TIME: {phase}[{model or 'all'}] {self.elapsed():.1f}")
        self.degrade(f"{phase}[{model}]")

    def degrade(self, at):
        proj, scale = self.projected_total()
        while proj > PC.BUDGET_CAP_SECONDS:
            nxt = next((s for s in PC.DEGRADATION_LADDER if s not in self.flags), None)
            if nxt is None:
                break
            self.flags.append(nxt)
            proj, scale = self.projected_total()
            self._log("ladder", at=at, step=nxt, projected_total_s=round(proj))
        self._log("projection", at=at, projected_total_s=round(proj), scale=round(scale, 3), ladder=list(self.flags))

    def hard_stop(self):
        return self.elapsed() > PC.HARD_STOP_SECONDS

    def _log(self, kind, **kw):
        kw = {"kind": kind, "elapsed_s": round(self.elapsed(), 1), **kw}
        self.log.append(kw)
        print("BUDGET: " + " ".join(f"{k}={v}" for k, v in kw.items()))

    def summary(self):
        return {"elapsed_s": self.elapsed(), "rates": self.rates, "ladder": self.flags, "log": self.log}

    def dump(self, path):
        path.write_text(json.dumps(self.summary(), indent=1, default=str))

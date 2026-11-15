"""12-hour budget on the GB10 (spec section 11).

Work per stage is counted in five units (greedy generations, teacher-forced sequences, sampled sequences,
backward passes, echo-pass questions) and converted to seconds with rates measured by the timing probe at
each model's first load (CPU-only stages have fixed estimates from the config). After every stage the
projected end time (elapsed + remaining estimates x the actual/planned ratio of finished stages) is written
to logs/timing.jsonl. If the projection exceeds 10.5 h, or a stage overruns its estimate by more than 50%,
cuts are applied in the pre-registered order until the projection fits (a cut affects only stages that have
not started):
  1 vllm_tries           use vLLM for tries if it is importable (no design change)
  2 drop_reference_rows  drop R1-R3
  3 drop_placebos        drop VES-placebo and ECS-placebo
  4 shrink_ves_grid      VES dev grid -> l_s = 50%, alpha in {0.2, 0.4}
  5 shrink_test          test set -> 1,500 questions (TriviaQA questions removed, seeded)
B3, B5, B7, VES, ECS-1P, ECS-exact and the integrity checks are never cut. No stage starts after 11.5 h.
"""
import os
import time
from datetime import datetime, timedelta, timezone

from vesecs.io import append_jsonl, write_json

DEFAULT_RATES = {"gen": 8.0, "tf": 12.0, "samp": 20.0, "bwd": 4.0, "echo": 6.0}   # SQ2's GB10 rates (Llama)
MIN_PLANNED_FOR_SCALE = 600.0      # seconds of finished planned work before the actual/planned ratio is used
MIN_STAGE_FOR_OVERRUN = 60.0       # stages estimated below this do not count as overrunning


class Budget:
    def __init__(self, cfg, run, n_models):
        self.cfg, self.run = cfg, run
        b = cfg["budget"]
        self.limit = float(b["project_limit_hours"]) * 3600
        self.hard = float(b["hard_stop_hours"]) * 3600
        self.overrun = float(b["overrun_factor"])
        self.cpu = b["cpu_seconds"]
        self.t0 = time.time()
        self.rates = {}
        self.cuts = []
        self.cut_unavailable = []
        self.done = set()
        self.started = set()
        self.planned_done = 0.0
        self.actual_done = 0.0
        self._t = {}
        self.models = []
        self.n_models = n_models
        self.answered_frac = 0.75
        self.sizes = {"pool": 3000, "dev": int(cfg["data"]["dev_size"]), "test": int(cfg["data"]["test_size"])}
        self.log_path = run.logs / "timing.jsonl"
        prev = run.logs / "budget.json"
        if prev.exists():                      # resumed run: keep the clock, the measured rates and the cuts
            from vesecs.io import read_json
            p = read_json(prev)
            if os.environ.get("VES_RESET_CLOCK") != "1":
                self.t0 -= float(p.get("elapsed_s") or 0.0)
            self.rates = p.get("rates") or {}
            self.cuts = list(p.get("cuts") or [])
            self.cut_unavailable = list(p.get("cuts_unavailable") or [])
            self._log({"event": "resume", "previous_elapsed_s": p.get("elapsed_s"), "cuts": self.cuts})

    def elapsed(self):
        return time.time() - self.t0

    def hard_stop(self):
        return self.elapsed() > self.hard

    # ------------------------------------------------------------ work model
    def r(self, model):
        return self.rates.get(model) or next(iter(self.rates.values()), None) or DEFAULT_RATES

    def ves_grid(self):
        v = self.cfg["ves"]
        if "shrink_ves_grid" in self.cuts:
            g = v["shrunk_grid"]
            return [(d, a) for d in g["steer_depths"] for a in g["alphas"]]
        return [(d, a) for d in v["steer_depths"] for a in v["alphas"]]

    def n_test(self):
        return int(self.cfg["budget"]["shrink_test_to"]) if "shrink_test" in self.cuts else self.sizes["test"]

    def work(self, stage, split):
        """Expected work units of one stage for one model."""
        n = {"pool": self.sizes["pool"], "dev": self.sizes["dev"], "test": self.n_test()}.get(split, 0)
        a = self.answered_frac
        K = int(self.cfg["ves"]["K"])
        rounds = a * n * (1 + 0.8 + 0.6) * K / 3.0
        nt = int(self.cfg["decoding"]["n_tries"])
        placebo = "drop_placebos" not in self.cuts
        if stage == "verify":
            return {"tf": int(self.cfg["checks"]["state_verify_n"])}
        if stage == "detectors":
            return {"cpu": self.cpu["pool_detectors"]}
        if stage == "ubar":
            return {"bwd": int(self.cfg["ves"]["n_ubar"])}
        if stage == "lens":
            return {"cpu": 30}
        if stage == "echo":
            return {"echo": n}
        if stage == "completions":
            return {"gen": 2.3 * n, "tf": (0 if split == "pool" else 2.0 * n)}
        if stage == "echo_detectors":
            return {"cpu": self.cpu["echo_detectors"]}
        if stage == "drafts":
            return {"gen": n}
        if stage == "states":
            return {"tf": 2 * n}
        if stage == "tries":
            return {"samp": n * nt, "vllm": "vllm_tries" in self.cuts}
        if stage == "b5":
            return {"tf": a * n * 8 * 0.5}
        if stage == "ves":
            n_cfg = len(self.ves_grid()) if split == "dev" else 1
            return {"gen": rounds * n_cfg, "tf": 2 * 0.8 * rounds * n_cfg, "bwd": 0.8 * rounds * n_cfg}
        if stage == "ves_free":
            n_cfg = len(self.cfg["ves_free"]["alphas"]) if split == "dev" else 1
            return {"gen": rounds * n_cfg, "tf": 0.8 * rounds * n_cfg}
        if stage == "ves_placebo":
            return {"gen": rounds, "tf": 0.8 * rounds} if placebo else {}
        if stage == "b7":
            return {"gen": rounds, "tf": 0.8 * rounds}
        if stage == "checks":
            return {"gen": 128, "tf": 128, "bwd": 12, "echo": 16}
        if stage == "tuning":
            return {"cpu": self.cpu["dev_tuning"]}
        if stage == "analysis":
            return {"cpu": self.cpu["analysis"]}
        return {}

    def seconds(self, w, model):
        r = self.r(model)
        samp_rate = r["samp"] * (4.0 if w.get("vllm") else 1.0)
        return (w.get("gen", 0) / r["gen"] + w.get("tf", 0) / r["tf"] + w.get("samp", 0) / samp_rate
                + w.get("bwd", 0) / r["bwd"] + w.get("echo", 0) / r["echo"] + w.get("cpu", 0))

    def plan(self):
        """Every (stage, model, split) of the run, in execution order."""
        keys = []
        pool = ["verify", "detectors", "ubar", "lens", "echo", "completions", "echo_detectors"]
        dev = ["drafts", "states", "tries", "b5", "echo", "completions", "ves", "ves_free", "ves_placebo", "b7", "checks",
               "tuning"]
        test = ["drafts", "states", "tries", "b5", "echo", "completions", "ves", "ves_free", "ves_placebo", "b7"]
        for m in self.models or [f"model{i}" for i in range(self.n_models)]:
            keys += [(s, m, "pool") for s in pool] + [(s, m, "dev") for s in dev]
        for m in self.models or [f"model{i}" for i in range(self.n_models)]:
            keys += [(s, m, "test") for s in test]
        return keys + [("analysis", None, "all")]

    def scale(self, rest):
        """actual / planned seconds of finished stages, used only once enough planned work has finished (at least
        10 min, or 5% of the plan) so that a short early stage cannot dominate; clamped to [0.5, 4]."""
        if self.planned_done < max(MIN_PLANNED_FOR_SCALE, 0.05 * (self.planned_done + rest)):
            return 1.0
        return float(min(4.0, max(0.5, self.actual_done / self.planned_done)))

    def projected_total(self):
        rest = sum(self.seconds(self.work(s, sp), m) for s, m, sp in self.plan() if (s, m, sp) not in self.done)
        scale = self.scale(rest)
        return self.elapsed() + scale * rest, scale

    # ------------------------------------------------------------ events
    def set_rates(self, model, rates):
        self.rates[model] = rates
        self._log({"event": "rates", "model": model, "rates": rates})
        self.check(f"rates[{model}]", overran=False)

    def start(self, stage, model, split):
        self.started.add((stage, model, split))
        self._t[(stage, model, split)] = time.time()

    def finish(self, stage, model, split, ok=True):
        key = (stage, model, split)
        self.done.add(key)
        took = time.time() - self._t.pop(key, time.time())
        est = self.seconds(self.work(stage, split), model)
        overran = est >= MIN_STAGE_FOR_OVERRUN and took > self.overrun * est
        self.planned_done += est
        self.actual_done += took
        proj, scale = self.projected_total()
        self._log({"event": "stage", "stage": stage, "model": model, "split": split, "ok": ok, "seconds": round(took, 1),
                   "estimate_s": round(est, 1), "overran": overran, "elapsed_s": round(self.elapsed(), 1),
                   "projected_total_s": round(proj), "scale": round(scale, 3),
                   "projected_end_utc": (datetime.now(timezone.utc) + timedelta(seconds=proj - self.elapsed())).isoformat(),
                   "cuts": list(self.cuts)})
        print(f"TIME: {stage}[{model or 'all'}/{split}] {took:.1f}s (est {est:.0f}s) elapsed {self.elapsed() / 3600:.2f} h, "
              f"projected {proj / 3600:.2f} h")
        self.check(f"{stage}[{model}/{split}]", overran)

    def skip(self, stage, model, split, why):
        self.done.add((stage, model, split))
        self._log({"event": "skipped", "stage": stage, "model": model, "split": split, "why": why})

    def check(self, at, overran):
        proj, _ = self.projected_total()
        if not (proj > self.limit or overran):
            return
        for cut in self.cfg["budget"]["cuts"]:
            if proj <= self.limit:
                break
            if cut in self.cuts or cut in self.cut_unavailable:
                continue
            if cut == "vllm_tries" and not vllm_available():
                self.cut_unavailable.append(cut)
                self._log({"event": "cut_unavailable", "cut": cut, "at": at})
                continue
            self.cuts.append(cut)
            proj, _ = self.projected_total()
            self._log({"event": "cut", "cut": cut, "at": at, "projected_total_s": round(proj)})
            print(f"BUDGET_CUT: {cut} at {at} (projected {proj / 3600:.2f} h)")

    def _log(self, row):
        row = {"t_utc": datetime.now(timezone.utc).isoformat(), **row}
        append_jsonl(self.log_path, row)

    def dump(self):
        write_json(self.run.logs / "budget.json", {"elapsed_s": self.elapsed(), "rates": self.rates, "cuts": self.cuts,
                                                   "cuts_unavailable": self.cut_unavailable}, indent=1)


def vllm_available():
    try:
        import vllm  # noqa: F401
        return True
    except Exception:
        return False

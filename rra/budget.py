"""12-hour budget guard on the GB10 (spec section 10, "Budget guard (fixed)").

  1 The projection uses only measured rates: each model's timing probe at its first load, then the actual
    per-item rates of completed work (running totals per model and work unit). Planning estimates are never used.
  2 Only the optional stages (E2, G) can be cut, and only if the measured projection exceeds 10.5 hours. They run
    after phase F has written the verdict; the decision is taken just before each starts, with every rate of the
    required stages already measured. Cut order: config budget.cut_order (G, then E2).
  3 Every cut is logged (logs/timing.jsonl) with the projection and the rates behind it.
  4 Hard stop at 11.5 hours: no stage starts after it; the report marks what is missing.

Work units: gen (greedy or token-ban generations), tf (teacher-forced sequences), samp (sampled sequences),
fit (one SQ2-recipe probe fit: full fit + 5-fold cross-fit on the pool), load (one model load), analysis (one
phase-F run). A rate that is not measured yet for a model is taken from the slowest model measured so far
(still a measurement; the row says so). Gemma (stage G) uses that rule until its own probe has run.
"""
import os
import time
from datetime import datetime, timedelta, timezone

from vesecs.io import append_jsonl, read_json, write_json

UNITS = ("gen", "tf", "samp", "fit", "load", "analysis")


class Budget:
    def __init__(self, cfg, run=None, clock=time.time, log_path=None):
        b = cfg["budget"]
        self.cfg = cfg
        self.limit = float(b["project_limit_hours"]) * 3600
        self.hard = float(b["hard_stop_hours"]) * 3600
        self.cut_order = list(b["cut_order"])
        self.clock = clock
        self.t0 = clock()
        self.totals = {}            # model -> unit -> [units, seconds]
        self.probe = {}             # model -> unit -> units / s
        self.cuts = []
        self.run = run
        self.log_path = log_path or (run.logs / "timing.jsonl" if run is not None else None)
        prev = run.logs / "budget.json" if run is not None else None
        if prev is not None and prev.exists():                  # resumed run: keep the clock and the measurements
            p = read_json(prev)
            if os.environ.get("RRA_RESET_CLOCK") != "1":
                self.t0 -= float(p.get("elapsed_s") or 0.0)
            self.totals = p.get("totals") or {}
            self.probe = p.get("probe") or {}
            self.cuts = list(p.get("cuts") or [])
            self._log({"event": "resume", "previous_elapsed_s": p.get("elapsed_s"), "cuts": self.cuts})

    # ------------------------------------------------------------ clock
    def elapsed(self):
        return self.clock() - self.t0

    def hard_stop(self):
        return self.elapsed() > self.hard

    # ------------------------------------------------------------ measurements
    def set_probe(self, model, rates):
        self.probe[model] = {k: float(v) for k, v in rates.items() if v and v > 0}
        self._log({"event": "timing_probe", "model": model, "rates": self.probe[model]})

    def record(self, model, unit, n_units, seconds):
        """Actual work done: n_units of `unit` in `seconds` (only completed work is recorded)."""
        if n_units <= 0 or seconds <= 0:
            return
        t = self.totals.setdefault(model, {}).setdefault(unit, [0.0, 0.0])
        t[0] += float(n_units)
        t[1] += float(seconds)

    def rate(self, model, unit):
        """(units per second, source) from measurements only; (None, None) if nothing is measured."""
        t = self.totals.get(model, {}).get(unit)
        if t and t[0] > 0 and t[1] > 0:
            return t[0] / t[1], "measured"
        p = self.probe.get(model, {}).get(unit)
        if p:
            return p, "probe"
        others = []
        for m in set(self.totals) | set(self.probe):
            if m == model:
                continue
            t = self.totals.get(m, {}).get(unit)
            if t and t[0] > 0 and t[1] > 0:
                others.append(t[0] / t[1])
            elif self.probe.get(m, {}).get(unit):
                others.append(self.probe[m][unit])
        if others:
            return min(others), "slowest other model"
        return None, None

    def seconds(self, work):
        """work: [(model, unit, n)] -> (seconds, list of (model, unit) without any measured rate)."""
        s, missing = 0.0, []
        for m, u, n in work:
            if n <= 0:
                continue
            r, _ = self.rate(m, u)
            if r is None:
                missing.append((m, u))
                continue
            s += n / r
        return s, missing

    def rates_table(self, work):
        out = {}
        for m, u, _ in work:
            r, src = self.rate(m, u)
            out[f"{m}/{u}"] = {"rate": r, "source": src}
        return out

    # ------------------------------------------------------------ projection and cuts
    def projection(self, required_rest, optional):
        """elapsed + remaining required work + optional stages not cut (all at measured rates)."""
        s, miss = self.seconds(required_rest)
        for name, work in optional.items():
            if name in self.cuts:
                continue
            so, mo = self.seconds(work)
            s += so
            miss += mo
        return self.elapsed() + s, miss

    def log_stage(self, stage, model, split, seconds, required_rest, optional):
        proj, miss = self.projection(required_rest, optional)
        row = {"event": "stage", "stage": stage, "model": model, "split": split, "seconds": round(seconds, 1),
               "elapsed_s": round(self.elapsed(), 1), "projected_total_s": round(proj),
               "projected_end_utc": (datetime.now(timezone.utc) + timedelta(seconds=proj - self.elapsed())).isoformat(),
               "unmeasured": [f"{m}/{u}" for m, u in miss], "cuts": list(self.cuts)}
        self._log(row)
        print(f"TIME: {stage}[{model or 'all'}/{split}] {seconds:.1f}s elapsed {self.elapsed() / 3600:.2f} h, "
              f"projected {proj / 3600:.2f} h")
        return proj

    def decide(self, name, required_rest, optional):
        """Called just before optional stage `name` starts. Cuts optional stages (config order) while the measured
        projection exceeds the limit. Returns True if `name` may run."""
        if name in self.cuts:
            return False
        if self.hard_stop():
            self._cut(name, "hard stop", required_rest, optional)
            return False
        proj, miss = self.projection(required_rest, optional)
        self._log({"event": "decide", "stage": name, "projected_total_s": round(proj), "limit_s": self.limit,
                   "unmeasured": [f"{m}/{u}" for m, u in miss]})
        for cut in self.cut_order:
            if proj <= self.limit:
                break
            if cut in self.cuts or cut not in optional:
                continue
            self._cut(cut, f"measured projection {proj / 3600:.2f} h > {self.limit / 3600:.2f} h", required_rest,
                      optional)
            proj, _ = self.projection(required_rest, optional)
        return name not in self.cuts

    def _cut(self, cut, why, required_rest, optional):
        self.cuts.append(cut)
        proj, _ = self.projection(required_rest, optional)
        work = list(required_rest) + [w for ws in optional.values() for w in ws]
        self._log({"event": "cut", "cut": cut, "why": why, "projected_total_after_s": round(proj),
                   "rates": self.rates_table(work)})
        print(f"BUDGET_CUT: {cut} ({why}; projection after the cut {proj / 3600:.2f} h)")

    # ------------------------------------------------------------ io
    def _log(self, row):
        if self.log_path is None:
            return
        append_jsonl(self.log_path, {"t_utc": datetime.now(timezone.utc).isoformat(), **row})

    def dump(self):
        if self.run is None:
            return
        write_json(self.run.logs / "budget.json", {"elapsed_s": self.elapsed(), "totals": self.totals,
                                                   "probe": self.probe, "cuts": self.cuts}, indent=1)

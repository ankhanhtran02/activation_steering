"""Integrity checks (spec section 9). Each writes {pass, ...details} into results/integrity.json; the fresh-test
phase refuses to start unless every required check passed and configs/frozen_config.json exists.

  1 isolation              fresh questions vs every other set (rra.data.isolation_report)
  2 frozen_D               replay set: s1 recomputed (A1 teacher-forced again, frozen D) vs the stored scores,
                           max abs difference < 1e-3 (per model)
    b3_replay              B3 with its VES/ECS dev-tuned threshold reproduces the reported replay U1 (39.0)
  3 state_reproduction     teacher-forcing stored pool answers reproduces the stored states (max abs < 1e-2;
                           first SQ2 batch, so the batch composition is SQ2's; per model)
  4 fallback_reproduction  regenerated A2 vs stored B7 round 1 on replay questions (first B7 batch). Reported, not
                           blocking: the spec's consequence of > 2% differences is a reporting rule (stored A2 is
                           always used for replay and VES/ECS dev, regenerated A2 elsewhere)
  5 pairing                A1 and A2 are written once (SHA-256 registered at creation or, for VES/ECS sets, at
                           load) and every arm reads the same records; re-checked before the test phase and in F
  6 shuffled_labels_R      R on shuffled pool labels: cross-fitted AUROC in [0.45, 0.55] (per model)
  7 verdict_code           synthetic inputs give each of the four verdicts (rra.tests)
  8 budget_guard           synthetic stage timings: no cut while the measured projection is under 10.5 h (rra.tests)
  9 freeze                 the fresh-test phase needs configs/frozen_config.json; its SHA-256 is logged
Failure policy: a GPU memory error during a check is retried with half the batch size, up to 3 times; then the
run stops (rra.engine.with_oom_retry). A check is never skipped.
"""
from vesecs.io import read_json, write_json

REQUIRED = ["isolation", "verdict_code", "budget_guard", "state_reproduction", "frozen_D", "b3_replay",
            "shuffled_labels_R", "pairing"]
PER_MODEL = {"state_reproduction", "frozen_D", "shuffled_labels_R", "pairing", "fallback_reproduction"}
INFORMATIONAL = {"fallback_reproduction"}


class Integrity:
    def __init__(self, path):
        self.path = path
        self.d = read_json(path) if path.exists() else {}

    def record(self, name, model, result):
        self.d.setdefault(name, {})[model or "all"] = result
        write_json(self.path, self.d, indent=1)

    def has(self, name, model=None):
        return (model or "all") in self.d.get(name, {})

    def status(self, models):
        """(all required passed, list of failures / missing)."""
        bad = []
        for name in REQUIRED:
            rows = self.d.get(name, {})
            keys = models if name in PER_MODEL else ["all"]
            for k in keys:
                r = rows.get(k)
                if r is None:
                    bad.append(f"{name}[{k}] missing")
                elif not r.get("pass"):
                    bad.append(f"{name}[{k}] failed")
        return not bad, bad

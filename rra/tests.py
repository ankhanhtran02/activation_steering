"""Unit tests (CPU, seconds). Run by phase A (integrity checks 7 and 8) and standalone:

    python -m rra.tests
"""
import itertools
import traceback

import numpy as np

from scoring import A_, C_, W_

RULE = {"min_recoverable": 150, "h_stop": 2.0, "main_go": 1.0, "mech_go": 0.5}


def _c(v, lo, hi):
    return {"value": v, "ci_low": lo, "ci_high": hi}


# ----------------------------------------------------------------------------- verdicts (integrity check 7)
def test_verdicts():
    from rra.verdicts import rra_verdict
    n = {"a": 300, "b": 200}
    H = _c(8.0, 6.0, 10.0)
    cases = [
        ({"n_recoverable": {"a": 300, "b": 149}, "H": H, "d_main": _c(3, 2, 4), "d_mech": _c(2, 1, 3)}, "INSUFFICIENT_DATA"),
        ({"n_recoverable": n, "H": _c(1.0, 0.5, 1.9), "d_main": _c(3, 2, 4), "d_mech": _c(2, 1, 3)}, "STOP"),
        ({"n_recoverable": n, "H": H, "d_main": _c(0.4, -0.2, 0.9), "d_mech": _c(2, 1, 3)}, "STOP"),
        ({"n_recoverable": n, "H": H, "d_main": _c(3, 2, 4), "d_mech": _c(-0.6, -1.0, -0.1)}, "STOP"),
        ({"n_recoverable": n, "H": H, "d_main": _c(1.5, 0.4, 2.6), "d_mech": _c(0.8, 0.2, 1.4)}, "GO"),
        ({"n_recoverable": n, "H": H, "d_main": _c(1.5, -0.1, 3.1), "d_mech": _c(0.8, 0.2, 1.4)}, "BORDERLINE"),
        ({"n_recoverable": n, "H": H, "d_main": _c(1.5, 0.4, 2.6), "d_mech": _c(0.3, -0.2, 0.8)}, "BORDERLINE"),
    ]
    for inp, want in cases:
        got = rra_verdict(inp, RULE)
        assert got["verdict"] == want, f"{inp} -> {got}, want {want}"
    note = rra_verdict(cases[-1][0], RULE)
    assert "simpler routing" in note.get("note", ""), "BORDERLINE note missing when only Delta_main passes"
    assert "note" not in rra_verdict(cases[-2][0], RULE), "note only when Delta_main passes"
    nan = float("nan")
    und = rra_verdict({"n_recoverable": n, "H": _c(nan, nan, nan), "d_main": _c(nan, nan, nan),
                       "d_mech": _c(nan, nan, nan)}, RULE)
    assert und["verdict"] == "BORDERLINE" and "undefined" in und["reason"]


# ----------------------------------------------------------------------------- budget guard (integrity check 8)
class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _budget(limit=10.5, hard=11.5):
    from rra.budget import Budget
    cfg = {"budget": {"project_limit_hours": limit, "hard_stop_hours": hard, "cut_order": ["G", "E2"]}}
    clk = _Clock()
    return Budget(cfg, None, clock=clk), clk


def test_budget_guard():
    h = 3600.0
    # measured: 10 gen/s, 20 samp/s for model a; elapsed 3.3 h
    b, clk = _budget()
    b.record("a", "gen", 1000, 100)
    b.record("a", "samp", 2000, 100)
    clk.t = 3.3 * h
    opt = {"E2": [("a", "samp", 20 * 40000)], "G": [("g", "gen", 10 * 36000)]}      # 40000 s + 36000 s = 21.1 h
    small = {"E2": [("a", "samp", 20 * 3600)], "G": [("g", "gen", 10 * 7200)]}       # 1 h + 2 h -> 6.3 h total
    assert b.decide("E2", [], small) and b.decide("G", [], small) and not b.cuts, "cut while the projection < 10.5 h"
    # synthetic stage timings that stay under the limit never cut, at any point
    for t in np.linspace(0.1, 7.0, 20):
        clk.t = t * h
        assert b.decide("E2", [], small) and not b.cuts, f"cut at {t:.1f} h with a projection under 10.5 h"
    # over the limit: G is cut first, then E2 only if still over
    b2, clk2 = _budget()
    b2.record("a", "gen", 1000, 100)
    b2.record("a", "samp", 2000, 100)
    clk2.t = 3.3 * h
    mid = {"E2": [("a", "samp", 20 * 3600)], "G": [("g", "gen", 10 * 36000)]}       # 3.3 + 1 + 10 h
    assert b2.decide("E2", [], mid) and b2.cuts == ["G"], f"expected only G cut, got {b2.cuts}"
    b3, clk3 = _budget()
    b3.record("a", "samp", 2000, 100)
    b3.record("a", "gen", 1000, 100)
    clk3.t = 9.0 * h
    assert not b3.decide("E2", [], opt) and set(b3.cuts) == {"G", "E2"}
    # required work is never cut: a required projection over the limit cuts only optional stages
    b4, clk4 = _budget()
    b4.record("a", "gen", 1000, 100)
    clk4.t = 1.0 * h
    assert not b4.decide("G", [("a", "gen", 10 * 11 * 3600)], {"G": [("g", "gen", 100)]}) and b4.cuts == ["G"]
    # rates come only from measurements: an unmeasured model borrows the slowest measured one
    b5, _ = _budget()
    b5.set_probe("a", {"gen": 10.0})
    b5.set_probe("b", {"gen": 5.0})
    assert b5.rate("g", "gen") == (5.0, "slowest other model")
    b5.record("a", "gen", 100, 50)
    assert b5.rate("a", "gen") == (2.0, "measured"), "completed work replaces the probe rate"
    s, miss = b5.seconds([("a", "fit", 1)])
    assert s == 0 and miss == [("a", "fit")], "a unit nobody measured must be reported, not guessed"
    # hard stop: no optional stage starts after 11.5 h
    b6, clk6 = _budget()
    b6.record("a", "gen", 1000, 100)
    clk6.t = 11.6 * h
    assert b6.hard_stop() and not b6.decide("E2", [], small)


# ----------------------------------------------------------------------------- arms and tuning
def _recs(N=80, seed=0, peer=True):
    from rra.arms import Recs
    rng = np.random.default_rng(seed)
    a1_idk = rng.random(N) < 0.15
    a1_lab = np.where(a1_idk, A_, np.where(rng.random(N) < 0.55, C_, W_))
    a2_null = a1_idk | (rng.random(N) < 0.2)
    a2_lab = np.where(a2_null, A_, np.where(rng.random(N) < 0.4, C_, W_))
    s1 = rng.normal(size=N) + (a1_lab == C_)
    s2 = rng.normal(size=N) + (a2_lab == C_)
    dA2 = np.where(a2_null, np.nan, rng.normal(size=N) + (a2_lab == C_))
    r = Recs([f"q{i}" for i in range(N)], [{"qid": f"q{i}", "source": "tqa"} for i in range(N)],
             [f"a{i}" for i in range(N)], a1_lab, a1_lab, a1_idk, s1, s2,
             [None if a1_idk[i] else f"b{i}" for i in range(N)], a2_lab, a2_lab, a2_null,
             ["x" if a2_null[i] else None for i in range(N)], dA2)
    if peer:
        pidk = rng.random(N) < 0.1
        plab = np.where(pidk, A_, np.where(rng.random(N) < 0.5, C_, W_))
        r.set_peer([f"p{i}" for i in range(N)], plab, plab, pidk, rng.normal(size=N) + (plab == C_))
    r.set_platt(1 / (1 + np.exp(-s1)), 1 / (1 + np.exp(-s2)))
    return r


def test_arms_rules():
    from rra.arms import outputs
    r = _recs()
    ans = r.answered
    t = {"t1": 0.5, "t2": 0.2, "t3": 0.1, "tb": -0.3, "tp": 0.0}
    for i in range(r.N):
        exp = {}
        if not ans[i]:
            exp = {k: A_ for k in ("B3", "RRA", "R-check", "R-band", "R-all", "RRA+check", "Oracle")}
        else:
            keep = r.s1[i] >= t["t1"]
            a2 = A_ if r.a2_null[i] else r.a2_lab[i]
            exp["B3"] = r.a1_lab[i] if keep else A_
            exp["R-all"] = r.a1_lab[i] if keep else a2
            exp["R-band"] = r.a1_lab[i] if keep else (a2 if r.s1[i] >= t["tb"] else A_)
            exp["RRA"] = r.a1_lab[i] if keep else (a2 if r.s2[i] >= t["t2"] else A_)
            chk = (not r.a2_null[i]) and r.dA2[i] >= t["t3"]
            exp["R-check"] = r.a1_lab[i] if keep else (a2 if chk else A_)
            exp["RRA+check"] = r.a1_lab[i] if keep else (a2 if (r.s2[i] >= t["t2"] and chk) else A_)
            exp["Oracle"] = C_ if (r.a1_lab[i] == C_ or (not r.a2_null[i] and r.a2_lab[i] == C_)) else A_
        for arm, want in exp.items():
            got = outputs(r, arm, t).label[i]
            assert got == want, f"{arm} q{i}: got {got}, want {want}"
    o = outputs(r, "RRA", t)
    fl = ans & ~(r.s1 >= t["t1"]) & (r.s2 >= t["t2"])
    assert np.array_equal(o.routed, fl), "RRA retries exactly where s1 < t1 and s2 >= t2"
    assert np.allclose(o.fpe, 1.0 + fl), "RRA FPE = 1 + 1 per routed retry"
    oc = outputs(r, "R-check", t)
    assert np.allclose(oc.fpe, 1.0 + (ans & ~(r.s1 >= t["t1"])) * 1.0 + (ans & ~(r.s1 >= t["t1"]) & ~r.a2_null) * 0.5)
    # RRA-EU: keep / retry / IDK by expected utility
    oe = outputs(r, "RRA-EU", {}, lam=1)
    for i in np.flatnonzero(ans):
        eu = [2 * r.p1[i] - 1, 2 * r.p2[i] - 1, 0.0]
        k = int(np.argmax(eu))
        want = r.a1_lab[i] if k == 0 else ((A_ if r.a2_null[i] else r.a2_lab[i]) if k == 1 else A_)
        assert oe.label[i] == want


def test_tuning_matches_brute_force():
    from rra.arms import grids, outputs, tune, utility
    for seed in (1, 2):
        r = _recs(N=70, seed=seed)
        step = 0.25
        T = tune(r, [1, 3], step)
        g = grids(r, step)
        axes = {"B3": ["t1"], "R-all": ["t1"], "R-band": ["t1", "tb"], "R-check": ["t1", "t3"], "RRA": ["t1", "t2"],
                "RRA+check": ["t1", "t2", "t3"], "RRA-peer": ["t1", "tp"]}
        for arm, ax in axes.items():
            for lam in (1, 3):
                best, bt = None, None
                for combo in itertools.product(*[g[a] for a in ax]):
                    th = dict(zip(ax, combo))
                    u = utility(outputs(r, arm, th, lam).label, lam)
                    if best is None or u > best + 1e-9:
                        best, bt = u, th
                got = T[arm][f"lam{lam}"]
                assert abs(got["U_tune"] - best) < 1e-9, f"{arm} lam{lam}: tune {got['U_tune']} vs brute {best}"
                assert all(got[k] == bt[k] for k in ax), f"{arm} lam{lam}: thresholds {got} vs brute {bt}"
                assert abs(utility(outputs(r, arm, got, lam).label, lam) - got["U_tune"]) < 1e-9


def test_null_reason():
    from rra.arms import null_reason
    assert null_reason("Paris", "Lyon") is None
    assert null_reason("Paris", "I don't know.") == "idk"
    assert null_reason("Paris", "!!! ???") == "garbled"
    assert null_reason("Paris", "Lyon or Marseille") == "hedged"
    assert null_reason("Paris", "paris.") == "same_as_A1"
    assert null_reason("Paris", "Paris, France") == "contains_A1"
    assert null_reason("New York City", "York") == "contains_A1"
    assert null_reason("Paris", "Parisian") is None, "containment is whole words only"
    assert null_reason("Paris", None) == "not_generated"


def test_platt():
    from rra.probes import calibration_curve, platt_apply, platt_fit
    rng = np.random.default_rng(0)
    s = rng.normal(size=4000) * 2 + 1
    y = rng.random(4000) < 1 / (1 + np.exp(-(1.5 * s - 0.7)))
    p = platt_fit(s, y)
    assert abs(p["a"] - 1.5) < 0.15 and abs(p["b"] + 0.7) < 0.15, p
    cal = calibration_curve(platt_apply(p, s), y)
    assert max(abs(c["p_mean"] - c["y_rate"]) for c in cal) < 0.08
    assert platt_fit([0.1, 0.2], [1, 1])["a"] == 0.0, "one-class sample: constant fit"


def test_R_recipe():
    """R = the SQ2 recipe (vesecs.detectors.fit_family with the 'both / before / after' moments)."""
    import torch
    from rra.probes import fit_R
    g = torch.Generator().manual_seed(0)
    n, nL, d = 150, 3, 16
    Rp, Ra = torch.randn(n, nL, d, generator=g), torch.randn(n, nL, d, generator=g)
    y = (Ra[:, 1, 0] + 0.3 * torch.randn(n, generator=g) > 0).numpy()
    ans = np.ones(n, bool)
    ans[:10] = False
    folds = np.arange(n) % 5
    probe, oof, infos = fit_R(Rp.half(), Ra.half(), y, ans, folds, [4, 8, 12])
    assert probe.layer == 8 and probe.moment in ("answer", "both"), (probe.layer, probe.moment)
    assert np.isnan(oof[:10]).all() and np.isfinite(oof[10:]).all() and len(infos) == 5


TESTS = [test_verdicts, test_budget_guard, test_arms_rules, test_tuning_matches_brute_force, test_null_reason,
         test_platt, test_R_recipe]


def run_all(verbose=True):
    failures, n_pass = [], 0
    passed = set()
    for t in TESTS:
        try:
            t()
            n_pass += 1
            passed.add(t.__name__)
            if verbose:
                print(f"PASS {t.__name__}")
        except Exception as e:
            failures.append(f"{t.__name__}: {type(e).__name__}: {e}")
            if verbose:
                traceback.print_exc()
                print(f"FAIL {t.__name__}")
    return {"n_tests": len(TESTS), "n_pass": n_pass, "failures": failures,
            "verdict_tests_pass": "test_verdicts" in passed, "budget_tests_pass": "test_budget_guard" in passed}


if __name__ == "__main__":
    r = run_all()
    print(f"{r['n_pass']}/{r['n_tests']} passed")
    raise SystemExit(0 if not r["failures"] else 1)

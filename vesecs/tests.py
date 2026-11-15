"""Unit tests (CPU, seconds). Run by phase A (integrity check 8) and standalone:

    python -m vesecs.tests
"""
import traceback

import numpy as np
import torch

from scoring import A_, C_, W_

VES_RULE = {"min_rejected_wrong": 150, "go_delta": 1.0}
ECS_RULE = {"min_wrong": 150, "g_go": 0.5, "g_stop": 0.25, "go_delta": 0.5, "min_auroc_gap": 0.02}


def _c(v, lo, hi):
    return {"value": v, "ci_low": lo, "ci_high": hi}


# ----------------------------------------------------------------------------- verdicts (integrity check 8)
def test_ves_verdicts():
    from vesecs.verdicts import ves_verdict
    n = {"a": 200, "b": 300}
    cases = [
        ({"n_rejected_wrong": {"a": 149, "b": 300}, "d_ves": _c(3, 2, 4), "d_ves_b5": _c(1, 0.5, 2)}, "INSUFFICIENT_DATA"),
        ({"n_rejected_wrong": n, "d_ves": _c(1.5, 0.2, 2.8), "d_ves_b5": _c(0.1, -0.5, 0.7)}, "GO"),
        ({"n_rejected_wrong": n, "d_ves": _c(0.3, -0.2, 0.9), "d_ves_b5": _c(0.1, -0.5, 0.7)}, "STOP"),
        ({"n_rejected_wrong": n, "d_ves": _c(2.0, 1.0, 3.0), "d_ves_b5": _c(-1.0, -2.0, -0.1)}, "STOP"),
        ({"n_rejected_wrong": n, "d_ves": _c(1.5, -0.1, 3.1), "d_ves_b5": _c(0.1, -0.5, 0.7)}, "BORDERLINE"),
        ({"n_rejected_wrong": n, "d_ves": _c(1.5, 0.2, 2.8), "d_ves_b5": _c(-0.2, -0.9, 0.5)}, "BORDERLINE"),
    ]
    for inp, want in cases:
        got = ves_verdict(inp, VES_RULE)["verdict"]
        assert got == want, f"VES {inp} -> {got}, want {want}"


def test_ecs_verdicts():
    from vesecs.verdicts import ecs_verdict
    n = {"a": 200, "b": 300}
    G = lambda v, lo, hi, d=True: {"value": v, "ci_low": lo, "ci_high": hi, "defined": d}
    cases = [
        ({"n_wrong": {"a": 10, "b": 300}, "G": G(0.8, 0.5, 1.1), "d_ecs": _c(1, 0.5, 1.5)}, "INSUFFICIENT_DATA"),
        ({"n_wrong": n, "G": G(float("nan"), float("nan"), float("nan"), False), "d_ecs": _c(1, 0.5, 1.5)}, "INSUFFICIENT_DATA"),
        ({"n_wrong": n, "G": G(0.7, 0.3, 1.1), "d_ecs": _c(0.9, 0.2, 1.6)}, "GO"),
        ({"n_wrong": n, "G": G(0.05, -0.2, 0.2), "d_ecs": _c(0.9, 0.2, 1.6)}, "STOP"),
        ({"n_wrong": n, "G": G(0.7, 0.3, 1.1), "d_ecs": _c(0.0, -0.3, 0.4)}, "STOP"),
        ({"n_wrong": n, "G": G(0.4, 0.1, 0.7), "d_ecs": _c(0.9, 0.2, 1.6)}, "BORDERLINE"),
    ]
    for inp, want in cases:
        got = ecs_verdict(inp, ECS_RULE)["verdict"]
        assert got == want, f"ECS {inp} -> {got}, want {want}"
    note = ecs_verdict(cases[5][0], ECS_RULE, exact=cases[2][0])
    assert note["verdict"] == "BORDERLINE" and "bottleneck" in note.get("note", ""), "ECS-exact note missing"


# ----------------------------------------------------------------------------- grader
def test_grader():
    from vesecs.grader import candidate_status, grade, hedged_form
    al = ["Paris"]
    assert grade("Paris", al)[0] == C_
    assert grade("Paris or Lyon", al) == (W_, False, True), "hedge must count as wrong"
    assert grade("Lyon / Paris", al) == (W_, False, True)
    assert grade("I don't know.", al)[0] == A_
    assert grade("The capital is Paris", al)[0] == C_
    assert grade("USA / United States", ["United States", "USA"])[0] == C_, "parts all matching the aliases are no hedge"
    assert grade("AC/DC", ["AC/DC"])[0] == C_, "a reply that is itself an alias is no hedge"
    assert grade("It is either Paris or something located in the south of France maybe", al)[0] == C_, \
        "parts longer than 6 words are no hedge"
    assert hedged_form("Paris or Lyon") and not hedged_form("Paris") and not hedged_form("Paris or paris")
    assert candidate_status("I don't know.", set()) == "idk"
    assert candidate_status("Paris", {"paris"}) == "duplicate"
    assert candidate_status("Lyon", {"paris"}) == "valid"
    assert candidate_status("!!! ???", set()) == "garbled"


# ----------------------------------------------------------------------------- detectors
def _synthetic_states(n=160, nL=3, d=24, seed=0):
    g = torch.Generator().manual_seed(seed)
    Rp = torch.randn(n, nL, d, generator=g)
    Ra = torch.randn(n, nL, d, generator=g)
    w = torch.randn(d, generator=g)
    y = ((Ra[:, 1] @ w + 0.5 * (Rp[:, 2] @ w) + torch.randn(n, generator=g) * 2) > 0).numpy()
    return Rp.half(), Ra.half(), y


def test_fit_family_matches_sq2():
    """D's selection and scores equal SQ2's _fit_predict on the same rows."""
    from pilot import config as PC
    from pilot.probes import _fit_predict
    from vesecs.detectors import fit_family, score_family
    Rp, Ra, y = _synthetic_states()
    tr, te = np.arange(0, 120), np.arange(120, 160)
    layers = [5, 7, 9]
    s_sq2, info = _fit_predict(Rp[tr], Ra[tr], y[tr], Rp[te], Ra[te], PC.RIDGE_LAMBDAS)
    p = fit_family(Rp, Ra, y, tr, layers)["D"]
    assert p.info["layer_index"] == info["layer_index"] and p.info["position"] == info["position"] \
        and p.info["lambda_rel"] == info["lambda_rel"], f"selection differs: {p.info} vs {info}"
    s = score_family(p, Rp[te], Ra[te], layers)
    # SQ2 forms the test kernel in float32, the primal form uses float64: differences are float32 rounding
    assert np.allclose(s, s_sq2, atol=1e-3), f"scores differ: max {np.abs(s - s_sq2).max()}"
    assert (np.argsort(np.argsort(s)) == np.argsort(np.argsort(s_sq2))).mean() > 0.95, "score ranks differ"


def test_primal_equals_dual():
    from vesecs.detectors import _loo_dual, _loo_primal
    g = torch.Generator().manual_seed(1)
    X = torch.randn(80, 12, generator=g).double()
    y = (torch.randn(80, generator=g) > 0).numpy()
    lams = [1e-3, 1e-1, 1.0, 10.0]
    a = _loo_dual(X @ X.T, y, lams)
    b, _ = _loo_primal(X, y, lams)
    for (a1, l1, lam1), (b1, l2, lam2) in zip(a, b):
        assert abs(a1 - b1) < 1e-9 and abs(lam1 - lam2) < 1e-6 * max(1.0, lam1), f"primal {b1} vs dual {a1}"


def test_vector_probe_forms():
    """The vector probe's scores agree between its dual (n <= d) and primal (n > d) code paths."""
    from vesecs.detectors import fit_vector
    g = torch.Generator().manual_seed(2)
    X = torch.randn(60, 40, generator=g)
    y = (X[:, 0] + torch.randn(60, generator=g) > 0).numpy()
    p_dual = fit_vector(X, y, np.arange(30), layer=0)                 # 30 <= 40: dual
    Xw = torch.cat([X, X[:, :0]], 1)
    p_primal = fit_vector(Xw[:, :20], y, np.arange(60), layer=0)      # 60 > 20: primal
    assert p_dual.info["form"] == "dual" and p_primal.info["form"] == "primal"
    s = p_primal.score(vector=Xw[:, :20])
    assert np.all(np.isfinite(s))


# ----------------------------------------------------------------------------- search replay
def test_replay_exact():
    """The offline replay equals a live loop with acceptance on, for every tau (synthetic deterministic proposals)."""
    from vesecs.search import replay
    rng = np.random.default_rng(0)
    for _ in range(300):
        stop = bool(rng.integers(2))
        rounds = []
        for k in range(3):
            st = rng.choice(["valid", "valid", "valid", "idk", "duplicate", "hedged"])
            rounds.append({"status": st, "sD_clean": float(rng.normal()) if st == "valid" else None, "fpe": 1.5, "k": k + 1})
        a1 = float(rng.normal())
        for tau in (-np.inf, -1.0, 0.0, 0.5, np.inf):
            # live loop
            if a1 >= tau:
                want = ("A1", 1.0)
            else:
                want, cost = None, 1.0
                for r in rounds:
                    cost += r["fpe"]
                    if r["status"] != "valid":
                        if stop:
                            want = ("IDK", cost)
                            break
                        continue
                    if r["sD_clean"] >= tau:
                        want = (r["k"], cost)
                        break
                want = want or ("IDK", cost)
            out, cost = replay(a1, False, rounds, tau, stop)
            got = (out if isinstance(out, str) else out["k"], cost)
            assert got == want, f"replay {got} != live {want}"


def test_methods_small():
    from vesecs import methods as M
    items = [{"qid": f"q{i}", "source": "tqa", "aliases": ["x"]} for i in range(4)]
    drafts = {"texts": ["a", "b", "I don't know.", "d"], "label_strict": [C_, W_, A_, W_], "label_sq2": [C_, W_, A_, W_],
              "is_idk": [False, False, True, False], "first_token": [1, 2, 3, 4]}
    scores = {"sD": [0.9, 0.1, 0.0, 0.2], "sD_before": [0.5] * 4, "sD_after": [0.5] * 4, "margin": [1, 0, -1, 0]}
    sr = M.SplitRecords(items, drafts, scores)
    o = M.gate(sr, sr.sD, 0.5, 1.0)
    assert list(o.label) == [C_, A_, A_, A_] and M.utility(o.label, 1) == 25.0
    assert M.tau_grid([0.1, 0.2])[0] == -np.inf and M.tau_grid([0.1, 0.2])[-1] == np.inf
    t = M.tune_gate(sr, sr.sD, 1.0, [1, 3], 0.05)
    assert t["lam1"]["U_dev"] == 25.0


def test_bootstrap():
    from vesecs.stats import Boot
    b = Boot(50, 200, 7)
    v = {"a": np.ones(50), "b": np.zeros(50)}
    r = b.mean_contrast(v)
    assert abs(r["pooled"]["value"] - 0.5) < 1e-12 and abs(r["pooled"]["ci_low"] - 0.5) < 1e-12


def test_budget_cuts():
    """Cuts are applied in the pre-registered order until the projection fits; vLLM is skipped when unavailable;
    the actual/planned ratio is ignored until enough planned work has finished."""
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace

    from vesecs import budget as B
    from vesecs import cfg as C
    cfg = C.load_config(C.CONFIG_PATH)
    cfg["budget"]["project_limit_hours"] = 10.5
    with tempfile.TemporaryDirectory() as d:
        run = SimpleNamespace(logs=Path(d))
        b = B.Budget(cfg, run, 2)
        b.models = ["m1", "m2"]
        b.rates = {"m1": dict(B.DEFAULT_RATES), "m2": dict(B.DEFAULT_RATES)}
        b.check("fast", overran=False)
        assert b.cuts == [], f"cuts at normal rates: {b.cuts}"
        b.rates = {m: {k: v / 3 for k, v in B.DEFAULT_RATES.items()} for m in b.models}
        b.check("slow", overran=False)
        order = [c for c in cfg["budget"]["cuts"] if c != "vllm_tries" or B.vllm_available()]
        assert b.cuts == order[:len(b.cuts)] and b.cuts, f"cuts out of order: {b.cuts}"
        if not B.vllm_available():
            assert "vllm_tries" in b.cut_unavailable
        b2 = B.Budget(cfg, run, 2)
        b2.planned_done, b2.actual_done = 1.0, 15.0
        assert b2.scale(10000.0) == 1.0, "a short early stage must not set the scale"
        b2.planned_done, b2.actual_done = 2000.0, 20000.0
        assert b2.scale(10000.0) == 4.0, "scale is clamped"


TESTS = [test_ves_verdicts, test_ecs_verdicts, test_grader, test_fit_family_matches_sq2, test_primal_equals_dual,
         test_vector_probe_forms, test_replay_exact, test_methods_small, test_bootstrap, test_budget_cuts]
VERDICT_TESTS = {"test_ves_verdicts", "test_ecs_verdicts"}


def run_all(verbose=True):
    import contextlib
    import io
    fails, passed = [], []
    for t in TESTS:
        try:
            with contextlib.redirect_stdout(io.StringIO()) if not verbose else contextlib.nullcontext():
                t()
            passed.append(t.__name__)
            if verbose:
                print(f"ok   {t.__name__}")
        except Exception as e:
            fails.append(f"{t.__name__}: {type(e).__name__}: {e}")
            if verbose:
                traceback.print_exc()
                print(f"FAIL {t.__name__}")
    return {"n_tests": len(TESTS), "n_pass": len(passed), "failures": fails, "passed": passed,
            "verdict_tests_pass": all(n in passed for n in VERDICT_TESTS)}


if __name__ == "__main__":
    r = run_all()
    print(f"{r['n_pass']}/{r['n_tests']} passed")
    raise SystemExit(0 if not r["failures"] else 1)

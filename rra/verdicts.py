"""RRA decision rule (spec section 8), a pure function of the saved metrics. Conditions are checked in the order
of the spec's table: INSUFFICIENT_DATA, STOP, GO, BORDERLINE. Pooled values on the fresh test.
Units: 1.0 point of U_1 = one more (right - wrong) answer per 100 questions.

  INSUFFICIENT_DATA  for either model, fewer than 150 fresh-test questions where A1 is wrong and A2 is right
  STOP               CI upper bound of H < 2.0, or CI upper bound of Delta_main < 1.0, or CI upper bound of
                     Delta_mech < 0
  GO                 Delta_main >= 1.0 with CI lower bound > 0, and Delta_mech >= 0.5 with CI lower bound > 0
  BORDERLINE         otherwise; with a note when Delta_main passes but Delta_mech does not (the gain then comes
                     from simpler routing, not from the recoverability probe)
"""
import numpy as np


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def rra_verdict(inp, rule):
    """inp: n_recoverable {model: int}; H, d_main, d_mech: {value, ci_low, ci_high} (pooled)."""
    nmin = min(inp["n_recoverable"].values()) if inp["n_recoverable"] else 0
    H, Hhi = _f(inp["H"]["value"]), _f(inp["H"]["ci_high"])
    dm, dlo, dhi = _f(inp["d_main"]["value"]), _f(inp["d_main"]["ci_low"]), _f(inp["d_main"]["ci_high"])
    dx, xlo, xhi = _f(inp["d_mech"]["value"]), _f(inp["d_mech"]["ci_low"]), _f(inp["d_mech"]["ci_high"])
    h_stop, main_go, mech_go = float(rule["h_stop"]), float(rule["main_go"]), float(rule["mech_go"])
    if nmin < int(rule["min_recoverable"]):
        return {"verdict": "INSUFFICIENT_DATA",
                "reason": f"a model has only {nmin} fresh-test questions where A1 is wrong and A2 is right "
                          f"(< {rule['min_recoverable']})"}
    stop = []
    if Hhi < h_stop:
        stop.append(f"CI upper bound of H {Hhi:.2f} < {h_stop} (nothing to recover)")
    if dhi < main_go:
        stop.append(f"CI upper bound of Delta_main {dhi:.2f} < {main_go}")
    if xhi < 0:
        stop.append(f"CI upper bound of Delta_mech {xhi:.2f} < 0")
    if stop:
        return {"verdict": "STOP", "reason": "; ".join(stop)}
    main_ok = dm >= main_go and dlo > 0
    mech_ok = dx >= mech_go and xlo > 0
    if main_ok and mech_ok:
        return {"verdict": "GO", "reason": f"Delta_main = {dm:.2f} (CI {dlo:.2f} to {dhi:.2f}) and Delta_mech = "
                                           f"{dx:.2f} (CI {xlo:.2f} to {xhi:.2f})"}
    out = {"verdict": "BORDERLINE",
           "reason": (f"H = {H:.2f} (CI upper {Hhi:.2f}); Delta_main = {dm:.2f} (CI {dlo:.2f} to {dhi:.2f}); "
                      f"Delta_mech = {dx:.2f} (CI {xlo:.2f} to {xhi:.2f})")
           if all(np.isfinite([dm, dlo, dhi, dx, xlo, xhi])) else "a contrast or its CI is undefined"}
    if main_ok and not mech_ok:
        out["note"] = ("Delta_main passes but Delta_mech does not: the gain comes from simpler routing, not from the "
                       "recoverability probe")
    return out

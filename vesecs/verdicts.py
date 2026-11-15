"""Decision rules (spec section 9), as pure functions of the saved metrics. Conditions are checked in the order
INSUFFICIENT_DATA, GO, STOP, BORDERLINE. Units: 1.0 point of U_1 = one more (right - wrong) answer per 100
questions."""
import numpy as np


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return x


def ves_verdict(inp, rule):
    """inp: n_rejected_wrong {model: int}; d_ves {value, ci_low, ci_high}; d_ves_b5 {value, ci_low, ci_high}."""
    nmin = min(inp["n_rejected_wrong"].values()) if inp["n_rejected_wrong"] else 0
    d, lo, hi = _f(inp["d_ves"]["value"]), _f(inp["d_ves"]["ci_low"]), _f(inp["d_ves"]["ci_high"])
    d5, hi5 = _f(inp["d_ves_b5"]["value"]), _f(inp["d_ves_b5"]["ci_high"])
    go = float(rule["go_delta"])
    if nmin < int(rule["min_rejected_wrong"]):
        return {"verdict": "INSUFFICIENT_DATA",
                "reason": f"a model has only {nmin} test drafts that are answered, wrong and rejected at the frozen "
                          f"tau (< {rule['min_rejected_wrong']})"}
    if d >= go and lo > 0 and d5 >= 0:
        return {"verdict": "GO", "reason": f"Delta_VES = {d:.2f} (CI {lo:.2f} to {hi:.2f}) >= {go} with CI above 0, "
                                           f"and VES >= B5 at matched compute ({d5:+.2f})"}
    if hi < go or hi5 < 0:
        why = []
        if hi < go:
            why.append(f"CI upper bound of Delta_VES {hi:.2f} < {go}")
        if hi5 < 0:
            why.append(f"CI upper bound of Delta_VES,B5 {hi5:.2f} < 0 (the value comes from searching, not steering)")
        return {"verdict": "STOP", "reason": "; ".join(why)}
    if not all(np.isfinite([d, lo, hi, d5, hi5])):
        return {"verdict": "BORDERLINE", "reason": "a contrast or its CI is undefined"}
    return {"verdict": "BORDERLINE", "reason": f"Delta_VES = {d:.2f} (CI {lo:.2f} to {hi:.2f}); "
                                               f"Delta_VES,B5 = {d5:.2f} (CI upper {hi5:.2f})"}


def ecs_verdict(inp, rule, exact=None):
    """inp: n_wrong {model: int}; G {value, ci_low, ci_high, defined}; d_ecs {value, ci_low, ci_high}.
    exact: the same inputs for ECS-exact (for the bottleneck note)."""
    out = _ecs_rule(inp, rule)
    if exact is not None and out["verdict"] != "GO" and _ecs_rule(exact, rule)["verdict"] == "GO":
        out["note"] = ("ECS-exact would pass GO while ECS-1P does not: the echo approximation, not the idea, "
                       "is the bottleneck")
    return out


def _ecs_rule(inp, rule):
    nmin = min(inp["n_wrong"].values()) if inp["n_wrong"] else 0
    G, glo, ghi = _f(inp["G"]["value"]), _f(inp["G"]["ci_low"]), _f(inp["G"]["ci_high"])
    d, lo, hi = _f(inp["d_ecs"]["value"]), _f(inp["d_ecs"]["ci_low"]), _f(inp["d_ecs"]["ci_high"])
    if nmin < int(rule["min_wrong"]):
        return {"verdict": "INSUFFICIENT_DATA", "reason": f"a model has only {nmin} answered wrong test drafts "
                                                          f"(< {rule['min_wrong']})"}
    if not inp["G"].get("defined", np.isfinite(G)):
        return {"verdict": "INSUFFICIENT_DATA", "reason": f"G undefined (A_aft - A_bef < {rule['min_auroc_gap']} "
                                                          "for a model)"}
    if G >= rule["g_go"] and glo > 0 and d >= rule["go_delta"] and lo > 0:
        return {"verdict": "GO", "reason": f"G = {G:.2f} (CI {glo:.2f} to {ghi:.2f}); Delta_ECS = {d:.2f} "
                                           f"(CI {lo:.2f} to {hi:.2f})"}
    if ghi < rule["g_stop"] or hi < rule["go_delta"]:
        why = []
        if ghi < rule["g_stop"]:
            why.append(f"CI upper bound of G {ghi:.2f} < {rule['g_stop']}")
        if hi < rule["go_delta"]:
            why.append(f"CI upper bound of Delta_ECS {hi:.2f} < {rule['go_delta']}")
        return {"verdict": "STOP", "reason": "; ".join(why)}
    return {"verdict": "BORDERLINE", "reason": f"G = {G:.2f} (CI {glo:.2f} to {ghi:.2f}); Delta_ECS = {d:.2f} "
                                               f"(CI {lo:.2f} to {hi:.2f})"}

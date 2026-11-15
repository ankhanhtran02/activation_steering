"""Integrity checks (spec section 10). Every check writes {pass, ...details} into results/integrity.json; the test
phase refuses to start unless all required checks passed (and frozen_config.json exists).

  1 isolation        vesecs.data.isolation_report (phase A)
  2 pairing          drafts are written once (a stage never overwrites them) and every method reads them; the
                     drafts' SHA-256 is registered at creation and re-checked before the test phase and in phase F
  3 zero_steering    VES with alpha = 0 reproduces A1 exactly (first dev batch, same batch composition as the
                     drafts; at least 50 questions)
  4 hook_reset       after the steered calls, an unsteered pass reproduces the stored before-states (max abs
                     difference < 1e-2; first dev batch, same composition)
  5 gradient         directional derivative g . u vs (L(eps u) - L(0)) / eps within 10%, random unit u,
                     eps = 1e-2 rho, 3 questions. A forward difference at this eps carries a curvature term that
                     can exceed 10% for a correct gradient (g . u is small for a random direction; measured on
                     the smoke model in float32), so the check records, in order: the forward difference in the
                     model's dtype (the spec's test), then in float32, then a central difference
                     (L(eps u) - L(-eps u)) / (2 eps) in float32. It passes at the first step that is within
                     tolerance for all questions; the step is recorded.
  6 identity_echo    an ECS hook that writes position T+1's own state back reproduces ECS-exact
    echo_equivalence (added) the KV-cache echo equals the spec's full-forward simulation (min cosine)
  7 shuffled_labels  D and E_1P trained on shuffled pool labels: cross-fitted AUROC in [0.45, 0.55]
  8 verdict_code     vesecs.tests: synthetic inputs give each of the four verdicts for each method
  9 freeze           the test phase needs configs/frozen_config.json; its SHA-256 is logged
"""
import numpy as np

from vesecs.io import read_json, write_json

REQUIRED = ["isolation", "verdict_code", "zero_steering", "hook_reset", "gradient", "identity_echo",
            "echo_equivalence", "shuffled_labels_D", "shuffled_labels_E1P", "pairing"]
PER_MODEL = {"zero_steering", "hook_reset", "gradient", "identity_echo", "echo_equivalence", "shuffled_labels_D",
             "shuffled_labels_E1P", "pairing"}


class Integrity:
    def __init__(self, path):
        self.path = path
        self.d = read_json(path) if path.exists() else {}

    def record(self, name, model, result):
        self.d.setdefault(name, {})[model or "all"] = result
        write_json(self.path, self.d, indent=1)

    def status(self, models):
        """(all_passed, list of failures / missing)."""
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


def zero_steering(eng, prompts, drafts, layer, n_min):
    n = min(eng.bs, len(prompts))
    g = eng.generate(prompts[:n], add=(layer, np.zeros((n, eng.d), np.float32)))
    match = [list(a) == list(b) for a, b in zip(g.token_ids, drafts["token_ids"][:n])]
    return {"pass": bool(all(match) and n >= min(n_min, len(prompts))), "n_compared": n, "n_match": int(sum(match)),
            "layer": layer}


def hook_reset(eng, prompts, conts, stored, layers, tol):
    n = min(eng.bs, len(prompts))
    o = eng.tf(prompts[:n], conts[:n], layers=layers)
    diff = float((o["res_prompt"].float() - stored[:n].float()).abs().max()) if n else 0.0
    return {"pass": bool(diff < tol), "max_abs_diff": diff, "n": n}


def gradient(eng, prompts, conts, layer, rho, eps_rel, tol, seed=0):
    d = eng.d
    u = np.random.default_rng(seed).normal(size=d)
    u = (u / np.linalg.norm(u)).astype(np.float32)
    eps = float(eps_rel) * float(rho)
    n = len(prompts)
    U = np.tile(eps * u, (n, 1))
    steps = []

    def run(central):
        g = eng.pull_vectors(prompts, conts, [layer])[layer]
        gu = g @ u
        L0 = eng.logprob_with_delta(prompts, conts, layer, np.zeros((n, d), np.float32))
        Lp = eng.logprob_with_delta(prompts, conts, layer, U)
        fd = ((Lp - eng.logprob_with_delta(prompts, conts, layer, -U)) / (2 * eps)) if central else (Lp - L0) / eps
        rel = np.abs(fd - gu) / np.maximum(np.abs(gu), 1e-12)
        return {"g_dot_u": gu.tolist(), "finite_difference": fd.tolist(), "rel_err": rel.tolist(),
                "pass": bool(np.all(rel <= tol))}

    dtype = str(next(eng.model.parameters()).dtype)
    r = run(False)
    steps.append({"step": f"forward difference ({dtype})", **r})
    if not r["pass"]:
        with eng.float32():
            r = run(False)
            steps.append({"step": "forward difference (float32)", **r})
            if not r["pass"]:
                r = run(True)
                steps.append({"step": "central difference (float32)", **r})
    return {"pass": steps[-1]["pass"], "passed_at": steps[-1]["step"] if steps[-1]["pass"] else None,
            "spec_forward_difference_pass": steps[0]["pass"], "eps": eps, "layer": layer, "steps": steps}


def identity_echo(eng, prompts, first, a, b, allowed, t_idk, k, rho_a):
    _, st = eng.echo(prompts, first, a, b, allowed, t_idk, k, rho_a, [("exact", None), ("identity", None)])
    x, y = st[("exact", None)].astype(np.float32), st[("identity", None)].astype(np.float32)
    diff = float(np.abs(x - y).max()) if len(x) else 0.0
    return {"pass": bool(diff <= 1e-3), "max_abs_diff": diff, "n_rows": int(len(x))}


def echo_equivalence(eng, prompts, first, a, b, allowed, t_idk, k, rho_a, beta, min_cos):
    q, st = eng.echo(prompts, first, a, b, allowed, t_idk, k, rho_a, [("1p", beta)])
    kv = st[("1p", beta)].astype(np.float32)
    rows = [(i, c) for i, r in enumerate(q) for c in r["slots"]]
    emb = eng.embed_unit([c for _, c in rows])
    state = {"n": 0}

    def fn(hT):
        n0, B = state["n"], hT.shape[0]
        state["n"] += B
        return hT.float() + float(beta) * float(rho_a) * emb[n0:n0 + B].to(hT.device)

    full = eng.echo_full([prompts[i] for i, _ in rows], [c for _, c in rows], a, b, fn)
    cos = (full * kv).sum(1) / np.maximum(np.linalg.norm(full, axis=1) * np.linalg.norm(kv, axis=1), 1e-12)
    return {"pass": bool(len(cos) > 0 and cos.min() >= min_cos), "min_cos": float(cos.min()) if len(cos) else None,
            "max_abs_diff": float(np.abs(full - kv).max()) if len(cos) else None, "n_rows": int(len(cos))}

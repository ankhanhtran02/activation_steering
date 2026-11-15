"""results/results_report.tex, in the SQ2 report's structure (abstract with both verdicts; setup as built; main
table; contrasts; strata; diagnostics; integrity checks; limitations; reproducing). Generated from summary.json."""
import math

LABELS = {"B0": "B0 Greedy", "B1": "B1 Confidence gate", "B3": "B3 Detector gate", "B3b": "B3b Before-only gate",
          "B5@1": "B5 Resampling, $N{=}1$", "B5@2": "B5 Resampling, $N{=}2$", "B5@4": "B5 Resampling, $N{=}4$",
          "B5@8": "B5 Resampling, $N{=}8$", "B5@Nm": "B5 Resampling, $N_{\\mathrm{matched}}$", "B7": "B7 Token-ban search",
          "VES": "\\textbf{VES}", "VES_free": "VES-free", "VES_placebo": "VES-placebo", "VES_unclean": "VES-unclean",
          "ECS_1p": "\\textbf{ECS-1P}", "ECS_exact": "ECS-exact", "ECS_placebo": "ECS-placebo",
          "R1": "R1 Judge A gate", "R2": "R2 Judge B gate", "R3": "R3 Majority vote"}


def esc(s):
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                 ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def f(x, nd=1, sign=False):
    if isinstance(x, float) and math.isinf(x):
        return r"$\infty$" if x > 0 else r"$-\infty$"
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ci(d, nd=2, sign=True):
    if not d:
        return "--"
    return f"{f(d.get('value'), nd, sign)} [{f(d.get('ci_low'), nd, sign)}, {f(d.get('ci_high'), nd, sign)}]"


def short(m):
    return {"llama31_8b_instruct": "Llama", "qwen25_7b_instruct": "Qwen"}.get(m, esc(m))


def write_report(S, path):
    models = S.get("models", [])
    V = S.get("verdicts", {})
    L = []
    a = L.append
    a(r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.4cm]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage{float}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\setlist{itemsep=2pt, topsep=4pt}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcommand{\IDK}{\textsc{idk}}
\title{Do VES and ECS Reduce Hallucination?\\[2pt]\large Results}
\author{Khanh Tran}
\date{\today}
\begin{document}
\maketitle
""")
    vv, ve = V.get("VES", {}), V.get("ECS", {})
    con = S.get("contrasts", {})
    a(r"\begin{abstract}\noindent")
    a(f"This report gives the results of the experiment specified in \\texttt{{ves-ces-experiment.tex}} on "
      f"{len(models)} model(s) and {S.get('n_test', 0)} test questions. "
      f"\\textbf{{VES verdict: {esc(vv.get('verdict', 'NOT RUN'))}}} ({esc(vv.get('reason', ''))}). "
      f"\\textbf{{ECS verdict: {esc(ve.get('verdict', 'NOT RUN'))}}} ({esc(ve.get('reason', ''))})."
      + (f" {esc(ve['note'])}." if ve.get("note") else "")
      + (" \\textit{This is a CPU smoke-test run with tiny models; its numbers mean nothing.}" if S.get("smoke") else ""))
    a(r"\end{abstract}")

    a(r"\section{The question}")
    a(r"Do VES (Verify-and-Exclude Steering) and ECS (Echo-Check Steering) reduce hallucination beyond what the "
      r"SQ2 hidden-state detector already achieves on its own? Both methods are the detector gate B3 plus an add-on, so "
      r"every primary contrast is \emph{method minus B3} on the same test questions with the same greedy drafts $A_1$. "
      r"$U_\lambda = 100\,(R - \lambda W)/N$; 1.0 point of $U_1$ is one more (right $-$ wrong) answer per 100 questions.")

    # setup
    a(r"\section{Setup, as built}")
    q = S.get("questions", {})
    comp = q.get("composition", {})
    a(r"\paragraph{Questions.} " + "; ".join(
        f"{esc(s)}: {c['n']} ({', '.join(f'{esc(k)} {v}' for k, v in c['by_source'].items())})" for s, c in comp.items())
      + f". Eligible PopQA items after the isolation rules: {q.get('n_popqa_eligible', '--')}; TriviaQA questions dropped "
        f"as text duplicates: {q.get('tqa_dropped_duplicate_text', '--')}. "
        + ("PopQA object ids were available for the pair rule." if q.get("popqa_ids_file") else
           "PopQA object ids were \\emph{not} available: the pair rule used entity names only."))
    rel = {}
    for s, c in comp.items():
        for k, v in c["by_relation"].items():
            rel.setdefault(k, {})[s] = v
    if rel:
        a(r"\begin{table}[H]\centering\small\begin{tabular}{lrr}\toprule Relation & Dev & Test \\ \midrule")
        for k in sorted(rel):
            a(f"{esc(k)} & {rel[k].get('dev', 0)} & {rel[k].get('test', 0)} \\\\")
        a(r"\bottomrule\end{tabular}\caption{Composition by source and relation.}\end{table}")
    a(r"\paragraph{Detectors and hyperparameters.} Table~\ref{tab:built} lists the detector selections made on the "
      r"pool and the configuration chosen on dev (frozen before the test phase).")
    a(r"\begin{table}[H]\centering\small\begin{tabularx}{\textwidth}{@{}lL@{}}\toprule Model & As built \\ \midrule")
    for m in S.get("pool", {}):
        p = S["pool"][m]
        det = p.get("detectors", {})
        sel = det.get("selected", {})
        parts = []
        lay = det.get("layers", [])
        for k in ("D", "D_before", "D_after"):
            if k in sel:
                li = sel[k]["layer_index"]
                layer = lay[li] if li < len(lay) else f"index {li}"
                pos = {"prompt": "before", "answer": "after"}.get(sel[k]["position"], sel[k]["position"])
                parts.append(f"{esc(k)}: layer {layer}, {esc(pos)}, "
                             f"$\\lambda$ {sel[k]['lambda_rel']:g}, cross-fitted pool AUROC "
                             f"{f(det.get('crossfit_auroc', {}).get(k), 3)}")
        if "sq2_stored_crossfit_auroc" in det:
            parts.append(f"SQ2's stored cross-fitted AUROC {f(det['sq2_stored_crossfit_auroc'], 3)}")
        ln = p.get("lens", {})
        if ln:
            parts.append(f"ECS: agreement {', '.join(f'{k}: {f(v, 3)}' for k, v in ln.get('agreement', {}).items())} "
                         f"$\\Rightarrow a = {ln.get('a_depth')}$ (layer {ln.get('a_layer')}), $b = {ln.get('b_depth')}$ "
                         f"(layer {ln.get('b_layer')}); $t_{{\\IDK}}$ = ``{esc(ln.get('t_idk_text', ''))}''")
        es = p.get("ecs_dev_selection", {})
        if es:
            parts.append(f"$\\beta^* = {es.get('beta_star')}$ (dev AUROC by $\\beta$: "
                         f"{', '.join(f'{k}: {f(v, 3)}' for k, v in es.get('beta_auroc_dev', {}).items())})")
        fz = S.get("frozen", {}).get("models", {}).get(m, {}).get("methods", {})
        if "VES" in fz:
            v = fz["VES"]["lam1"]
            parts.append(f"VES: $\\ell_s$ = {v['depth']} depth, $\\alpha$ = {v['alpha']}, $\\tau$ = {f(v['tau'], 3)}")
        if "B5" in fz and "N_matched" in fz["B5"]:
            parts.append(f"$N_{{\\mathrm{{matched}}}}$ = {fz['B5']['N_matched']['N']}"
                         + ("" if fz["B5"]["N_matched"]["matched"] else " (no $N \\le 8$ reaches VES's FPE)"))
        a(f"{short(m)} & " + "; ".join(parts) + r" \\")
    a(r"\bottomrule\end{tabularx}\caption{Detector selections (SQ2 recipe on the pool, strict labels) and dev choices.}"
      r"\label{tab:built}\end{table}")
    gc_ = S.get("grader_changes", {})
    if gc_ or S.get("pool"):
        a(r"\paragraph{Grader.} Primary labels use the strict grader (SQ2 grader plus the hedge rule). Labels that "
          r"change relative to the SQ2 grader: " + "; ".join(
            [f"{short(m)} pool {p.get('grader', {}).get('n_changed_vs_sq2', '--')} of {p.get('grader', {}).get('n', '--')}"
             for m, p in S.get('pool', {}).items()]
            + [f"{esc(k)} {v['changed']} of {v['n']}" for k, v in gc_.items()]) + ".")

    # main table
    a(r"\section{Results}")
    a(r"\subsection{Main table}")
    a(r"{\small\begin{longtable}{llrrrrrr}")
    a(r"\caption{Test results per 100 questions. Right / wrong / \IDK, $U_1$ and FPE use the $\lambda{=}1$-tuned "
      r"configuration; $U_3$ uses the $\lambda{=}3$-tuned one. Every thresholded method uses its own dev-tuned "
      r"threshold(s).}\\")
    a(r"\toprule Method & Model & Right & Wrong & \IDK & $U_1$ & $U_3$ & FPE \\ \midrule\endfirsthead")
    a(r"\toprule Method & Model & Right & Wrong & \IDK & $U_1$ & $U_3$ & FPE \\ \midrule\endhead")
    a(r"\bottomrule\endfoot")
    for name, row in S.get("main_table", {}).items():
        first = True
        for m, r in row["per_model"].items():
            a(f"{LABELS.get(name, esc(name)) if first else ''} & {short(m)} & {f(r['right'])} & {f(r['wrong'])} & "
              f"{f(r['idk'])} & {f(r['U1'])} & {f(r['U3'])} & {f(r['fpe'], 2)} \\\\")
            first = False
        p = row["pooled"]
        if p.get("U1") is not None and len(row["per_model"]) > 1:
            a(f" & pooled & {f(p['right'])} & {f(p['wrong'])} & {f(p['idk'])} & {f(p['U1'])} & {f(p['U3'])} & "
              f"{f(p['fpe'], 2)} \\\\")
        a(r"\midrule")
    a(r"\end{longtable}}")
    a(r"\begin{table}[H]\centering\small\begin{tabular}{lrr}\toprule Method & $U_1$ strict grader & $U_1$ SQ2 grader \\ \midrule")
    for name, row in S.get("main_table", {}).items():
        p = row["pooled"]
        a(f"{LABELS.get(name, esc(name))} & {f(p.get('U1'))} & {f(p.get('U1_sq2_grader'))} \\\\")
    a(r"\bottomrule\end{tabular}\caption{Pooled $U_1$ under the strict and the SQ2 grader (same outputs).}\end{table}")

    a(r"\subsection{Contrasts}")
    a(r"\begin{table}[H]\centering\small\begin{tabularx}{\textwidth}{@{}lLLL@{}}\toprule Contrast & " +
      " & ".join(short(m) for m in models) + (" & Pooled" if len(models) > 1 else "") + r" \\ \midrule")
    names = {"d_ves": r"$\Delta_{\mathrm{VES}}$ = VES $-$ B3", "d_ves_b5": r"$\Delta_{\mathrm{VES,B5}}$ = VES $-$ B5@$N_m$",
             "d_ecs": r"$\Delta_{\mathrm{ECS}}$ = ECS-1P $-$ B3", "d_ecs_exact": "ECS-exact $-$ B3",
             "d_ecs_placebo": "ECS-placebo $-$ B3", "d_ves_free": "VES-free $-$ B3", "d_ves_placebo": "VES-placebo $-$ B3",
             "d_ves_unclean": "VES-unclean $-$ B3", "d_b7": "B7 $-$ B3", "d_b5_matched": "B5@$N_m$ $-$ B3",
             "d_ves_b7": "VES $-$ B7", "d_ves_lam3": "VES $-$ B3 ($U_3$)", "d_ecs_lam3": "ECS-1P $-$ B3 ($U_3$)"}
    for k, lab in names.items():
        c = con.get(k)
        if not c:
            continue
        cells = [ci(c["per_model"][m]) for m in models] + ([ci(c["pooled"])] if len(models) > 1 else [])
        a(f"{lab} & " + " & ".join(cells) + r" \\")
    a(r"\bottomrule\end{tabularx}\caption{Test contrasts in points of $U_1$ (95\% paired bootstrap CI, "
      + f"{S.get('n_boot', 2000):,} " + r"resamples of test questions, both models moving together).}\end{table}")

    a(r"\subsection{ECS feasibility: gap closure $G$}")
    a(r"\begin{table}[H]\centering\small\begin{tabular}{llrrrrr}\toprule Echo & Model & $A_{\mathrm{bef}}$ & "
      r"$A_{\mathrm{aft}}$ & $A_{\mathrm{echo}}$ & $G$ & pooled $G$ [CI] \\ \midrule")
    for name, gg in S.get("G", {}).items():
        first = True
        for m, p in gg["per_model"].items():
            a(f"{esc(name) if first else ''} & {short(m)} & {f(p['auroc_before'], 3)} & {f(p['auroc_after'], 3)} & "
              f"{f(p['auroc_echo'], 3)} & {f(p['G'], 2)} & {ci(gg['pooled'], 2, False) if first else ''} \\\\")
            first = False
    a(r"\bottomrule\end{tabular}\caption{AUROC for $A_1$'s correctness on answered test drafts. "
      r"$G = (A_{\mathrm{echo}} - A_{\mathrm{bef}})/(A_{\mathrm{aft}} - A_{\mathrm{bef}})$, undefined when "
      r"$A_{\mathrm{aft}} - A_{\mathrm{bef}} < 0.02$.}\end{table}")

    a(r"\subsection{Verdicts}")
    for k in ("VES", "ECS"):
        v = V.get(k, {})
        a(f"\\paragraph{{{k}: {esc(v.get('verdict', 'NOT RUN'))}.}} {esc(v.get('reason', ''))}"
          + (f" {esc(v['note'])}." if v.get("note") else "") + (f" \\textit{{{esc(v['integrity_note'])}}}" if v.get("integrity_note") else ""))
        inp = v.get("inputs") or {}
        bits = []
        for kk, lab in (("n_rejected_wrong", "answered, wrong and rejected test drafts"), ("n_wrong", "answered wrong test drafts")):
            if kk in inp:
                bits.append(lab + ": " + ", ".join(f"{short(m)} {n}" for m, n in inp[kk].items()))
        for kk, lab in (("d_ves", r"$\Delta_{\mathrm{VES}}$"), ("d_ves_b5", r"$\Delta_{\mathrm{VES,B5}}$"),
                        ("d_ecs", r"$\Delta_{\mathrm{ECS}}$"), ("G", "$G$")):
            if isinstance(inp.get(kk), dict):
                bits.append(f"{lab} {ci(inp[kk])}")
        if bits:
            a("Inputs: " + "; ".join(bits) + ".")

    # strata
    a(r"\section{Strata}")
    a(r"\begin{table}[H]\centering\small\begin{tabularx}{\textwidth}{@{}lrLLL@{}}\toprule Stratum & $n$ & "
      r"$\Delta_{\mathrm{VES}}$ & $\Delta_{\mathrm{VES,B5}}$ & $\Delta_{\mathrm{ECS}}$ \\ \midrule")
    for sname, row in S.get("strata", {}).items():
        n = "/".join(str(v) for v in row["n"].values())
        a(f"{esc(sname)} & {n} & {ci((row.get('d_ves') or {}).get('pooled'))} & {ci((row.get('d_ves_b5') or {}).get('pooled'))} & "
          f"{ci((row.get('d_ecs') or {}).get('pooled'))} \\\\")
    a(r"\bottomrule\end{tabularx}\caption{Pooled contrasts by stratum ($n$ per model). Confident errors: $A_1$ wrong "
      r"and at least 8 of tries 1--10 match $A_1$ (per model). Disagreement: SQ2 definition across the two models.}"
      r"\end{table}")

    # diagnostics
    a(r"\section{Diagnostics}")
    D = S.get("diagnostics", {})
    a(r"\begin{table}[H]\centering\small\begin{tabular}{llrrr}\toprule Model & Method & Hit rate & Hit rate (conf.\ errors) "
      r"& Distinct rate \\ \midrule")
    for m, hr in D.get("hit_rate", {}).items():
        for meth, h in hr.items():
            a(f"{short(m)} & {esc(meth)} & {f(h['all'], 3)} & {f(h['confident_errors'], 3)} & "
              f"{f(D.get('distinct_rate', {}).get(m, {}).get(meth), 3)} \\\\")
    a(r"\bottomrule\end{tabular}\caption{Hit rate: among wrong answered test drafts, the share where any of the first "
      r"$K$ proposals is correct. Distinct rate: share of rounds giving a new, non-\IDK, non-garbled answer.}\end{table}")
    for m, fo in D.get("fooling", {}).items():
        a(f"\\paragraph{{Fooling ({short(m)}).}} Mean steered $s_D$ $-$ clean $s_D$ over VES candidates: "
          f"{f(fo['mean_steered_minus_clean'], 3, True)} (n = {fo['n']}); $U_1$ of VES-unclean {f(fo['VES_unclean_U1'])} "
          f"vs VES {f(fo['VES_U1'])}.")
    for m, e in D.get("ecs", {}).items():
        a(f"\\paragraph{{ECS ({short(m)}).}} Agreement $\\Pr[\\mathrm{{top}}_a = f(A_1)]$ = "
          f"{f(e['agreement_top_a_eq_first_token'], 3)}; switches {e['n_switches']}: wrong$\\to$right "
          f"{e['switch_wrong_to_right']}, right$\\to$wrong {e['switch_right_to_wrong']}, \\IDK$\\to$right "
          f"{e['switch_idk_to_right']}, right$\\to$\\IDK{{}} {e['switch_right_to_idk']}; net {e['net']:+d}.")
    nf = D.get("net_fixes_vs_B3", {})
    if nf:
        a(r"\begin{table}[H]\centering\small\begin{tabular}{l" + "r" * len(nf) + r"}\toprule Method & "
          + " & ".join(short(m) for m in nf) + r" \\ \midrule")
        meths = list(next(iter(nf.values())).keys())
        for meth in meths:
            a(f"{LABELS.get(meth, esc(meth))} & " + " & ".join(
                f"{nf[m][meth]['net']:+d} ({nf[m][meth]['fixes']}/{nf[m][meth]['breaks']})" if meth in nf[m] else "--"
                for m in nf) + r" \\")
        a(r"\bottomrule\end{tabular}\caption{Net fixes versus B3: (wrong or \IDK $\to$ right) $-$ (right $\to$ wrong or "
          r"\IDK), with (fixes/breaks).}\end{table}")
    ov = S.get("ecs_overhead", {})
    for m, o in ov.items():
        if "error" not in o:
            a(f"\\paragraph{{ECS overhead ({short(m)}).}} $k(b-a)/L$ = {f(o['frac_of_one_token'], 3)} of one token's "
              f"forward pass; about {f(100 * o['frac_of_generation'], 2)}\\% of the whole generation "
              f"($P$ = {f(o['P'])}, $n$ = {f(o['n'])}). Latency was not measured.")
    a(r"Risk--coverage curves (descriptive, thresholds swept on test) are in \texttt{results/summary.json} under "
      r"\texttt{diagnostics.risk\_coverage}.")

    # integrity
    a(r"\section{Integrity checks}")
    a(r"\begin{table}[H]\centering\small\begin{tabularx}{\textwidth}{@{}llL@{}}\toprule Check & Scope & Result \\ \midrule")
    for name, rows in S.get("integrity", {}).items():
        for scope, r in rows.items():
            det = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                   if k in ("max_abs_diff", "min_cos", "auroc", "n_match", "n_compared",
                                                        "passed_at", "spec_forward_difference_pass", "n_pass", "n_tests")}
            res = "pass" if r.get("pass") else r"\textbf{FAIL}"
            scope_s = short(scope) if scope != "all" else "all"
            a(f"{esc(name)} & {scope_s} & {res} {esc(str(det)) if det else ''} " + r"\\")
    a(r"\bottomrule\end{tabularx}\caption{Integrity checks (results/integrity.json). The gradient check records the "
      r"spec's forward difference and, if that is outside 10\%, the float32 and central-difference fallbacks.}\end{table}")
    st = S.get("integrity_status", {})
    if st and not st.get("all_passed"):
        a(r"\textbf{Not all checks passed:} " + esc("; ".join(st.get("problems", []))) + ".")

    # limitations
    a(r"\section{Limitations}")
    a(r"\begin{itemize}")
    a(r"\item The strict grader adds a hedge rule to the SQ2 grader; label changes are listed in the setup. The audit "
      r"files (\texttt{audit/human\_check\_<model>.csv}) list 100 test outputs per model that differ from $A_1$, for a "
      r"manual check after the run.")
    b = S.get("budget", {})
    a(r"\item Budget cuts applied: " + (esc(", ".join(b.get("cuts", []))) or "none")
      + (f" (unavailable: {esc(', '.join(b['cuts_unavailable']))})" if b.get("cuts_unavailable") else "")
      + f". Elapsed {f(b.get('elapsed_h'), 2)} h.")
    if S.get("missing"):
        a(r"\item Missing or failed parts: " + esc("; ".join(S["missing"])) + ".")
    a(r"\item ECS is simulated with separate calls (exactly equivalent outputs; the echo side rows share the prompt's "
      r"key/value cache, checked against the full-sequence simulation). Latency is not measured.")
    a(r"\item CIs cover question sampling only; detectors and thresholds are fixed.")
    a(r"\end{itemize}")
    a(r"\section{Reproducing the numbers}")
    a(f"Run folder: \\path{{{S.get('run', '')}}}. \\texttt{{python -m vesecs.analysis -{{}}-run-dir <run>}} recomputes "
      r"every number from the saved per-question records.")
    a(r"\end{document}")
    path.write_text("\n".join(L), encoding="utf-8")

"""results/results_report.tex in the SQ2 report's structure: abstract with the verdict; the question; setup as
built; main table; contrasts; verdict; probe diagnostics; strata; replay (diagnostic); integrity checks; optional
stages; limitations; reproducing. Generated from summary.json."""
import math

from vesecs.report import esc

LABELS = {"B0": "B0 Greedy", "B3": "B3 Detector gate", "R-all": "R-all (retry every flagged)",
          "R-band": "R-band (detector band)", "R-check": "R-check (= B7, one round)", "RRA": "\\textbf{RRA}",
          "RRA-EU": "RRA-EU", "RRA+check": "RRA+check", "RRA-peer": "RRA-peer (exploratory)",
          "Oracle": "Oracle (ceiling)", "Oracle-noswitch": "Oracle without switching"}
SHORT = {"llama31_8b_instruct": "Llama", "qwen25_7b_instruct": "Qwen", "gemma2_9b_it": "Gemma"}


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
    return SHORT.get(m, esc(m))


def pct(x):
    return "--" if x is None else f"{100 * x:.0f}\\%"


def _arm_table(a, T, models, caption):
    a(r"\begin{table}[H]\centering\small")
    a(r"\begin{tabular}{@{}llrrrrrrrr@{}}\toprule")
    a(r"Arm & Model & Right & Wrong & \IDK & $U_1$ & $U_3$ & Retry \% & Routed prec. & FPE \\ \midrule")
    for arm, row in T.items():
        first = True
        for m in list(models) + ["pooled"]:
            r = row["pooled"] if m == "pooled" else row["per_model"].get(m)
            if r is None or (m == "pooled" and len(models) < 2):
                continue
            a(f"{LABELS.get(arm, esc(arm)) if first else ''} & {'pooled' if m == 'pooled' else short(m)} & "
              f"{f(r.get('right'))} & {f(r.get('wrong'))} & {f(r.get('idk'))} & {f(r.get('U1'))} & {f(r.get('U3'))} & "
              f"{f(r.get('retry_rate'))} & {pct(r.get('routed_precision'))} & {f(r.get('fpe'), 2)} \\\\")
            first = False
        a(r"\addlinespace[2pt]")
    a(r"\bottomrule\end{tabular}")
    a(f"\\caption{{{caption}}}\\end{{table}}")


def _contrast_rows(a, con, models, keys):
    a(r"\begin{table}[H]\centering\small")
    a(r"\begin{tabularx}{\textwidth}{@{}l" + "L" * (len(models) + 1) + r"@{}}\toprule")
    a("Contrast ($U_1$ points) & " + " & ".join(short(m) for m in models) + r" & Pooled \\ \midrule")
    for k, label in keys:
        c = con.get(k)
        if not c:
            continue
        a(f"{label} & " + " & ".join(ci(c['per_model'].get(m)) for m in models) + f" & {ci(c['pooled'])} \\\\")
    a(r"\bottomrule\end{tabularx}")


def write_report(S, path):
    models = S.get("models", [])
    V = S.get("verdicts", {}).get("RRA", {})
    con = S.get("contrasts", {})
    comp = (S.get("comparator") or {}).get("lam1", {}).get("arm", "--")
    L = []
    a = L.append
    a(r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{float}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
\setlist{itemsep=2pt, topsep=4pt}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\newcommand{\IDK}{\textsc{idk}}
\title{Recoverability-Routed Answering (RRA)\\[2pt]\large Results}
\author{Khanh Tran}
\date{\today}
\begin{document}
\maketitle
""")
    a(r"\begin{abstract}\noindent")
    a(f"This report gives the results of the experiment specified in \\texttt{{rra-experiment-design.tex}} on "
      f"{len(models)} model(s) and {S.get('n_test', 0)} fresh test questions. RRA predicts, from the hidden states of "
      f"the model's first answer, whether its next-best answer (first token banned) would be right, and then keeps, "
      f"switches or says ``I don't know''. \\textbf{{Verdict: {esc(V.get('verdict', 'NOT RUN'))}}} "
      f"({esc(V.get('reason', ''))})." + (f" {esc(V['note'])}." if V.get("note") else "")
      + (f" \\textit{{Note: {esc(V['integrity_note'])}.}}" if V.get("integrity_note") else "")
      + (" \\textit{This is a CPU smoke-test run with tiny models; its numbers mean nothing.}" if S.get("smoke") else ""))
    a(r"\end{abstract}")

    a(r"\section{The question}")
    a(r"Can the model's hidden states decide \emph{when a retry will work}, well enough to beat the detector gate B3? "
      r"RRA never scores a retry: the recoverability probe $R$ reads the before and after states of the first answer "
      r"$A_1$ and predicts whether the fallback $A_2$ (greedy decoding with $A_1$'s first token banned at the first "
      r"step) will be right. $U_\lambda = 100\,(R - \lambda W)/N$; 1.0 point of $U_1$ is one more (right $-$ wrong) "
      r"answer per 100 questions. Contrasts: $H$ = Oracle $-$ Oracle without switching (headroom), "
      r"$\Delta_{\mathrm{main}}$ = RRA $-$ B3, $\Delta_{\mathrm{mech}}$ = RRA $-$ $C$, where $C$ is the better of R-band "
      f"and R-check on the tuning set (chosen before the fresh test: \\textbf{{{esc(comp)}}}).")

    # ---------------------------------------------------------------- setup
    a(r"\section{Setup, as built}")
    q = S.get("questions") or {}
    comp_q = q.get("composition", {})
    if comp_q:
        a(r"\paragraph{Fresh questions.} " + "; ".join(
            f"{esc(s)}: {c['n']} ({', '.join(f'{esc(k)} {v}' for k, v in c['by_source'].items())})"
            for s, c in comp_q.items()) + ". PopQA relations: " + ", ".join(
            f"{esc(k)} {v}" for k, v in comp_q.get("test_fresh", {}).get("by_relation", {}).items()) + " (test). "
            f"Dropped while building: {esc(q.get('dropped', {}))}.")
    a(r"\paragraph{Reused.} Models, prompt, greedy decoding ($\le 16$ tokens), strict grader and the frozen detector "
      f"$D$ come from the VES/ECS run \\texttt{{{esc(S.get('ves_run', ''))}}}; the pool (answers, labels, folds, "
      r"states at five depths) from the SQ2 run. Tuning set = VES/ECS dev (600) + fresh dev; replay = VES/ECS test.")
    pool = S.get("pool", {})
    rows = []
    for m, d in pool.items():
        R = d.get("R", {})
        sel = R.get("selected", {})
        rows.append(f"{short(m)} & {sel.get('layer', '--')} ({f(sel.get('depth'), 2)}) & {esc(sel.get('moment', '--'))} & "
                    f"{f(sel.get('lambda_rel'), 3)} & {R.get('n_train', '--')} & {R.get('n_pos', '--')} & "
                    f"{f(R.get('crossfit_auroc'), 3)} & {f(R.get('shuffled_auroc'), 3)} \\\\")
    if rows:
        a(r"\begin{table}[H]\centering\small\begin{tabular}{@{}lrlrrrrr@{}}\toprule")
        a(r"Model & Layer (depth) & Moment & Ridge (rel.) & $n$ train & $y_2{=}1$ & Cross-fit AUROC & Shuffled \\ \midrule")
        L.extend(rows)
        a(r"\bottomrule\end{tabular}\caption{Recoverability probe $R$ on the pool (moment: prompt = before, answer = "
          r"after).}\end{table}")
    tun = S.get("tuning") or {}
    if tun:
        a(r"\begin{table}[H]\centering\small\begin{tabular}{@{}llrrrrrr@{}}\toprule")
        a(r"Model & Arm & $\tau_1$ & $\tau_2$ & $\tau_b$ & $\tau_3$ & $\tau_p$ & $U_1$ (tuning) \\ \midrule")
        for m, arms in tun.items():
            for arm, v in arms.items():
                t = v.get("lam1", {})
                a(f"{short(m)} & {LABELS.get(arm, esc(arm))} & {f(t.get('t1'), 3)} & {f(t.get('t2'), 3)} & "
                  f"{f(t.get('tb'), 3)} & {f(t.get('t3'), 3)} & {f(t.get('tp'), 3)} & {f(t.get('U_tune'))} \\\\")
            a(r"\addlinespace[2pt]")
        a(r"\bottomrule\end{tabular}\caption{Frozen thresholds ($\lambda = 1$), tuned on the tuning set.}\end{table}")
        cmp = S.get("comparator") or {}
        if cmp:
            a("Comparator choice on the tuning set (pooled $U_1$): " + "; ".join(
                f"{esc(k)}: {esc(v['arm'])} (" + ", ".join(f"{esc(x)} {f(y)}" for x, y in v["U_tune_pooled"].items()) + ")"
                for k, v in cmp.items()) + ".")

    # ---------------------------------------------------------------- main results
    a(r"\section{Main results (fresh test)}")
    T = S.get("main_table", {})
    if T:
        _arm_table(a, T, models, "Each arm on the fresh test. Right / wrong / \\IDK{} per 100 questions; $U_3$ uses the "
                                 "$\\lambda = 3$ thresholds; retry \\% = questions where $A_2$ is generated; routed "
                                 "precision = right / (right + wrong) among fallback outputs (above "
                                 "$\\lambda/(1+\\lambda)$ = 50\\% a retry adds utility at $\\lambda = 1$).")
    else:
        a("No fresh-test results. " + esc("; ".join(S.get("missing", []))))
    if con:
        _contrast_rows(a, con, models, [("H", "$H$ (switching headroom)"), ("d_main", r"$\Delta_{\mathrm{main}}$ = RRA $-$ B3"),
                                        ("d_mech", f"$\\Delta_{{\\mathrm{{mech}}}}$ = RRA $-$ {esc(comp)}"),
                                        ("RRA-EU - B3", "RRA-EU $-$ B3"), ("RRA+check - B3", "RRA+check $-$ B3"),
                                        ("RRA+check - RRA", "RRA+check $-$ RRA"), ("R-all - B3", "R-all $-$ B3"),
                                        ("R-band - B3", "R-band $-$ B3"), ("R-check - B3", "R-check $-$ B3"),
                                        ("RRA - R-band", "RRA $-$ R-band"), ("RRA - R-check", "RRA $-$ R-check"),
                                        ("RRA-peer - B3", "RRA-peer $-$ B3 (exploratory)"),
                                        ("d_main_lam3", r"$\Delta_{\mathrm{main}}$ at $\lambda = 3$ ($U_3$)"),
                                        ("d_mech_lam3", r"$\Delta_{\mathrm{mech}}$ at $\lambda = 3$ ($U_3$)")])
        a(f"\\caption{{Contrasts on the fresh test with 95\\% paired-bootstrap intervals ({S.get('n_boot', '--')} "
          f"resamples over questions, both models together; pooled = mean of the models).}}\\end{{table}}")

    a(r"\section{Verdict}")
    inp = V.get("inputs", {})
    a(f"\\textbf{{{esc(V.get('verdict', 'NOT RUN'))}.}} {esc(V.get('reason', ''))}."
      + (f" {esc(V['note'])}." if V.get("note") else ""))
    if inp.get("n_recoverable"):
        a(" Questions where $A_1$ is wrong and $A_2$ is right (need $\\ge 150$ per model): " + ", ".join(
            f"{short(m)} {n}" for m, n in inp["n_recoverable"].items()) + ".")
    a(r" Rule: INSUFFICIENT\_DATA if a model has fewer than 150 such questions; STOP if the CI upper bound of $H$ "
      r"is below 2.0, of $\Delta_{\mathrm{main}}$ below 1.0, or of $\Delta_{\mathrm{mech}}$ below 0; GO if "
      r"$\Delta_{\mathrm{main}} \ge 1.0$ and $\Delta_{\mathrm{mech}} \ge 0.5$, both with CI lower bound above 0; "
      r"otherwise BORDERLINE.")

    # ---------------------------------------------------------------- diagnostics
    D = S.get("diagnostics", {})
    if D:
        a(r"\section{Probe diagnostics}")
        a(r"\begin{table}[H]\centering\small\begin{tabular}{@{}llrrrrrr@{}}\toprule")
        a(r"Model & Questions & $n$ & Successes & AUROC $s_2$ & AUROC $D(A_2)$ & AUROC $D(A_2)$, non-null & AUROC $-s_1$ \\ \midrule")
        for m, d in D.items():
            for nm in ("flagged", "answered"):
                x = d[nm]
                a(f"{short(m)} & {nm} & {x['n']} & {x['n_success']} & {f(x['auroc_s2'], 3)} & {f(x['auroc_D_A2'], 3)} & "
                  f"{f(x['auroc_D_A2_non_null'], 3)} & {f(x['auroc_minus_s1'], 3)} \\\\")
        a(r"\bottomrule\end{tabular}\caption{Predicting a successful retry ($A_2$ not null and right) on the fresh test. "
          r"Flagged = answered and $s_1 < \tau_1$ (RRA's $\tau_1$). A null $A_2$ gets the lowest $D(A_2)$.}\end{table}")
        for m, d in D.items():
            a(f"\\paragraph{{{short(m)}: fallbacks.}} " + esc(d["A2"]) + ".")
            cal = d.get("calibration_test", {}).get("p2", [])
            if cal:
                a(" Calibration of $p_2$ on the fresh test (mean $p_2$ / observed rate / $n$ per decile): " + "; ".join(
                    f"{f(c['p_mean'], 2)}/{f(c['y_rate'], 2)}/{c['n']}" for c in cal) + ".")

    st = S.get("strata", {})
    if st:
        a(r"\section{Strata}")
        a(r"\begin{table}[H]\centering\small\begin{tabularx}{\textwidth}{@{}lLLLL@{}}\toprule")
        a(r"Stratum & $n$ per model & $H$ & $\Delta_{\mathrm{main}}$ & $\Delta_{\mathrm{mech}}$ \\ \midrule")
        for name, row in st.items():
            a(f"{esc(name)} & {', '.join(f'{short(m)} {n}' for m, n in row['n'].items())} & "
              f"{ci(row.get('H')['pooled']) if row.get('H') else '--'} & "
              f"{ci(row.get('d_main')['pooled']) if row.get('d_main') else '--'} & "
              f"{ci(row.get('d_mech')['pooled']) if row.get('d_mech') else '--'} \\\\")
        a(r"\bottomrule\end{tabularx}\caption{Strata of the fresh test (pooled contrasts; confident errors need the optional "
          r"stage E2: $A_1$ wrong and $\ge 8$ of tries 1--10 match $A_1$; disagreement: the two models' $A_1$ "
          r"correctness differs).}\end{table}")

    rp = S.get("replay")
    if rp:
        a(r"\section{Replay on the VES/ECS test set (diagnostic only)}")
        a(r"RRA was designed after seeing aggregate results on this set, so it is not confirmatory. $A_2$ = the stored "
          r"B7 round-1 answer.")
        rm = list(next(iter(rp["main_table"].values()))["per_model"]) if rp.get("main_table") else []
        _arm_table(a, {k: v for k, v in rp["main_table"].items() if k in ("B0", "B3", "R-all", "R-band", "R-check", "RRA",
                                                                          "RRA-EU", "RRA+check", "Oracle")},
                   rm, f"Replay set ({rp['n']} questions), frozen thresholds.")
        _contrast_rows(a, rp.get("contrasts", {}), rm, [("H", "$H$"), ("d_main", r"$\Delta_{\mathrm{main}}$"),
                                                         ("d_mech", r"$\Delta_{\mathrm{mech}}$")])
        a(r"\caption{Replay contrasts (diagnostic).}\end{table}")

    # ---------------------------------------------------------------- integrity, budget, optional
    a(r"\section{Integrity checks}")
    I = S.get("integrity", {})
    a(r"\begin{table}[H]\centering\small\begin{tabularx}{\textwidth}{@{}llL@{}}\toprule")
    a(r"Check & Model & Result \\ \midrule")
    for name, rows_ in I.items():
        for m, r in rows_.items():
            det = {k: v for k, v in r.items() if k in ("max_abs_diff", "auroc", "rate", "n_differ", "n", "U1_recomputed",
                                                       "U1_expected", "batch_size", "test_ves", "n_pass", "n_tests")}
            a(f"{esc(name)} & {short(m) if m != 'all' else '--'} & {'pass' if r.get('pass') else 'FAIL'}"
              + (" (reported, not blocking)" if r.get("informational") else "") + f"; {esc(det)} \\\\")
    a(r"\bottomrule\end{tabularx}\caption{Integrity checks (results/integrity.json).}\end{table}")
    ist = S.get("integrity_status", {})
    if ist and not ist.get("all_passed"):
        a("\\textbf{Not all required checks passed:} " + esc("; ".join(ist.get("problems", []))) + ".")
    b = S.get("budget", {})
    a(f"\\paragraph{{Budget.}} Elapsed {f(b.get('elapsed_h'), 2)} h. Cuts: {esc(b.get('cuts') or 'none')}. "
      "Measured rates per model and unit are in summary.json and logs/timing.jsonl.")
    if S.get("missing"):
        a(r"\paragraph{Missing parts.} " + esc("; ".join(S["missing"])) + ".")

    G = S.get("generality")
    if G:
        a(r"\section{Generality: Gemma-2-9B-it (optional stage G, not part of the verdict)}")
        a(f"Gemma's own detector $D$ (cross-fitted pool AUROC {f(G['D'].get('crossfit_auroc'), 3)}) and $R$ (layer "
          f"{G['R']['selected'].get('layer')}, {esc(G['R']['selected'].get('moment'))}, cross-fitted AUROC "
          f"{f(G['R'].get('crossfit_auroc'), 3)}), tuned on the fresh dev set ({G['n_dev']}) and evaluated on the fresh "
          f"test ({G['n_test']}); recoverable questions: {G['n_recoverable']}.")
        _arm_table(a, {k: v for k, v in G["main_table"].items() if k != "RRA-peer"}, [G["model"]], "Gemma on the fresh test.")
        _contrast_rows(a, G.get("contrasts", {}), [G["model"]], [("H", "$H$"), ("d_main", r"$\Delta_{\mathrm{main}}$"),
                                                                  ("d_mech", r"$\Delta_{\mathrm{mech}}$")])
        a(r"\caption{Gemma contrasts (bootstrap over its fresh-test questions).}\end{table}")

    a(r"\section{Limitations}")
    a(r"\begin{itemize}")
    a(r"\item The fresh PopQA relations (occupation, director, producer, screenwriter, composer, author) differ from the "
      r"pool's, so $R$ and $D$ are evaluated off their training relations there; results are reported by source.")
    a(r"\item One fallback only (first-token ban). RRA's value is bounded by the headroom $H$ of that fallback.")
    a(r"\item The strict grader is automatic; \texttt{audit/human\_check\_<model>.csv} holds 100 routed retries per "
      r"model for a hand check.")
    a(r"\end{itemize}")
    a(r"\section{Reproducing}")
    a(r"\texttt{bash run\_rra.sh} (everything); \texttt{bash run\_rra.sh analyze outputs\_rra/run\_<id>} re-runs this "
      r"analysis on CPU. Frozen configuration: \texttt{configs/frozen\_config.json} (SHA-256 in "
      r"\texttt{logs/frozen\_config\_sha256.json}).")
    a(r"\end{document}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

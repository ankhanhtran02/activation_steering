# Pre-registration amendment, 2026-11-07: the final run

Amends `experiment_plan.yaml` as already amended by `preregistration_amendment_2026-11-04.md`. Where this
document and either earlier text differ, this document governs the final run. `main.py` prints the
SHA-256 of both amendment files (`AMENDMENT:` lines) and stores them in `frozen_config.json`.

## 0. Status and purpose

- **Not blind.** Written after the results of run 1 (2026-11-03) and run 2 (`run_20261104T153241Z`) and
  three independent reviews of run 2 were read. Every change below fixes a validity problem that a
  review identified, or removes work that cannot change a verdict. No change is motivated by a score.
- **Purpose.** This is the last screening run. After it, the authors decide whether to keep the current
  hypotheses (H1–H4) or move to new ones. The run is designed so that each registered hypothesis ends
  with a verdict (`SUPPORTED`, `REJECTED`, `MIXED` or `UNTESTABLE`) computed by pre-specified rules
  (section 5), written to `verdicts.json` and printed as `VERDICT:` lines.
- **Unchanged:** the models and their pinned revisions; the prompts; greedy decoding and batching; the
  primary metric (`abstention_score`, λ = 1) and answer matching; the seed-0 train / dev draws; the dev
  selection rule for (L\*, c\*) with the A11 caps; every registered condition that a verdict depends on.

## 1. What run 2 left unsettled

| Open question after run 2 | Settled here by |
|---|---|
| PopQA test items were reused across runs, after analysis choices were made on them | F1 |
| H1c was decided by the dose rule (gate covers ~90% of items; gated ≈ ungated steering) | F2 |
| H1b had no valid placebo: no random direction reached the knowledge dose; the shuffled-label null was not a null | F3, F4 |
| Is D_ku a knowledge direction or a question-type (relation / template) axis? | F5 |
| H1a's equivalence was measured at one dev-chosen coverage point | F12 (coverage sweep) |
| H1e was computed over doses where Llama's output was 100% degenerate | F6 |
| H2c's validity rule was one-sided; the three ablation variants differed in layer; H2d had 100 prompts | F7 |
| "Don't know." was scored as a wrong answer (arm-dependent bias up to 1.5 points) | F8 |
| No output was ever checked by a judge or a person | F9 |
| D_wq ran at 18σ (Llama) / 58σ (Qwen) of its own spread: its null could be an overdose | F10 |
| H3's verdict depended on a comparator chosen after run 1; its signal was picked on test | F11 |
| The seed rule (same sign) cannot judge a null result; no verdict was written anywhere | F12 |
| The 7–9B confirmation gate | section 4 (removed at the authors' request) |

## 2. Changes

**F1. Fresh test data.** `previously_scored_qids.json` lists every question id that run 1 or run 2
scored on a test split (PopQA 1,200; EntityQuestions 600; TriviaQA 975). The loaders exclude them and
assert it.
- PopQA test = every remaining test-pool question: **1,715 questions**, 149–198 per popularity decile,
  subject-disjoint from train and dev as before. Never cut by the budget (the `test_1200` rung is gone).
- EntityQuestions: the next 25 questions per relation in the registered permutation that no earlier
  run scored. TriviaQA: 1,000 new test questions (repeated question ids removed, disjoint from the
  unchanged 300-question threshold dev split).
- Train, train_reserve, dev and the seed redraws are unchanged (they are not test data).

**F2. H1c re-specified.** Registered H1c runs gated steering at c\*, which was chosen for *ungated*
steering under a 5–30-point abstention cap. With the gate firing on ~90% of items the outcome was fixed
by that dose. Gated steering now gets its own strength: the dev-clean dose of the knowledge ladder at L\*
that maximises dev S1 of gated steering (ties within 0.5 points → smaller c). H1c compares that arm with
gated refusal (confirmatory family H1). The c\* version is reported as exploratory. H1c is also reported
at every ladder dose.

**F3. Dose-matched random placebos (H1b, registered comparison).** 8 isotropic and 4 covariance-shaped
random directions. Each takes the sign that lowers the teacher-forced margin on 64 dev questions
(sign-aligned), then is run on the 300-item dev subset at ascending multiples of α\* (0.5, 1, 2, 4, 8, 16;
smaller if 0.5 overshoots), stopping at the knowledge arm's abstention increase or at > 10% degenerate
output, with one geometric midpoint. The clean multiple closest to the target is frozen. Matched =
within ±3% of the subset, compared as **integer item counts** (run 2 lost a match to floating point).
If no random direction can be matched, that is reported as the pre-registered branch "random directions
cannot induce clean abstention; H1a decides".

**F4. The uniform abstention push (H1b, primary placebo).** D_force = mean over train questions of the
last-prompt residual under "…Do not answer this question. Reply exactly: I don't know." minus under the
base prompt. Every question contributes to both sides, so the direction carries "being told to abstain"
and no known/unknown contrast. Dose-matched as in F3 (sign fixed), at every seed. H1b's primary contrast
is knowledge_dir_steer's surplus minus D_force's surplus (each against its own coverage-matched
threshold): equivalence means the learned direction selects no better than a push toward abstention that
knows nothing about the question.

**F5. D_ku decomposition.** With μ_r the mean train residual of PopQA relation r, D_ku splits exactly into
a between-relation part (relation composition of the known / unknown classes) and a within-relation part.
Both are steered (dose-matched as F3) and reported with their norm shares. This settles whether D_ku's
effect comes from question type.

**F6. H1e on the valid dose range.** The teacher-forced grid is {0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64}
restricted to doses that are clean on dev at (L\*, c) (degenerate ≤ 5%). CV(gain) uses |mean|.
Comparison directions on the same grid: two sign-aligned random placebos (same absolute norm) and D_force
(at its dose-matched scale). The margin matrices are saved.

**F7. H2.**
- Every ablation variant uses the registered arm's L_ref, so the variants differ only in method.
- H2c validity is two-sided (|ablation-only abstention change| ≤ 5 points). Two baseline-robust forms are
  pre-specified:
  - conditional: on items both the unsteered and the ablated model answer, 1 − (steering-induced
    abstention with ablation) / (without);
  - log-odds.
- The registered form is primary when valid, otherwise the conditional form.
- Refusal-direction cosines are reported at every candidate layer.
- H2d uses 300 held-out benign prompts (the first 100 are run 2's), with a Fisher test.

**F8. Abstention rule.** "Don't know" / "Do not know" without a subject now count as abstentions (they
were scored as wrong answers). The registered phrase list is kept as a sensitivity analysis of every
verdict-relevant result.

**F9. LLM judge.** Phase D labels every distinct (question, output) of every test-split condition with a
pinned local judge, Qwen/Qwen2.5-7B-Instruct @ a09a3545 (not a model under test). It uses the same
prompt as `audit.py`.
- Automatic labels stay primary. Every verdict is recomputed under judge labels and flagged
  `label_robust` true / false.
- `judge_disagreements.csv` holds up to 300 disagreement rows per model, with an empty `human_label`
  column. If a verdict is label-sensitive, the human labels on these rows decide which labelling it
  follows.

**F10. D_wq at its own dose.** A dev ladder of c·σ_{L\*}(D_wq), c ∈ {1, 2, 4, 8, 16, 32}, at L\* and the
shared positions. The selected dose has the largest net dev (W→C − C→W) among clean doses that pass the
A11 caps. It runs on test with 5 label-permuted and 5 isotropic random nulls at the same norm. H4b is
judged at both the registered shared site and this dose.

**F11. H3.**
- The internal signal is pre-specified on dev: the highest AUROC among answered dev items. Run 2 picked
  both arms of H3a on test.
- Train projections on D_ku / D_read used by the combination models are cross-fitted.
- The registered comparator (popularity + surprisal) is primary. Amendment A6's (popularity + relation)
  is secondary and reported as such. H3 is also recomputed with judge-labelled test correctness.

**F12. Inference outputs.**
- H1a gets an outcome category: `superior` / `equivalent` / `inferior` / `non_superior` /
  `inconclusive` (definitions in `analysis_core.h1a_category`).
- Every H1a estimate is also given on the answered-accuracy scale.
- H1a is reported at every ladder dose (Holm within the sweep, exploratory).
- Replication across seeds means every seed has the seed-0 category. This replaces A8's same-sign rule,
  which cannot judge an equivalence.
- TriviaQA H1d gets bootstrap CIs and TOST ±3 pp at 70% coverage, plus D_force as its placebo.
- `knowledge − always_abstain` is reported with a CI.
- The verdict table is computed by `verdicts.py` and written to `verdicts.json`.

## 3. Work dropped (settled or unable to change a verdict)

- Answer-position D_wq ladder and arms: W→C = 0 on dev at every strength in both models in run 2.
- Last-prompt-only / generated-only, alternative-layer and between-question ablations.
- D_wq refits r = 1–4 and the shared-norm D_wq dose ladder.
- Run 2's three dose-matched nulls (shuffled-label, covariance random, −D_ku): replaced by F3–F4.
- Transfer-set random subsets.
- The 20 norm-matched isotropic draws on PopQA test are reduced to 10. They remain the registered
  norm-matched H1b comparison and H4b's shared-site null.
- Random-direction ablations go from 11 to 7.

## 4. Kill rule and decision rules (amended at the authors' request)

The plan required a programme-reorganising result (e.g. threshold or placebo matching learned steering)
to be confirmed on one 7–9B model before it is acted on. **That gate is removed.**
- A verdict on which both 3B families agree is final at screening and is acted on.
- A verdict on which they disagree is `MIXED`, meaning "not established". It is no longer forwarded to a
  confirmation tier.
- A 7–9B replication becomes an optional first step of any decision to continue with the current
  hypotheses.

Decision rules after this run (the plan's, adapted to the new gate):
- H1 `SUPPORTED`: steering does not beat its own detector. Accepted now. Later steering work must carry
  `random_dir_placebo` and `confidence_threshold_margin`.
- H1 `REJECTED`: gated steering becomes the main study.
- H3 `SUPPORTED` (frequency suffices): report the gate as API-compatible routing, and restate
  mechanistic gate claims as frequency tracking.
- H4 `SUPPORTED`: advance the recovering intervention with its nulls. H4 `REJECTED`: report the observed
  W→C and knockout rates as upper bounds.

## 5. Verdict criteria (implemented in `verdicts.py`; thresholds are the plan's)

Per model, then combined: both models `REJECTED` → `REJECTED`; both `SUPPORTED` → `SUPPORTED`; otherwise
`MIXED` (`UNTESTABLE` if either model cannot run the test).

- **H1** (confirmatory H1a + H1c)
  - `REJECTED` if the H1a surplus is ≥ +5 with 95% CI > 0, or tuned gated steering beats gated refusal
    with 95% CI > 0.
  - `SUPPORTED` if both of these hold:
    - H1a is `equivalent` (90% CI within ±3), or `non_superior` / `inferior` (95% CI upper < +3). The
      latter is labelled "non-superiority: the equivalence prediction missed";
    - no dev-clean ladder dose is `superior`, and H1c is not rejected.
- **H1b**: learned minus uniform push.
  - `SUPPORTED` if equivalent (TOST ±3) or the 95% CI upper < +3.
  - `REJECTED` if ≥ +5 with CI > 0.
  - `UNMATCHED` if D_force could not be dose-matched on dev.
- **H1d** (TriviaQA, 70% coverage): `SUPPORTED` if steered minus unsteered selective accuracy is within
  ±3 pp (TOST); `REJECTED` if its 95% CI lies outside the band on one side.
- **H1e**: `SUPPORTED` if Spearman ≥ 0.7, CV(gain) ≤ 0.35 and ≥ 80% monotone; `REJECTED` if CV > 0.7 or
  > 30% non-monotone.
- **H2**
  - `REJECTED` (the plan's conjunction) if all hold: positive control passes; bottom-tercile abstention
    drops < 15% relative; primary H2c fraction removed ≤ 0.20; benign spillover < 2 points.
  - `SUPPORTED` if every H2a–H2d prediction holds.
  - Otherwise `PARTIAL`, with the predictions met listed.
- **H3** (registered comparator)
  - `REJECTED` if the dev-selected internal signal has within-decile AUROC ≥ 0.65 and beats popularity
    + surprisal by ≥ 0.05.
  - `SUPPORTED` if all hold: internal − comparator ≤ 0.03; every internal within-decile AUROC < 0.60;
    increment over popularity ≤ 0.03.
  - The A6 comparator verdict is reported alongside.
- **H4** (recovery half)
  - `REJECTED` if both hold: knockout recovery < 10% in stored_outcompeted; D_wq W→C not above both
    nulls, at either dose.
  - `SUPPORTED` if either holds:
    - knockout ≥ 15% (stored_outcompeted), < 3% (missing), and detection AUROC ≥ 0.70 and ≥ 0.07 above
      A1;
    - D_wq is above both nulls.
  - **H4a**: W→C in stored_outcompeted < 5% and W→A ≥ 3× W→C.

## 6. Budget

Expected about 8 hours on the GB10 for both models, including the judge (about 15 GB, loaded after the
3B arms). Run limit 12 h, cap 11 h. The degradation ladder cuts, in order and only if the projection
exceeds the cap:
1. instruction-set random draws, 5 → 2;
2. random ablations, 7 → 3;
3. norm-matched draws, 10 → 5;
4. shared-site label-permuted D_wq, 5 → 3;
5. own-dose D_wq nulls, 5 + 5 → 3 + 3;
6. random placebos, 8 + 4 → 4 + 2.

The fresh test set, the seeds, the dose ladder, D_force, the D_ku decomposition, the H2 arms and the
judge are never cut.

## 7. What this run cannot settle

- **Scale.** No 7–9B model is run.
- **Human agreement (κ ≥ 0.8).** The judge measures label robustness. Agreement with people still
  needs the human labels in `judge_disagreements.csv` / `audit_h1a.csv`.
- **Unverified Llama weights.** The Llama weights come from the `unsloth` mirror and have not been
  checked against Meta's repository.
- **Single platform.** Everything runs on one hardware platform.
- **Contamination.** No pretraining-contamination probe is run.

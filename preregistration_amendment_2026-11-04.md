# Pre-registration amendment, 2026-11-04

Amends `experiment_plan.yaml` ("Does hallucination steering beat its own detector?", screening tier).
The plan file is left unchanged on disk; where this document and the plan differ, this document
governs the second run. `main.py` prints this file's SHA-256 (`AMENDMENT:` line) and stores it in
`frozen_config.json`, so every result of the second run is tied to this text.

## 0. Status

- **First run.** Completed 2026-11-03 (`outputs/`, 9,321 s). Its results were read by the authors and
  by a reviewer before this amendment was written. This amendment is therefore **not blind** to
  first-run outcomes. To limit that, it changes only what is needed to fix a reviewer issue, a code
  bug, or a validity problem. A higher abstention score is not a reason for any change.
- **Second run.** Not started. All of its configurations are selected on dev and frozen before any
  test-split forward pass (A2).
- **Unchanged:**
  - the primary metric (`abstention_score`, higher is better) and the scoring code;
  - answer matching and the abstention phrase list;
  - models (now pinned to the commits the first run used) and the prompt templates;
  - greedy decoding, batch size and the seed-0 data splits;
  - every registered condition and control, including the 20 norm-matched random draws, the
    confidence threshold, the confidence prompt and the popularity gate.

## 1. Corrections to the record of the first run

### 1.1 False log line

The first run printed, for each model, `FROZEN_CONFIG ... (written before any test forward pass)`.
That line was false in two ways:

1. **Each model's own unsteered test passes came before its freeze.** These were the probe pass,
   greedy generation, gold-alias scoring, 8-sample labelling and answer activations on PopQA test
   and EntityQuestions. They ran in `label_unsteered`, before `freeze`. No selection used them: dev
   selection read only dev, thresholds were fit on dev, and `tau_store` on train.
2. **Qwen's configuration was written after Llama's test conditions had run.** The models were
   processed one after the other.

No test data entered any selection, but the claim as logged was untrue. The second run freezes
every model and seed in one write before the first test-split pass. Any earlier request raises an
error (A2).

### 1.2 Withdrawn first-run results and their replacements

These first-run results are withdrawn. `reanalyze.py` recomputes them from the saved per-item
records with the amended statistics (A1). The output is in `outputs/reanalysis_2026-11-04/`.

| Withdrawn | Why it is invalid | Replacement (first-run data, A1 statistics) |
|---|---|---|
| All `PAIRED:` p-values and `holm_p` | McNemar on correct vs not-correct does not test the reported score difference. Holm ran over all 41 rows instead of within the registered families. Two "pooled" rows were Llama-only, because Qwen's comparison was undefined and was dropped without a flag. | Sign-flip permutation p on per-question score differences, Holm within families. Pooled rows are `undefined` when any model is undefined. H1 family: knowledge vs coverage-matched threshold: Llama −1.33 [−3.33, 0.67], p = 0.28; Qwen −2.00 [−3.33, −0.33], p = 0.013, Holm 0.038. Gated steering vs gated refusal: Llama −7.33 [−9.67, −5.00]; Qwen −0.42 [−1.33, 0.50], p = 0.47. |
| H1b CI and flag | The coverage-matched CI came from 78 of 2,000 resamples (Llama). The "best" draw was chosen with NaN surpluses passed to `max()`. The percentile flag compared against a NaN percentile. No isotropic draw reached the knowledge dose on dev in either model. | Undefined in both models. Norm-matched: 12/20 (Llama) and 6/20 (Qwen) draws answer more items than unsteered, so their surplus is undefined (< 95% defined). Coverage-matched: no draw was matched on dev. |
| H1d AURC claim ("steered AURC lower") | Each curve was integrated over its own coverage range (Llama: steered 1–29% vs unsteered 1–51%). Lower-coverage ranges have mechanically lower risk. | Common grid 1–26%. Llama: steered AURC 0.376 vs unsteered 0.324, a difference of +0.053 [0.012, 0.098]. **Steering is worse**, the opposite of the withdrawn claim. Qwen: undefined, because knowledge steering answers only 6% of items (3 grid points). |
| Llama H2c (fraction removed = 0.71) | Directional ablation alone raised Llama's abstention by 40 points (49% → 90%) and cut top-tercile accuracy by 14 points. The ratio measures a damaged model near the abstention ceiling, not the refusal actuator. | Undefined (A1 validity rule). Llama's H2b is flagged `ablation_damages_model`. Qwen's H2c (−0.78 [−0.98, −0.61]) passes the rule. |

### 1.3 Substitutions not disclosed in the plan

1. **EntityQuestions in place of TriviaQA / NQ-open.** The goal specification named TriviaQA or
   NQ-open as the second QA set. The plan used EntityQuestions without saying it was a substitute.
   Both are low-coverage sets for 3B models (unsteered coverage 33–51% on PopQA), so the 70% and
   85% coverage points of H1d were unreachable. EntityQuestions stays as registered. TriviaQA is
   added (A10). NQ-open is not used.
2. **DoLa baseline dropped.** The goal specification listed DoLa (contrastive layer decoding) as a
   baseline. It was not implemented in the first run and is **not** added in the second. Neither
   run supports any claim relative to DoLa.
3. **Latency cap not measured.** The goal specification set a latency cap for deployable
   interventions. Neither run measures per-query latency, and no latency cap is evaluated. `TIME:`
   lines record wall-clock per condition for budgeting only.

## 2. Amendments for the second run

Each item names the review item or bug it addresses.

**A1. Statistics** (review 1; bugs in `report.paired_lines`, H1b, H1d, H2c)
- Effect sizes use the paired bootstrap over questions: 2,000 resamples, seed 7, with coverage
  matching and curve construction redone in every resample.
- A CI is reported as `undefined` if fewer than 95% of resamples are valid. It is never computed
  from the survivors.
- p-values come from a two-sided paired sign-flip permutation test on per-question score
  differences: 10,000 flips, seed 11, conditional on the full-sample coverage matching.
- Holm correction applies within the registered families: H1 {H1a, H1c}, H2 {H2b, H2c, H2d}, H3
  {H3a, H3c}, H4 {H4b, H4c, H4d}, and generalisation {H1a, H4b, H4d on EntityQuestions and
  TriviaQA}. The per-model rows are the inference units; pooled rows are descriptive. Other rows
  are exploratory, Holm-corrected within their own family.
- McNemar is kept as a secondary test of accuracy only.
- AURC is compared on one common coverage grid: 1% steps up to the smaller maximum coverage minus
  3 points, fixed from the full sample. It is undefined below 5 grid points.
- H2c is undefined if:
  - the positive control fails; or
  - the ablation alone raises abstention by more than 5 points; or
  - steering raises abstention by less than 5 points.
- `PAIRED:` lines keep their fields and add `perm_p`, `family`, `valid_frac` and `status`. `holm_p`
  now refers to `perm_p` within the family.

**A2. Freezing order** (review 2)
- All pre-test work for both models and all seeds runs first.
- `frozen_config.json` is then written once, with a per-model SHA-256, and the file hash is logged.
- Only then are the test splits unlocked. A guard raises on any earlier test-split pass.
- A `TEST_GUARD` line logs the first test pass.
- TriviaQA's dev split, used for its threshold only, is labelled before the freeze.

**A3. Tables** (review 3). Every condition table and stratum table gains:
- `always_abstain` (score 0);
- gains and losses as item counts and points, and net points;
- surplus per answered item;
- the common-grid AURC difference against unsteered.

The per-condition `aurc` line remains but is marked "own range, not comparable".

**A4. Measurement audit** (review 4)
- `audit_h1a.csv` holds up to 50 rows per (H1a arm × transition vs unsteered), for
  `knowledge_dir_steer` and the coverage-matched threshold. Threshold withdrawals are abstentions
  by policy and are not audited.
- `audit.py judge` labels every output of both arms with an LLM judge. Backends: Anthropic API, or
  a local HF model that is not one of the models under test.
- `audit.py score` reports Cohen's κ for human vs automatic, strict and judge labels. It also
  rescores H1a with both arms under each labelling.
- The registered automatic labels stay primary. Strict and judge labels are sensitivity analyses.
  The κ ≥ 0.8 target and the one-time rule revision clause are unchanged.

**A5. Implementation suspects** (review 5)
- **IDK reference string.**
  - The prompt asks for exactly "I don't know.", and both unsteered models abstain with the period
    (`samples_unsteered.jsonl`).
  - The first run scored `[I, don, 't, know] + end-of-turn` for both tokenizers, i.e.
    end-of-turn right after "know", where the models write ".". This depressed every margin's IDK
    term by an item-dependent amount.
  - The reference is now "I don't know.". The margin's abstain term is
    log[P("I don't know." + eot) + P("I don't know" + eot)]. Qwen under knowledge steering writes
    the form without the period (23 of 25 sampled abstentions), and the second term covers it.
  - `IDK_FORM_CHECK` logs the share of each form.
- **Qwen semantic-entropy probe** (test AUROC 0.41).
  - The probe's sign is applied consistently in the code. The problem is the label: all abstained
    samples formed one cluster, so a question answered "I don't know." 8 times out of 8 had entropy
    0 ("certain").
  - For abstention-heavy Qwen (67% unsteered abstention), low entropy therefore marked unknown
    questions, and the probe learned "will answer" instead of "knows".
  - Abstained and degenerate samples are now singleton clusters. `SEP_LABEL_CHECK` logs whether the
    label predicts train correctness in the expected direction. The probe is never sign-flipped on
    the basis of data.
- **Qwen layer-14 knowledge "no-op"** (dev S1 identical at all three strengths).
  - Every hook now counts its prefill and decode calls.
  - The dev sweep logs, per configuration:
    - first-token KL on 64 dev prompts;
    - the fraction of outputs changed;
    - the added norm, in σ units and as a fraction of rbar;
    - a `no_op` flag (outputs changed < 1% and KL < 1e-3).
  - Direction diagnostics log each direction's top-1 and top-5 coordinate energy and its alignment
    with the residual's dominant coordinates.
  - CPU reproduction (Qwen2.5-3B, pinned revision, greedy-only labels on 128 train questions,
    32 dev prompts, first-run dose 0.4·rbar): see §3.

**A6. H3** (review 6)
- AUROCs predict unsteered correctness among answered items (C vs W). Abstained items no longer
  inflate them.
- New baselines, fitted on answered train items: relation-only (one-hot PopQA property) and
  popularity + relation.
- H3c increments are reported over popularity and over popularity + relation.
- H3a compares the best internal signal with the stronger of `popularity_surprisal_logreg` and
  popularity + relation.
- The one canonical within-decile AUROC is the probability that a random (correct, wrong) pair
  from the same decile is ranked correctly: the pair-weighted mean of per-decile AUROCs. It has a
  bootstrap CI and replaces the unweighted mean over deciles with ≥ 10 per class.
- `rejects_H3_if` now reads: "internal gates keep the canonical within-decile AUROC ≥ 0.65 and beat
  the stronger frequency baseline by ≥ 0.05 in both models".
- First-run data, answered items only (margin only; relation baselines cross-fitted on test, since
  train outcomes were not saved):

  | Model | Margin AUROC | Popularity + relation | Increment of margin over it |
  |---|---|---|---|
  | Llama | 0.80 | 0.81 | +0.034 [0.012, 0.057] |
  | Qwen | 0.77 | 0.88 | +0.002 [−0.008, 0.011] |

  Qwen's relation alone reaches 0.84.

**A7. Gated knockout** (review 7; exploratory)
- On items the margin gate withdraws, output the forced second guess (first-token ban) instead of
  abstaining.
- A per-relation chance baseline estimates the accuracy of a second guess drawn from the relation's
  answer prior (leave-one-out over PopQA, excluding the first answer).
- First-run data:

  | Model | Change vs gated refusal | Second-guess accuracy | Chance accuracy |
  |---|---|---|---|
  | Llama | −27.4 [−30.3, −24.5] points | 8.4% | 4.0% |
  | Qwen | −15.8 [−18.3, −13.1] points | 11.3% | 5.0% |

  This is labelled exploratory and is not part of any family.

**A8. Seeds** (review 8)
- Seeds s ∈ {0, 1, 2}. Seed s redraws train (100/decile), train_reserve and dev (40/decile) from the
  same subject-disjoint pools with rng(10,000 + s); seed 0 reproduces the registered draw exactly.
- Seed s also uses label-sampling seed 1234 + 1000·s, a bootstrap resample of the 128 + 128
  refusal instructions (s ≥ 1), and its own dev selection of (L*, c*), thresholds and null
  multipliers.
- Seeded arms:
  - unsteered;
  - `confidence_threshold_margin`;
  - `knowledge_dir_steer`, `read_dir_steer`, `refusal_dir_steer`;
  - `margin_gated_steer_knowledge`;
  - the three dose-matched nulls.
- D_wq is refit 5 times: r = 0 is registered, r = 1–4 are question-level bootstraps of the gray
  zone. Every refit runs on test at the shared site and at the answer-position site.
- `SEED_CHECK` / `IDENTICAL_PER_SEED` flag any two (condition, seed) cells with identical per-item
  outputs, any seeded condition identical across seeds, and any two conditions with identical
  per-seed scores.
- **Decision rule.**
  - The confirmatory H1a and H1c tests use seed 0 (the registered draw).
  - A seed-0 result counts as *replicated* only if seeds 1 and 2 give a point estimate of the same
    sign. Otherwise it is *inconclusive* and is forwarded to confirmation under the plan's rules.
  - The mean and SD across seeds are reported for every seeded arm.
  - The plan's kill rule (both families) is unchanged.

**A9. Added controls** (review 9)
- `read_dir_steer`: D_read = mean last-prompt residual of unsteered-wrong minus unsteered-correct
  answered train items, at the shared site and norm.
- `dose_matched_null_{shuffled_ku, cov_random, negated_ku}`: shuffled-label D_ku, a
  covariance-shaped random direction z ~ N(0, Σ_L*), and −D_ku.
  - Each is run at the multiple of α* in {0.5, 1, 1.5, 2, 3, 4} whose dev-subset abstention increase
    is closest to knowledge_dir_steer's, with degenerate output ≤ 10%.
  - A null is "matched" if it lands within ±3 points.
  - H1b uses these nulls when no registered isotropic coverage-matched draw is matched on dev. In
    that case the H1b coverage-matched comparison is `undefined` and the null comparison is
    reported in its place.
- `within_question_dir_steer_answer_positions`: D_wq added at generated (answer) positions only.
  - It has its own dev ladder: candidate layers × c ∈ {2, 4, 8, 16, 32, 64} σ^ans, where σ^ans is the SD
    of the answer-token projection.
  - Selection: highest dev S1 with degenerate output ≤ 5% and the A11 caps. Ties go to the smaller
    c, then the middle layer.
  - A matched random ladder (3 draws) runs at the same layer, positions and norms. Its test draws
    serve as the H4b null for this arm.
- `majority_vote_k8` and `best_of_k8` use the 8 saved temperature-1 samples (dev samples are added
  to fit their gates).
  - Majority vote takes the plurality cluster; abstained and degenerate samples are singletons.
  - Best-of-k takes the highest teacher-forced log P(sample + eot).
  - Each is gated at its own τ_dev(λ).

**A10. TriviaQA** (review 10)
- Source: `mandarjoshi/trivia_qa` rc.nocontext validation at commit 0f7faf33.
- 300 dev questions (threshold only) and 1,000 test questions, disjoint, seed 0.
- Directions are transferred from PopQA. Arms: unsteered, knowledge, read, wq and refusal steering
  (with steered margins), threshold, gated knowledge steering, and 20 random draws on a 300-item
  subset.
- H1a, H1c and common-grid H1d with selective accuracy at 50/70/85% are reported. This is the set
  for the ≥ 70% coverage regime.

**A11. Strength and dev-selection caps** (review 11)
- **Strength unit.** Strength is c·σ_L(v), where σ_L(v) is the SD of the unsteered last-prompt
  residual's projection on v̂ over PopQA train.
- **Grid.** c ∈ {2, 4, 8, 16, 32, 64}. The first run chose the largest grid value (0.4·rbar) in
  both models. Along Qwen's layer-14 D_ku, σ is 1.5% of rbar (§3), so the first run's doses
  (0.1–0.4·rbar) are about 7–27σ. The new grid spans about 0.03–0.94·rbar there, past the old
  maximum. Doses too large for a model are ineligible on dev (degeneration or the caps). `alpha_over_rbar` is logged for every configuration so
  doses can be compared with the first run.
- **Norm matching is unchanged.** The shared site adds the absolute norm α* = c*·σ_L*(D_ku) for every
  primary direction and every norm-matched random draw. Dose-ladder rungs add c·σ_L*(D_ku) for every
  direction.
- **Dev eligibility.** Besides the registered criteria (degenerate ≤ 5%, abstention increase 5–30
  points), a configuration must also have:
  - a paired MMLU-dev accuracy delta ≥ −1.5 points, on 300 questions from the cais/mmlu validation
    split (disjoint from the 500 test questions); and
  - a perplexity ratio ≤ 1.05 on 200 dev articles of non-memorised reference text.

  The fallback rules apply the same caps.
- **Reference text.** BBC News articles from May 2025 (`RealTimeData/bbc_news_alltime`, config
  2025-05, commit 8dd1ecdc), published after both models were released. They replace the Alpaca
  outputs in the dev cap (200 dev articles) and in the reported perplexity side effect (200 test
  articles).
- The teacher-forced dose response uses c ∈ {0, 2, 4, 8, 16, 32, 48, 64} σ.

**A12. Refusal ablation** (review 12)
- The registered arm (directional ablation on embed + every block at L_ref) is kept.
- Added variants:
  - mean-ablation on embed + every block: the component along r̂ is set to its mean over the 128
    harmless training prompts, per module;
  - mean-ablation at L_ref only.
- Each variant gets its own positive control and its own `_plus_knowledge_steer` arm.
- Random-direction ablations: 5 (directional), 3 and 3 (mean variants).
- **Primary H2b/H2c arm:** the registered directional variant if it passes A1's validity rule;
  otherwise mean-ablation on all layers; otherwise mean-ablation at L_ref. All three are always
  reported.

**A13. Budget** (review 13; the first run's ladder fired from a cold calibration)
- The first run calibrated on a cold first batch and measured 1.55× the planned time, against
  about 1.0 in reality. It then applied all six degradation steps at once, including cutting test
  from 1,500 to 1,200 items, and finished in 9,321 s of a 14,400 s limit.
- Now:
  - calibration runs after a warm-up batch;
  - the time scale is re-estimated from completed work;
  - each ladder step is decided only when its work is reached, in ladder order.
- Ladder order:
  1. EntityQuestions random subset 300 → 150
  2. TriviaQA random subset 300 → 150
  3. random ablations 11 → 4
  4. drop the alternative layer
  5. label-permuted D_wq 5 → 3
  6. coverage-matched draws 5 → 3
  7. answer-position random draws 3 → 1
  8. PopQA test 1,500 → 1,200 (decided at the freeze, and only if all earlier steps together cannot
     close the gap)
- Test is restored to 1,500 items.
- Run limit 36,000 s, cap 28,800 s. Seeds and the 20 norm-matched draws are never cut.
- Phase B runs the H1a-critical arms for every seed for both models before any ablation.

**A14. Reproducibility** (review 14)
- Model and dataset revisions are pinned in `config.py`:
  - PopQA 098765c7;
  - MMLU c30699e8;
  - Alpaca dce01c9b;
  - TriviaQA 0f7faf33;
  - BBC News 8dd1ecdc;
  - AdvBench at llm-attacks commit 098262ed, sha256 6cd1a5c6…;
  - EntityQuestions zip sha256 015eb64a….
- `setup.py` records the sha256 of every file in `data_cache/`. `main.py` refuses to run on any
  mismatch and logs `DATA_CHECKSUM`, `CODE_HASH` (all sources, the plan, this amendment, the
  lockfile), `AMENDMENT` and `LOCKFILE`.
- `run.sh` writes `requirements.lock` (`pip freeze`) on first setup and installs from it afterwards.
- Every run writes to its own folder `outputs/run_<UTC timestamp>/`, and `main.py` refuses a folder
  that already holds results, so no run can overwrite another. The first run's files stay in
  `outputs/`.
- Every per-item output is saved under `<run folder>/per_item/<model>/`:
  - every condition, seed and split, including dev sweeps;
  - unsteered labels;
  - the 8 samples per question.

## 3. Qwen layer-14 diagnosis (CPU reproduction)

Question: is the first run's Qwen layer-14 "no-op" (dev S1 −13.75 at all three strengths) an
implementation fault?

**Setup**
- Model: Qwen2.5-3B-Instruct at the pinned commit, bf16 on CPU, using this repository's
  hook and steering code.
- D_ku was fitted with greedy-only labels (Qwen's first-run fallback) on 128 questions of the
  registered PopQA train draw: 16 correct, 81 abstained, 31 wrong.
- The direction was added at the last prompt token at 0.1·rbar and 0.4·rbar (the first run's c*). The
  effect was measured on the next-token distribution over 32 dev prompts, against a norm-matched
  isotropic random direction.
- Only layer 14 was completed. Layers 18 and 22 were stopped for time; the first run's dev sweep
  already shows their effect.

**Result at layer 14**

| | KL (first token) | Argmax changed | Δ log P("I", first IDK token) |
|---|---|---|---|
| D_ku, 0.4·rbar | 0.035 | 9% of prompts | −0.06 |
| random, 0.4·rbar | 0.022 | 6% | −0.09 |
| D_ku, 0.1·rbar | 0.003 | 0% | +0.04 |
| random, 0.1·rbar | 0.004 | 0% | −0.04 |

- **The hook fires on every pass** (2 prefill calls per pass), so this is not a hook that never
  runs.
- **The direction is not an outlier-channel direction.**
  - At the last prompt token, one coordinate (dim 1874) carries 32% of the residual's mean
    square, so rbar_14 = 49.1 is inflated by that coordinate.
  - D_ku puts only 0.36% of its energy on the residual's five largest coordinates.
  - Its own top-5 coordinates carry 3.3% of its energy.
- **Spread:** σ_14(D_ku) = 0.72, i.e. 1.5% of rbar. The first run's c = 0.4 dose was about 27σ.
- **Effect:** at 0.4·rbar, D_ku perturbs the model about as much as the random direction, and
  neither raises the probability of the abstention token.

**Conclusion**
- The layer-14 "no-op" is not a code fault. At layer 14 the known/unknown contrast is not read out
  as abstention by the later layers: adding it acts like a random perturbation of the same norm.
  This is consistent with the first run's dev sweep, where c = 0.4 changed abstention by +0.25
  points at layer 14, +13.5 at layer 18 and +25.8 at layer 22.
- No code change follows. The hook and direction diagnostics (A5) stay in, so the second run
  measures this at full size for every layer, model and seed.
- The measurement also shows that rbar is dominated by one coordinate, which is the reason for the
  σ-based strength unit and the {2, …, 64}σ grid (A11).
- Caveats: 128 training questions with greedy-only labels, 32 prompts, one layer.

## 4. Budget statement for the second run

Planned cost at plan rates is about 6–7 hours for two models, against a cap of 8 hours and a limit
of 10. The first run's measured generation rate (about 25 questions/s) matched the plan rate.
Ladder decisions are logged as `BUDGET: at=ladder:<step>` lines and stored in `results.json`.

# Pilot pre-registration: does activation steering do anything a confidence rule can't?

Filed 2026-11-09, before any pilot data were generated. The thresholds below are the constants in
`pilot/config.py`; the rules are implemented in `pilot/signals.py`. Nothing here is tuned on pilot data.

## 1. Purpose

A cheap screen of five sub-questions (SQ1-SQ5). For each one the pilot asks two things: can it be
measured cheaply, and is there any sign of a large effect? Each gets a go / stop signal. A go earns a
full study; it is not a finding. A stop on SQ2 or SQ3 is taken seriously, because the rest depends on them.

The research problem: steering often "reduces hallucination" by making the model say "I don't know"
more often, which a confidence rule (answer only when confident) already does. Steering could add value
in two ways: by catching mistakes the confidence score misses (SQ1, SQ2, SQ4), or by turning wrong
answers into right ones (SQ3, SQ5).

## 2. Shared setup

| Item | Pilot choice |
|---|---|
| Models | Llama-3.1-8B-Instruct (ungated `unsloth` mirror, rev `4699cc75`) and Qwen2.5-7B-Instruct (rev `a09a3545`), bf16, frozen |
| Questions | 3,000: 1,500 TriviaQA (rc.nocontext validation; 100 of them year questions for SQ5) + 1,500 PopQA from 8 relations with single-entity answers (place of birth 250, country 250, capital 200, capital of 150, sport 200, religion 150, father 200, mother 100). No relation above 10% of the pool. Questions the earlier study scored on a test split, and its TriviaQA dev questions, are excluded |
| Prompt | the earlier study's: answer with only the answer, or reply exactly "I don't know." |
| Main answer | greedy, at most 16 new tokens |
| Tries | 18 T=1 draws per question: the first 10 define disagreement questions (SQ2), the last 8 measure answer consistency. Independent draws; the two sets never overlap |
| Confidence | m = log P(answer + end-of-turn) - log P("I don't know." + end-of-turn), teacher-forced |
| Internal state | residual stream at 5 layers (0.25 / 0.4 / 0.5 / 0.6 / 0.75 of depth), at the last prompt token and at the last answer token; attention-head outputs at the last answer token on the SQ1 training folds (for ITI) |
| Grading | automatic alias matching (`scoring.py`) |
| Grading check | two-stage hand check of random answered main answers per model (section 2.1). If it fails, signals are withheld until the grader is fixed and the saved answers re-graded. Signals are not shown before the check (`python -m pilot.analyze`) |
| SQ2 screening extension | 2,969 more questions (same sources and exclusions; questions and PopQA subjects disjoint from the pool) get only the 10 tries per model. The disagreement questions found there then get the full record (main answer, confidence, states, 8 separate tries) |
| Folds | 5 rotating folds, grouped by PopQA subject entity (TriviaQA: by question) |
| One-time use | every pilot question id is written to `pilot_qids.json`; a full study must exclude them |

### 2.1 Grading check: two-stage plan

The summary's bar is "more than about 5% disagreement on answered questions". A single check of 100
answers is coarse at that bar, so the check has two stages. Each model's sheet has 200 random answered
main answers.

1. Label the 100 stage-1 rows.
   - ≤ 3 disagreements: pass.
   - ≥ 8 disagreements: fail.
2. Otherwise, also label the 100 stage-2 rows. Pass if there are ≤ 10 disagreements in all 200 (5%).

Probability of passing, by the grader's true disagreement rate:

| True rate | 2% | 3% | 5% | 7% | 8% | 10% |
|---|---|---|---|---|---|---|
| Two-stage plan | 1.00 | 0.96 | 0.60 | 0.19 | 0.09 | 0.01 |
| Single check of 100, pass if ≤ 5 | 0.98 | 0.92 | 0.62 | 0.29 | 0.18 | 0.06 |

A good grader usually needs only the first 100 labels (about 115 expected at a 2% rate, about 160 at 5%).
Near exactly 5% no affordable sample size can decide reliably.

## 3. Sub-questions and signal rules

### SQ1: do steering methods know anything beyond confidence?

Directions, fitted on folds 0-1 only, at 0.4 depth unless noted:

- **CAA**: mean difference of the residual at the answer letter, A/B prompts completed with the
  "I don't know." option vs the model's own wrong answer (option order randomised), on questions the
  model gets wrong.
- **ITI**: top 48 attention heads by probe accuracy (right vs wrong main answers), mass-mean shift
  scaled by each head's SD, added to those layers' attention output.
- **push**: a plain "say I don't know" push: the mean last-prompt residual under an instruction to
  decline, minus the same under the normal instruction.
- **20 random directions**, which should show no extra knowledge.
- **smart_probe**: the push applied only where the cross-fitted internal mistake detector flags a
  likely error. The flag rate equals the error rate on the training folds.
- **smart_oracle** (added): a planted control. The push is applied where the true label says "wrong",
  with 25% of labels flipped.

Every direction takes the sign that lowers confidence. Its dose is set so that the mean confidence drop
on 128 training-fold questions equals the 30th percentile of their positive margins (within 15%), capped
at the mean residual norm (ITI: alpha <= 240).

**Cheap test** (no generation). On up to 800 held-out questions (answered ones only), delta = steered
minus unsteered confidence. Selectivity = AUROC of delta for right vs wrong main answers, using only pairs
in the same decile of unsteered confidence (0.5 = no knowledge beyond confidence).

**Adjusted selectivity** (robustness). Before the AUROC is computed, delta is regressed on:

- the starting confidence: a cubic, the decile bins, and a piecewise-linear spline with 19 knots;
- the main answer's length in tokens;
- the dataset / PopQA relation.

The AUROC of the residual cannot carry information those covariates carry. It rules out three ways a
direction could look selective without knowing anything: wrong answers sitting lower inside a decile,
being longer, or coming from harder relations. In 20 simulated datasets of the real evaluation size:

| Fake steering effect | Selectivity | Adjusted selectivity |
|---|---|---|
| tracks confidence shape | 0.533 | 0.503 |
| tracks answer length | 0.760 | 0.506 |
| tracks relation | 0.607 | 0.502 |
| real right/wrong knowledge | 0.759 | 0.698 |
| pure noise | 0.501 | 0.501 |

The adjusted check is conservative: real knowledge that overlaps with the covariates loses some signal.
It needs at least 5 answered evaluation questions per covariate (about 40 covariates, so about 200
answers; the plan expects about 560). With fewer, it is undefined and SQ1 cannot reach GO.

**Check.** On 300 of them, real steered answers are generated for CAA, ITI, push, both smart versions and
5 random directions. Two numbers come out of this:

- real selectivity: the same within-decile AUROC, with "still answers" vs "declines" as the score;
- item AUROC: how well the cheap drop predicts each actual switch to "I don't know".

**Signal (per model; overall GO if both GO, STOP if both STOP, else BORDERLINE).**

| Signal | Condition |
|---|---|
| GO | smart_probe selectivity > all 20 random directions, AND its adjusted selectivity > all 20 random directions' adjusted selectivity, AND median item AUROC >= 0.70, AND real smart_probe selectivity > all 5 generated random directions |
| BORDERLINE | smart_probe separates but not after adjustment (suspected artifact); or it separates but the real-answer check fails; or smart_probe does not separate while the planted control does (the test has power, but the detector adds nothing within confidence strata) |
| STOP | neither smart_probe nor the planted control separates from the random directions: the cheap test does not work, and the audit would need full evaluations |

### SQ2: does the inside of the model know about its mistakes when its outputs don't?

Disagreement questions are those where one model is right in >= 8 of its 10 tries and the other in
<= 2. They come from the pool and from the screening extension. Disagreement between two similar-sized
models is rare, so the pool alone might not reach 150. For each model, two logistic regressions predict
whether the main answer is right:

- **outputs only**: margin, mean token log-prob, and, from the separate 8 tries, agreement with the main
  answer, the number of distinct answers, and the share of declines;
- **outputs + detector**: the same plus the detector score. The detector is a linear ridge probe; its
  layer, position and strength are chosen by leave-one-out AUROC inside the training folds.

Both regressions are trained on answered pool questions only, and evaluated on disagreement questions
with an answered main answer:

- pool questions are scored out-of-fold;
- extension questions are scored by models fitted on the whole pool (the extension is never used for
  training, and its PopQA subjects never appear in the pool). The effect is the mean over the two models of AUROC(outputs +
detector) - AUROC(outputs). Its 95% CI comes from a bootstrap over disagreement questions.

| Signal | Condition |
|---|---|
| INSUFFICIENT_DATA | fewer than 150 disagreement questions (the full study needs a much larger pool) |
| GO | gain >= 0.03 and CI lower bound > 0 |
| STOP | CI upper bound < 0.03 |
| BORDERLINE | otherwise |

### SQ3: can combining internal states across rewordings fix wrong answers?

Setup: 500 PopQA questions × 5 wordings (the original + 4 templates per relation). The hand-check
sheet covers every template at least once.

1. **Room for improvement.** Among wrong answers to the original wording, the share answered correctly
   under at least one other wording. For context, the same share using 4 T=1 tries is also reported.
2. **Tuning, on 200 questions.** At the last prompt token, h ← (1 - beta) h + beta × the mean of the 5
   wordings' residuals. The grid is layers {0.4, 0.5, 0.6} of depth × beta {0.5, 1.0}; the setting with
   the most net fixes wins.
3. **Test, on 300 questions.** The best setting is compared with:
   - voting across wordings;
   - voting across repeated tries (the original answer + 4 tries);
   - averaging answer probabilities across wordings;
   - the control: averaging with the 4 rewordings of a different question (same relation, different
     subject).

Counts are fixes (not right → right), breaks (right → not right) and net, relative to the original
wording. They are pooled over the two models.

| Signal | Condition (checked in this order) |
|---|---|
| STOP | fixable share < 10%; or probability averaging's net exceeds internal averaging's by >= 2% of test questions; or the control's net >= internal averaging's |
| GO | fixable share >= 10%, internal fixes >= probability-averaging fixes, and internal breaks <= internal fixes |
| BORDERLINE | otherwise |

### SQ4: can training teach the model to use hidden information?

**Gate.** SQ4 runs in the same job only if SQ2's provisional signal (automatic labels) is GO or
BORDERLINE. Its result is interpreted only after the grading check.

**Design.** On Llama-3.1-8B, two LoRA adapters are trained (rank 16 on q/v, 2 epochs, lr 1e-4, batch
16, same initialisation and data order) on folds 0-2:

- **real-label**: answer iff the main answer is correct;
- **confidence-label**: answer iff the margin is above a threshold that gives exactly the same number of
  answer targets.

The answer target is the model's own main answer. On folds 3-4, each adapter's surplus over the
confidence rule at the same answer rate is computed.

| Signal | Condition |
|---|---|
| GO | surplus(real) - surplus(confidence) > 0 with bootstrap CI lower bound > 0 |
| STOP | CI inside ±2 points (the two match) |
| BORDERLINE | otherwise |

### SQ5: can the model give a vaguer but correct answer?

**Questions.** 200 in total:

- 100 TriviaQA year questions; the coarse answer is the decade;
- 100 PopQA place-of-birth questions; the coarse answer is the city's country (Wikidata P17).

**Scoring.** Fine correct +1, coarse correct +0.5, "I don't know" 0, wrong -1.

**What is compared.** Two routers decide, per question, whether to keep the fine answer or ask the
coarse question instead. Both are a threshold on one score, tuned on one half of the questions and
applied to the other. Both also tune whether to ask the coarse question when the fine answer was
"I don't know".

- **Simple prompt**: the score is the confidence (margin).
- **Attainable ideal pick**: the score is the cross-fitted probability that the fine answer is right,
  from the confidence plus the internal mistake detector. It is trained out-of-fold on the pool; no
  SQ5 labels are used.
- **Oracle pick** (reported as an upper bound only): the better of the model's own fine and coarse
  answers. It needs the right answers, so it is always far ahead, and a go judged against it would be
  automatic.

Signal from the gap, attainable minus simple. The gap is averaged over models, with a bootstrap CI over
the 200 questions:

| Signal | Condition |
|---|---|
| STOP | gap <= 2 points |
| GO | gap >= 5 points and CI lower bound > 0 |
| BORDERLINE | otherwise |

If the gap is not STOP, one steering direction is tried on the questions the simple prompt sends to the
coarse question, and compared with the simple prompt. The direction is the mean residual under the
coarse question minus under the fine question, fitted on one half and applied on the other; its strength
is chosen from {4, 8} × SD.

## 4. Changes from the summary, and why

1. **12-hour GB10 budget.** Shared prompt prefill for the 18 tries: one prefill per question instead of
   18, checked against plain decoding at start-up, with a fallback to plain sampling. This cuts sampling
   cost by about 4× and leaves the samples' distribution unchanged. SQ4 runs in the same job. The
   projected total is about 4.8 h, 7.3 h if the sampler falls back, and 9.3 h at half the assumed
   throughput. The cap is 11 h and the hard limit 12 h. If the projection exceeds the cap,
   pre-registered ladder steps apply, in order:
   1. 20 → 10 random directions;
   2. SQ3 on one model;
   3. SQ1 evaluation 800 → 500 questions;
   4. 1 LoRA epoch;
   5. SQ2 screening extension halved;
   6. SQ1 on one model;
   7. skip SQ4.
2. **No LLM judge.** Qwen2.5-7B, the earlier study's judge, is now a model under test. The grading check
   is a hand check, enforced by `pilot.analyze`.
3. **Two-stage grading check** (section 2.1) instead of a single 100-answer check. It keeps the summary's
   5% bar and its 100 answers as the first stage, but lets an 8% grader through half as often (9% vs
   18%), and it asks for 100 more labels only when the first 100 are ambiguous.
4. **SQ2 screening extension** (+2,969 questions, 10 tries each). It protects the 150-question floor
   for about 25 minutes of GPU time. Disagreement status always comes from the screening tries; the
   follow-up draws its own, separate 8 tries for consistency.
5. **SQ5 uses an attainable ideal.** The summary's ideal pick knows which answers are right, so a "go"
   would be automatic and uninformative. The decision now compares the simple prompt with the best pick
   the model's own signals allow (confidence + internal detector). This asks the research problem's
   question, whether reaching inside beats the confidence rule, for the choice of detail. The oracle gap
   is still reported as an upper bound.
6. **CAA and ITI are re-implemented** following the papers and fitted on pilot data: CAA on "decline vs
   own wrong answer" A/B pairs, ITI on the model's own right and wrong answers. Using the papers' own
   datasets (CAA's A/B sets, TruthfulQA) would need extra downloads and targets other behaviours.
7. **Planted positive control in SQ1.** The detector-gated "smart" version carries extra knowledge only
   if the detector does, which is exactly what SQ2 tests. The planted control lets a failure of the test
   be told apart from a detector with no extra signal.
8. **Adjusted selectivity in SQ1** (section 3). Matching on confidence deciles alone leaves room for
   artifacts; a GO now also needs the separation to survive adjustment for confidence shape, answer
   length and dataset.
9. **Topic cap.** It is enforced on PopQA relations. TriviaQA has no topic labels and is treated as
   mixed-topic. The 100 year questions put about 7% year answers in the TriviaQA half (natural rate
   about 4%).
10. **SQ2 evaluation set.** Only disagreement questions whose main answer is an answer are used: a main
    answer of "I don't know" is neither right nor wrong.
11. **SQ3 rewordings are PopQA-only.** These questions can be templated; TriviaQA would need a model to
    paraphrase it. Probability averaging chooses among the actual answers ("I don't know." only if every
    wording declined), so that it is a strong fixer and not a disguised abstainer.
12. **SQ5 thresholds.** "Well short of the ideal" is made concrete as a gap of >= 5 points (CI lower
    bound > 0), with 2-5 points as borderline.
13. **Operational go / stop rules.** The summary's "clearly", "near zero with a tight margin" and
    "match" are made concrete as in section 3.

## 5. What the pilot can and cannot tell

It can tell whether each sub-question is measurable, whether there is enough data (disagreement
questions, fixable wrong answers) and whether a large effect is plausible. It cannot tell whether a small
effect is real, or whether results hold on other models.

Known weak points:

- Near the 5% bar the grading check is still uncertain: a grader at exactly 5% passes 60% of the time.
- SQ2 may still fall short of 150 if the two models agree more than expected; it then reports
  INSUFFICIENT_DATA. The extension questions are scored by models fitted on the whole pool rather than
  out-of-fold, a small difference in training size.
- SQ5's attainable pick depends on the detector. If SQ2 finds nothing, SQ5 will most likely stop too,
  which is consistent, not independent evidence.
- Confounding inside confidence deciles: the adjusted selectivity removes confidence shape, answer length
  and dataset / relation (linearly, in the covariates listed). Other differences between right and wrong
  answers that a specific direction happens to track, and random directions do not, are not ruled out.
- The cheap SQ1 test sees declines, not changed answers.
- At matched doses, random directions may need large norms.
- Wikidata gives today's country, not historical ones.
- UK cities accept any constituent country.
- Signals are screening decisions, with no multiplicity correction.

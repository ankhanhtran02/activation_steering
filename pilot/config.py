"""Every constant of the pilot. The go / stop thresholds below are the ones written in
pilot/PREREGISTRATION.md; they are fixed before anything runs and are never tuned on pilot data.

Layer convention (as in model_utils): layer L is model.model.layers[L]; its output residual is
hidden_states[L + 1].
"""
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SMOKE = os.environ.get("PILOT_SMOKE") == "1"    # CPU smoke test of the code paths only (pilot/smoke_test.py)
DATA_DIR = Path(os.environ["STEER_DATA_DIR"]) if os.environ.get("STEER_DATA_DIR") else ROOT / "data_cache"
OUTPUTS_ROOT = ROOT / "outputs_pilot"
RUN_ID = os.environ.get("PILOT_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = Path(os.environ["PILOT_OUT_DIR"]) if os.environ.get("PILOT_OUT_DIR") else OUTPUTS_ROOT / f"run_{RUN_ID}"
MANIFEST_PATH = Path(os.environ["PILOT_MANIFEST"]) if os.environ.get("PILOT_MANIFEST") else ROOT / "pilot_manifest.json"
PREVIOUSLY_SCORED_PATH = ROOT / "previously_scored_qids.json"   # test questions of the earlier study, excluded
PILOT_QIDS_PATH = ROOT / "pilot_qids.json"                       # every pilot question: a full study must exclude them
RESUME = os.environ.get("PILOT_RESUME") == "1"                   # reuse finished phases found in OUT_DIR


def latest_run_dir():
    runs = sorted(p for p in OUTPUTS_ROOT.glob("run_*") if (p / "results.json").exists())
    return runs[-1] if runs else None


PILOT_SEED = 20261109

# ---------------------------------------------------------------- models
# Two open models of similar size (the summary's example pair). Revisions pinned 2026-11-09 from the
# Hugging Face API. The Llama weights come from the ungated unsloth mirror of
# meta-llama/Llama-3.1-8B-Instruct (no token needed); Qwen2.5-7B-Instruct was the earlier study's judge
# and is already in the GPU machine's cache.
MODELS = [
    {"name": "llama31_8b_instruct", "hf_id": "unsloth/Llama-3.1-8B-Instruct",
     "revision": "4699cc75b550f9c6f3173fb80f4703b62d946aa5",
     "n_layers": 32, "d_model": 4096, "n_heads": 32, "head_dim": 128},
    {"name": "qwen25_7b_instruct", "hf_id": "Qwen/Qwen2.5-7B-Instruct",
     "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
     "n_layers": 28, "d_model": 3584, "n_heads": 28, "head_dim": 128},
]
MODEL_FILE_PATTERNS = ["*.json", "*.jinja", "*.safetensors", "tokenizer*", "*.model", "*.txt"]

# Depth fractions. Recorded states: 5 layers. Steering (CAA, push, random, SQ5): 0.4 depth (CAA's choice
# of layer 13/32 for 7B models). SQ3 averages states at 0.4 / 0.5 / 0.6 depth. All are recorded layers.
RECORD_FRACTIONS = (0.25, 0.4, 0.5, 0.6, 0.75)
STEER_FRACTION = 0.4
SQ3_FRACTIONS = (0.4, 0.5, 0.6)


def layer_plan(spec):
    n = spec["n_layers"]
    rec = sorted({round(f * n) for f in RECORD_FRACTIONS})
    plan = {"record_layers": rec, "steer_layer": round(STEER_FRACTION * n),
            "sq3_layers": [round(f * n) for f in SQ3_FRACTIONS]}
    assert plan["steer_layer"] in rec and all(L in rec for L in plan["sq3_layers"])
    return plan


# ---------------------------------------------------------------- data
DATA_FILES = {"popqa_full": "pilot_popqa_full.jsonl",          # PopQA with object ids (setup_pilot.py)
              "triviaqa": "triviaqa_validation.jsonl",          # same file and pin as the earlier study
              "wikidata_country": "pilot_wikidata_city_country.jsonl"}
POPQA_PIN = {"repo": "akariasai/PopQA", "split": "test", "revision": "098765c79ea10a2cb19c828324e33281b8336ec0"}
TRIVIAQA_PIN = {"repo": "mandarjoshi/trivia_qa", "config": "rc.nocontext", "split": "validation",
                "revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"}
WIKIDATA_API = "https://www.wikidata.org/w/api.php"   # wbgetentities: P17 (country), best-ranked statements
HTTP_USER_AGENT = "activation-steering-pilot-setup/1.0 (research pilot)"

# About 3,000 short factual questions: 1,500 TriviaQA + 1,500 PopQA. PopQA relations with clean,
# single-entity answers only (the earlier study found genre / producer / director / screenwriter /
# composer / author labels ambiguous, and occupation has many valid answers). No relation exceeds
# 10% of the pool. TriviaQA carries no topic labels; it is treated as mixed-topic.
N_TQA = 1500
POPQA_QUOTAS = {"place of birth": 250, "country": 250, "capital": 200, "capital of": 150, "sport": 200,
                "religion": 150, "father": 200, "mother": 100}
TOPIC_CAP_FRAC = 0.10
N_SQ5_YEAR = 100          # TriviaQA questions with a year answer (coarse answer: the decade)
N_SQ5_POB = 100           # PopQA place-of-birth questions (coarse answer: the city's country, Wikidata P17)
N_SQ3 = 500               # PopQA questions reworded with templates
N_SQ3_TUNE = 200
N_FOLDS = 5               # rotating folds, grouped by PopQA subject entity (TriviaQA: by question)

# ---------------------------------------------------------------- decoding / recording
QA_MAX_NEW_TOKENS = 16
N_DIS_TRIES = 10          # T=1 tries that define disagreement questions (SQ2)
N_CON_TRIES = 8           # separate T=1 tries for answer consistency (never used to pick questions)
SAMPLE_SEED = 1234
SAMPLE_BATCH_PROMPTS = 32  # 32 prompts x 18 tries = 576 sequences per sampling batch

# Grading check: a two-stage plan around the summary's 5% bar (100 answers, then 100 more only when
# the first 100 are ambiguous). Probability of passing at a true disagreement rate of 2% / 3% / 5% /
# 7% / 8% / 10%: 1.00 / 0.96 / 0.60 / 0.19 / 0.09 / 0.01 (a single check of 100 with "<= 5" passes
# 0.98 / 0.92 / 0.62 / 0.29 / 0.18 / 0.06). Expected labels: ~115 at 2%, ~160 at 5%.
GRADING_STAGE_N = 100            # answers per stage per model (200 rows are written; stage 2 only if asked)
GRADING_STAGE1_PASS_MAX = 3      # <= 3 disagreements in the first 100: pass
GRADING_STAGE1_FAIL_MIN = 8      # >= 8 in the first 100: fail
GRADING_MAX_DISAGREE = 0.05      # otherwise label stage 2: pass if <= 5% of all labelled rows (10 of 200)

# SQ2 screening extension: disagreement questions are rare between two similar-sized models, so 3,000
# more questions (disjoint questions and PopQA subjects) are screened with the 10 tries only. Only the
# disagreement questions found there get the full record (main answer, confidence, states, 8 separate
# tries). They are never used to train the detector or the combiners (fold EXT_FOLD).
N_EXT_TQA = 1500
EXT_POPQA_QUOTAS = {"place of birth": 250, "country": 300, "capital": 250, "capital of": 150, "sport": 250,
                    "religion": 125, "father": 125, "mother": 50}   # filled up to what is available
EXT_FOLD = 5
EXT_SAMPLE_SEED = 4321
FOLLOWUP_SAMPLE_SEED = 5678
EXT_EXPECTED_DISAGREE = 300      # budget estimate only

# ---------------------------------------------------------------- SQ1: cheap selectivity test
SQ1_TRAIN_FOLDS = (0, 1)   # directions, ITI heads and doses come from these folds; evaluation from 2-4
SQ1_N_TUNE = 128           # answered train-side items used to set each direction's dose
SQ1_N_EVAL = 800           # evaluation questions (answered ones enter the test)
SQ1_N_GEN = 300            # of those, answered questions that also get real steered answers
SQ1_N_RANDOM = 20
SQ1_N_RANDOM_GEN = 5       # random directions that also get real steered answers
SQ1_RANDOM_SEED_BASE = 5000
SQ1_FORCE_N = 500          # questions for the plain "say I don't know" push
SQ1_CAA_MAX_PAIRS = 400
SQ1_CAA_MIN_PAIRS = 20
# Dose: every direction is scaled so that its mean confidence drop on the tune items equals the drop
# that, applied uniformly, would turn 30% of the confidently answered tune items into "I don't know"
# (the 30th percentile of positive unsteered margins). Rank statistics below do not depend on the dose;
# matching keeps the real-answer check comparable across directions.
SQ1_DOSE_FLIP_FRAC = 0.30
SQ1_DOSE_TOL = 0.15
SQ1_DOSE_MAX_ITERS = 6
SQ1_ALPHA_INIT_SIGMA = 4.0     # first try: 4 SD of the residual's projection on the direction
SQ1_ALPHA_MAX_RNORM = 1.0      # cap: the mean residual norm at the steering layer
ITI_TOP_K = 48                 # ITI defaults for 7B models
ITI_ALPHA_INIT = 15.0
ITI_ALPHA_MAX = 240.0
ITI_VAL_FRAC = 0.3
ITI_MIN_CLASS = 10             # right and wrong answers needed on the training folds
SQ1_N_BINS = 10                # confidence strata (deciles of the unsteered margin)
SQ1_ORACLE_NOISE = 0.25        # planted control: gate on the true error label with 25% of labels flipped
SQ1_ADJ_MIN_ROWS_PER_COV = 5   # adjusted selectivity needs >= 5 answered questions per covariate (else undefined: no GO)
SQ1_GO_ITEM_AUROC = 0.70       # cheap confidence drop must predict the real "I don't know" flips
SQ1_MIN_FLIPS = 10

# ---------------------------------------------------------------- SQ2: internal detector vs outputs
SQ2_HI, SQ2_LO = 8, 2          # one model right >= 8/10, the other <= 2/10
SQ2_MIN_DISAGREE = 150
SQ2_GO_DELTA = 0.03            # AUROC gain of outputs + detector over outputs only
RIDGE_LAMBDAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]   # x mean kernel eigenvalue, chosen by LOO inside training folds
N_BOOT = 2000
BOOT_SEED = 7

# ---------------------------------------------------------------- SQ3: averaging across rewordings
SQ3_STRENGTHS = [0.5, 1.0]     # h <- (1 - beta) h + beta * mean over the 5 wordings (last prompt token)
SQ3_N_VOTE_TRIES = 4           # canonical answer + 4 of the disagreement tries = 5 votes, as for wordings
SQ3_MIN_FIXABLE = 0.10
SQ3_CLEAR_LOSS = 0.02          # "clearly loses": probability averaging's net gain exceeds internal's by >= 2% of test questions
SQ3_N_PARA_CHECK = 60

# ---------------------------------------------------------------- SQ4: LoRA with real vs confidence labels
SQ4_MODEL = MODELS[0]["name"]
SQ4_TRAIN_FOLDS = (0, 1, 2)
LORA_RANK = 16
LORA_ALPHA = 32
LORA_TARGETS = ("q_proj", "v_proj")
LORA_LR = 1e-4
LORA_EPOCHS = 2
LORA_BATCH = 16
LORA_SEED = 99
SQ4_SESOI = 2.0                # points of the +1/0/-1 score: "the two match" if the 95% CI is inside +/- 2
SQ4_RUN_ON = ("GO", "BORDERLINE")   # provisional SQ2 signals that unlock SQ4

# ---------------------------------------------------------------- SQ5: coarser answers
COARSE_CREDIT = 0.5            # fine correct +1, coarse correct +0.5, "I don't know" 0, wrong -1
# The "ideal pick" that decides SQ5 is attainable: the same coarse prompt, routed by the model's own
# cross-fitted confidence + internal-detector score instead of confidence alone. (An oracle that knows
# which answers are right is always far ahead, so a go against it would be automatic; it is reported as
# an upper bound only.)
SQ5_STOP_GAP = 2.0             # points: simple prompt within 2 points of the attainable pick -> stop
SQ5_GO_GAP = 5.0               # points: >= 5 points short (with CI lower bound > 0) -> go
SQ5_STEER_C = [4.0, 8.0]       # strengths (x SD of the projection) for the single steering direction

# ---------------------------------------------------------------- compute budget (GB10, 12 h)
RUN_LIMIT_SECONDS = 12 * 3600
BUDGET_CAP_SECONDS = 11 * 3600
HARD_STOP_SECONDS = 11.5 * 3600     # no new phase starts after this
ANALYSIS_RESERVE_SECONDS = 900
# Degradation ladder, pre-registered order; a step is applied only if the projection exceeds the cap,
# and only to phases that have not started.
DEGRADATION_LADDER = [
    "sq1_random_10",       # SQ1 random directions 20 -> 10
    "sq3_single_model",    # SQ3 on the first model only
    "sq1_eval_500",        # SQ1 evaluation questions 800 -> 500
    "sq4_one_epoch",       # SQ4 LoRA epochs 2 -> 1
    "sq2_extension_half",  # SQ2 screening extension 3,000 -> 1,500 questions
    "sq1_single_model",    # SQ1 on the first model only
    "sq4_skip",            # SQ4 not run
]

# ---------------------------------------------------------------- smoke test (CPU, tiny models)
if SMOKE:
    MODELS = []
    for _entry in [e for e in os.environ.get("PILOT_SMOKE_MODELS", "").split(";") if e]:
        _name, _hf, _rev, _nl, _d, _nh, _hd = _entry.split("|")
        MODELS.append({"name": _name, "hf_id": _hf, "revision": _rev, "n_layers": int(_nl), "d_model": int(_d),
                       "n_heads": int(_nh), "head_dim": int(_hd)})
    SQ4_MODEL = MODELS[0]["name"] if MODELS else SQ4_MODEL
    N_TQA = 60
    POPQA_QUOTAS = {k: 8 for k in POPQA_QUOTAS}
    N_SQ5_YEAR, N_SQ5_POB = 6, 6
    N_SQ3, N_SQ3_TUNE = 24, 10
    QA_MAX_NEW_TOKENS = 8
    SAMPLE_BATCH_PROMPTS = 8
    GRADING_STAGE_N = 10
    N_EXT_TQA = 40
    EXT_POPQA_QUOTAS = {k: 5 for k in EXT_POPQA_QUOTAS}
    EXT_EXPECTED_DISAGREE = 10
    SQ1_N_TUNE, SQ1_N_EVAL, SQ1_N_GEN = 12, 60, 16
    SQ1_N_RANDOM, SQ1_N_RANDOM_GEN = 3, 2
    SQ1_FORCE_N, SQ1_CAA_MAX_PAIRS, SQ1_CAA_MIN_PAIRS = 20, 20, 3
    SQ1_DOSE_MAX_ITERS = 3
    ITI_TOP_K = 6
    ITI_MIN_CLASS = 3
    SQ1_ADJ_MIN_ROWS_PER_COV = 1
    SQ1_MIN_FLIPS = 2
    SQ2_MIN_DISAGREE = 5
    N_BOOT = 200
    SQ3_N_PARA_CHECK = 8
    LORA_EPOCHS, LORA_BATCH = 1, 8
    SQ4_RUN_ON = ("GO", "BORDERLINE", "STOP", "INSUFFICIENT_DATA", "UNTESTABLE")   # always exercise SQ4

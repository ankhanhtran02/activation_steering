"""Pre-registered constants from experiment_plan.yaml (single source for every module), plus the
changes filed in preregistration_amendment_2026-11-04.md (marked "amendment Ax") and in
preregistration_amendment_2026-11-07_final_run.md (the final, decisive run; marked "final Fx")."""
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SMOKE = os.environ.get("STEER_SMOKE") == "1"     # CPU smoke test of the code paths only (see smoke_test.py)
DATA_DIR = Path(os.environ["STEER_DATA_DIR"]) if os.environ.get("STEER_DATA_DIR") else ROOT / "data_cache"
# Every run writes to its own folder outputs/run_<UTC timestamp>/ (run.sh passes STEER_OUT_DIR and
# STEER_RUN_ID and points outputs/latest at it); main.py refuses a folder that already holds results.
OUTPUTS_ROOT = ROOT / "outputs"
RUN_ID = os.environ.get("STEER_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = Path(os.environ["STEER_OUT_DIR"]) if os.environ.get("STEER_OUT_DIR") else OUTPUTS_ROOT / f"run_{RUN_ID}"
MANIFEST_PATH = Path(os.environ["STEER_MANIFEST"]) if os.environ.get("STEER_MANIFEST") else ROOT / "run_manifest.json"
FROZEN_CONFIG_PATH = OUT_DIR / "frozen_config.json"
RESULTS_PATH = OUT_DIR / "results.json"
AUDIT_PATH = OUT_DIR / "audit.csv"
AUDIT_H1A_PATH = OUT_DIR / "audit_h1a.csv"
JUDGE_DISAGREEMENTS_PATH = OUT_DIR / "judge_disagreements.csv"   # final F9: rows for the human tie-break
VERDICTS_PATH = OUT_DIR / "verdicts.json"                        # final F12: the decision table
PER_ITEM_DIR = OUT_DIR / "per_item"                     # amendment A14: every per-item output
AMENDMENT_PATH = ROOT / "preregistration_amendment_2026-11-04.md"
FINAL_AMENDMENT_PATH = ROOT / "preregistration_amendment_2026-11-07_final_run.md"
AMENDMENT_PATHS = [AMENDMENT_PATH, FINAL_AMENDMENT_PATH]
# final F1: every question id that run 1 or run 2 scored on a test split; the final run never scores them
PREVIOUSLY_SCORED_PATH = ROOT / "previously_scored_qids.json"
LOCKFILE_PATH = ROOT / "requirements.lock"


def latest_run_dir():
    """Newest outputs/run_* folder that holds a results.json (the offline tools' default). The first
    run (2026-11-03) predates per-run folders and wrote straight into outputs/, which is the fallback."""
    runs = sorted(p for p in OUTPUTS_ROOT.glob("run_*") if (p / "results.json").exists())
    return runs[-1] if runs else OUTPUTS_ROOT


GLOBAL_SEED = 0

# Model revisions are pinned (amendment A14); they equal the commits the first run resolved.
MODELS = [
    {
        "name": "llama32_3b_instruct",
        "hf_id": "unsloth/Llama-3.2-3B-Instruct",  # ungated mirror of meta-llama/Llama-3.2-3B-Instruct
        "revision": "006f5dcd1393c3add266de40994ba96225e9689d",
        "n_layers": 28,
        "d_model": 3072,
        "steering_candidate_layers": [11, 14, 17],
        "refusal_candidate_layers": [8, 11, 14, 17],
    },
    {
        "name": "qwen25_3b_instruct",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "n_layers": 36,
        "d_model": 2048,
        "steering_candidate_layers": [14, 18, 22],
        "refusal_candidate_layers": [11, 14, 18, 22],
    },
]

# ---------------------------------------------------------------- data sources (amendment A14)
# HF datasets are loaded at a pinned commit; URL sources are verified byte-for-byte. setup.py
# records the sha256 of every file it writes to data_cache/ in run_manifest.json and main.py
# refuses to run if a file no longer matches.
DATASET_PINS = {
    "popqa": {"repo": "akariasai/PopQA", "config": None, "split": "test",
              "revision": "098765c79ea10a2cb19c828324e33281b8336ec0"},
    "mmlu": {"repo": "cais/mmlu", "config": "all", "split": ["test", "validation"],
             "revision": "c30699e8356da336a370243923dbaf21066bb9fe"},
    "alpaca": {"repo": "tatsu-lab/alpaca", "config": None, "split": "train",
               "revision": "dce01c9b08f87459cf36a430d809084718273017"},
    "triviaqa": {"repo": "mandarjoshi/trivia_qa", "config": "rc.nocontext", "split": "validation",
                 "revision": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"},
    # BBC News articles published in May 2025: after the release of both models (Llama-3.2 Sep 2024,
    # Qwen2.5 Sep 2024), so the text cannot be memorised (amendment A11).
    "bbc_news": {"repo": "RealTimeData/bbc_news_alltime", "config": "2025-05", "split": "train",
                 "revision": "8dd1ecdc92ac43f9c04a3da3e945537dbb08179b"},
    "entity_questions": {"url": "https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip",
                         "sha256": "015eb64a12fa2f9b2f95d2c78bba69779e0d200dbbd4b0772f3ce57bc00da9c5"},
    "advbench": {"url": ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
                         "098262edf85f807224e70ecd87b9d83716bf6b73/data/advbench/harmful_behaviors.csv"),
                 "sha256": "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1"},
}
DATA_FILES = {"popqa": "popqa.jsonl", "entity_questions": "entity_questions_test.jsonl",
              "advbench": "advbench.jsonl", "alpaca": "alpaca_noinput.jsonl", "mmlu": "mmlu_test.jsonl",
              "mmlu_dev": "mmlu_validation.jsonl", "triviaqa": "triviaqa_validation.jsonl",
              "bbc_news": "bbc_news_2025_05.jsonl"}

# final F9: local LLM judge for every test-split output (not one of the models under test; a third
# model size, different weights). Pinned like the models; downloaded by setup.py.
# files fetched from every model repo by setup.py (the run.sh preflight checks the cache with the same list)
MODEL_FILE_PATTERNS = ["*.json", "*.jinja", "*.safetensors", "tokenizer*", "*.model", "*.txt"]
JUDGE = {"name": "qwen25_7b_instruct_judge", "hf_id": "Qwen/Qwen2.5-7B-Instruct",
         "revision": "a09a35458c702b33eeacc393d103063234e8bc28"}
JUDGE_BATCH_SIZE = 64
JUDGE_MAX_NEW_TOKENS = 3
JUDGE_N_DISAGREEMENT_ROWS = 300      # per model, stratified by condition, for the human tie-break

# ---------------------------------------------------------------- prompts
SYSTEM_PROMPT = "You are a concise assistant that answers factual questions."
USER_TEMPLATE_BASE = (
    "Answer the question with only the answer (a name, place, date, or short phrase). "
    "If you do not know the answer, reply exactly: I don't know.\n"
    "Question: {question}"
)
USER_TEMPLATE_CONFIDENCE = (
    "Answer the question with only the answer (a name, place, date, or short phrase). "
    "Scoring: a correct answer earns 1 point, \"I don't know\" earns 0 points, and a wrong "
    "answer loses {lam} point(s). Answer only if you are more than {t_pct}% confident that "
    "your answer is correct; otherwise reply exactly: I don't know.\n"
    "Question: {question}"
)
# Not specified by the plan; fixed here and used for every MMLU configuration.
USER_TEMPLATE_MMLU = (
    "Answer the following multiple-choice question. Reply with only the letter "
    "(A, B, C, or D).\n\n{question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}"
)
# Neutral framing for the non-memorised perplexity text (amendment A11).
USER_TEMPLATE_REFERENCE = "Write a short news article with the headline: {title}"
# Amendment A5: the prompt asks for exactly "I don't know." and both unsteered models emit it with
# the period; the first run scored the margin against "I don't know" without it. The canonical
# reference now carries the period; the margin's abstain term is the log-sum-exp over both written
# forms, so a steered model that drops the period is still measured against its own abstention.
# final F4: the uniform abstention push D_force is the mean (within question) of the last-prompt
# residual under this instruction minus under the base instruction; it carries "being told to
# abstain" and no known/unknown contrast (every train question contributes once to both sides).
USER_TEMPLATE_FORCE_IDK = (
    "Answer the question with only the answer (a name, place, date, or short phrase). "
    "If you do not know the answer, reply exactly: I don't know.\n"
    "Question: {question}\n"
    "Do not answer this question. Reply exactly: I don't know."
)
IDK_STRING = "I don't know."
IDK_VARIANTS = ["I don't know.", "I don't know"]
LLAMA_DATE_STRING = "26 Jul 2024"
CONFIDENCE_PROMPT_INSTANCES = {
    "confidence_prompt_t050": {"lam": 1, "t_pct": 50},
    "confidence_prompt_t075": {"lam": 3, "t_pct": 75},
}

# ---------------------------------------------------------------- decoding
QA_MAX_NEW_TOKENS = 16
INSTR_MAX_NEW_TOKENS = 32
BATCH_SIZE = 64
SAMPLE_BATCH_PROMPTS = 32          # 32 prompts x 8 samples = 256 sequences per sampling batch
N_LABEL_SAMPLES = 8
SAMPLE_SEED = 1234

# ---------------------------------------------------------------- data
SPLIT_SALT = "S9-split"
SPLIT_FRACTIONS = (0.60, 0.20, 0.20)
# final F1: the PopQA test set is every test-pool question that neither earlier run scored (runs 1
# and 2 both scored the first 120 per decile of the seed-0 test permutation): 1,715 questions, at
# least 149 per decile. TEST_MAX_PER_DECILE caps it (None = all; only the smoke test sets a cap).
N_PER_DECILE = {"train": 100, "train_reserve": 100, "dev": 40}
TEST_MAX_PER_DECILE = None
DEV_SUBSET_PER_DECILE = 30          # 300-item coverage-matching dev subset
DOSE_ITEMS_PER_DECILE = 30          # 300 unsteered-answered test items for the dose analysis
EQ_PER_RELATION = 25
EQ_RELATIONS = ["P17", "P19", "P20", "P26", "P36", "P40", "P50", "P69", "P106", "P112",
                "P127", "P131", "P136", "P159", "P170", "P175", "P176", "P264", "P276",
                "P407", "P413", "P495", "P740", "P800"]
EQ_SUBSET_N = 300
N_ADV_TRAIN, N_ADV_HELDOUT = 128, 100
# final F7: 300 held-out benign Alpaca instructions (was 100; the first 100 are the same items), so
# H2d's +5-point spillover prediction is testable (run 2: 5/100 vs 0/100, Fisher p ~ 0.06)
N_ALPACA_TRAIN, N_ALPACA_BENIGN = 128, 300
PPL_MAX_RESPONSE_TOKENS = 128
N_MMLU = 500                        # test (side effects), cais/mmlu "all" test split
N_MMLU_DEV = 300                    # amendment A11: dev cap, cais/mmlu "all" validation split (disjoint)
N_REF_DEV, N_REF_TEST = 200, 200    # amendment A11: BBC 2025-05 articles for the perplexity cap / report
REF_MIN_CHARS = 600
# amendment A10: TriviaQA (rc.nocontext validation) as a high-coverage transfer set
TQA_DEV_N, TQA_TEST_N, TQA_SUBSET_N = 300, 1000, 300

# ---------------------------------------------------------------- seeds (amendment A8)
# A seed s redraws the PopQA train / train_reserve / dev questions from their subject-disjoint
# pools (seed 0 reproduces the registered draw), uses label-sampling seed SAMPLE_SEED + 1000 s, and
# resamples the refusal-direction instructions (s >= 1). The test sets never change.
N_SEEDS = 3
SEED_SPLIT_BASE = 10_000
SEED_SAMPLE_STRIDE = 1000
# final F3/F5: the seeded arms are the H1-critical ones (the three old dose-matched nulls are replaced
# by the dose-matched uniform push D_force; gated steering also runs at its dev-tuned strength)
SEEDED_CONDITIONS = ["unsteered", "confidence_threshold_margin", "knowledge_dir_steer", "read_dir_steer",
                     "refusal_dir_steer", "margin_gated_steer_knowledge", "margin_gated_steer_knowledge_tuned",
                     "dose_matched_placebo_force"]
N_WQ_REFITS = 1                     # final: D_wq refits r >= 1 dropped (stability is not a registered question)
WQ_REFIT_SEED_BASE = 6000

# ---------------------------------------------------------------- steering
# Amendment A11: the strength of every direction v at layer L is alpha = c * sigma_L(v), where
# sigma_L(v) is the SD over PopQA train of the unsteered residual's projection on v_hat (last prompt
# token; answer tokens for the answer-position D_wq). The grid extends past the old maximum. The
# shared site still adds one absolute norm alpha* = c* sigma_{L*}(D_ku) for every primary direction
# and the norm-matched random draws, so the registered norm matching is unchanged.
# Grid set from the CPU measurement in amendment section 3: along Qwen's layer-14 D_ku, sigma is
# 1.5 % of rbar, so the first run's doses 0.1-0.4 rbar are about 7-27 sigma; {2..64} sigma spans
# about 0.03-0.94 rbar there, and doses that are too large for a model are ineligible on dev.
C_GRID = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
# final F6: the teacher-forced dose response uses a finer grid restricted to doses that are clean on
# dev at (L*, c) (degenerate <= 5%); run 2 fitted slopes up to 64 sigma where Llama was 100% degenerate
DOSE_C_GRID = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]
COV_MATCH_MULTIPLIERS = [1, 2, 4]
# final F3: 10 norm-matched isotropic draws (registered null of H1b-norm-matched and H4b); run 2 showed
# they barely move abstention, so the dose-matched placebos below carry H1b
N_RANDOM_DRAWS = 10
N_TRANSFER_RANDOM_DRAWS = 0         # final: transfer-set random subsets dropped (uninformative in run 2)
MIN_RANDOM_DRAWS = 5
RANDOM_SEED_BASE = 1000
N_COV_MATCHED_DRAWS = 5
N_LABEL_PERM_DRAWS = 5
LABEL_PERM_SEED_BASE = 2000
RANDOM_ABLATION_SEED_BASE = 3000
COSINE_NULL_SEED = 4000
N_COSINE_NULL = 1000
N_INSTR_RANDOM_DRAWS = 5           # draws 0-4 on instruction sets
N_PPL_RANDOM_DRAWS = 2             # draws 0-1 on perplexity text
MIN_CLASS_KNOWN_UNKNOWN = 100
MIN_GRAY_QUESTIONS = 80
MIN_GRAY_AFTER_TOPUP = 50

# amendment A9: dose-matched nulls (one of each per seed), strengths as multiples of alpha*
NULL_TYPES = ["shuffled_ku", "cov_random", "negated_ku"]
NULL_MULTIPLIERS = [0.5, 1, 1.5, 2, 3, 4]
NULL_SEED_BASE = 7000
DOSE_MATCH_TOL = 3.0               # points of dev abstention increase
# amendment A9: D_wq at answer (generated) positions with its own ladder and a matched random ladder
WQ_ANSWER_C_GRID = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
N_WQ_ANSWER_RANDOM = 3
WQ_ANSWER_RANDOM_SEED_BASE = 8000
# amendment A12: refusal ablation variants (directional = registered arm)
REFUSAL_ABLATION_VARIANTS = {
    "refusal_dir_ablation": {"mode": "directional", "layers": "all"},
    "refusal_dir_mean_ablation": {"mode": "mean", "layers": "all"},
    "refusal_dir_mean_ablation_single_layer": {"mode": "mean", "layers": "single"},
}
N_RANDOM_ABLATION_DRAWS = {"refusal_dir_ablation": 3, "refusal_dir_mean_ablation": 2,
                           "refusal_dir_mean_ablation_single_layer": 2}

# final F3: dose-matched placebos for H1b. Each direction is run on the 300-item dev subset at
# multiples of alpha* (ascending, stop once knowledge_dir_steer's abstention increase is reached or
# output degenerates > 10%; then one geometric midpoint), and the clean multiple closest to the
# target is frozen. Matched = within +/- 3 points, compared as item counts (run 2 compared floats and
# lost a match to 3.0000000000000036). Random directions take the sign that lowers the teacher-forced
# margin on 64 dev prompts (sign-aligned), so they can push toward abstention at all.
N_PLACEBO_ISO = 8                    # sign-aligned isotropic random directions
N_PLACEBO_COV = 4                    # sign-aligned covariance-shaped random directions
PLACEBO_SEED_BASE = 11000
PLACEBO_MULTIPLIERS = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
PLACEBO_LOW_MULTIPLIERS = [0.25, 0.125]
SIGN_PROXY_MULTIPLIER = 2.0
DOSE_MATCH_TOL_FRAC = 0.03
# final F10: D_wq at its own scale (c * sigma_{L*}(D_wq)); dev selection = largest net W->C - C->W
# with degenerate <= 5% and the A11 caps; nulls at the same norm and positions
WQ_OWN_C_GRID = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
N_WQ_OWN_NULLS = 5
WQ_OWN_RANDOM_SEED_BASE = 12000
# final F11: relations whose gold labels are often ambiguous or too narrow (titles shared by several
# works, genre granularity); a sensitivity analysis excludes them
AMBIGUOUS_RELATIONS = ("genre", "producer", "director", "screenwriter", "composer", "author")

DEV_ELIGIBLE_MAX_DEGENERATE = 0.05
DEV_ELIGIBLE_ABST_INC = (5.0, 30.0)  # percentage points
DEV_TIE_POINTS = 0.5
DEV_FALLBACK_TARGET_INC = 15.0
MMLU_CAP_POINTS = -1.5               # amendment A11: dev eligibility, paired MMLU-dev delta >= -1.5
PPL_CAP_RATIO = 1.05                 # amendment A11: dev eligibility, reference-text perplexity ratio
COV_MATCH_MAX_DEGENERATE = 0.10
ZERO_VECTOR_MIN_IDENTICAL = 0.995
NO_OP_MAX_FRAC_CHANGED = 0.01        # amendment A5: hook diagnostics
NO_OP_MAX_KL = 1e-3
DIAG_N_PROMPTS = 64

# ---------------------------------------------------------------- metrics / statistics
LAMBDAS = [0.5, 1.0, 2.0, 3.0]
LAMBDA_PRIMARY = 1.0
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 7
N_PERMUTATIONS = 10_000             # amendment A1: sign-flip permutation test
PERMUTATION_SEED = 11
MIN_VALID_RESAMPLE_FRAC = 0.95      # amendment A1: otherwise the CI is "undefined"
AURC_GRID_STEP = 0.01               # amendment A1: common coverage grid
AURC_GRID_SLACK = 0.03
AURC_MIN_POINTS = 5
TOST_MARGIN = 3.0                   # points of S_1
SESOI = 4.0
COVERAGE_POINTS = [0.50, 0.70, 0.85]
PROBE_C_GRID = [0.01, 0.1, 1.0, 10.0]
N_SPLIT_RESAMPLES = 5
N_SAMPLES_PER_CONDITION = 50
AUDIT_PER_MODEL = 300
AUDIT_ROWS_PER_CELL = 50            # amendment A4: per (condition, transition) cell of the H1a arms
H2C_MAX_ABLATION_ABST_RISE = 5.0    # amendment A1: H2c undefined if the ablation alone raises abstention
H2C_MIN_STEER_INCREASE = 5.0        # ... or if steering raises abstention by < 5 points
H2B_MAX_TOP_TERCILE_DAMAGE = 3.0

# ---------------------------------------------------------------- compute budget (amendment A13)
# The first run projected its cost from one cold calibration batch (scale 1.55 instead of ~1.0) and
# applied all six degradation steps at once, including the 1,500 -> 1,200 test cut. Now: a warm-up
# batch precedes calibration, the scale is re-estimated from completed work, and every ladder step
# is decided only when its work is reached, in ladder order (see pipeline.Budget).
RUN_LIMIT_SECONDS = 12 * 3600       # final: expected ~8 h; limit 12 h, cap 11 h
BUDGET_CAP_SECONDS = 11 * 3600
PLAN_ANALYSIS_SECONDS = 3600            # analysis + phase D judge (~31k distinct outputs, ~30 min)
PLAN_GEN_RATE = 25.0                # QA questions / s at <= 16 new tokens, batch 64
PLAN_INSTR_RATE = 12.0              # instruction prompts / s at 32 new tokens
PLAN_FWD_RATE = 60.0                # teacher-forced sequences / s
PLAN_SAMPLE_RATE = 14.0             # prompts / s for 8 samples with shared prefill
CALIB_N_GEN, CALIB_N_FWD = 256, 512
# Degradation ladder in pre-registered order: (flag, what it cuts). Savings are computed from the
# work plan in pipeline.Budget; test_1200 is only applied if every earlier step cannot close the gap.
# final: only work that no verdict depends on can be cut; the fresh test set, the seeds, the
# dose ladder, the placebos D_force / D_ku decomposition, the judge and the H2 arms are never cut.
DEGRADATION_LADDER = [
    "instr_random_2",           # instruction-set random draws 5 -> 2
    "random_ablation_reduced",  # random-direction ablations 7 -> 3
    "norm_random_5",            # norm-matched isotropic draws on PopQA test 10 -> 5
    "label_perm_3",             # shared-site label-permuted D_wq 5 -> 3
    "wq_own_nulls_3",           # own-dose D_wq nulls 5 + 5 -> 3 + 3
    "placebo_reduced",          # sign-aligned random placebos 8 + 4 -> 4 + 2
]

PLANNED_CONDITIONS = [
    "unsteered", "confidence_prompt", "confidence_threshold_margin", "popularity_gate",
    "knowledge_dir_steer", "within_question_dir_steer", "refusal_dir_steer",
    "margin_gated_steer", "random_dir_placebo",
]
STEERING_CONDITIONS = ["knowledge_dir_steer", "within_question_dir_steer", "refusal_dir_steer"]
# amendment A3/A9 and final F3/F5/F10: conditions added to every table
ADDED_CONDITIONS = ["always_abstain", "read_dir_steer", "dose_matched_placebo", "knowledge_decomposition",
                    "within_question_dir_steer_own", "sample_vote"]

# ---------------------------------------------------------------- smoke test (CPU, tiny models)
# STEER_SMOKE=1 runs every code path end to end on two small instruct models and tiny splits. It
# exists to catch crashes before the GPU run; its numbers mean nothing and are never reported.
if SMOKE:
    _sm = os.environ.get("STEER_SMOKE_MODELS", "")
    MODELS = []
    for _entry in [e for e in _sm.split(";") if e]:
        _name, _hf, _rev, _nl, _d, _cand, _ref = _entry.split("|")
        MODELS.append({"name": _name, "hf_id": _hf, "revision": _rev, "n_layers": int(_nl), "d_model": int(_d),
                       "steering_candidate_layers": [int(x) for x in _cand.split(",")],
                       "refusal_candidate_layers": [int(x) for x in _ref.split(",")]})
    _j = os.environ.get("STEER_SMOKE_JUDGE", "").split("|")
    if len(_j) == 3:
        JUDGE = {"name": _j[0], "hf_id": _j[1], "revision": _j[2]}
    N_PER_DECILE = {"train": 8, "train_reserve": 4, "dev": 4}
    TEST_MAX_PER_DECILE = 3
    DEV_SUBSET_PER_DECILE = 3
    DOSE_ITEMS_PER_DECILE = 2
    EQ_PER_RELATION = 1
    TQA_DEV_N, TQA_TEST_N = 20, 24
    N_ADV_TRAIN, N_ADV_HELDOUT = 12, 8
    N_ALPACA_TRAIN, N_ALPACA_BENIGN = 12, 8
    N_MMLU, N_MMLU_DEV = 20, 16
    N_REF_DEV, N_REF_TEST = 6, 6
    REF_MIN_CHARS = 100
    PPL_MAX_RESPONSE_TOKENS = 32
    QA_MAX_NEW_TOKENS, INSTR_MAX_NEW_TOKENS = 8, 8
    BATCH_SIZE, SAMPLE_BATCH_PROMPTS = 16, 8
    N_LABEL_SAMPLES = 8
    N_SEEDS = 2
    C_GRID = [2.0, 8.0, 32.0]
    DOSE_C_GRID = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    WQ_OWN_C_GRID = [2.0, 8.0, 32.0]
    N_RANDOM_DRAWS, N_INSTR_RANDOM_DRAWS, N_PPL_RANDOM_DRAWS = 3, 2, 2
    N_RANDOM_ABLATION_DRAWS = {"refusal_dir_ablation": 1, "refusal_dir_mean_ablation": 1,
                               "refusal_dir_mean_ablation_single_layer": 1}
    N_LABEL_PERM_DRAWS = 2
    N_PLACEBO_ISO, N_PLACEBO_COV = 2, 1
    PLACEBO_MULTIPLIERS = [0.5, 2.0, 8.0]
    N_WQ_OWN_NULLS = 2
    MIN_CLASS_KNOWN_UNKNOWN, MIN_GRAY_QUESTIONS, MIN_GRAY_AFTER_TOPUP = 3, 3, 2
    DIAG_N_PROMPTS = 8
    N_BOOTSTRAP, N_PERMUTATIONS, N_COSINE_NULL = 200, 500, 100
    CALIB_N_GEN, CALIB_N_FWD = 16, 32
    AUDIT_PER_MODEL, AUDIT_ROWS_PER_CELL, JUDGE_N_DISAGREEMENT_ROWS = 20, 5, 20
    JUDGE_BATCH_SIZE = 8
    N_SPLIT_RESAMPLES = 2

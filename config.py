"""Pre-registered constants from experiment_plan.yaml (single source for every module)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data_cache"
OUT_DIR = ROOT / "outputs"
MANIFEST_PATH = ROOT / "run_manifest.json"
FROZEN_CONFIG_PATH = OUT_DIR / "frozen_config.json"
RESULTS_PATH = OUT_DIR / "results.json"
AUDIT_PATH = OUT_DIR / "audit.csv"

GLOBAL_SEED = 0

MODELS = [
    {
        "name": "llama32_3b_instruct",
        "hf_id": "unsloth/Llama-3.2-3B-Instruct",  # ungated mirror of meta-llama/Llama-3.2-3B-Instruct
        "n_layers": 28,
        "d_model": 3072,
        "steering_candidate_layers": [11, 14, 17],
        "refusal_candidate_layers": [8, 11, 14, 17],
    },
    {
        "name": "qwen25_3b_instruct",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "n_layers": 36,
        "d_model": 2048,
        "steering_candidate_layers": [14, 18, 22],
        "refusal_candidate_layers": [11, 14, 18, 22],
    },
]

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
IDK_STRING = "I don't know"
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
N_PER_DECILE = {"train": 100, "train_reserve": 100, "dev": 40, "test": 150}
DEV_SUBSET_PER_DECILE = 30          # 300-item coverage-matching dev subset
DOSE_ITEMS_PER_DECILE = 30          # 300 unsteered-answered test items for the dose analysis
EQ_PER_RELATION = 25
EQ_RELATIONS = ["P17", "P19", "P20", "P26", "P36", "P40", "P50", "P69", "P106", "P112",
                "P127", "P131", "P136", "P159", "P170", "P175", "P176", "P264", "P276",
                "P407", "P413", "P495", "P740", "P800"]
EQ_SUBSET_N = 300
N_ADV_TRAIN, N_ADV_HELDOUT = 128, 100
N_ALPACA_TRAIN, N_ALPACA_BENIGN, N_ALPACA_PPL = 128, 100, 200
PPL_MAX_RESPONSE_TOKENS = 128
N_MMLU = 500

# ---------------------------------------------------------------- steering
C_GRID = [0.1, 0.2, 0.4]
DOSE_C_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6]
COV_MATCH_MULTIPLIERS = [1, 2, 4]
N_RANDOM_DRAWS = 20
MIN_RANDOM_DRAWS = 10
RANDOM_SEED_BASE = 1000
N_COV_MATCHED_DRAWS = 5
N_LABEL_PERM_DRAWS = 5
LABEL_PERM_SEED_BASE = 2000
N_RANDOM_ABLATION_DRAWS = 2
RANDOM_ABLATION_SEED_BASE = 3000
COSINE_NULL_SEED = 4000
N_COSINE_NULL = 1000
N_INSTR_RANDOM_DRAWS = 5           # draws 0-4 on instruction sets
N_PPL_RANDOM_DRAWS = 2             # draws 0-1 on perplexity text
MIN_CLASS_KNOWN_UNKNOWN = 100
MIN_GRAY_QUESTIONS = 80
MIN_GRAY_AFTER_TOPUP = 50

DEV_ELIGIBLE_MAX_DEGENERATE = 0.05
DEV_ELIGIBLE_ABST_INC = (5.0, 30.0)  # percentage points
DEV_TIE_POINTS = 0.5
DEV_FALLBACK_TARGET_INC = 15.0
COV_MATCH_MAX_DEGENERATE = 0.10
ZERO_VECTOR_MIN_IDENTICAL = 0.995

# ---------------------------------------------------------------- metrics / statistics
LAMBDAS = [0.5, 1.0, 2.0, 3.0]
LAMBDA_PRIMARY = 1.0
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 7
TOST_MARGIN = 3.0                   # points of S_1
SESOI = 4.0
COVERAGE_POINTS = [0.50, 0.70, 0.85]
PROBE_C_GRID = [0.01, 0.1, 1.0, 10.0]
N_SPLIT_RESAMPLES = 5
N_SAMPLES_PER_CONDITION = 50
AUDIT_PER_MODEL = 300

# ---------------------------------------------------------------- compute budget
RUN_LIMIT_SECONDS = 14400
BUDGET_CAP_SECONDS = 11520
PLAN_PER_MODEL_SECONDS = 4947
PLAN_ANALYSIS_SECONDS = 250
PLAN_GEN_RATE = 25.0                # questions / s at <= 16 new tokens, batch 64
PLAN_FWD_RATE = 60.0                # teacher-forced sequences / s
CALIB_N_GEN, CALIB_N_FWD = 256, 512
# Degradation ladder in pre-registered order: (flag, planned seconds saved per model).
DEGRADATION_LADDER = [
    ("eq_subset_150", 120),
    ("random_ablation_1", 60),
    ("drop_alt_layer", 60),
    ("label_perm_3", 120),
    ("cov_matched_3", 190),
    ("test_1200", 0.2 * (1635 + 1550)),
]

PLANNED_CONDITIONS = [
    "unsteered", "confidence_prompt", "confidence_threshold_margin", "popularity_gate",
    "knowledge_dir_steer", "within_question_dir_steer", "refusal_dir_steer",
    "margin_gated_steer", "random_dir_placebo",
]
STEERING_CONDITIONS = ["knowledge_dir_steer", "within_question_dir_steer", "refusal_dir_steer"]

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
# Strong model for spec extraction and judging (asymmetric architecture from AgentPex)
EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "gpt-4o")
# Cheaper model for per-constraint nl_check evaluation (can be overridden)
EVAL_MODEL: str = os.getenv("EVAL_MODEL", "gpt-4o-mini")
JUDGE_MODEL: str = os.getenv("JUDGE_MODEL", "gpt-4o")

# ── Dataset ───────────────────────────────────────────────────────────────────
HUGGINGFACE_REPO_ID: str = "mcemri/MAD"
MAD_FULL_FILE: str = "MAD_full_dataset.json"
MAD_HUMAN_FILE: str = "MAD_human_labelled_dataset.json"
RAW_DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data", "raw")

# ── Pipeline ──────────────────────────────────────────────────────────────────
# Max steps before switching to step-by-step judging (vs all-at-once)
LONG_TRACE_THRESHOLD: int = 70
# Constraint generation strategy: "step_by_step" | "one_shot"
CONSTRAINT_GENERATION_MODE: str = "step_by_step"
# Max dynamic constraints generated per step (to control API cost)
MAX_DYNAMIC_CONSTRAINTS_PER_STEP: int = 3
# Temperature for all LLM calls (0 = deterministic)
LLM_TEMPERATURE: float = 0.0
# Number of runs per trajectory for robustness (AgentRx used n=3)
N_RUNS: int = 1

# ── MAST Taxonomy ─────────────────────────────────────────────────────────────
MAST_FAILURE_FAMILIES = {
    "FC1": "Specification Issues",
    "FC2": "Inter-Agent Misalignment",
    "FC3": "Task Verification Failures",
}

MAST_FAILURE_MODES = {
    # FC1
    "DisobeyTaskSpec":     "FC1",
    "DisobeyRoleSpec":     "FC1",
    # FC2
    "IgnoredOtherAgentInput":    "FC2",
    "ReasoningActionMismatch":   "FC2",
    "PrematureTermination":      "FC2",
    "UnawareOfStoppingCond":     "FC2",
    "IncompleteVerification":    "FC2",
    "IncorrectVerification":     "FC2",
    # FC3
    "NoVerification":        "FC3",
    "Hallucination":         "FC3",
    "FaultyReasoning":       "FC3",
    "ContextLoss":           "FC3",
    "FailTaskSpec":          "FC3",
    "InsufficientInfo":      "FC3",
}

# ── Paths ─────────────────────────────────────────────────────────────────────
PROMPTS_DIR: str = os.path.join(os.path.dirname(__file__), "prompts")
RESULTS_DIR: str = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

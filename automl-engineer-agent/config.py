"""
config.py — central configuration for automl-engineer-agent.
All modules import from here so settings are changed in one place.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent / ".env")

# ── Anthropic ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# Key may be empty at import so the Streamlit app can load; set .env, sidebar, or Demo mode.

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
OUTPUT_DIR = ROOT_DIR / os.getenv("OUTPUT_DIR", "outputs")
REPORTS_DIR = ROOT_DIR / os.getenv("REPORTS_DIR", "reports")
DATASETS_DIR = ROOT_DIR / "datasets"

# Create directories if they don't exist
for d in [OUTPUT_DIR, REPORTS_DIR, DATASETS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Agent settings ────────────────────────────────────────────────────────────
MAX_TOKENS = 4096
MAX_AGENT_ITERATIONS = 20   # safety cap on tool-call loops

# ── ML settings ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Models to try for classification
CLASSIFICATION_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
]

# Models to try for regression
REGRESSION_MODELS = [
    "linear_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
]

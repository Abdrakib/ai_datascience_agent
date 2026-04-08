"""Project configuration (imported by agent tools and app)."""

from pathlib import Path

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 300
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42

_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = _ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

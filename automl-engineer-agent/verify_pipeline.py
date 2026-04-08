"""
verify_pipeline.py — verifies full pipeline wiring without a real Anthropic API key.
Run: python verify_pipeline.py
"""

import sys, types
from pathlib import Path

# Stub anthropic before any project import touches it
mock_anthropic = types.ModuleType("anthropic")
class _FakeClient:
    def __init__(self, api_key=None): pass
mock_anthropic.Anthropic = _FakeClient
sys.modules["anthropic"] = mock_anthropic

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from agent.core import AutoMLAgent

SEP = "-" * 60

def make_clf_df(n=300):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "age":      rng.integers(20, 80, n).astype(float),
        "bmi":      rng.normal(27, 5, n).round(1),
        "glucose":  rng.normal(100, 25, n).round(1),
        "smoker":   rng.choice(["Yes", "No"], n),
        "gender":   rng.choice(["Male", "Female"], n),
        "survived": rng.integers(0, 2, n),
    })
    df.loc[rng.choice(n, 15, replace=False), "bmi"] = np.nan
    return df

def make_reg_df(n=300):
    rng = np.random.default_rng(99)
    sqft = rng.integers(800, 4000, n)
    df = pd.DataFrame({
        "sqft":         sqft,
        "bedrooms":     rng.integers(1, 6, n).astype(float),
        "age_years":    rng.integers(0, 50, n).astype(float),
        "neighborhood": rng.choice(["Urban", "Suburban", "Rural"], n),
        "price":        sqft * 110 + rng.normal(0, 15000, n),
    })
    df.loc[rng.choice(n, 10, replace=False), "bedrooms"] = np.nan
    return df


class MockAgent:
    """Fire tools directly in the correct order, bypassing Claude API."""
    def __init__(self, df, user_message, target_col, task_type, run_id="verify"):
        inner = AutoMLAgent.__new__(AutoMLAgent)
        inner.df            = df
        inner.user_message  = user_message
        inner._eda_report   = None
        inner._task_result  = None
        inner._prep_result  = None
        inner._plan_result  = None
        inner._train_result = None
        inner._eval_result  = None
        inner._tune_result  = None
        inner.result        = {}
        self._inner   = inner
        self._target  = target_col
        self._task    = task_type
        self._run_id  = run_id

    def run(self):
        inner = self._inner
        steps = [
            ("run_eda",        {}),
            ("detect_task",    {"user_hint": inner.user_message}),
            ("preprocess",     {"target_col": self._target, "task_type": self._task}),
            ("plan_training",  {}),
            ("train_models",   {}),
            ("evaluate_model", {"run_id": self._run_id}),
        ]
        for name, inputs in steps:
            yield {"type": "tool", "name": name, "status": "running"}
            output = inner._dispatch(name, inputs)
            yield {"type": "tool", "name": name, "status": "done", "output": output}
        inner.result = inner._build_result()
        yield {"type": "done", "result": inner.result}


def verify(label, df, message, target_col, task_type, run_id,
           primary_metric, metric_threshold):
    print(SEP)
    print(f"VERIFY — {label}")
    print(SEP)

    agent  = MockAgent(df, message, target_col, task_type, run_id)
    events = list(agent.run())

    tools_done = [e["name"] for e in events if e["type"] == "tool" and e["status"] == "done"]
    print(f"Tools executed: {tools_done}")

    result = next(e["result"] for e in events if e["type"] == "done")

    print(f"Status:       {result['status']}")
    print(f"Target col:   {result['target_col']}")
    print(f"Task type:    {result['task_type']}")
    print(f"Best model:   {result['best_model_name']}")
    print(f"Metrics:      {result['best_metrics']}")
    print(f"Plots:")
    for name, path in result["plot_paths"].items():
        exists = Path(path).exists()
        print(f"  {name}: {'OK' if exists else 'MISSING'}")
        assert exists, f"Plot missing: {path}"

    score = result["best_metrics"].get(primary_metric, 0)
    print(f"\n{primary_metric.upper()} = {score:.4f}  (threshold ≥ {metric_threshold})")
    assert score >= metric_threshold, f"{primary_metric} {score:.4f} below threshold"

    assert result["status"] == "complete"
    assert result["target_col"] == target_col
    assert result["task_type"]  == task_type
    assert len(result["plot_paths"]) >= 2

    print("PASSED\n")


if __name__ == "__main__":
    verify(
        label          = "Classification — patient survival",
        df             = make_clf_df(),
        message        = "predict whether patient survived",
        target_col     = "survived",
        task_type      = "classification",
        run_id         = "verify_clf",
        primary_metric = "accuracy",
        metric_threshold = 0.45,
    )

    verify(
        label          = "Regression — house price",
        df             = make_reg_df(),
        message        = "predict house price",
        target_col     = "price",
        task_type      = "regression",
        run_id         = "verify_reg",
        primary_metric = "r2",
        metric_threshold = 0.80,
    )

    print(SEP)
    print("ALL PIPELINE VERIFICATIONS PASSED!")
    print("The agent core is fully wired.")
    print(f"\nNext: run LIVE_TEST=1 python test_step5.py")
    print(f"      (requires ANTHROPIC_API_KEY in .env)")
    print(SEP)

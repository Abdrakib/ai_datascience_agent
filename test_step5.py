"""
test_step5.py — tests the AutoMLAgent orchestrator.

Two modes:
  1. Mock mode (default) — replays a scripted sequence of tool calls
     without hitting the real API. Verifies the full pipeline wiring.
  2. Live mode — set env var LIVE_TEST=1 to use real Claude API.

Run:  python test_step5.py
Live: LIVE_TEST=1 python test_step5.py
"""

import sys, os
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))


# ── Sample data ───────────────────────────────────────────────────────────────

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


# ── Mock agent — bypasses Claude API, fires tools in fixed order ──────────────

class MockAutoMLAgent:
    """
    Mimics AutoMLAgent.run() but fires tools directly without Claude.
    Used to verify the full pipeline wiring in CI / offline.
    """
    def __init__(self, df, user_message):
        from agent.core import AutoMLAgent
        self._inner = AutoMLAgent.__new__(AutoMLAgent)
        self._inner.df = df
        self._inner.user_message = user_message
        self._inner._eda_report  = None
        self._inner._task_result = None
        self._inner._prep_result = None
        self._inner._plan_result = None
        self._inner._train_result= None
        self._inner._eval_result = None
        self._inner.result       = {}

    def run(self):
        inner = self._inner
        steps = [
            ("run_eda",        {}),
            ("detect_task",    {"user_hint": inner.user_message}),
            ("preprocess",     {"target_col": "survived", "task_type": "classification"}),
            ("plan_training",  {}),
            ("train_models",   {}),
            ("evaluate_model", {"run_id": "mock_run"}),
        ]
        for name, inputs in steps:
            yield {"type": "tool", "name": name, "status": "running"}
            output = inner._dispatch(name, inputs)
            yield {"type": "tool", "name": name, "status": "done", "output": output}

        inner.result = inner._build_result()
        yield {"type": "done", "result": inner.result}


# ── Tests ─────────────────────────────────────────────────────────────────────

SEP = "-" * 60

def test_mock_pipeline():
    print(SEP)
    print("TEST 1 — Full pipeline (mock, no API call)")
    print(SEP)

    df = make_clf_df()
    agent = MockAutoMLAgent(df, "predict whether patient survived")

    events = list(agent.run())

    tool_names_seen = [e["name"] for e in events if e["type"] == "tool" and e["status"] == "done"]
    print(f"Tools executed: {tool_names_seen}")
    assert "run_eda"        in tool_names_seen
    assert "detect_task"    in tool_names_seen
    assert "preprocess"     in tool_names_seen
    assert "train_models"   in tool_names_seen
    assert "evaluate_model" in tool_names_seen

    done_event = next(e for e in events if e["type"] == "done")
    result = done_event["result"]

    print(f"\nResult keys: {list(result.keys())}")
    assert result["status"]          == "complete"
    assert result["target_col"]      == "survived"
    assert result["task_type"]       == "classification"
    assert result["best_model_name"] is not None
    assert "metrics" in result
    assert "plot_paths" in result

    print(f"\nBest model:   {result['best_model_name']}")
    print(f"Metrics:      {result['best_metrics']}")
    print(f"Plots saved:  {list(result['plot_paths'].keys())}")

    # Verify plots actually exist on disk
    for plot_name, path in result["plot_paths"].items():
        assert Path(path).exists(), f"Plot missing: {path}"
        print(f"  {plot_name}: OK")

    print("\nPASSED\n")
    return result


def test_tool_output_formats():
    print(SEP)
    print("TEST 2 — Tool output is valid non-empty markdown")
    print(SEP)
    df = make_clf_df()
    agent = MockAutoMLAgent(df, "classify survival")
    events = [e for e in agent.run() if e["type"] == "tool" and e["status"] == "done"]
    for e in events:
        assert isinstance(e["output"], str), f"{e['name']} output is not a string"
        assert len(e["output"]) > 20,        f"{e['name']} output is too short"
        print(f"  {e['name']}: {len(e['output'])} chars — OK")
    print("PASSED\n")


def test_live_api():
    print(SEP)
    print("TEST 3 — Live Claude API (LIVE_TEST=1)")
    print(SEP)
    from agent.core import AutoMLAgent

    df = make_clf_df(n=200)
    agent = AutoMLAgent(df, "predict whether patient survived")

    events = list(agent.run())
    result = agent.result

    print(f"\nBest model:  {result.get('best_model_name')}")
    print(f"Task type:   {result.get('task_type')}")
    print(f"Target col:  {result.get('target_col')}")

    texts = [e["content"] for e in events if e["type"] == "text"]
    print(f"\nClaude narration ({len(texts)} messages):")
    for t in texts:
        print(f"  > {t[:120]}{'...' if len(t) > 120 else ''}")

    assert result["status"] == "complete"
    assert result["target_col"] == "survived"
    assert result["task_type"] == "classification"
    print("\nPASSED\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_mock_pipeline()
    test_tool_output_formats()

    if os.environ.get("LIVE_TEST") == "1":
        test_live_api()
    else:
        print(f"{SEP}")
        print("Skipping live API test (set LIVE_TEST=1 to enable).")
        print(f"{SEP}\n")

    print(SEP)
    print("ALL STEP 5 TESTS PASSED!")
    print(SEP)

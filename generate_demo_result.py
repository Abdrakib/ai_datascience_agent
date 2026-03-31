"""
Generate demo_result.json for Streamlit Demo Mode (no API key required in UI).

Run from project root:
    python generate_demo_result.py
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

os.environ.setdefault("ANTHROPIC_API_KEY", "stub-for-demo-json-generation")

# Stub anthropic before project imports
mock_anthropic = types.ModuleType("anthropic")


class _FakeClient:
    def __init__(self, api_key=None):
        pass


mock_anthropic.Anthropic = _FakeClient
sys.modules["anthropic"] = mock_anthropic

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

from agent.core import AutoMLAgent
from verify_pipeline import make_clf_df


def _new_pipeline_track() -> list[dict]:
    return [
        {"step": 1, "name": "run_eda", "label": "EDA", "status": "waiting",
         "data": None, "error": None},
        {"step": 2, "name": "detect_task", "label": "Task detection", "status": "waiting",
         "data": None, "error": None},
        {"step": 3, "name": "preprocess", "label": "Preprocessing", "status": "waiting",
         "data": None, "error": None},
        {"step": 4, "name": "plan_training", "label": "Training Plan", "status": "waiting",
         "data": None, "error": None},
        {"step": 5, "name": "train_models", "label": "Model training", "status": "waiting",
         "data": None, "error": None},
        {"step": 6, "name": "tune_model", "label": "Hyperparameter tuning", "status": "waiting",
         "data": None, "error": None},
        {"step": 7, "name": "evaluate_model", "label": "Evaluation", "status": "waiting",
         "data": None, "error": None},
        {"step": 8, "name": "final", "label": "Final recommendation", "status": "waiting",
         "data": None, "error": None},
    ]


def _pipeline_track_finalize(result: dict, track: list[dict]) -> None:
    for s in track:
        if s["name"] == "final":
            s["status"] = "done"
            s["data"] = result
            break


class DemoMockAgent:
    """Same as verify_pipeline.MockAgent but includes tune_model."""

    def __init__(self, df, message, target_col, task_type, run_id="demo_showcase"):
        inner = AutoMLAgent.__new__(AutoMLAgent)
        inner.df = df
        inner.user_message = message
        inner._eda_report = inner._task_result = inner._prep_result = None
        inner._plan_result = None
        inner._train_result = inner._eval_result = inner._tune_result = None
        inner.result = {}
        self._inner = inner
        self._target = target_col
        self._task = task_type
        self._run_id = run_id

    def run(self):
        inner = self._inner
        steps = [
            ("run_eda", {}),
            ("detect_task", {"user_hint": inner.user_message}),
            ("preprocess", {"target_col": self._target, "task_type": self._task, "scaler_type": "standard"}),
            ("plan_training", {}),
            ("train_models", {}),
            ("tune_model", {}),
            ("evaluate_model", {"run_id": self._run_id}),
        ]
        for name, inputs in steps:
            yield {"type": "tool", "name": name, "status": "running"}
            output = inner._dispatch(name, inputs)
            yield {"type": "tool", "name": name, "status": "done", "output": output}
        inner.result = inner._build_result()
        yield {"type": "done", "result": inner.result}


def _json_safe(obj):
    if obj is None:
        return None
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def _plot_paths_to_repo_relative(obj: Any, root: Path) -> None:
    """Rewrite plot_paths string values to paths relative to repo root (posix)."""
    root = root.resolve()
    if isinstance(obj, dict):
        if "plot_paths" in obj and isinstance(obj["plot_paths"], dict):
            for pk, pv in list(obj["plot_paths"].items()):
                if isinstance(pv, str) and pv:
                    p = Path(pv)
                    try:
                        rel = p.resolve().relative_to(root)
                        obj["plot_paths"][pk] = rel.as_posix()
                    except ValueError:
                        pass
        for v in obj.values():
            _plot_paths_to_repo_relative(v, root)
    elif isinstance(obj, list):
        for item in obj:
            _plot_paths_to_repo_relative(item, root)


def main() -> None:
    df = make_clf_df()
    message = "predict whether patient survived"
    target_col = "survived"
    task_type = "classification"
    run_id = "demo_showcase"

    agent = DemoMockAgent(df, message, target_col, task_type, run_id=run_id)
    events = list(agent.run())
    result = next(e["result"] for e in events if e["type"] == "done")
    inner = agent._inner

    track = _new_pipeline_track()
    tool_names = [
        "run_eda", "detect_task", "preprocess", "plan_training",
        "train_models", "tune_model", "evaluate_model",
    ]
    for s in track:
        if s["name"] in tool_names:
            s["status"] = "done"
            s["data"] = inner._get_step_data(s["name"])
    _pipeline_track_finalize(result, track)

    result_out = _json_safe(result)
    if isinstance(result_out, dict) and isinstance(result_out.get("comparison_df"), list):
        pass  # already records

    out = {
        "version": 1,
        "demo_dataset_path": "datasets/titanic.csv",
        "demo_goal": "Predict whether a patient survived (demo)",
        "result": result_out,
        "pipeline_track": _json_safe(track),
        "log_lines": [
            '<div class="log-line log-text"><span class="log-ts">00:00:00</span>'
            "Demo mode — pre-computed example run.</div>",
            '<div class="log-line log-tool-done"><span class="log-ts">00:00:01</span>'
            '✓ <span class="tool-chip">run_eda</span> — EDA complete</div>',
        ],
    }

    repo_root = Path(__file__).parent
    _plot_paths_to_repo_relative(out, repo_root)

    path = repo_root / "demo_result.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {path}")

    pp = result.get("plot_paths", {})
    for k, v in pp.items():
        ok = Path(v).exists()
        print(f"  plot {k}: {'OK' if ok else 'MISSING'} {v}")


if __name__ == "__main__":
    main()

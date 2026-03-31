"""
Generate all demo_result_*.json files for Streamlit Demo Mode (no API key).

Run from project root (this directory):
    python generate_all_demos.py
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

os.environ.setdefault("ANTHROPIC_API_KEY", "stub-for-all-demos-generation")

mock_anthropic = types.ModuleType("anthropic")


class _FakeClient:
    def __init__(self, api_key=None):
        pass


mock_anthropic.Anthropic = _FakeClient
sys.modules["anthropic"] = mock_anthropic

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

from agent.core import AutoMLAgent


class NumpyEncoder(json.JSONEncoder):
    """Serialize numpy scalars and arrays; delegate the rest to default."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


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
    """MockAgent-style run: same order as verify_pipeline + tune_model + evaluate."""

    def __init__(self, df, message, target_col, task_type, run_id: str):
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


_TITANIC_FEATURE_COLS = ("pclass", "sex", "age", "sibsp", "parch", "fare", "embarked")
_TITANIC_TARGET_COL = "survived"


def make_titanic_like(n: int = 800) -> pd.DataFrame:
    """Synthetic Titanic-style data: seven features only; survived is target only."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "pclass": rng.choice([1, 2, 3], n),
            "sex": rng.choice(["male", "female"], n),
            "age": rng.uniform(0.5, 80.0, n).round(1),
            "sibsp": rng.integers(0, 9, n),
            "parch": rng.integers(0, 7, n),
            "fare": rng.uniform(0, 500, n).round(2),
            "embarked": rng.choice(["C", "Q", "S"], n),
        }
    )
    df[_TITANIC_TARGET_COL] = rng.integers(0, 2, n)
    return df.loc[:, list(_TITANIC_FEATURE_COLS) + [_TITANIC_TARGET_COL]]


def make_diabetes_binary() -> pd.DataFrame:
    bunch = load_diabetes()
    X, y = bunch.data, bunch.target
    cols = [str(c).replace(" ", "_") for c in bunch.feature_names]
    df = pd.DataFrame(X, columns=cols)
    med = float(np.median(y))
    df["target_binary"] = (y > med).astype(int)
    return df


def _json_safe(obj: Any) -> Any:
    """Recursively convert result tree to JSON-safe structures; skip failures."""
    try:
        if obj is None:
            return None
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                try:
                    out[str(k)] = _json_safe(v)
                except Exception:
                    continue
            return out
        if isinstance(obj, (list, tuple)):
            out = []
            for v in obj:
                try:
                    out.append(_json_safe(v))
                except Exception:
                    continue
            return out
        if isinstance(obj, (str, int, float, bool)):
            return obj
        return str(obj)
    except Exception:
        return None


def _plot_paths_to_repo_relative(obj: Any, root: Path) -> None:
    root = root.resolve()
    if isinstance(obj, dict):
        if "plot_paths" in obj and isinstance(obj["plot_paths"], dict):
            for pk, pv in list(obj["plot_paths"].items()):
                if isinstance(pv, str) and pv:
                    p = Path(pv)
                    try:
                        obj["plot_paths"][pk] = p.resolve().relative_to(root).as_posix()
                    except ValueError:
                        pass
        for v in obj.values():
            _plot_paths_to_repo_relative(v, root)
    elif isinstance(obj, list):
        for item in obj:
            _plot_paths_to_repo_relative(item, root)


def build_demo_payload(
    *,
    label: str,
    df: pd.DataFrame,
    message: str,
    target_col: str,
    task_type: str,
    run_id: str,
    demo_dataset_path: str,
    demo_goal: str,
) -> dict:
    print(f"Generating demo for {label}...")
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
    out = {
        "version": 1,
        "demo_dataset_path": demo_dataset_path,
        "demo_goal": demo_goal,
        "result": result_out,
        "pipeline_track": _json_safe(track),
        "log_lines": [
            '<div class="log-line log-text"><span class="log-ts">00:00:00</span>'
            f"Demo mode — pre-computed run ({label}).</div>",
        ],
    }
    _plot_paths_to_repo_relative(out, ROOT)
    return out


def write_json(path: Path, payload: dict) -> None:
    text = json.dumps(payload, indent=2, cls=NumpyEncoder, default=str)
    path.write_text(text, encoding="utf-8")
    print(f"Saved {path.name}")


def main() -> None:
    datasets_dir = ROOT / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    titanic_df = make_titanic_like(800)
    titanic_csv = datasets_dir / "titanic_demo_synth.csv"
    titanic_df.to_csv(titanic_csv, index=False)

    diabetes_df = make_diabetes_binary()
    diabetes_csv = datasets_dir / "diabetes_sklearn_demo.csv"
    diabetes_df.to_csv(diabetes_csv, index=False)

    configs = [
        {
            "key": "healthcare",
            "label": "healthcare",
            "df": pd.read_csv(datasets_dir / "sample_healthcare_classification.csv"),
            "message": "predict whether the patient will be readmitted",
            "target_col": "readmitted",
            "task_type": "classification",
            "run_id": "healthcare",
            "demo_dataset_path": "datasets/sample_healthcare_classification.csv",
            "demo_goal": "Predict hospital readmission from patient features (demo)",
        },
        {
            "key": "housing",
            "label": "housing",
            "df": pd.read_csv(datasets_dir / "sample_housing_regression.csv"),
            "message": "predict the median house price",
            "target_col": "price",
            "task_type": "regression",
            "run_id": "housing",
            "demo_dataset_path": "datasets/sample_housing_regression.csv",
            "demo_goal": "Predict median house value from housing features (demo)",
        },
        {
            "key": "titanic",
            "label": "titanic",
            "df": titanic_df,
            "message": "predict whether the passenger survived",
            "target_col": "survived",
            "task_type": "classification",
            "run_id": "titanic",
            "demo_dataset_path": "datasets/titanic_demo_synth.csv",
            "demo_goal": "Predict passenger survival (demo — 800-row Titanic-style synthetic data)",
        },
        {
            "key": "diabetes",
            "label": "diabetes",
            "df": diabetes_df,
            "message": "predict the binary diabetes severity class",
            "target_col": "target_binary",
            "task_type": "classification",
            "run_id": "diabetes",
            "demo_dataset_path": "datasets/diabetes_sklearn_demo.csv",
            "demo_goal": "Predict diabetes progression class from sklearn diabetes features (demo)",
        },
    ]

    healthcare_payload: dict | None = None

    for cfg in configs:
        payload = build_demo_payload(
            label=cfg["label"],
            df=cfg["df"],
            message=cfg["message"],
            target_col=cfg["target_col"],
            task_type=cfg["task_type"],
            run_id=cfg["run_id"],
            demo_dataset_path=cfg["demo_dataset_path"],
            demo_goal=cfg["demo_goal"],
        )
        out_path = ROOT / f"demo_result_{cfg['key']}.json"
        write_json(out_path, payload)
        if cfg["key"] == "healthcare":
            healthcare_payload = payload

    if healthcare_payload is not None:
        write_json(ROOT / "demo_result.json", healthcare_payload)

    print("All demo files generated successfully")
    print(
        "Run: git add demo_result*.json && git commit -m 'update demo files' && git push"
    )


if __name__ == "__main__":
    main()

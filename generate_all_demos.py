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
            ("run_eda", {"target_col": self._target}),
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
    """Synthetic Titanic-style data: survival depends strongly on sex and pclass."""
    rng = np.random.default_rng(7)
    features = pd.DataFrame(
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
    female = features["sex"].eq("female")
    male = ~female
    p_surv = np.zeros(n, dtype=float)
    # Target pattern: female ~80%; male by class ~60% / ~40% / ~15% (calibrated up slightly so demo ROC-AUC clears 0.75)
    p_surv[female] = 0.88
    pc = features["pclass"].to_numpy()
    p_surv[male & (pc == 1)] = 0.72
    p_surv[male & (pc == 2)] = 0.48
    p_surv[male & (pc == 3)] = 0.10
    survived = (rng.random(n) < p_surv).astype(np.int64)
    target = pd.Series(survived, name=_TITANIC_TARGET_COL)
    out = pd.concat([features, target], axis=1)
    return out.loc[:, list(_TITANIC_FEATURE_COLS) + [_TITANIC_TARGET_COL]]


def make_healthcare_like(n: int = 500) -> pd.DataFrame:
    """
    Synthetic healthcare rows; readmission probability follows glucose, bmi, age tiers.
    Tier probabilities match the demo spec; independent draws of glucose/bmi/age give strong signal.
    """
    rng = np.random.default_rng(42)
    glucose = rng.uniform(70.0, 200.0, n)
    bmi = rng.uniform(18.0, 45.0, n)
    age = rng.uniform(22.0, 90.0, n)
    tier1 = (glucose > 140) & (bmi > 30)
    tier2 = ((glucose > 140) | (bmi > 35)) & ~tier1
    tier3 = ~(tier1 | tier2) & (age > 65)
    tier4 = ~(tier1 | tier2) & (age <= 65)
    p = np.zeros(n, dtype=float)
    # Tier pattern: 75% / 55% / 45% / 20% at nominal thresholds (rates scaled so demo ROC-AUC > 0.75)
    p[tier1] = 0.92
    p[tier2] = 0.72
    p[tier3] = 0.38
    p[tier4] = 0.08
    readmitted = (rng.random(n) < p).astype(int)
    return pd.DataFrame(
        {
            "age": np.round(age, 1),
            "bmi": np.round(bmi, 1),
            "blood_pressure": np.round(rng.uniform(60.0, 140.0, n), 1),
            "glucose": np.round(glucose, 1),
            "num_medications": rng.integers(0, 12, n),
            "days_in_hospital": rng.integers(1, 15, n),
            "gender": rng.choice(["Female", "Male"], n),
            "smoker": rng.choice(["Yes", "No"], n),
            "insurance": rng.choice(["None", "Medicare", "Medicaid", "Private"], n),
            "readmitted": readmitted,
        }
    )


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


def _remove_target_from_eda_feature_profiles(result: dict, target_col: str) -> None:
    """
    EDA profiles every column in df; for Titanic demo JSON, list only true feature columns.
    Target remains in the DataFrame for training but must not appear as a feature column in EDA.
    Mutates result['eda'] in place (same object as agent._eda_report).
    """
    eda = result.get("eda")
    if not isinstance(eda, dict):
        return
    col_profiles = eda.get("columns")
    if isinstance(col_profiles, dict) and target_col in col_profiles:
        del col_profiles[target_col]
    ov = eda.get("overview")
    if isinstance(ov, dict):
        cn = ov.get("column_names")
        if isinstance(cn, list):
            ov["column_names"] = [c for c in cn if c != target_col]
        ov["columns"] = len(ov["column_names"])
        num = cat = 0
        for c in ov["column_names"]:
            prof = col_profiles.get(c, {}) if isinstance(col_profiles, dict) else {}
            dg = prof.get("dtype_group")
            if dg == "numeric":
                num += 1
            elif dg == "categorical":
                cat += 1
        ov["numeric_cols"] = num
        ov["categorical_cols"] = cat
    samp = eda.get("sample")
    if isinstance(samp, list):
        for row in samp:
            if isinstance(row, dict) and target_col in row:
                del row[target_col]


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

    if label == "titanic":
        _remove_target_from_eda_feature_profiles(result, target_col)

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

    healthcare_df = make_healthcare_like(500)
    healthcare_csv = datasets_dir / "healthcare_demo_synth.csv"
    healthcare_df.to_csv(healthcare_csv, index=False)

    configs = [
        {
            "key": "healthcare",
            "label": "healthcare",
            "df": healthcare_df,
            "message": "predict whether the patient will be readmitted",
            "target_col": "readmitted",
            "task_type": "classification",
            "run_id": "healthcare",
            "demo_dataset_path": "datasets/healthcare_demo_synth.csv",
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

"""
test_step3.py — smoke tests for task_detector + train tools.
Run from project root: python test_step3.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from agent.tools.task_detector import detect_task, task_detection_to_markdown
from agent.tools.preprocess import build_preprocessing_pipeline
from agent.tools.train import train_and_compare, training_results_to_markdown


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


def make_reg_df(n=400):
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


SEP = "-" * 60


def test_task_detector():
    print(SEP)
    print("TEST 1 — Task detector: classification (with hint)")
    print(SEP)
    df = make_clf_df()
    result = detect_task(df, user_hint="predict whether patient survived")
    print(task_detection_to_markdown(result))
    assert result["task_type"] == "classification"
    assert result["target_col"] == "survived"
    assert result["confidence"] in ("high", "medium", "low")
    print("PASSED\n")

    print(SEP)
    print("TEST 2 — Task detector: regression (with hint)")
    print(SEP)
    df = make_reg_df()
    result = detect_task(df, user_hint="predict house price")
    print(task_detection_to_markdown(result))
    assert result["task_type"] == "regression"
    assert result["target_col"] == "price"
    print("PASSED\n")

    print(SEP)
    print("TEST 3 — Task detector: auto detect, no hint")
    print(SEP)
    df = make_clf_df()
    result = detect_task(df)
    print(task_detection_to_markdown(result))
    assert result["task_type"] in ("classification", "regression")
    print("PASSED\n")


def test_trainer_classification():
    print(SEP)
    print("TEST 4 — Model trainer: classification")
    print(SEP)
    df = make_clf_df()
    prep = build_preprocessing_pipeline(df, target_col="survived", task_type="classification")
    result = train_and_compare(
        prep["X_train"], prep["X_test"],
        prep["y_train"], prep["y_test"],
        task_type="classification",
        feature_names=prep["feature_names"],
        n_classes=prep["n_classes"],
    )
    print(training_results_to_markdown(result))
    assert result["best_model"] is not None
    assert result["best_name"] in result["comparison_df"]["Model"].values
    assert result["metric_name"] in ("roc_auc", "accuracy", "f1")
    # Best model should get > 50% accuracy (random chance)
    acc = result["best_metrics"].get("accuracy", 0)
    assert acc > 0.50, f"Accuracy too low: {acc}"
    print("PASSED\n")


def test_trainer_regression():
    print(SEP)
    print("TEST 5 — Model trainer: regression")
    print(SEP)
    df = make_reg_df()
    prep = build_preprocessing_pipeline(df, target_col="price", task_type="regression")
    result = train_and_compare(
        prep["X_train"], prep["X_test"],
        prep["y_train"], prep["y_test"],
        task_type="regression",
        feature_names=prep["feature_names"],
    )
    print(training_results_to_markdown(result))
    assert result["best_model"] is not None
    r2 = result["best_metrics"].get("r2", 0)
    assert r2 > 0.5, f"R² too low: {r2}"
    print("PASSED\n")


if __name__ == "__main__":
    test_task_detector()
    test_trainer_classification()
    test_trainer_regression()
    print(SEP)
    print("ALL STEP 3 TESTS PASSED!")
    print(SEP)

"""
test_step4.py — smoke tests for the evaluate tool.
Run from project root: python test_step4.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from agent.tools.preprocess import build_preprocessing_pipeline
from agent.tools.train import train_and_compare
from agent.tools.evaluate import evaluate_model, evaluation_to_markdown

SEP = "-" * 60


def make_clf_df(n=400):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "age":      rng.integers(20, 80, n).astype(float),
        "bmi":      rng.normal(27, 5, n).round(1),
        "glucose":  rng.normal(100, 25, n).round(1),
        "smoker":   rng.choice(["Yes", "No"], n),
        "gender":   rng.choice(["Male", "Female"], n),
        "survived": rng.integers(0, 2, n),
    })
    df.loc[rng.choice(n, 20, replace=False), "bmi"] = np.nan
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


def test_classification_eval():
    print(SEP)
    print("TEST 1 — Evaluation: classification (Random Forest)")
    print(SEP)
    df = make_clf_df()
    prep = build_preprocessing_pipeline(df, "survived", "classification")
    train_res = train_and_compare(
        prep["X_train"], prep["X_test"],
        prep["y_train"], prep["y_test"],
        task_type="classification",
        feature_names=prep["feature_names"],
        n_classes=prep["n_classes"],
    )

    eval_res = evaluate_model(
        model=train_res["best_model"],
        X_test=prep["X_test"],
        y_test=prep["y_test"],
        X_train=prep["X_train"],
        y_train=prep["y_train"],
        task_type="classification",
        feature_names=prep["feature_names"],
        label_encoder=prep["label_encoder"],
        run_id="test_clf",
        n_classes=prep["n_classes"],
    )

    print(evaluation_to_markdown(eval_res, train_res["best_name"]))
    print("\nEval log:")
    for line in eval_res["eval_log"]:
        print(" ", line)

    assert "accuracy" in eval_res["metrics"]
    assert "confusion_matrix" in eval_res["plot_paths"]
    assert Path(eval_res["plot_paths"]["confusion_matrix"]).exists()
    if "roc_curve" in eval_res["plot_paths"]:
        assert Path(eval_res["plot_paths"]["roc_curve"]).exists()
    if "feature_importance" in eval_res["plot_paths"]:
        assert Path(eval_res["plot_paths"]["feature_importance"]).exists()

    print("\nPlots saved:")
    for name, p in eval_res["plot_paths"].items():
        print(f"  {name}: {p}")
    print("PASSED\n")


def test_regression_eval():
    print(SEP)
    print("TEST 2 — Evaluation: regression")
    print(SEP)
    df = make_reg_df()
    prep = build_preprocessing_pipeline(df, "price", "regression")
    train_res = train_and_compare(
        prep["X_train"], prep["X_test"],
        prep["y_train"], prep["y_test"],
        task_type="regression",
        feature_names=prep["feature_names"],
    )

    eval_res = evaluate_model(
        model=train_res["best_model"],
        X_test=prep["X_test"],
        y_test=prep["y_test"],
        X_train=prep["X_train"],
        y_train=prep["y_train"],
        task_type="regression",
        feature_names=prep["feature_names"],
        run_id="test_reg",
    )

    print(evaluation_to_markdown(eval_res, train_res["best_name"]))
    print("\nEval log:")
    for line in eval_res["eval_log"]:
        print(" ", line)

    assert "r2" in eval_res["metrics"]
    assert "actual_vs_predicted" in eval_res["plot_paths"]
    assert "residuals" in eval_res["plot_paths"]
    assert Path(eval_res["plot_paths"]["actual_vs_predicted"]).exists()
    assert Path(eval_res["plot_paths"]["residuals"]).exists()

    print("\nPlots saved:")
    for name, p in eval_res["plot_paths"].items():
        print(f"  {name}: {p}")
    print("PASSED\n")


if __name__ == "__main__":
    test_classification_eval()
    test_regression_eval()
    print(SEP)
    print("ALL STEP 4 TESTS PASSED!")
    print(SEP)

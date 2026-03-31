"""
test_step2.py — smoke test for EDA + preprocessor tools.
Run from project root: python test_step2.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from agent.tools.eda import run_eda, eda_to_markdown
from agent.tools.preprocess import build_preprocessing_pipeline, preprocessing_log_to_markdown


def make_sample_clf_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "age":      rng.integers(20, 80, n).astype(float),
        "bmi":      rng.normal(27, 5, n).round(1),
        "glucose":  rng.normal(100, 25, n).round(1),
        "smoker":   rng.choice(["Yes", "No"], n),
        "gender":   rng.choice(["Male", "Female"], n),
        "survived": rng.integers(0, 2, n),
    })
    # Inject missing values
    df.loc[rng.choice(n, 15, replace=False), "bmi"] = np.nan
    df.loc[rng.choice(n, 10, replace=False), "glucose"] = np.nan
    return df


def make_sample_reg_df() -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = 300
    sqft = rng.integers(800, 4000, n)
    df = pd.DataFrame({
        "sqft":          sqft,
        "bedrooms":      rng.integers(1, 6, n).astype(float),
        "neighborhood":  rng.choice(["Urban", "Suburban", "Rural"], n),
        "age_years":     rng.integers(0, 50, n).astype(float),
        "price":         sqft * 110 + rng.normal(0, 15000, n),
    })
    df.loc[rng.choice(n, 12, replace=False), "bedrooms"] = np.nan
    return df


def run_tests():
    sep = "-" * 60

    print(sep)
    print("TEST 1 — EDA on classification dataset")
    print(sep)
    df_clf = make_sample_clf_df()
    eda_report = run_eda(df_clf, target_col="survived")
    md = eda_to_markdown(eda_report)
    print(md)
    assert eda_report["overview"]["rows"] == 200
    assert "survived" in eda_report.get("target_info", {}).get("column", "survived")
    assert eda_report["target_info"]["inferred_task"] == "classification"
    print("PASSED\n")

    print(sep)
    print("TEST 2 — EDA on regression dataset")
    print(sep)
    df_reg = make_sample_reg_df()
    eda_reg = run_eda(df_reg, target_col="price")
    print(eda_to_markdown(eda_reg))
    assert eda_reg["target_info"]["inferred_task"] == "regression"
    print("PASSED\n")

    print(sep)
    print("TEST 3 — Preprocessor on classification dataset")
    print(sep)
    result = build_preprocessing_pipeline(df_clf, target_col="survived", task_type="classification")
    print(preprocessing_log_to_markdown(result))
    assert result["X_train"].shape[0] > 0
    assert result["X_test"].shape[0] > 0
    assert result["task_type"] == "classification"
    assert result["label_encoder"] is not None
    assert len(result["feature_names"]) == result["X_train"].shape[1]
    print(f"Feature names: {result['feature_names']}")
    print("PASSED\n")

    print(sep)
    print("TEST 4 — Preprocessor on regression dataset")
    print(sep)
    result_reg = build_preprocessing_pipeline(df_reg, target_col="price", task_type="regression")
    print(preprocessing_log_to_markdown(result_reg))
    assert result_reg["task_type"] == "regression"
    assert result_reg["label_encoder"] is None
    print(f"Feature names: {result_reg['feature_names']}")
    print("PASSED\n")

    print(sep)
    print("ALL TESTS PASSED — Step 2 complete!")
    print(sep)


if __name__ == "__main__":
    run_tests()

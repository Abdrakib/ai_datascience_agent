"""
predict.py — Save, load, and run inference with a trained model.

Usage
-----
# After running the agent, save the best model:
python predict.py save --run-id my_run

# Predict on new data:
python predict.py predict --model outputs/my_run_model.pkl --input new_data.csv

# Show model info:
python predict.py info --model outputs/my_run_model.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from config import OUTPUT_DIR

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _infer_pipeline_input_columns(pipeline) -> list[str]:
    """Column names expected by a fitted ColumnTransformer (before encoding)."""
    cols: list[str] = []
    for name, _, ccols in getattr(pipeline, "transformers_", []) or []:
        if name == "remainder":
            continue
        if ccols == "drop" or ccols is None:
            continue
        cols.extend(list(ccols))
    return cols


def _compute_training_stats(
    X: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any], dict[str, list]]:
    """Means, medians (numeric), modes (categorical), and unique category lists per cat column."""
    means: dict[str, float] = {}
    medians: dict[str, float] = {}
    modes: dict[str, Any] = {}
    cat_values: dict[str, list] = {}
    for col in X.columns:
        s = X[col]
        if pd.api.types.is_numeric_dtype(s):
            medians[col] = float(np.nanmedian(s.astype(float)))
            means[col] = float(np.nanmean(s.astype(float)))
        else:
            m = s.mode()
            modes[col] = m.iloc[0] if len(m) else None
            cat_values[col] = sorted(s.astype(str).unique().tolist())
    return means, medians, modes, cat_values


def _extract_feature_importance(model, feature_names: list[str]) -> dict[str, float]:
    """Best-effort feature importance for UI when SHAP is unavailable."""
    out: dict[str, float] = {}
    try:
        if hasattr(model, "feature_importances_"):
            imp = np.asarray(model.feature_importances_, dtype=float).ravel()
            for i, n in enumerate(feature_names):
                if i < len(imp):
                    out[str(n)] = float(imp[i])
        elif hasattr(model, "coef_"):
            coef = np.asarray(model.coef_, dtype=float).ravel()
            for i, n in enumerate(feature_names):
                if i < len(coef):
                    out[str(n)] = float(abs(coef[i]))
    except Exception:
        pass
    return out


def get_model_summary(bundle: dict) -> dict[str, Any]:
    """Human-readable model information from a loaded bundle."""
    metrics = bundle.get("best_metrics") or {}
    metrics_fmt: dict[str, str] = {}
    for k, v in metrics.items():
        if k == "train_time_s":
            continue
        if isinstance(v, (float, int, np.floating, np.integer)):
            metrics_fmt[str(k)] = f"{float(v):.4f}" if isinstance(v, float) else str(v)
        else:
            metrics_fmt[str(k)] = str(v)

    orig = bundle.get("original_columns")
    if not orig:
        orig = _infer_pipeline_input_columns(bundle["pipeline"])

    feat_names = bundle.get("feature_names") or []

    return {
        "model_name": bundle.get("model_name", "—"),
        "task_type": bundle.get("task_type", "—"),
        "target_col": bundle.get("target_col", "—"),
        "n_features": len(feat_names),
        "original_features": list(orig),
        "training_date": bundle.get("training_date", "—"),
        "n_training_rows": bundle.get("n_training_rows"),
        "metrics": metrics_fmt,
        "expected_input_columns": list(orig),
    }


# ── Save ──────────────────────────────────────────────────────────────────────


def save_model(
    pipeline,  # fitted ColumnTransformer
    model,  # fitted sklearn estimator
    label_encoder,  # fitted LabelEncoder or None
    feature_names: list,
    task_type: str,
    target_col: str,
    best_metrics: dict,
    model_name: str,
    run_id: str = "model",
    X_train: pd.DataFrame | None = None,
    num_cols: list[str] | None = None,
    cat_cols: list[str] | None = None,
    n_training_rows: int | None = None,
) -> str:
    """
    Bundle everything needed for inference into a single .pkl file.
    Returns the saved path.

    X_train: raw feature DataFrame (before transform) — used for imputation stats and metadata.
    num_cols / cat_cols: optional lists from preprocessing (for manual inference forms).
    n_training_rows: optional count of training rows (e.g. train set size); overrides len(X_train).
    """
    feature_means: dict[str, float] = {}
    feature_medians: dict[str, float] = {}
    feature_modes: dict[str, Any] = {}
    categorical_uniques: dict[str, list] = {}
    original_columns: list[str] = []
    n_rows_stored: int | None = n_training_rows

    if X_train is not None and len(X_train.columns) > 0:
        original_columns = list(X_train.columns)
        feature_means, feature_medians, feature_modes, categorical_uniques = _compute_training_stats(X_train)
        if n_rows_stored is None:
            n_rows_stored = int(len(X_train))
    else:
        original_columns = _infer_pipeline_input_columns(pipeline)

    training_date = datetime.now().isoformat(timespec="seconds")

    bundle = {
        "pipeline": pipeline,
        "model": model,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "task_type": task_type,
        "target_col": target_col,
        "best_metrics": best_metrics,
        "model_name": model_name,
        "run_id": run_id,
        "feature_means": feature_means,
        "feature_medians": feature_medians,
        "feature_modes": feature_modes,
        "original_columns": original_columns,
        "training_date": training_date,
        "n_training_rows": n_rows_stored,
        "categorical_uniques": categorical_uniques,
        "num_cols": list(num_cols) if num_cols is not None else None,
        "cat_cols": list(cat_cols) if cat_cols is not None else None,
        "bundle_feature_importances": _extract_feature_importance(model, feature_names),
    }
    path = OUTPUT_DIR / f"{run_id}_model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    print(f"Model saved → {path}")
    return str(path)


# ── Load ──────────────────────────────────────────────────────────────────────


def load_model(pkl_path: str) -> dict:
    """Load a saved model bundle."""
    path = Path(pkl_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {pkl_path}")
    bundle = joblib.load(path)
    return bundle


# ── Predict ───────────────────────────────────────────────────────────────────


def prepare_transformed_features(bundle: dict, df: pd.DataFrame):
    """
    Align input columns like predict(), return transformed design matrix.
    Used for SHAP / explanation in the UI.
    """
    pipeline = bundle["pipeline"]
    target_col = bundle["target_col"]
    feature_medians = bundle.get("feature_medians") or {}
    feature_modes = bundle.get("feature_modes") or {}
    expected = bundle.get("original_columns")
    if not expected:
        expected = _infer_pipeline_input_columns(pipeline)

    fill_log: list[str] = []

    X = df.drop(columns=[target_col], errors="ignore").copy()
    extra = [c for c in X.columns if c not in expected]
    if extra:
        X = X.drop(columns=extra, errors="ignore")
        fill_log.append(f"Dropped extra columns not used in training: {extra}")

    for col in expected:
        if col not in X.columns:
            if col in feature_medians:
                val = feature_medians[col]
                X[col] = val
                fill_log.append(f"Filled missing column '{col}' with training median: {val}")
            elif col in feature_modes:
                val = feature_modes[col]
                X[col] = val
                fill_log.append(f"Filled missing column '{col}' with training mode: {val}")
            else:
                if col in (bundle.get("num_cols") or []):
                    X[col] = 0.0
                    fill_log.append(f"Filled missing column '{col}' with 0.0 (no training stat)")
                else:
                    X[col] = ""
                    fill_log.append(f"Filled missing column '{col}' with empty string (no training stat)")

    X = X[[c for c in expected if c in X.columns]]
    X_transformed = pipeline.transform(X)
    return X_transformed, fill_log


def predict(bundle: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Run inference on new data.

    Returns
    -------
    result_df : DataFrame with original columns + 'prediction' (+ 'probability' for binary clf)
    fill_log  : human-readable lines for imputed missing columns
    """
    pipeline = bundle["pipeline"]
    model = bundle["model"]
    label_encoder = bundle.get("label_encoder")
    task_type = bundle["task_type"]
    target_col = bundle["target_col"]

    try:
        X_transformed, fill_log = prepare_transformed_features(bundle, df)
    except Exception as e:
        logger.exception("Pipeline transform failed")
        raise RuntimeError(f"Preprocessing failed: {e}") from e

    try:
        preds_raw = model.predict(X_transformed)
    except Exception as e:
        logger.exception("Model predict failed")
        raise RuntimeError(f"Prediction failed: {e}") from e

    result = df.copy()

    if task_type == "classification" and label_encoder is not None:
        pr = np.asarray(preds_raw).ravel()
        if pr.dtype.kind in "fiu":
            pr = pr.astype(int)
        result["prediction"] = label_encoder.inverse_transform(pr)
        if hasattr(model, "predict_proba") and len(label_encoder.classes_) == 2:
            proba = model.predict_proba(X_transformed)[:, 1]
            result["probability"] = np.asarray(proba).round(4)
    else:
        result["prediction"] = preds_raw

    return result, fill_log


# ── Info ──────────────────────────────────────────────────────────────────────


def print_model_info(bundle: dict) -> None:
    print("\n" + "=" * 50)
    print("  Model bundle info")
    print("=" * 50)
    print(f"  Run ID:       {bundle.get('run_id', '—')}")
    print(f"  Model:        {bundle.get('model_name', '—')}")
    print(f"  Task type:    {bundle.get('task_type', '—')}")
    print(f"  Target col:   {bundle.get('target_col', '—')}")
    print(f"  Features:     {len(bundle.get('feature_names', []))}")
    print()
    metrics = bundle.get("best_metrics", {})
    if metrics:
        print("  Metrics on test set:")
        for k, v in metrics.items():
            if k != "train_time_s" and isinstance(v, float):
                print(f"    {k}: {v:.4f}")
    print("=" * 50 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cli_save(args):
    """Load agent result and save model bundle — called after a pipeline run."""

    print("Note: to save a model programmatically, call save_model() directly")
    print("from your code after agent.run() completes.")
    print()
    print("Example:")
    print("  from predict import save_model")
    print("  save_model(")
    print("      pipeline      = result['prep']['pipeline'],")
    print("      model         = agent._train_result['best_model'],")
    print("      label_encoder = result['prep'].get('label_encoder'),")
    print("      feature_names = result['prep']['feature_names'],")
    print("      task_type     = result['task_type'],")
    print("      target_col    = result['target_col'],")
    print("      best_metrics  = result['best_metrics'],")
    print("      model_name    = result['best_model_name'],")
    print("      run_id        = 'my_run',")
    print("  )")


def _cli_predict(args):
    print(f"Loading model: {args.model}")
    bundle = load_model(args.model)
    print_model_info(bundle)

    print(f"Loading input: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Input shape: {df.shape}")

    result_df, fill_log = predict(bundle, df)
    for line in fill_log:
        print(line)

    out_path = Path(args.output) if args.output else Path(args.input).parent / (Path(args.input).stem + "_predictions.csv")

    result_df.to_csv(out_path, index=False)
    print(f"\nPredictions saved → {out_path}")
    print(result_df[["prediction"] + (["probability"] if "probability" in result_df else [])].head(10))


def _cli_info(args):
    bundle = load_model(args.model)
    print_model_info(bundle)
    print("Feature names used during training:")
    for i, name in enumerate(bundle.get("feature_names", []), 1):
        print(f"  {i:2d}. {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Explainable ML Pipeline Agent — model persistence and inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command")

    # save
    p_save = sub.add_parser("save", help="Instructions for saving a trained model")
    p_save.add_argument("--run-id", default="model", help="Run identifier")

    # predict
    p_pred = sub.add_parser("predict", help="Run inference on new data")
    p_pred.add_argument("--model", required=True, help="Path to .pkl model file")
    p_pred.add_argument("--input", required=True, help="Path to input CSV")
    p_pred.add_argument("--output", default=None, help="Path to save predictions CSV")

    # info
    p_info = sub.add_parser("info", help="Show model info")
    p_info.add_argument("--model", required=True, help="Path to .pkl model file")

    args = parser.parse_args()

    if args.command == "save":
        _cli_save(args)
    elif args.command == "predict":
        _cli_predict(args)
    elif args.command == "info":
        _cli_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

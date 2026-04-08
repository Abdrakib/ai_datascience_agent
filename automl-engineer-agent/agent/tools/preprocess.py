"""
agent/tools/preprocess.py — Smart preprocessing pipeline.

Automatically handles:
  - Duplicate removal
  - Log1p transform for highly skewed numeric columns (when values are positive)
  - Missing value imputation (numeric + categorical)
  - Categorical encoding (binary, one-hot, ordinal)
  - Numeric scaling
  - SMOTE / SMOTEENN for class imbalance (when applicable)
  - Feature/target splitting
  - Train/test splitting

The pipeline is built with sklearn so it can be saved alongside the model
and used consistently at inference time.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

from config import RANDOM_STATE, TEST_SIZE

warnings.filterwarnings("ignore")


class Log1pTransformer(BaseEstimator, TransformerMixin):
    """Apply numpy log1p to selected columns (by index within the numeric block)."""

    def __init__(self, indices: list[int] | None = None):
        self.indices = indices if indices is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        out = X.copy()
        for i in self.indices:
            if i < out.shape[1]:
                out[:, i] = np.log1p(out[:, i])
        return out


# ── Public entry point ────────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "auto",            # "classification" | "regression" | "auto"
    scaler_type: str = "standard",      # "standard" | "minmax" | "none"
    drop_duplicates: bool = True,
    high_cardinality_threshold: int = 15,
) -> dict[str, Any]:
    """
    Build and fit a full preprocessing pipeline, split data, return everything
    the training step needs.

    Returns
    -------
    {
      "X_train", "X_test", "y_train", "y_test",
      "pipeline"          : fitted ColumnTransformer (for saving/inference),
      "feature_names"     : column names after encoding,
      "label_encoder"     : fitted LabelEncoder (classification only),
      "task_type"         : "classification" | "regression",
      "dropped_cols"      : columns removed before preprocessing,
      "encoding_summary"  : what was done to each column,
      "preprocessing_log" : list of human-readable steps taken,
      "log_transformed_cols" : list of column names log1p-transformed in the numeric pipe,
      "smote_applied"     : whether SMOTE/SMOTEENN resampling was applied,
      "smote_log"         : human-readable SMOTE outcome,
    }
    """
    log: list[str] = []
    df = df.copy()

    # 1. Drop duplicates
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        dropped = before - len(df)
        if dropped:
            log.append(f"Dropped {dropped} duplicate rows.")

    # 2. Validate target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    # 3. Infer task type
    if task_type == "auto":
        task_type = _infer_task(df[target_col])
        log.append(f"Inferred task type: {task_type}.")

    # 4. Separate features and target
    y_raw = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # 5. Drop constant / ID-like columns
    dropped_cols = []
    for col in X.columns:
        if X[col].nunique() <= 1:
            X = X.drop(columns=[col])
            dropped_cols.append(col)
            log.append(f"Dropped constant column: `{col}`.")
        elif _looks_like_id(col, X[col]):
            X = X.drop(columns=[col])
            dropped_cols.append(col)
            log.append(f"Dropped likely ID column: `{col}`.")

    # 6. Categorize columns
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    # Split high-cardinality categoricals out (drop for now, flag for user)
    high_card_cols = [
        c for c in cat_cols if X[c].nunique() > high_cardinality_threshold
    ]
    safe_cat_cols = [c for c in cat_cols if c not in high_card_cols]

    for col in high_card_cols:
        X = X.drop(columns=[col])
        dropped_cols.append(col)
        log.append(
            f"Dropped high-cardinality column `{col}` "
            f"({df[col].nunique()} unique values). "
            "Consider target encoding in a future upgrade."
        )

    encoding_summary: dict[str, str] = {}

    # 6b. Skewness-based log1p (tracked by column name; applied in numeric pipeline)
    log_transformed_cols: list[str] = []
    log1p_indices: list[int] = []
    for idx, col in enumerate(num_cols):
        series = X[col].dropna()
        if len(series) < 3:
            continue
        skew_val = float(series.skew())
        if abs(skew_val) <= 1.5:
            continue
        if series.min() <= 0:
            if series.min() < 0:
                log.append(
                    f"Skipped log transform on `{col}` — contains negative values "
                    f"(skewness {skew_val:.2f})"
                )
            else:
                log.append(
                    f"Skipped log transform on `{col}` — not all strictly positive "
                    f"(skewness {skew_val:.2f})"
                )
            continue
        log_transformed_cols.append(col)
        log1p_indices.append(idx)
        log.append(
            f"Applied log1p transform to `{col}` (skewness was {skew_val:.2f})"
        )

    # 7. Build sklearn pipelines per column group
    transformers = []

    # Numeric: log1p (optional) → impute → scale
    if num_cols:
        scaler = (
            StandardScaler() if scaler_type == "standard"
            else MinMaxScaler() if scaler_type == "minmax"
            else "passthrough"
        )
        num_pipe = Pipeline([
            ("log1p", Log1pTransformer(indices=log1p_indices)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ])
        transformers.append(("numeric", num_pipe, num_cols))
        for col in num_cols:
            encoding_summary[col] = f"median imputation + {scaler_type} scaling"
        log.append(
            f"Numeric columns ({len(num_cols)}): median imputation + {scaler_type} scaling."
        )

    # Categorical: impute → one-hot encode
    if safe_cat_cols:
        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", cat_pipe, safe_cat_cols))
        for col in safe_cat_cols:
            encoding_summary[col] = f"mode imputation + one-hot encoding"
        log.append(
            f"Categorical columns ({len(safe_cat_cols)}): mode imputation + one-hot encoding."
        )

    # Build ColumnTransformer
    if not transformers:
        raise ValueError("No usable feature columns remain after preprocessing.")

    pipeline = ColumnTransformer(transformers=transformers, remainder="drop")

    # 8. Encode target
    label_encoder = None
    if task_type == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        log.append(
            f"Target encoded with LabelEncoder. "
            f"Classes: {list(label_encoder.classes_)}"
        )
    else:
        y = y_raw.values.astype(float)
        # Handle missing target values
        valid_mask = ~np.isnan(y)
        if not valid_mask.all():
            n_bad = (~valid_mask).sum()
            log.append(f"Dropped {n_bad} rows with missing target values.")
            X = X[valid_mask]
            y = y[valid_mask]

    # 9. Train/test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if task_type == "classification" and len(np.unique(y)) > 1 else None,
    )
    log.append(
        f"Train/test split: {len(X_train_raw)} train rows, "
        f"{len(X_test_raw)} test rows ({int(TEST_SIZE*100)}% test)."
    )

    # 10. Fit pipeline on train, transform both
    X_train = pipeline.fit_transform(X_train_raw)
    X_test = pipeline.transform(X_test_raw)

    smote_applied = False
    smote_log = ""

    # 11. Class imbalance: SMOTE / SMOTEENN on transformed training features
    if task_type == "classification":
        y_tr = np.asarray(y_train)
        uniq, counts = np.unique(y_tr, return_counts=True)
        if len(counts) >= 2:
            imbalance_ratio = float(counts.max()) / float(counts.min())
            log.append(
                f"Class imbalance ratio (majority/minority): {imbalance_ratio:.2f}."
            )
            if imbalance_ratio > 2.0:
                X_train, y_train, smote_log, smote_applied = apply_smote(
                    X_train,
                    y_tr,
                    imbalance_ratio,
                    random_state=RANDOM_STATE,
                )
                if smote_log:
                    log.append(smote_log)
        else:
            imbalance_ratio = 1.0
    else:
        imbalance_ratio = 1.0

    # 12. Derive feature names after encoding
    feature_names = _get_feature_names(pipeline, num_cols, safe_cat_cols)
    log.append(f"Final feature matrix: {X_train.shape[1]} features.")

    return {
        "X_train":          X_train,
        "X_test":           X_test,
        "y_train":          y_train,
        "y_test":           y_test,
        "pipeline":         pipeline,
        "feature_names":    feature_names,
        "label_encoder":    label_encoder,
        "task_type":        task_type,
        "dropped_cols":     dropped_cols,
        "encoding_summary": encoding_summary,
        "preprocessing_log": log,
        "num_cols":         num_cols,
        "cat_cols":         safe_cat_cols,
        "n_classes":        len(np.unique(y_train)) if task_type == "classification" else None,
        "log_transformed_cols": log_transformed_cols,
        "smote_applied":    smote_applied,
        "smote_log":        smote_log,
    }


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    imbalance_ratio: float,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    """
    Optionally resample training data with SMOTE or SMOTEENN.

    Returns (X_train_out, y_train_out, log_message, applied).
    """
    try:
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
    except ImportError:
        msg = "SMOTE skipped — run: pip install imbalanced-learn"
        warnings.warn(msg, stacklevel=2)
        return X_train, y_train, msg, False

    if imbalance_ratio <= 2.0:
        return X_train, y_train, "", False

    try:
        if imbalance_ratio > 5.0:
            sampler = SMOTEENN(random_state=random_state)
            method_name = "SMOTEENN"
        else:
            sampler = SMOTE(random_state=random_state)
            method_name = "SMOTE"

        X_res, y_res = sampler.fit_resample(
            np.asarray(X_train), np.asarray(y_train).ravel()
        )

        def _dist(y: np.ndarray) -> str:
            u, c = np.unique(y, return_counts=True)
            parts = [f"class {int(a)}: {int(b)}" for a, b in zip(u, c)]
            return ", ".join(parts)

        msg = (
            f"Applied {method_name} (imbalance ratio was {imbalance_ratio:.2f}). "
            f"New class distribution: {_dist(y_res)}."
        )
        return X_res, y_res, msg, True
    except Exception as e:
        msg = f"SMOTE skipped due to error: {e}"
        warnings.warn(msg, stacklevel=2)
        return X_train, y_train, msg, False


def preprocessing_log_to_markdown(result: dict[str, Any]) -> str:
    """Format preprocessing result as readable markdown."""
    lines = ["## Preprocessing steps"]
    for step in result["preprocessing_log"]:
        lines.append(f"- {step}")
    lines += [
        "",
        f"**Train size:** {result['X_train'].shape[0]} rows × {result['X_train'].shape[1]} features",
        f"**Test size:**  {result['X_test'].shape[0]} rows × {result['X_test'].shape[1]} features",
        f"**Task type:**  {result['task_type']}",
    ]
    if result["dropped_cols"]:
        lines.append(f"**Dropped columns:** {', '.join(result['dropped_cols'])}")
    lt = result.get("log_transformed_cols") or []
    if lt:
        lines.append(f"**Log transforms applied:** {', '.join(lt)}")
    if result.get("smote_applied"):
        sl = result.get("smote_log") or ""
        lines.append(f"**Class balancing:** {sl}")
    return "\n".join(lines)


# ── Private helpers ───────────────────────────────────────────────────────────

def _infer_task(target: pd.Series) -> str:
    """Infer regression vs classification from the target column."""
    if not pd.api.types.is_numeric_dtype(target):
        return "classification"
    n_unique = target.nunique()
    n_rows = len(target)
    # Heuristic: if fewer than 20 unique values or < 5% of rows are unique → classify
    if n_unique <= 20 or (n_unique / n_rows) < 0.05:
        return "classification"
    return "regression"


def _looks_like_id(col_name: str, series: pd.Series) -> bool:
    """Heuristic: is this column an ID field we should drop?"""
    name_lower = col_name.lower()
    id_keywords = ["_id", "id_", " id", "uuid", "index", "row_num", "row_number"]
    if any(kw in name_lower for kw in id_keywords):
        # Only drop if nearly all values are unique
        if series.nunique() / len(series) > 0.95:
            return True
    return False


def _get_feature_names(
    pipeline: ColumnTransformer,
    num_cols: list[str],
    cat_cols: list[str],
) -> list[str]:
    """Extract feature names after ColumnTransformer fit."""
    names = []
    for name, transformer, cols in pipeline.transformers_:
        if name == "numeric":
            names.extend(cols)
        elif name == "categorical":
            try:
                encoder = transformer.named_steps["encoder"]
                ohe_names = encoder.get_feature_names_out(cols).tolist()
                names.extend(ohe_names)
            except Exception:
                names.extend(cols)
    return names

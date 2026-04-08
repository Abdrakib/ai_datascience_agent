"""
agent/tools/eda.py — Exploratory Data Analysis tool.

Called by the agent to profile any CSV dataset and return a structured
summary the LLM can reason about: shape, dtypes, missing values,
descriptive stats, correlations, class balance, and data quality flags.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ── Public entry point ────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame, target_col: str | None = None) -> dict[str, Any]:
    """
    Profile a dataframe and return a structured dict the agent can read.

    Parameters
    ----------
    df         : the loaded dataframe
    target_col : optional — if known, adds target-specific stats

    Returns
    -------
    dict with keys: overview, columns, missing, target_info, quality_flags
    """
    report: dict[str, Any] = {}

    report["overview"] = _overview(df)
    report["columns"]  = _column_profiles(df)
    report["missing"]  = _missing_summary(df)
    report["sample"]   = _sample_rows(df)

    if target_col and target_col in df.columns:
        report["target_info"] = _target_profile(df, target_col)

    report["quality_flags"] = _quality_flags(df, target_col)
    report["recommendations"] = _recommendations(report)

    return report


def eda_to_markdown(report: dict[str, Any]) -> str:
    """Convert EDA report dict to a clean markdown string for display."""
    lines = []

    ov = report["overview"]
    lines += [
        "## Dataset overview",
        f"- **Rows:** {ov['rows']:,}",
        f"- **Columns:** {ov['columns']}",
        f"- **Numeric columns:** {ov['numeric_cols']}",
        f"- **Categorical columns:** {ov['categorical_cols']}",
        f"- **Duplicate rows:** {ov['duplicate_rows']}",
        f"- **Memory usage:** {ov['memory_mb']:.2f} MB",
        "",
    ]

    miss = report["missing"]
    if miss["total_missing"] > 0:
        lines += ["## Missing values"]
        for col, info in miss["by_column"].items():
            lines.append(f"- `{col}`: {info['count']} missing ({info['pct']:.1f}%)")
        lines.append("")

    lines += ["## Column profiles"]
    for col, prof in report["columns"].items():
        if prof["dtype_group"] == "numeric":
            lines.append(
                f"- **{col}** (numeric): mean={prof['mean']:.2f}, "
                f"std={prof['std']:.2f}, min={prof['min']:.2f}, max={prof['max']:.2f}"
            )
        else:
            lines.append(
                f"- **{col}** (categorical): {prof['n_unique']} unique values, "
                f"top='{prof['top_value']}'"
            )
    lines.append("")

    if "target_info" in report:
        ti = report["target_info"]
        lines += ["## Target column", f"- **Column:** `{ti['column']}`",
                  f"- **Type:** {ti['inferred_task']}"]
        if ti["inferred_task"] == "classification":
            lines.append(f"- **Classes:** {ti['classes']}")
            lines.append(f"- **Class distribution:** {ti['class_distribution']}")
        else:
            lines.append(f"- **Range:** {ti['min']:.2f} → {ti['max']:.2f}")
            lines.append(f"- **Mean:** {ti['mean']:.2f}  Std: {ti['std']:.2f}")
        lines.append("")

    flags = report["quality_flags"]
    if flags:
        lines += ["## Quality flags"]
        for f in flags:
            lines.append(f"- {f}")
        lines.append("")

    recs = report["recommendations"]
    if recs:
        lines += ["## Preprocessing recommendations"]
        for r in recs:
            lines.append(f"- {r}")

    return "\n".join(lines)


# ── Private helpers ───────────────────────────────────────────────────────────

def _overview(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()
    return {
        "rows":             len(df),
        "columns":          len(df.columns),
        "numeric_cols":     len(num_cols),
        "categorical_cols": len(cat_cols),
        "duplicate_rows":   int(df.duplicated().sum()),
        "memory_mb":        df.memory_usage(deep=True).sum() / 1_048_576,
        "column_names":     df.columns.tolist(),
    }


def _column_profiles(df: pd.DataFrame) -> dict:
    profiles = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            profiles[col] = {
                "dtype":       str(series.dtype),
                "dtype_group": "numeric",
                "missing":     int(series.isna().sum()),
                "n_unique":    int(series.nunique()),
                "mean":        float(series.mean()) if series.notna().any() else None,
                "std":         float(series.std())  if series.notna().any() else None,
                "min":         float(series.min())  if series.notna().any() else None,
                "max":         float(series.max())  if series.notna().any() else None,
                "median":      float(series.median()) if series.notna().any() else None,
                "skewness":    float(series.skew()) if series.notna().any() else None,
            }
        else:
            vc = series.value_counts()
            profiles[col] = {
                "dtype":       str(series.dtype),
                "dtype_group": "categorical",
                "missing":     int(series.isna().sum()),
                "n_unique":    int(series.nunique()),
                "top_value":   str(vc.index[0]) if len(vc) > 0 else None,
                "top_freq":    int(vc.iloc[0])  if len(vc) > 0 else None,
                "value_counts": vc.head(10).to_dict(),
            }
    return profiles


def _missing_summary(df: pd.DataFrame) -> dict:
    missing_cols = {}
    total = 0
    for col in df.columns:
        cnt = int(df[col].isna().sum())
        if cnt > 0:
            pct = cnt / len(df) * 100
            missing_cols[col] = {"count": cnt, "pct": round(pct, 2)}
            total += cnt
    return {
        "total_missing": total,
        "columns_with_missing": len(missing_cols),
        "by_column": missing_cols,
    }


def _sample_rows(df: pd.DataFrame, n: int = 5) -> list[dict]:
    return df.head(n).to_dict(orient="records")


def _target_profile(df: pd.DataFrame, target_col: str) -> dict:
    series = df[target_col]
    info: dict[str, Any] = {"column": target_col}

    n_unique = series.nunique()
    is_numeric = pd.api.types.is_numeric_dtype(series)

    # Infer task type
    if not is_numeric or n_unique <= 20:
        info["inferred_task"] = "classification"
        vc = series.value_counts()
        info["classes"] = series.unique().tolist()
        info["n_classes"] = n_unique
        info["class_distribution"] = vc.to_dict()
        # Imbalance ratio
        if len(vc) >= 2:
            info["imbalance_ratio"] = round(vc.iloc[0] / vc.iloc[-1], 2)
    else:
        info["inferred_task"] = "regression"
        info["mean"]   = float(series.mean())
        info["std"]    = float(series.std())
        info["min"]    = float(series.min())
        info["max"]    = float(series.max())
        info["median"] = float(series.median())
        info["skewness"] = float(series.skew())

    return info


def _quality_flags(df: pd.DataFrame, target_col: str | None) -> list[str]:
    flags = []

    # High missing
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 30:
            flags.append(f"HIGH MISSING: `{col}` has {pct:.0f}% missing values — consider dropping.")
        elif pct > 10:
            flags.append(f"MODERATE MISSING: `{col}` has {pct:.0f}% missing values.")

    # Duplicates
    dupes = df.duplicated().sum()
    if dupes > 0:
        flags.append(f"DUPLICATES: {dupes} duplicate rows detected.")

    # High cardinality categoricals
    for col in df.select_dtypes(exclude="number").columns:
        if col == target_col:
            continue
        n_uniq = df[col].nunique()
        if n_uniq > 50:
            flags.append(f"HIGH CARDINALITY: `{col}` has {n_uniq} unique values — may need target encoding.")

    # Constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            flags.append(f"CONSTANT COLUMN: `{col}` has only one unique value — should be dropped.")

    # Class imbalance
    if target_col and target_col in df.columns:
        series = df[target_col]
        if not pd.api.types.is_numeric_dtype(series) or series.nunique() <= 20:
            vc = series.value_counts()
            if len(vc) >= 2:
                ratio = vc.iloc[0] / vc.iloc[-1]
                if ratio > 5:
                    flags.append(
                        f"CLASS IMBALANCE: target ratio is {ratio:.1f}:1 — "
                        "consider SMOTE or class_weight='balanced'."
                    )

    # Skewed numeric columns
    for col in df.select_dtypes(include="number").columns:
        if col == target_col:
            continue
        skew = df[col].skew()
        if abs(skew) > 2:
            flags.append(f"HIGH SKEW: `{col}` skewness={skew:.2f} — consider log transform.")

    return flags


def _recommendations(report: dict) -> list[str]:
    recs = []
    miss = report["missing"]

    if miss["total_missing"] > 0:
        recs.append("Impute missing numeric values (median recommended for skewed data).")
        recs.append("Impute missing categorical values with mode or a dedicated 'Unknown' category.")

    for col, prof in report["columns"].items():
        if prof["dtype_group"] == "categorical" and prof["n_unique"] == 2:
            recs.append(f"Binary-encode `{col}` (Label Encoding is sufficient).")
        elif prof["dtype_group"] == "categorical" and prof["n_unique"] <= 15:
            recs.append(f"One-hot encode `{col}` ({prof['n_unique']} categories).")

    num_cols = [c for c, p in report["columns"].items() if p["dtype_group"] == "numeric"]
    if len(num_cols) > 1:
        recs.append("Scale numeric features with StandardScaler or MinMaxScaler before training.")

    dupes = report["overview"]["duplicate_rows"]
    if dupes > 0:
        recs.append(f"Drop {dupes} duplicate rows before training.")

    return recs

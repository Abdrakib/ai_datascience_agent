"""
agent/tools/task_detector.py — Infers ML task type and target column.

Given a dataframe (and optional user hint), determines:
  - task_type : "classification" | "regression"
  - target_col: best candidate for the target column
  - confidence : "high" | "medium" | "low"
  - reasoning  : human-readable explanation of the decision
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


# ── Public entry point ────────────────────────────────────────────────────────

def detect_task(
    df: pd.DataFrame,
    user_hint: str = "",
) -> dict[str, Any]:
    """
    Auto-detect the ML task and target column.

    Parameters
    ----------
    df        : loaded dataframe
    user_hint : free-text from the user, e.g. "predict house price" or
                "classify whether a patient is readmitted"

    Returns
    -------
    {
      "target_col"  : str,
      "task_type"   : "classification" | "regression",
      "confidence"  : "high" | "medium" | "low",
      "reasoning"   : str,
      "alternatives": list of other plausible target columns,
    }
    """
    hint_lower = user_hint.lower()

    # 1. Try to find target from user hint first
    hint_target = _extract_target_from_hint(df, hint_lower)

    # 2. Score every column as a target candidate
    scores = {col: _score_column_as_target(df, col, hint_lower) for col in df.columns}
    ranked = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)

    # 3. Pick best candidate
    if hint_target and hint_target in df.columns:
        target_col = hint_target
        confidence = "high"
        reasoning  = f"Target column `{target_col}` matched from user description."
    elif ranked:
        target_col = ranked[0][0]
        top_score  = ranked[0][1]["score"]
        confidence = "high" if top_score >= 8 else "medium" if top_score >= 4 else "low"
        reasoning  = ranked[0][1]["reason"]
    else:
        raise ValueError("Could not identify any target column. Please specify it explicitly.")

    # 4. Infer task type
    task_type, task_reason = _infer_task_type(df[target_col], hint_lower)
    reasoning += f" {task_reason}"

    # 5. Alternatives (next best candidates, excluding winner)
    alternatives = [col for col, _ in ranked[1:4] if col != target_col]

    return {
        "target_col":   target_col,
        "task_type":    task_type,
        "confidence":   confidence,
        "reasoning":    reasoning,
        "alternatives": alternatives,
        "n_classes":    int(df[target_col].nunique()) if task_type == "classification" else None,
        "target_dtype": str(df[target_col].dtype),
    }


def task_detection_to_markdown(result: dict[str, Any]) -> str:
    lines = [
        "## Task detection",
        f"- **Target column:** `{result['target_col']}`",
        f"- **Task type:** {result['task_type']}",
        f"- **Confidence:** {result['confidence']}",
        f"- **Reasoning:** {result['reasoning']}",
    ]
    if result["task_type"] == "classification":
        lines.append(f"- **Number of classes:** {result['n_classes']}")
    if result["alternatives"]:
        lines.append(f"- **Alternative candidates:** {', '.join(f'`{c}`' for c in result['alternatives'])}")
    return "\n".join(lines)


# ── Private helpers ───────────────────────────────────────────────────────────

# Keywords that strongly suggest a column is a target
_CLASSIFICATION_TARGET_KEYWORDS = [
    "target", "label", "class", "category", "outcome", "result",
    "survived", "churn", "fraud", "default", "readmit", "diagnosis",
    "disease", "spam", "approved", "cancelled", "converted", "clicked",
    "purchased", "status", "flag", "is_", "has_", "will_",
]

_REGRESSION_TARGET_KEYWORDS = [
    "price", "cost", "salary", "revenue", "sales", "income", "value",
    "amount", "score", "rate", "count", "total", "profit", "loss",
    "age", "duration", "length", "size", "weight", "height", "area",
    "demand", "quantity", "yield", "return",
]

_TASK_CLASSIFICATION_HINTS = [
    "classif", "predict whether", "predict if", "binary", "categor",
    "label", "class", "churn", "fraud", "spam", "diagnosis",
    "survived", "approve", "readmit",
]

_TASK_REGRESSION_HINTS = [
    "regress", "predict price", "predict cost", "predict salary",
    "predict revenue", "how much", "how many", "forecast", "estimate",
    "continuous",
]


def _extract_target_from_hint(df: pd.DataFrame, hint: str) -> str | None:
    """Check if the user hint names a column directly."""
    for col in df.columns:
        # Exact match (case-insensitive, spaces→underscores)
        col_variants = [col.lower(), col.lower().replace("_", " ")]
        for variant in col_variants:
            if variant in hint:
                return col
    return None


def _score_column_as_target(
    df: pd.DataFrame, col: str, hint: str
) -> dict[str, Any]:
    """Score a column 0-12 on how likely it is to be the target."""
    score = 0
    reasons = []
    series = df[col]
    col_lower = col.lower()

    # Keyword match in column name
    for kw in _CLASSIFICATION_TARGET_KEYWORDS + _REGRESSION_TARGET_KEYWORDS:
        if kw in col_lower:
            score += 4
            reasons.append(f"column name matches keyword '{kw}'")
            break

    # Column position: last column is often the target
    if col == df.columns[-1]:
        score += 3
        reasons.append("last column in dataframe")

    # Binary numeric (0/1) → very likely classification target
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = series.dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            score += 4
            reasons.append("binary 0/1 column")
        elif series.nunique() <= 10:
            score += 2
            reasons.append(f"low cardinality numeric ({series.nunique()} unique values)")
    else:
        # Categorical with low cardinality
        if series.nunique() <= 15:
            score += 2
            reasons.append(f"low cardinality categorical ({series.nunique()} unique values)")

    # Penalize: looks like an ID
    if any(kw in col_lower for kw in ["_id", "id_", "uuid", "index"]):
        score -= 5
        reasons.append("looks like an ID column")

    # Penalize: high cardinality text
    if series.dtype == object and series.nunique() > 50:
        score -= 3
        reasons.append("high cardinality text")

    # Hint mentions this column name
    if col_lower in hint or col_lower.replace("_", " ") in hint:
        score += 5
        reasons.append("mentioned in user description")

    reason_str = (
        f"`{col}` selected as target: " + "; ".join(reasons) + "."
        if reasons else f"`{col}` selected by default."
    )
    return {"score": score, "reason": reason_str}


def _infer_task_type(
    target: pd.Series, hint: str
) -> tuple[str, str]:
    """Return (task_type, reasoning_sentence)."""

    # Check hint for explicit task keywords
    for kw in _TASK_CLASSIFICATION_HINTS:
        if kw in hint:
            return "classification", f"User description suggests classification ('{kw}')."

    for kw in _TASK_REGRESSION_HINTS:
        if kw in hint:
            return "regression", f"User description suggests regression ('{kw}')."

    # Infer from data
    if not pd.api.types.is_numeric_dtype(target):
        return "classification", "Target is non-numeric → classification."

    col_lower = str(target.name or "").lower()
    _continuous_name_keywords = (
        "price", "cost", "salary", "revenue", "income", "amount", "value",
        "total", "rate", "score", "age", "weight", "height", "area", "distance",
        "duration", "length", "size", "demand", "quantity", "yield", "return",
        "profit", "loss", "temperature", "humidity", "pressure",
    )
    if any(kw in col_lower for kw in _continuous_name_keywords):
        return "regression", (
            "Column name suggests a continuous numeric target → regression."
        )

    n_unique = target.nunique()
    n_rows   = len(target)

    if set(target.dropna().unique()).issubset({0, 1}):
        return "classification", "Target is binary (0/1) → classification."

    if n_unique <= 15:
        return "classification", (
            f"Target has only {n_unique} unique values → treated as classification."
        )

    if n_unique / n_rows > 0.05:
        return "regression", (
            f"Target is continuous numeric ({n_unique} unique values) → regression."
        )

    return "classification", (
        f"Target has {n_unique} unique values relative to {n_rows} rows → classification."
    )

"""
agent/tools/plan_training.py — Dataset-aware training configuration before model fitting.

Builds a structured plan from EDA, task detection, and preprocessing outputs.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

# Model names must match agent/tools/train.py exactly
CLF_BASELINE = "Logistic Regression"
REG_BASELINE = "Linear Regression"
RF = "Random Forest"
XGB = "XGBoost"
LGB = "LightGBM"


def _skewed_column_names(quality_flags: list | None) -> list[str]:
    """Extract column names from HIGH SKEW quality flag strings."""
    if not quality_flags:
        return []
    cols: list[str] = []
    for flag in quality_flags:
        if not isinstance(flag, str):
            continue
        if "HIGH SKEW" not in flag and "skewness" not in flag.lower():
            continue
        m = re.search(r"`([^`]+)`", flag)
        if m:
            cols.append(m.group(1))
    return cols


def _imbalance_from_y(y: Any) -> tuple[float, int, int]:
    """Return (majority/minority ratio, majority_count, minority_count) for classification."""
    y = np.asarray(y).ravel()
    if y.size == 0:
        return 1.0, 0, 0
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2:
        return 1.0, int(counts[0]), int(counts[0])
    maj = int(counts.max())
    mino = int(counts.min())
    ratio = float(maj) / float(mino) if mino > 0 else 1.0
    return ratio, maj, mino


def plan_training(
    eda_report: dict[str, Any],
    task_result: dict[str, Any],
    prep_result: dict[str, Any],
) -> dict[str, Any]:
    """
    Analyze dataset characteristics and return a training plan dict.
    Uses .get() throughout so missing keys do not raise.
    """
    overview = eda_report.get("overview") or {}
    n_rows = int(overview.get("rows", 0))

    feature_names = prep_result.get("feature_names") or []
    n_features = len(feature_names)

    task_type = str(task_result.get("task_type", "classification") or "classification")
    n_classes = task_result.get("n_classes")
    if n_classes is None:
        n_classes = prep_result.get("n_classes")
    if n_classes is None and task_type == "classification":
        n_classes = 2
    n_classes = int(n_classes) if n_classes is not None else 0

    imbalance_ratio = prep_result.get("imbalance_ratio")
    if imbalance_ratio is None and task_type == "classification":
        y_tr = prep_result.get("y_train")
        if y_tr is not None:
            imbalance_ratio, _, _ = _imbalance_from_y(y_tr)
    if imbalance_ratio is None:
        imbalance_ratio = 1.0
    else:
        imbalance_ratio = float(imbalance_ratio)

    missing_block = eda_report.get("missing") or {}
    has_missing = int(missing_block.get("total_missing", 0) or 0) > 0

    quality_flags = eda_report.get("quality_flags") or []
    skewed_cols = _skewed_column_names(
        quality_flags if isinstance(quality_flags, list) else []
    )

    smote_applied = bool(prep_result.get("smote_applied", False))

    is_small = n_rows < 1000
    is_large = n_rows > 10000
    is_wide = n_features > 50
    is_binary = task_type == "classification" and n_classes == 2

    warnings: list[str] = []
    notes: list[str] = []
    skip_models: list[str] = []
    skip_reasons: dict[str, str] = {}
    adjusted_params: dict[str, dict[str, Any]] = {}

    # --- Model inclusion ---
    if task_type == "classification":
        recommended = [CLF_BASELINE, RF]
        skip_boost = is_small and n_features < 5
        if skip_boost:
            skip_models = [XGB, LGB]
            skip_reasons[XGB] = (
                "Small dataset (<1000 rows) with fewer than 5 features — "
                "skipping tree boosting to limit overfitting risk."
            )
            skip_reasons[LGB] = skip_reasons[XGB]
        else:
            recommended.extend([XGB, LGB])
    else:
        recommended = [REG_BASELINE, RF]
        skip_boost = is_small and n_features < 5
        if skip_boost:
            skip_models = [XGB, LGB]
            skip_reasons[XGB] = (
                "Small dataset (<1000 rows) with fewer than 5 features — "
                "skipping tree boosting to limit overfitting risk."
            )
            skip_reasons[LGB] = skip_reasons[XGB]
        else:
            recommended.extend([XGB, LGB])

    if is_wide:
        notes.append(
            "High-dimensional data — linear models and LightGBM tend to perform well."
        )
    if is_large:
        notes.append("Large dataset — LightGBM recommended for speed over XGBoost.")

    # --- Parameter presets (small vs large are mutually exclusive by row count) ---
    if is_small:
        warnings.append(
            "Small dataset — using conservative parameters to reduce overfitting risk."
        )
        adjusted_params[RF] = {"n_estimators": 100, "max_depth": 5}
        adjusted_params[XGB] = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1}
        adjusted_params[LGB] = {"n_estimators": 100, "num_leaves": 31}
    elif is_large:
        adjusted_params[RF] = {"n_estimators": 300, "n_jobs": -1}
        adjusted_params[XGB] = {"n_estimators": 300, "tree_method": "hist"}
        adjusted_params[LGB] = {"n_estimators": 300}

    # --- Class imbalance & class weights ---
    if task_type == "classification" and imbalance_ratio > 2.0 and not smote_applied:
        notes.append("Class imbalance detected — using balanced class weights.")
        adjusted_params.setdefault(CLF_BASELINE, {})
        adjusted_params[CLF_BASELINE]["class_weight"] = "balanced"
        adjusted_params.setdefault(RF, {})
        adjusted_params[RF]["class_weight"] = "balanced"
        adjusted_params.setdefault(LGB, {})
        adjusted_params[LGB]["class_weight"] = "balanced"
        y_tr = prep_result.get("y_train")
        if y_tr is not None and is_binary:
            ratio, maj_c, min_c = _imbalance_from_y(y_tr)
            if min_c > 0:
                spw = float(maj_c) / float(min_c)
                adjusted_params.setdefault(XGB, {})
                adjusted_params[XGB]["scale_pos_weight"] = spw
    elif task_type == "classification" and imbalance_ratio > 2.0 and smote_applied:
        notes.append("SMOTE already applied — using default class weights.")
        adjusted_params.setdefault(CLF_BASELINE, {})
        adjusted_params[CLF_BASELINE]["class_weight"] = None
        adjusted_params.setdefault(RF, {})
        adjusted_params[RF]["class_weight"] = None
        adjusted_params.setdefault(LGB, {})
        adjusted_params[LGB]["class_weight"] = None
        adjusted_params.setdefault(XGB, {})
        adjusted_params[XGB]["scale_pos_weight"] = 1

    # --- Primary metric ---
    metric_reasoning = ""
    if task_type == "regression":
        primary_metric = "r2"
        metric_reasoning = "Regression task — R² measures explained variance."
    elif is_binary:
        primary_metric = "roc_auc"
        if imbalance_ratio > 3.0:
            metric_reasoning = (
                "Imbalanced classes — ROC-AUC is more meaningful than accuracy."
            )
        else:
            metric_reasoning = (
                "Binary classification — ROC-AUC summarizes ranking quality across thresholds."
            )
    else:
        primary_metric = "f1_weighted"
        metric_reasoning = (
            "Multiclass classification — weighted F1 balances precision and recall per class."
        )

    # --- Tuning budget ---
    if is_small:
        n_trials, timeout = 20, 60
        tune_note = (
            f"Small dataset ({n_rows} rows): Optuna budget set to {n_trials} trials "
            f"and {timeout}s timeout to avoid overfitting."
        )
    elif n_rows <= 10000:
        n_trials, timeout = 50, 120
        tune_note = (
            f"Medium dataset ({n_rows} rows): Optuna budget set to {n_trials} trials "
            f"and {timeout}s timeout."
        )
    else:
        n_trials, timeout = 100, 300
        tune_note = (
            f"Large dataset ({n_rows} rows): Optuna budget set to {n_trials} trials "
            f"and {timeout}s timeout for thorough search."
        )
    notes.append(tune_note)

    # --- Plan summary paragraph ---
    parts = [
        f"The dataset has {n_rows} rows and {n_features} features after preprocessing.",
        f"Task: {task_type}",
    ]
    if task_type == "classification":
        parts.append(f"with {n_classes} classes.")
    if has_missing:
        parts.append("Missing values were present in the raw data.")
    if skewed_cols:
        parts.append(f"EDA flagged skewed columns: {', '.join(skewed_cols)}.")
    parts.append(
        f"Recommended models: {', '.join(recommended)}."
        + (f" Skipping: {', '.join(skip_models)}." if skip_models else "")
    )
    parts.append(
        f"Primary evaluation metric: {primary_metric}. "
        f"Tuning budget: {n_trials} trials, {timeout}s."
    )
    plan_summary = " ".join(parts)

    dataset_profile = {
        "n_rows": n_rows,
        "n_features": n_features,
        "is_small": is_small,
        "is_large": is_large,
        "is_wide": is_wide,
        "is_binary": is_binary,
        "imbalance_ratio": imbalance_ratio,
        "smote_applied": smote_applied,
    }

    return {
        "recommended_models": recommended,
        "skip_models": skip_models,
        "skip_reasons": skip_reasons,
        "adjusted_params": adjusted_params,
        "primary_metric": primary_metric,
        "metric_reasoning": metric_reasoning,
        "n_trials": n_trials,
        "timeout": timeout,
        "warnings": warnings,
        "notes": notes,
        "plan_summary": plan_summary,
        "dataset_profile": dataset_profile,
    }


def plan_to_markdown(plan: dict[str, Any]) -> str:
    """Render a training plan as markdown for the agent / UI."""
    lines: list[str] = ["## Training plan", ""]

    prof = plan.get("dataset_profile") or {}
    lines.append("### Dataset profile")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    for key in (
        "n_rows",
        "n_features",
        "is_small",
        "is_large",
        "is_wide",
        "is_binary",
        "imbalance_ratio",
        "smote_applied",
    ):
        lines.append(f"| `{key}` | {prof.get(key, '—')} |")
    lines.append("")

    rec = plan.get("recommended_models") or []
    lines.append("### Recommended models")
    lines.append("")
    for m in rec:
        lines.append(f"- {m}")
    lines.append("")

    skip = plan.get("skip_models") or []
    reasons = plan.get("skip_reasons") or {}
    if skip:
        lines.append("### Skipped models")
        lines.append("")
        for m in skip:
            r = reasons.get(m, "Not selected by plan rules.")
            lines.append(f"- **{m}:** {r}")
        lines.append("")

    adj = plan.get("adjusted_params") or {}
    if adj:
        lines.append("### Parameter adjustments")
        lines.append("")
        lines.append("| Model | Overrides |")
        lines.append("|-------|-----------|")
        for model_name, params in sorted(adj.items()):
            lines.append(f"| {model_name} | `{params}` |")
        lines.append("")

    lines.append("### Primary metric")
    lines.append("")
    lines.append(f"- **Metric:** `{plan.get('primary_metric', '—')}`")
    lines.append(f"- **Reasoning:** {plan.get('metric_reasoning', '—')}")
    lines.append("")

    lines.append("### Tuning budget (Optuna)")
    lines.append("")
    lines.append(
        f"- **n_trials:** {plan.get('n_trials', '—')}  "
        f"· **timeout (s):** {plan.get('timeout', '—')}"
    )
    lines.append("")

    warns = plan.get("warnings") or []
    if warns:
        lines.append("### Warnings")
        lines.append("")
        for w in warns:
            lines.append(f"> ⚠️ {w}")
            lines.append("")

    note_list = plan.get("notes") or []
    if note_list:
        lines.append("### Notes")
        lines.append("")
        for n in note_list:
            lines.append(f"- {n}")
        lines.append("")

    lines.append("### Summary")
    lines.append("")
    lines.append(plan.get("plan_summary", ""))

    return "\n".join(lines).strip()

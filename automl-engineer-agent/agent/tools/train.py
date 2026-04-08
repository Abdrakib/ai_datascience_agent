"""
agent/tools/train.py — Multi-model training and selection.

Trains all configured models for the detected task type, evaluates each
on the held-out test set, and returns:
  - all model results ranked by primary metric
  - the best fitted model
  - a comparison table for the report

Upstream preprocessing may apply log1p to skewed numeric columns and
SMOTE/SMOTEENN for class imbalance; this module consumes the resulting arrays.
"""

from __future__ import annotations

import math
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from config import CV_FOLDS, RANDOM_STATE


# ── Public entry point ────────────────────────────────────────────────────────

def train_and_compare(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    feature_names: list[str] | None = None,
    n_classes: int = 2,
    adjusted_params: dict[str, dict[str, Any]] | None = None,
    skip_models: list[str] | None = None,
    skip_reasons: dict[str, str] | None = None,
    primary_metric: str | None = None,
    cv_folds: int = CV_FOLDS,
) -> dict[str, Any]:
    """
    Train all available models and return a ranked comparison.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : ndarray
        Preprocessed arrays; preprocess may apply log1p and SMOTE/SMOTEENN.
    task_type    : "classification" | "regression"
    feature_names: column names (for feature importance)
    n_classes    : number of target classes (classification only)

    Returns
    -------
    {
      "results"     : list of per-model dicts, sorted by primary metric desc,
      "best_model"  : sklearn estimator (fitted),
      "best_name"   : str,
      "best_metrics": dict,
      "metric_name" : primary metric used for ranking,
      "comparison_df": pd.DataFrame for display,
      "training_log" : list of str,
    }
    """
    log: list[str] = []
    skip_reasons = skip_reasons or {}
    models = _get_models(
        task_type,
        n_classes,
        adjusted_params=adjusted_params,
        skip_models=skip_models,
    )
    log.append(f"Training {len(models)} models for {task_type} task.")
    if skip_models:
        for sm in skip_models:
            log.append(
                f"  Skipped {sm}: {skip_reasons.get(sm, 'Excluded by training plan.')}"
            )
    if adjusted_params:
        log.append(
            "  Parameter overrides applied for: "
            + ", ".join(sorted(adjusted_params.keys()))
        )

    primary = primary_metric if primary_metric else _primary_metric(task_type)
    gap_threshold = OVERFIT_GAP_THRESHOLD_CLF if task_type == "classification" else OVERFIT_GAP_THRESHOLD_REG
    cv_folds_used = int(cv_folds) if cv_folds and cv_folds > 0 else 0
    if cv_folds_used == 0:
        log.append("Cross-validation skipped (cv_folds=0) — single train/test split only.")

    results = []
    for name, model in models.items():
        log.append(f"  Training {name}...")
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            elapsed = time.time() - t0
            test_metrics = _evaluate(model, X_test, y_test, task_type, n_classes)
            train_metrics = _evaluate(model, X_train, y_train, task_type, n_classes)

            train_score = _get_primary_score(train_metrics, task_type, primary)
            test_score = _get_primary_score(test_metrics, task_type, primary)
            gap = train_score - test_score
            overfit = gap > gap_threshold

            test_metrics["train_time_s"] = round(elapsed, 3)
            test_metrics["train_score"] = train_score
            test_metrics["test_score"] = test_score
            test_metrics["generalization_gap"] = gap
            test_metrics["overfit"] = overfit

            cv_block = _cross_validate_model(
                model, X_train, y_train, task_type, n_classes, cv_folds_used, log, name
            )

            results.append({
                "name":                 name,
                "model":                model,
                "metrics":              test_metrics,
                "train_score":          float(train_score),
                "generalization_gap":   float(gap),
                "overfit":              bool(overfit),
                **cv_block,
            })
            _log_metrics(log, name, test_metrics, task_type, elapsed)
        except Exception as e:
            log.append(f"  {name} FAILED: {e}")

    if not results:
        raise RuntimeError("All models failed to train. Check your data.")

    # Overfitting warnings (plain English for UI)
    metric_label = _primary_metric_label(task_type)
    overfitting_warnings: list[str] = []
    for r in results:
        m = r["metrics"]
        if m.get("overfit", False):
            ts = r.get("train_score", m.get("train_score", 0))
            tss = m.get("test_score", 0)
            gap = m.get("generalization_gap", 0)
            overfitting_warnings.append(
                f"{r['name']} is overfitting — train {metric_label} {ts:.4f} vs test {metric_label} {tss:.4f}, gap {gap:.4f}"
            )
        if r.get("cv_overfit") and r.get("cv_mean") is not None:
            ctm = r.get("cv_train_mean", 0.0)
            cvm = r.get("cv_mean", 0.0)
            overfitting_warnings.append(
                f"{r['name']} shows consistent overfitting across CV folds — "
                f"CV train mean {ctm:.4f} vs CV test mean {cvm:.4f}"
            )

    # Select best model: exclude severely overfit (gap > 0.25) on single split
    severely_overfit = [
        r for r in results
        if r["metrics"].get("overfit") and r["metrics"].get("generalization_gap", 0) > SEVERE_OVERFIT_GAP
    ]
    candidates = [r for r in results if r not in severely_overfit]

    if candidates:
        candidates.sort(
            key=lambda r: _ranking_score(r, task_type, primary),
            reverse=True,
        )
        best = candidates[0]
    else:
        # All severely overfit — pick least overfit
        results.sort(key=lambda r: r["metrics"].get("generalization_gap", 999))
        best = results[0]
        overfitting_warnings.append(
            f"All models showed severe overfitting. Selected {best['name']} (least overfit)."
        )

    log.append(
        f"\nBest model: {best['name']} "
        f"({primary}={best['metrics'].get(primary, 0):.4f})"
    )
    if overfitting_warnings:
        log.append("Overfitting warnings: " + "; ".join(overfitting_warnings))

    cv_summary = _build_cv_summary(best, cv_folds_used, primary)
    log.append(cv_summary)

    importances = _get_feature_importances(best["model"], feature_names)
    comparison_df = _build_comparison_df(results, task_type)

    return {
        "results":             results,
        "best_model":          best["model"],
        "best_name":           best["name"],
        "best_metrics":        best["metrics"],
        "metric_name":         primary,
        "task_type":           task_type,
        "comparison_df":       comparison_df,
        "feature_importances": importances,
        "training_log":        log,
        "overfitting_warnings": overfitting_warnings,
        "cv_summary":          cv_summary,
        "cv_folds_used":       cv_folds_used,
    }


def training_results_to_markdown(result: dict[str, Any]) -> str:
    lines = ["## Model training results", ""]

    df = result["comparison_df"]
    lines.append(df.to_markdown(index=False))
    task_type = result.get("task_type")
    if not task_type:
        task_type = (
            "classification"
            if result["metric_name"] in ("roc_auc", "f1", "f1_weighted", "accuracy")
            else "regression"
        )
    metric_name = result.get("metric_name")
    primary_val = _get_primary_score(result["best_metrics"], task_type, metric_name)
    label = _metric_display_name(metric_name) if metric_name else _primary_metric_label(task_type)
    lines += [
        "",
        f"**Best model:** {result['best_name']}",
        f"**Primary metric ({label}):** "
        f"{primary_val:.4f}",
    ]
    cv_sum = result.get("cv_summary")
    if cv_sum:
        lines += ["", f"**Cross-validation:** {cv_sum}"]

    if result["feature_importances"]:
        lines += ["", "## Top 10 feature importances"]
        for feat, imp in list(result["feature_importances"].items())[:10]:
            bar = "█" * int(imp * 40)
            lines.append(f"- `{feat}`: {imp:.4f}  {bar}")

    return "\n".join(lines)


# ── Model definitions ─────────────────────────────────────────────────────────

def _get_models(
    task_type: str,
    n_classes: int,
    adjusted_params: dict[str, dict[str, Any]] | None = None,
    skip_models: list[str] | None = None,
) -> dict[str, Any]:
    """Return a dict of model_name → unfitted estimator."""
    skip_set = set(skip_models or [])
    overrides = adjusted_params or {}
    models: dict[str, Any] = {}

    def merge(name: str, base: dict[str, Any]) -> dict[str, Any]:
        m = dict(base)
        m.update(overrides.get(name, {}))
        return m

    if task_type == "classification":
        lr_kw = merge(
            "Logistic Regression",
            {
                "max_iter": 1000,
                "random_state": RANDOM_STATE,
                "class_weight": "balanced",
            },
        )
        if "Logistic Regression" not in skip_set:
            models["Logistic Regression"] = LogisticRegression(**lr_kw)

        rf_kw = merge(
            "Random Forest",
            {
                "n_estimators": 200,
                "random_state": RANDOM_STATE,
                "class_weight": "balanced",
                "n_jobs": -1,
            },
        )
        if "Random Forest" not in skip_set:
            models["Random Forest"] = RandomForestClassifier(**rf_kw)

        if HAS_XGB and "XGBoost" not in skip_set:
            xgb_kw = merge(
                "XGBoost",
                {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "random_state": RANDOM_STATE,
                    "eval_metric": "logloss",
                    "verbosity": 0,
                    "use_label_encoder": False,
                },
            )
            models["XGBoost"] = XGBClassifier(**xgb_kw)

        if HAS_LGB and "LightGBM" not in skip_set:
            lgb_kw = merge(
                "LightGBM",
                {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "random_state": RANDOM_STATE,
                    "class_weight": "balanced",
                    "verbose": -1,
                },
            )
            models["LightGBM"] = LGBMClassifier(**lgb_kw)

    else:  # regression
        if "Linear Regression" not in skip_set:
            lr_reg_kw = merge("Linear Regression", {})
            models["Linear Regression"] = LinearRegression(**lr_reg_kw)

        if "Random Forest" not in skip_set:
            rf_reg_kw = merge(
                "Random Forest",
                {
                    "n_estimators": 200,
                    "random_state": RANDOM_STATE,
                    "n_jobs": -1,
                },
            )
            models["Random Forest"] = RandomForestRegressor(**rf_reg_kw)

        if HAS_XGB and "XGBoost" not in skip_set:
            xgb_reg_kw = merge(
                "XGBoost",
                {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "random_state": RANDOM_STATE,
                    "verbosity": 0,
                },
            )
            models["XGBoost"] = XGBRegressor(**xgb_reg_kw)

        if HAS_LGB and "LightGBM" not in skip_set:
            lgb_reg_kw = merge(
                "LightGBM",
                {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "random_state": RANDOM_STATE,
                    "verbose": -1,
                },
            )
            models["LightGBM"] = LGBMRegressor(**lgb_reg_kw)

    return models


# ── Evaluation ────────────────────────────────────────────────────────────────

OVERFIT_GAP_THRESHOLD_CLF = 0.15
OVERFIT_GAP_THRESHOLD_REG = 0.20
SEVERE_OVERFIT_GAP = 0.25


def _cv_scoring_str(task_type: str, n_classes: int) -> str:
    if task_type == "classification":
        return "roc_auc" if n_classes == 2 else "f1_weighted"
    return "r2"


def _cross_validate_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    n_classes: int,
    cv_folds: int,
    log: list[str],
    name: str,
) -> dict[str, Any]:
    """Run cross_validate on a clone of the fitted model; never raises."""
    empty = {
        "cv_scores":         None,
        "cv_mean":           None,
        "cv_std":            None,
        "cv_train_scores":   None,
        "cv_train_mean":     None,
        "cv_overfit":        False,
    }
    if cv_folds < 2:
        return empty
    n = len(X_train)
    if n < cv_folds:
        log.append(f"  {name}: CV skipped — not enough samples for {cv_folds} folds.")
        return empty
    try:
        scoring = _cv_scoring_str(task_type, n_classes)
        if task_type == "classification":
            cv_split = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE
            )
        else:
            cv_split = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        est = clone(model)
        cv_res = cross_validate(
            est,
            X_train,
            y_train,
            cv=cv_split,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )
        test_scores = np.asarray(cv_res["test_score"], dtype=float)
        train_scores = np.asarray(cv_res["train_score"], dtype=float)
        cv_mean = float(np.mean(test_scores))
        cv_std = float(np.std(test_scores))
        train_mean = float(np.mean(train_scores))
        thr = OVERFIT_GAP_THRESHOLD_CLF if task_type == "classification" else OVERFIT_GAP_THRESHOLD_REG
        cv_overfit = (train_mean - cv_mean) > thr
        return {
            "cv_scores":       [float(x) for x in test_scores],
            "cv_mean":         cv_mean,
            "cv_std":          cv_std,
            "cv_train_scores": [float(x) for x in train_scores],
            "cv_train_mean":   train_mean,
            "cv_overfit":      bool(cv_overfit),
        }
    except Exception as e:
        log.append(f"  {name}: cross-validation failed ({e}) — using single split only.")
        return empty


def _ranking_score(
    r: dict[str, Any],
    task_type: str,
    primary: str | None,
) -> float:
    """Prefer CV mean when available; otherwise single split test score."""
    cm = r.get("cv_mean")
    if cm is not None:
        try:
            v = float(cm)
            if not (isinstance(v, float) and math.isnan(v)):
                return v
        except (TypeError, ValueError):
            pass
    return _get_primary_score(r["metrics"], task_type, primary)


def _build_cv_summary(best: dict[str, Any], cv_folds_used: int, primary: str) -> str:
    if cv_folds_used < 2 or best.get("cv_mean") is None:
        return (
            "Cross-validation not available — best model selected using single "
            f"held-out test score ({primary}={best['metrics'].get(primary, 0):.4f})."
        )
    cm = best.get("cv_mean", 0.0)
    cs = best.get("cv_std", 0.0)
    ts = best["metrics"].get("test_score", 0.0)
    return (
        f"{cv_folds_used}-fold cross-validation results: best model {best['name']} achieved "
        f"CV mean {cm:.4f} ± {cs:.4f} vs single test score {ts:.4f}"
    )


def _evaluate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    n_classes: int,
) -> dict[str, float]:
    """Evaluate model on given X, y. Returns metrics dict."""
    y_pred = model.predict(X)

    if task_type == "classification":
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1":       float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        }
        if n_classes == 2 and hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y, y_prob))
            except Exception:
                pass
    else:
        mse  = mean_squared_error(y, y_pred)
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(y, y_pred))
        r2   = float(r2_score(y, y_pred))
        metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    return metrics


def _primary_metric(task_type: str) -> str:
    return "roc_auc" if task_type == "classification" else "r2"


def _get_primary_score(
    metrics: dict[str, Any],
    task_type: str,
    primary_metric: str | None = None,
) -> float:
    """Score for ranking: follows primary_metric when set; else legacy defaults."""
    pm = primary_metric or _primary_metric(task_type)
    if task_type == "classification":
        if pm in ("f1_weighted", "f1"):
            return float(metrics.get("f1", 0.0))
        if pm == "accuracy":
            return float(metrics.get("accuracy", 0.0))
        if pm == "roc_auc":
            v = metrics.get("roc_auc")
            if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
                return float(v)
            return float(metrics.get("accuracy", 0.0))
        v = metrics.get("roc_auc")
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            return float(v)
        return float(metrics.get("accuracy", 0.0))
    return float(metrics.get("r2", 0.0))


def _primary_metric_label(task_type: str) -> str:
    return "ROC-AUC" if task_type == "classification" else "R²"


def _metric_display_name(metric_name: str | None) -> str:
    if not metric_name:
        return ""
    labels = {
        "roc_auc": "ROC-AUC",
        "r2": "R²",
        "f1_weighted": "F1 (weighted)",
        "f1": "F1 (weighted)",
        "accuracy": "Accuracy",
    }
    return labels.get(metric_name, metric_name)


def _log_metrics(
    log: list[str],
    name: str,
    metrics: dict[str, float],
    task_type: str,
    elapsed: float,
) -> None:
    if task_type == "classification":
        msg = (
            f"    {name}: acc={metrics['accuracy']:.3f}, "
            f"f1={metrics['f1']:.3f}"
        )
        if "roc_auc" in metrics:
            msg += f", auc={metrics['roc_auc']:.3f}"
    else:
        msg = (
            f"    {name}: r2={metrics['r2']:.3f}, "
            f"rmse={metrics['rmse']:.2f}, mae={metrics['mae']:.2f}"
        )
    msg += f"  [{elapsed:.2f}s]"
    log.append(msg)


# ── Comparison table ──────────────────────────────────────────────────────────

def _build_comparison_df(results: list[dict], task_type: str) -> pd.DataFrame:
    """Model, split scores, CV columns, overfit flags, train time."""
    rows = []
    for r in results:
        m = r["metrics"]
        train_s = r.get("train_score", m.get("train_score"))
        test_s = m.get("test_score")
        gap = m.get("generalization_gap", 0.0)
        cv_mean = r.get("cv_mean")
        cv_std = r.get("cv_std")
        cv_tr = r.get("cv_train_mean")
        has_cv = cv_mean is not None
        rows.append({
            "Model":           r["name"],
            "Train Score":     round(float(train_s), 4) if train_s is not None else None,
            "Test Score":      round(float(test_s), 4) if test_s is not None else None,
            "Gap":             round(float(gap), 4),
            "CV Mean":         round(float(cv_mean), 4) if has_cv else None,
            "CV Std":          round(float(cv_std), 4) if has_cv and cv_std is not None else None,
            "CV Train Mean":   round(float(cv_tr), 4) if has_cv and cv_tr is not None else None,
            "CV Overfit":      ("Yes" if r.get("cv_overfit") else "No") if has_cv else "—",
            "Overfit":         "Yes" if m.get("overfit") else "No",
            "Train Time(s)":   round(float(m.get("train_time_s", 0)), 2),
        })
    return pd.DataFrame(rows)


# ── Feature importances ───────────────────────────────────────────────────────

def _get_feature_importances(
    model: Any,
    feature_names: list[str] | None,
) -> dict[str, float]:
    """Extract feature importances if the model supports it."""
    importances: dict[str, float] = {}

    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw = np.abs(model.coef_).flatten()
    else:
        return importances

    names = feature_names if feature_names and len(feature_names) == len(raw) \
        else [f"feature_{i}" for i in range(len(raw))]

    # Normalize to sum to 1
    total = raw.sum()
    if total > 0:
        raw = raw / total

    paired = sorted(zip(names, raw), key=lambda x: x[1], reverse=True)
    return {name: float(imp) for name, imp in paired}


def tuning_result_to_markdown(tune_result: dict[str, Any]) -> str:
    """Format Optuna tuning output for agent / UI markdown."""
    if not tune_result.get("success", False):
        err = tune_result.get("error", "Unknown error")
        return f"## Hyperparameter tuning\n\n**Status:** failed\n\n{err}\n"

    lines = [
        "## Hyperparameter tuning (Optuna)",
        "",
        f"- **Score before tuning:** {tune_result.get('baseline_score', 0):.4f}",
        f"- **Score after tuning (test):** {tune_result.get('best_score', 0):.4f}",
        f"- **Improvement:** {tune_result.get('improvement', 0):+.4f}",
        f"- **Trials run:** {tune_result.get('n_trials_run', 0)}",
        f"- **Tuning time:** {tune_result.get('tuning_time_s', 0):.1f}s",
        f"- **Generalization gap (train − test):** {tune_result.get('generalization_gap', 0):.4f}",
        f"- **Overfitting flag:** {'yes' if tune_result.get('overfit') else 'no'}",
        "",
    ]
    bp = tune_result.get("best_params") or {}
    if bp:
        lines.append("### Best hyperparameters")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for k, v in sorted(bp.items()):
            lines.append(f"| `{k}` | {v} |")
        lines.append("")

    for line in tune_result.get("tuning_log", []):
        lines.append(f"- {line}")

    if tune_result.get("overfit"):
        lines.append("")
        lines.append(
            "**Warning:** Tuned model still shows elevated train–test gap; "
            "consider more data or stronger regularization."
        )

    return "\n".join(lines)

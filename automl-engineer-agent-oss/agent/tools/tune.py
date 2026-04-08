"""
Optuna hyperparameter tuning for the best model selected by train_and_compare.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

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

from config import RANDOM_STATE

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    from optuna.trial import FixedTrial

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    FixedTrial = None  # type: ignore[misc, assignment]


def _n_classes(y: np.ndarray) -> int:
    return int(len(np.unique(y)))


def _score_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    n_cls: int,
) -> dict[str, float]:
    y_pred = model.predict(X)
    if task_type == "classification":
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)),
        }
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
                if n_cls == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
                else:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y, proba, multi_class="ovr", average="weighted")
                    )
            except Exception:
                pass
    else:
        mse = mean_squared_error(y, y_pred)
        metrics = {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y, y_pred)),
            "r2": float(r2_score(y, y_pred)),
        }
    return metrics


def _primary_score(metrics: dict[str, float], task_type: str) -> float:
    if task_type == "classification":
        v = metrics.get("roc_auc")
        if isinstance(v, (int, float)):
            return float(v)
        return float(metrics.get("accuracy", 0.0))
    return float(metrics.get("r2", 0.0))


def _build_estimator_from_trial(
    trial: Any,
    model_name: str,
    task_type: str,
    n_cls: int,
) -> Any:
    """Instantiate sklearn / boosting estimator from Optuna trial suggestions."""
    if task_type == "classification":
        if model_name == "Logistic Regression":
            C = trial.suggest_float("C", 0.01, 10.0, log=True)
            max_iter = trial.suggest_int("max_iter", 200, 1000)
            solver = trial.suggest_categorical("solver", ["lbfgs", "saga"])
            multi_class = "multinomial" if n_cls > 2 else "auto"
            return LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver=solver,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                multi_class=multi_class,
            )

        if model_name == "Random Forest":
            return RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )

        if model_name == "XGBoost":
            if not HAS_XGB:
                raise RuntimeError("xgboost not installed")
            xgb_obj = "binary:logistic" if n_cls == 2 else "multi:softprob"
            xgb_em = "logloss" if n_cls == 2 else "mlogloss"
            return XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                random_state=RANDOM_STATE,
                verbosity=0,
                objective=xgb_obj,
                eval_metric=xgb_em,
            )

        if model_name == "LightGBM":
            if not HAS_LGB:
                raise RuntimeError("lightgbm not installed")
            return LGBMClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 20, 100),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=RANDOM_STATE,
                class_weight="balanced",
                verbose=-1,
            )

    else:  # regression
        if model_name == "Linear Regression":
            alpha = trial.suggest_float("alpha", 0.01, 10.0, log=True)
            return Ridge(alpha=alpha, random_state=RANDOM_STATE)

        if model_name == "Random Forest":
            return RandomForestRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

        if model_name == "XGBoost":
            if not HAS_XGB:
                raise RuntimeError("xgboost not installed")
            return XGBRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
                random_state=RANDOM_STATE,
                verbosity=0,
            )

        if model_name == "LightGBM":
            if not HAS_LGB:
                raise RuntimeError("lightgbm not installed")
            return LGBMRegressor(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 20, 100),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                random_state=RANDOM_STATE,
                verbose=-1,
            )

    raise ValueError(f"unsupported model: {model_name} / {task_type}")


def _supported_model(model_name: str, task_type: str) -> bool:
    if task_type == "classification":
        return model_name in ("Logistic Regression", "Random Forest", "XGBoost", "LightGBM")
    return model_name in ("Linear Regression", "Random Forest", "XGBoost", "LightGBM")


def _deps_ok(model_name: str) -> bool:
    if model_name == "XGBoost" and not HAS_XGB:
        return False
    if model_name == "LightGBM" and not HAS_LGB:
        return False
    return True


def tune_best_model(
    model_name: str,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    baseline_score: float,
    n_trials: int = 50,
    timeout: int = 120,
) -> dict[str, Any]:
    """
    Bayesian hyperparameter search with Optuna for the given best model name.
    """
    log: list[str] = []
    t0_all = time.time()

    if not HAS_OPTUNA:
        return {
            "success": False,
            "error": "optuna not installed — run: pip install optuna",
        }

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if not _supported_model(model_name, task_type):
        return {
            "success": False,
            "error": f"Model not supported for tuning: {model_name}",
        }

    if not _deps_ok(model_name):
        return {
            "success": False,
            "error": f"Required library missing for {model_name}",
        }

    if len(X_train) < 1000:
        n_trials = 20
        log.append(f"Dataset has {len(X_train)} training rows (<1000); using n_trials={n_trials}.")

    n_cls = _n_classes(y_train) if task_type == "classification" else 2

    direction = "maximize"
    sampler = TPESampler(seed=42)
    pruner = MedianPruner()
    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        try:
            est = _build_estimator_from_trial(trial, model_name, task_type, n_cls)
            est.fit(X_train, y_train)
            te = _score_model(est, X_test, y_test, task_type, n_cls)
            return _primary_score(te, task_type)
        except Exception:
            return float("-inf")

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    n_trials_run = len(study.trials)
    if n_trials_run == 0:
        return {
            "success": False,
            "error": "No Optuna trials executed.",
        }
    try:
        _ = study.best_params
    except ValueError:
        return {
            "success": False,
            "error": "Optuna did not complete any successful trials.",
        }
    if study.best_value in (float("-inf"), None) or (
        isinstance(study.best_value, float) and study.best_value <= float("-1e100")
    ):
        return {
            "success": False,
            "error": "All Optuna trials failed during training or scoring.",
        }

    best_params = study.best_params.copy()

    # Refit best on full training data using fixed best params
    fixed = FixedTrial(study.best_params)
    tuned_model = _build_estimator_from_trial(fixed, model_name, task_type, n_cls)
    tuned_model.fit(X_train, y_train)

    tuning_time_s = time.time() - t0_all

    train_m = _score_model(tuned_model, X_train, y_train, task_type, n_cls)
    test_m = _score_model(tuned_model, X_test, y_test, task_type, n_cls)
    train_sc = _primary_score(train_m, task_type)
    test_sc = _primary_score(test_m, task_type)
    gap = train_sc - test_sc
    overfit_clf = gap > 0.15
    overfit_reg = gap > 0.20
    overfit = overfit_clf if task_type == "classification" else overfit_reg

    improvement = test_sc - baseline_score
    log.append(f"Optuna completed {n_trials_run} trials in {tuning_time_s:.1f}s.")
    log.append(f"Best test score: {test_sc:.4f} (baseline {baseline_score:.4f}, improvement {improvement:+.4f}).")

    return {
        "success": True,
        "tuned_model": tuned_model,
        "best_params": best_params,
        "best_score": float(test_sc),
        "baseline_score": float(baseline_score),
        "improvement": float(improvement),
        "n_trials_run": n_trials_run,
        "tuning_time_s": float(tuning_time_s),
        "generalization_gap": float(gap),
        "overfit": bool(overfit),
        "tuning_log": log,
        "test_metrics_full": test_m,
        "train_metrics_full": train_m,
    }

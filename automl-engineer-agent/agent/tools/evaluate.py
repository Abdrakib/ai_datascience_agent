"""
agent/tools/evaluate.py — Model evaluation, plots, and SHAP explainability.

Produces:
  - Full metric report (classification or regression)
  - Confusion matrix plot (classification)
  - Actual vs Predicted plot (regression)
  - Feature importance bar chart
  - SHAP bar summary (mean |SHAP|), beeswarm summary, dependence plots (top 3 features), waterfall (local)
  - Residuals plot (regression)
  - ROC curve (binary classification)
  - All plots saved to outputs/ as PNG files
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

from config import OUTPUT_DIR

# ── Colour palette (consistent across all plots) ──────────────────────────────
PALETTE = {
    "primary":   "#534AB7",   # purple
    "secondary": "#1D9E75",   # teal
    "accent":    "#D85A30",   # coral
    "neutral":   "#888780",   # gray
    "light":     "#F1EFE8",   # background
}


# ── Public entry point ────────────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    task_type: str,
    feature_names: list[str] | None = None,
    label_encoder: Any = None,
    run_id: str = "run",
    n_classes: int = 2,
) -> dict[str, Any]:
    """
    Full evaluation pipeline.  Returns a dict with metrics, plot paths,
    SHAP values, and a markdown summary.

    Parameters
    ----------
    model         : fitted sklearn estimator
    X_test/y_test : held-out test data
    X_train/y_train: training data (needed for SHAP background)
    task_type     : "classification" | "regression"
    feature_names : column names after encoding
    label_encoder : fitted LabelEncoder (classification only)
    run_id        : short string used in saved filenames
    n_classes     : number of target classes
    """
    y_pred = model.predict(X_test)
    plot_paths: dict[str, str] = {}
    eval_log: list[str] = []

    # ── Metrics ───────────────────────────────────────────────────────────────
    if task_type == "classification":
        metrics = _clf_metrics(y_test, y_pred, model, X_test, n_classes)
    else:
        metrics = _reg_metrics(y_test, y_pred)

    eval_log.append("Computed evaluation metrics.")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if task_type == "classification":
        cm_path = _plot_confusion_matrix(
            y_test, y_pred, label_encoder, run_id
        )
        plot_paths["confusion_matrix"] = cm_path
        eval_log.append(f"Saved confusion matrix → {cm_path}")

        if n_classes == 2 and hasattr(model, "predict_proba"):
            roc_path = _plot_roc_curve(model, X_test, y_test, run_id)
            plot_paths["roc_curve"] = roc_path
            eval_log.append(f"Saved ROC curve → {roc_path}")
    else:
        avp_path = _plot_actual_vs_predicted(y_test, y_pred, run_id)
        plot_paths["actual_vs_predicted"] = avp_path
        eval_log.append(f"Saved actual vs predicted → {avp_path}")

        res_path = _plot_residuals(y_test, y_pred, run_id)
        plot_paths["residuals"] = res_path
        eval_log.append(f"Saved residuals plot → {res_path}")

    # Feature importance chart
    fi_path = _plot_feature_importance(model, feature_names, run_id)
    if fi_path:
        plot_paths["feature_importance"] = fi_path
        eval_log.append(f"Saved feature importance chart → {fi_path}")

    # ── SHAP ──────────────────────────────────────────────────────────────────
    shap_values = None
    shap_explanation_text = ""
    if HAS_SHAP:
        try:
            shap_values, shap_paths_extra, shap_explanation_text = _compute_shap(
                model, X_train, X_test, feature_names, task_type, run_id
            )
            for k, p in shap_paths_extra.items():
                plot_paths[k] = p
                eval_log.append(f"Saved SHAP plot → {p} ({k})")
        except Exception as e:
            eval_log.append(f"SHAP skipped: {e}")
    else:
        eval_log.append("SHAP not installed — skipping (pip install shap).")

    return {
        "metrics":      metrics,
        "plot_paths":   plot_paths,
        "shap_values":  shap_values,
        "eval_log":     eval_log,
        "task_type":    task_type,
        "y_pred":       y_pred,
        "has_shap":     HAS_SHAP and bool(
            plot_paths.get("shap_summary")
            or plot_paths.get("shap_bar")
            or any(k.startswith("shap_dependence_") for k in plot_paths)
        ),
        "shap_explanation_text": shap_explanation_text,
    }


def evaluation_to_markdown(
    result: dict[str, Any],
    model_name: str = "",
    train_result: dict[str, Any] | None = None,
) -> str:
    lines = [f"## Evaluation results  {('— ' + model_name) if model_name else ''}"]
    m = result["metrics"]

    if result["task_type"] == "classification":
        lines += [
            f"- **Accuracy:** {m.get('accuracy', 0):.4f}",
            f"- **F1 (weighted):** {m.get('f1', 0):.4f}",
        ]
        if "roc_auc" in m:
            lines.append(f"- **ROC-AUC:** {m['roc_auc']:.4f}")
        if "classification_report" in m:
            lines += ["", "```", m["classification_report"], "```"]
    else:
        lines += [
            f"- **R²:** {m.get('r2', 0):.4f}",
            f"- **RMSE:** {m.get('rmse', 0):.4f}",
            f"- **MAE:** {m.get('mae', 0):.4f}",
            f"- **MAPE:** {m.get('mape', 0):.2f}%",
        ]

    if train_result and model_name:
        base_name = str(model_name).replace(" (tuned)", "").strip()
        cv_row = None
        for r in train_result.get("results") or []:
            if r.get("name") == base_name or r.get("name") == model_name:
                cv_row = r
                break
        scores = (cv_row or {}).get("cv_scores")
        if scores:
            lines += ["", "## Cross-validation results", "", "| Fold | Score |", "|------|-------|"]
            for i, sc in enumerate(scores, start=1):
                lines.append(f"| Fold {i} | {float(sc):.4f} |")
            cv_mean = (cv_row or {}).get("cv_mean")
            cv_std = (cv_row or {}).get("cv_std")
            if cv_mean is not None and cv_std is not None:
                lines.append(f"| Mean | {float(cv_mean):.4f} ± {float(cv_std):.4f} |")
            lines.append("")
            std = float(cv_std) if cv_std is not None else 0.0
            if std < 0.02:
                rel = "Low variance across folds — metrics are reliable"
            elif std <= 0.05:
                rel = "Moderate variance — metrics are reasonably stable"
            else:
                rel = "High variance across folds — metrics may not be stable. Consider more data or simpler model."
            lines.append(f"**Reliability:** {rel}")

    if result["plot_paths"]:
        lines += ["", "**Plots saved:**"]
        for name, path in result["plot_paths"].items():
            lines.append(f"- `{name}`: `{path}`")

    expl = result.get("shap_explanation_text") or ""
    if expl.strip():
        lines += ["", "**SHAP — what drove one prediction (example row)**", expl]

    return "\n".join(lines)


# ── Metric helpers ────────────────────────────────────────────────────────────

def _clf_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    model: Any,
    X_test: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    m: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1":       float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }
    if n_classes == 2 and hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
            m["roc_auc"] = float(roc_auc_score(y_test, y_prob))
            m["y_prob"]  = y_prob
        except Exception:
            pass
    return m


def _reg_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse  = mean_squared_error(y_test, y_pred)
    mask = y_test != 0
    mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100) if mask.any() else 0.0
    return {
        "r2":   float(r2_score(y_test, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae":  float(mean_absolute_error(y_test, y_pred)),
        "mape": mape,
    }


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(PALETTE["light"])
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12, color="#2C2C2A")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color="#5F5E5A")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color="#5F5E5A")
    ax.tick_params(colors="#5F5E5A", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#D3D1C7")


def _save_fig(fig: plt.Figure, name: str) -> str:
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def _plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Any,
    run_id: str,
) -> str:
    cm = confusion_matrix(y_test, y_pred)
    labels = label_encoder.classes_ if label_encoder else [str(i) for i in range(len(cm))]

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")
    im = ax.imshow(cm, interpolation="nearest", cmap="Purples")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="white" if cm[i, j] > thresh else "#2C2C2A",
            )

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    _style_ax(ax, "Confusion matrix", "Predicted label", "True label")
    fig.tight_layout()
    return _save_fig(fig, f"{run_id}_confusion_matrix")


def _plot_roc_curve(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_id: str,
) -> str:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")
    ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"], lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["primary"])
    _style_ax(ax, "ROC curve", "False Positive Rate", "True Positive Rate")
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    return _save_fig(fig, f"{run_id}_roc_curve")


def _plot_actual_vs_predicted(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    run_id: str,
) -> str:
    y_test = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.patch.set_facecolor("white")
    ax.scatter(
        y_test,
        y_test,
        alpha=0.6,
        s=45,
        color="#1f77b4",
        edgecolors="none",
        label="Actual",
        zorder=2,
    )
    ax.scatter(
        y_test,
        y_pred,
        alpha=0.6,
        s=45,
        color="#2ca02c",
        edgecolors="none",
        label="Predicted",
        zorder=3,
    )
    mn = float(min(y_test.min(), y_pred.min()))
    mx = float(max(y_test.max(), y_pred.max()))
    pad = (mx - mn) * 0.04 if mx > mn else 1.0
    ax.plot(
        [mn - pad, mx + pad],
        [mn - pad, mx + pad],
        color="#d62728",
        lw=2.0,
        linestyle="--",
        label="Perfect Prediction",
        zorder=1,
    )
    ax.set_xlim(mn - pad, mx + pad)
    ax.set_ylim(mn - pad, mx + pad)
    ax.set_aspect("equal", adjustable="box")
    _style_ax(ax, "Actual vs predicted", "Actual (target)", "Predicted (model)")
    r2 = r2_score(y_test, y_pred)
    ax.text(
        0.05,
        0.93,
        f"R² = {r2:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        color="#2C2C2A",
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=10, frameon=True, framealpha=0.95, edgecolor="#D3D1C7")
    fig.tight_layout()
    return _save_fig(fig, f"{run_id}_actual_vs_predicted")


def _plot_residuals(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    run_id: str,
) -> str:
    y_test = np.asarray(y_test).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_test - y_pred
    pos = residuals >= 0
    neg = ~pos
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.patch.set_facecolor("white")
    if np.any(pos):
        ax.scatter(
            y_pred[pos],
            residuals[pos],
            alpha=0.6,
            s=45,
            color="#2ca02c",
            edgecolors="none",
            label="Underestimation",
            zorder=2,
        )
    if np.any(neg):
        ax.scatter(
            y_pred[neg],
            residuals[neg],
            alpha=0.6,
            s=45,
            color="#d62728",
            edgecolors="none",
            label="Overestimation",
            zorder=2,
        )
    ax.axhline(0, color="#333333", lw=1.2, linestyle="-", zorder=1)
    _style_ax(
        ax,
        "Residuals",
        "Predicted value",
        "Residual (actual − predicted)",
    )
    ax.legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.95, edgecolor="#D3D1C7")
    fig.tight_layout()
    return _save_fig(fig, f"{run_id}_residuals")


def _plot_feature_importance(
    model: Any,
    feature_names: list[str] | None,
    run_id: str,
    top_n: int = 15,
) -> str | None:
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw = np.abs(model.coef_).flatten()
    else:
        return None

    names = (
        feature_names if feature_names and len(feature_names) == len(raw)
        else [f"feature_{i}" for i in range(len(raw))]
    )
    paired = sorted(zip(names, raw), key=lambda x: x[1], reverse=True)[:top_n]
    names_top, imps_top = zip(*paired)

    fig, ax = plt.subplots(figsize=(7, max(4, len(names_top) * 0.38)))
    fig.patch.set_facecolor("white")
    colors = [PALETTE["primary"] if i == 0 else PALETTE["secondary"] for i in range(len(names_top))]
    bars = ax.barh(list(reversed(names_top)), list(reversed(imps_top)),
                   color=list(reversed(colors)), edgecolor="none", height=0.65)
    _style_ax(ax, f"Feature importance (top {len(names_top)})", "Importance", "")
    ax.set_facecolor(PALETTE["light"])
    for bar, val in zip(bars, list(reversed(imps_top))):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color="#5F5E5A")
    fig.tight_layout()
    return _save_fig(fig, f"{run_id}_feature_importance")


def _normalize_shap_matrix(shap_vals: Any, task_type: str) -> np.ndarray:
    """Return (n_samples, n_features) SHAP matrix for plotting and explanations."""
    if isinstance(shap_vals, np.ndarray):
        if shap_vals.ndim == 3 and task_type == "classification":
            # (n_samples, n_features, n_classes) — use positive class (index 1)
            return np.asarray(shap_vals[:, :, 1])
        if shap_vals.ndim == 3:
            return np.asarray(shap_vals[:, :, 0])
        return np.asarray(shap_vals)
    if isinstance(shap_vals, list):
        if len(shap_vals) == 2:
            return np.asarray(shap_vals[1])
        if len(shap_vals) > 2:
            return np.mean(
                np.stack([np.asarray(v) for v in shap_vals], axis=0), axis=0
            )
        return np.asarray(shap_vals[0])
    return np.asarray(shap_vals)


def _expected_value_for_waterfall(explainer: Any, shap_vals: Any, task_type: str) -> float:
    ev = explainer.expected_value
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        ev_a = np.asarray(ev).ravel()
        if ev_a.size > 1 and task_type == "classification":
            return float(ev_a[1])
        return float(ev_a[0])
    if isinstance(shap_vals, list) and len(shap_vals) > 2:
        ev_a = np.asarray(ev).ravel()
        return float(np.mean(ev_a))
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        ev_a = np.asarray(ev).ravel()
        if ev_a.size > 1:
            return float(ev_a[1])
        return float(ev_a[0])
    ev_a = np.asarray(ev).ravel()
    return float(ev_a[0])


def _sanitize_shap_feature_filename(name: str, idx: int) -> str:
    s = re.sub(r"[^\w\-.]+", "_", str(name)).strip("_")
    return s if s else f"feature_{idx}"


def _apply_shap_brand_colors() -> None:
    """Bias matplotlib toward the project palette for SHAP figures."""
    import matplotlib as mpl

    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"]]
    )


def _compute_shap(
    model: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str] | None,
    task_type: str,
    run_id: str,
    max_background: int = 100,
    max_explain: int = 200,
) -> tuple[Any, dict[str, str], str]:
    """Compute SHAP values; save bar summary, beeswarm, dependence (top 3), waterfall, explanation."""
    background = shap.sample(X_train, min(max_background, len(X_train)))
    explain_data = np.asarray(X_test[:max_explain])
    plot_paths_out: dict[str, str] = {}
    explanation_text = ""

    model_type = type(model).__name__.lower()
    if any(t in model_type for t in ["forest", "xgb", "lgbm", "boost", "tree"]):
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(explain_data)
    else:
        explainer = shap.KernelExplainer(model.predict, background)
        shap_vals = explainer.shap_values(explain_data, nsamples=100)

    shap_arr = _normalize_shap_matrix(shap_vals, task_type)
    if shap_arr.size == 0 or not np.all(np.isfinite(shap_arr)):
        return shap_vals, plot_paths_out, explanation_text

    n_feat = shap_arr.shape[1]
    names = (
        feature_names if feature_names and len(feature_names) == n_feat
        else [f"feature_{i}" for i in range(n_feat)]
    )
    X_df = pd.DataFrame(explain_data, columns=names)
    mean_abs = np.abs(shap_arr).mean(axis=0)

    _apply_shap_brand_colors()

    # --- Mean |SHAP| bar chart (shap.summary_plot bar) ---
    plt.figure(figsize=(10, 6.8))
    try:
        shap.summary_plot(
            shap_arr,
            X_df,
            plot_type="bar",
            show=False,
            max_display=min(20, n_feat),
            color=PALETTE["primary"],
        )
    except TypeError:
        shap.summary_plot(
            shap_arr,
            X_df,
            plot_type="bar",
            show=False,
            max_display=min(20, n_feat),
        )
    fig_bar = plt.gcf()
    fig_bar.suptitle(
        "Mean |SHAP| (global impact)",
        fontsize=14,
        fontweight="bold",
        color="#2C2C2A",
        y=1.02,
    )
    fig_bar.subplots_adjust(top=0.88)
    plot_paths_out["shap_bar"] = _save_fig(fig_bar, f"{run_id}_shap_bar")

    # --- Beeswarm summary ---
    plt.figure(figsize=(10, 6.8))
    shap.summary_plot(
        shap_arr,
        X_df,
        show=False,
        max_display=min(20, n_feat),
    )
    fig_bee = plt.gcf()
    fig_bee.suptitle(
        "Feature Impact Distribution",
        fontsize=15,
        fontweight="bold",
        color="#2C2C2A",
        y=1.02,
    )
    fig_bee.text(
        0.5,
        0.02,
        "Each dot represents a data point. Red = high feature value, Blue = low. "
        "Position shows impact on prediction.",
        ha="center",
        fontsize=9.5,
        color="#5F5E5A",
        wrap=True,
        transform=fig_bee.transFigure,
    )
    fig_bee.subplots_adjust(bottom=0.14, top=0.90)
    plot_paths_out["shap_summary"] = _save_fig(fig_bee, f"{run_id}_shap_summary")

    # --- Dependence plots: top 3 features by mean |SHAP| ---
    n_test_rows = int(len(np.asarray(X_test)))
    if n_test_rows >= 20 and n_feat > 0:
        order = np.argsort(-mean_abs)
        top_idx = [int(order[i]) for i in range(min(3, n_feat))]
        used_keys: set[str] = set()
        for feat_i in top_idx:
            safe = _sanitize_shap_feature_filename(names[feat_i], feat_i)
            key = f"shap_dependence_{safe}"
            if key in used_keys:
                key = f"{key}_{feat_i}"
            used_keys.add(key)
            fname = f"{run_id}_shap_dependence_{safe}"
            try:
                plt.figure(figsize=(8.5, 5.5))
                try:
                    shap.dependence_plot(
                        feat_i,
                        shap_arr,
                        X_df,
                        interaction_index="auto",
                        show=False,
                    )
                except TypeError:
                    shap.dependence_plot(
                        feat_i,
                        shap_arr,
                        X_df,
                        show=False,
                    )
                fig_dep = plt.gcf()
                plot_paths_out[key] = _save_fig(fig_dep, fname)
            except Exception:
                try:
                    if plt.get_fignums():
                        plt.close(plt.gcf())
                except Exception:
                    pass

    # --- Waterfall (local explanation) ---
    try:
        ev_scalar = _expected_value_for_waterfall(explainer, shap_vals, task_type)
        exp = shap.Explanation(
            values=shap_arr[0],
            base_values=ev_scalar,
            data=explain_data[0],
            feature_names=np.array(names, dtype=object),
        )
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, max_display=15, show=False)
        fig_w = plt.gcf()
        plot_paths_out["shap_waterfall"] = _save_fig(fig_w, f"{run_id}_shap_waterfall")
    except Exception:
        if plt.get_fignums():
            plt.close(plt.gcf())

    row = shap_arr[0]
    order = np.argsort(np.abs(row))[::-1]
    top3 = [names[i] for i in order[:3] if i < len(names)]
    if len(top3) >= 3:
        explanation_text = (
            "This prediction is mainly influenced by "
            f"{top3[0]}, {top3[1]}, and {top3[2]}."
        )
    elif len(top3) == 2:
        explanation_text = (
            "This prediction is mainly influenced by "
            f"{top3[0]} and {top3[1]}."
        )
    elif len(top3) == 1:
        explanation_text = (
            f"This prediction is mainly influenced by {top3[0]}."
        )

    return shap_vals, plot_paths_out, explanation_text

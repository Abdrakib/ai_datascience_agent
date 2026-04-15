"""
Open Source AutoML Agent — Streamlit UI (local LLM via transformers, no API key).
"""
from __future__ import annotations

import html as html_mod
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from agent.core import OssAutoMLAgent, load_llm_pipeline  # noqa: E402
from agent.report import (  # noqa: E402
    _build_html,
    _build_markdown,
    _generate_next_steps,
    count_embedded_plots_html,
    generate_report,
)
from agent.tools.eda import run_eda  # noqa: E402
from config import OUTPUT_DIR  # noqa: E402
from predict import (  # noqa: E402
    get_model_summary,
    load_model,
    predict,
    prepare_transformed_features,
    save_model,
)

st.set_page_config(
    page_title="AutoML Engineer OSS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_llm_pipeline():
    return load_llm_pipeline()


def _init_state() -> None:
    defaults = {
        "df": None,
        "filename": "",
        "goal": "",
        "pipeline_events": [],
        "final_result": None,
        "load_error": None,
        "run_error": None,
        "log_lines_oss": [],
        "report_export": None,
        "saved_model_path": None,
        "oss_saved_model_path": None,
        "oss_inference_bundle": None,
        "oss_inference_predictions": None,
        "oss_model_pkl_bytes": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _sample_path(name: str) -> Path:
    return APP_ROOT / "datasets" / name


def _log_html(lines: list[str]) -> str:
    parts = []
    for line in lines:
        ts, _, rest = line.partition("] ")
        parts.append(
            f'<div class="log-line"><span class="log-ts">{html_mod.escape(ts + "]")}</span>'
            f'<span class="log-text">{html_mod.escape(rest if rest else line)}</span></div>'
        )
    return f'<div class="log-container">{"".join(parts)}</div>'


def _step_card_header(name: str, step: int | str) -> None:
    badge = (
        '<span style="background:#1e3a2a;color:#4ade80;padding:2px 8px;border-radius:4px;'
        'font-size:11px;border:1px solid #2d5e3e;">Done</span>'
    )
    st.markdown(
        f'<div class="step-card"><div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
        f'<strong style="color:#e2e0d8;">{html_mod.escape(str(name))}</strong> {badge}'
        f'<span style="color:#444441;font-size:12px;">Step {html_mod.escape(str(step))}</span></div>',
        unsafe_allow_html=True,
    )


def _render_step_body(ev: dict) -> None:
    name = ev.get("name", "")
    r = ev.get("result") or {}
    expl = r.get("explanation", "")

    if name == "EDA":
        d = {k: v for k, v in r.items() if k != "explanation"}
        render_step_1_eda(d)
    elif name == "Task detection":
        d = {k: v for k, v in r.items() if k != "explanation"}
        render_step_2_task(d)
    elif name == "Step 2b: Domain Research":
        render_step_2b_domain_research(r)
    elif name == "Preprocessing":
        render_step_3_prep(r["prep"])
    elif name == "Training plan":
        d = {k: v for k, v in r.items() if k != "explanation"}
        render_step_4_plan_training(d)
    elif name == "Training":
        render_step_4_train(r["train"])
    elif name == "Tuning":
        render_step_4b_tune(r["tune"])
    elif name == "Evaluation":
        render_step_5_eval(r["eval"])
    elif name == "Final summary":
        fr = r.get("full") or {}
        render_step_8_final(fr, r.get("explanation"))
    else:
        st.json(r)

    if expl and name != "Final summary":
        st.markdown(
            f'<p style="color:#a8a49c;line-height:1.5;">{html_mod.escape(str(expl))}</p>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _original_feature_frame(df: pd.DataFrame, prep: dict, target_col: str) -> pd.DataFrame | None:
    num = prep.get("num_cols") or []
    cat = prep.get("cat_cols") or []
    cols = list(num) + list(cat)
    if not cols:
        return None
    X = df.drop(columns=[target_col], errors="ignore")
    use = [c for c in cols if c in X.columns]
    if not use:
        return None
    return X[use].copy()


def pal() -> dict[str, str]:
    return {
        "text": "#e2e0d8",
        "muted": "#888780",
        "muted2": "#444441",
        "accent": "#7c3aed",
        "accent_soft": "#c084fc",
        "blue": "#60a5fa",
        "green": "#4ade80",
        "red": "#fb7185",
        "amber": "#fbbf24",
        "border": "#2a2a2e",
        "card_bg": "#141416",
        "pre_bg": "#0a0a0c",
        "section": "#444441",
        "summary_bg": "#1a1a22",
        "summary_border": "#2f2f38",
    }


def esc(s: object) -> str:
    return html_mod.escape(str(s), quote=True) if s is not None else ""


def gap_threshold_for_task(task_type: str) -> float:
    return 0.15 if task_type == "classification" else 0.20


def cv_reliability_label(cv_std: float | None) -> tuple[str, str]:
    p = pal()
    if cv_std is None:
        return "—", p["muted"]
    if cv_std < 0.02:
        return "Reliable", p["green"]
    if cv_std <= 0.05:
        return "Moderate", p["amber"]
    return "Unstable", p["red"]


def plan_dataset_size_label(n_rows: int) -> str:
    if n_rows < 1000:
        return "Small (< 1000 rows)"
    if n_rows <= 10000:
        return "Medium"
    return "Large (> 10000 rows)"


def plan_tuning_budget_why(n_rows: int) -> str:
    if n_rows < 1000:
        return (
            f"Only {n_rows} rows — fewer Optuna trials and a short timeout to limit "
            "overfitting risk and keep the UI responsive."
        )
    if n_rows <= 10000:
        return f"At {n_rows} rows, a mid-sized budget balances search quality with runtime."
    return (
        f"Large dataset ({n_rows:,} rows) — a higher trial count and longer timeout "
        "let Optuna explore the hyperparameter space properly."
    )


def plan_skip_dataset_hook(plan: dict) -> str:
    dp = plan.get("dataset_profile") or {}
    n = int(dp.get("n_rows", 0))
    nf = int(dp.get("n_features", 0))
    return f"Context: your processed data has {n:,} rows and {nf} features after preprocessing."


def plan_why_included(model_name: str, plan: dict) -> str:
    dp = plan.get("dataset_profile") or {}
    n = int(dp.get("n_rows", 0))
    nf = int(dp.get("n_features", 0))
    ir = float(dp.get("imbalance_ratio", 1.0) or 1.0)
    is_small = bool(dp.get("is_small"))
    is_large = bool(dp.get("is_large"))
    is_wide = bool(dp.get("is_wide"))
    is_bin = bool(dp.get("is_binary"))
    smote = bool(dp.get("smote_applied"))
    adj = (plan.get("adjusted_params") or {}).get(model_name, {})
    pm = plan.get("primary_metric") or ""

    if model_name == "Logistic Regression":
        if is_bin and ir > 2 and not smote:
            return (
                f"Included: your {n:,} rows show a {ir:.2f}:1 class ratio — "
                f"a linear model with balanced weights gives a clear, fast baseline before trees."
            )
        return (
            f"Included: with {n:,} rows and {nf} encoded features, "
            f"logistic regression is a strong, interpretable baseline for comparison."
        )
    if model_name == "Linear Regression":
        return (
            f"Included: {n:,} rows × {nf} features — a closed-form linear fit "
            f"anchors the leaderboard before non-linear models."
        )
    if model_name == "Random Forest":
        parts = [f"Included: {n:,} rows and {nf} features suit tree ensembles that handle mixed data."]
        if adj.get("max_depth") is not None:
            parts.append(
                f" We capped max_depth={adj['max_depth']} because the sample is small "
                f"({n} rows) to curb overfitting."
            )
        elif is_large and adj.get("n_estimators"):
            parts.append(
                f" At this size we use {adj.get('n_estimators', 300)} trees for stable estimates."
            )
        return "".join(parts)
    if model_name == "XGBoost":
        if is_small and nf < 5:
            return ""
        if is_large and adj.get("tree_method") == "hist":
            return (
                f"Included: {n:,} rows justify gradient boosting; "
                f"tree_method=hist keeps each trial fast on this volume."
            )
        return (
            f"Included: {n:,} rows × {nf} features — XGBoost captures non-linear "
            f"interactions typical in tabular benchmarks."
        )
    if model_name == "LightGBM":
        if is_wide:
            return (
                f"Included: {nf} features is relatively wide — LightGBM scales well "
                f"to many columns on your {n:,} rows."
            )
        if is_large:
            return (
                f"Included: at {n:,} rows, LightGBM is preferred for speed while "
                f"still fitting strong tree ensembles."
            )
        return (
            f"Included: complements other models on {n:,} rows × {nf} features "
            f"with efficient leaf-wise growth."
        )
    return f"Included for this run ({n:,} rows, {nf} features, primary metric {pm})."


def render_step_1_eda(eda: dict) -> None:
    ov = eda.get("overview", {})
    miss = eda.get("missing", {})
    cols_prof = eda.get("columns", {})
    flags = eda.get("quality_flags", [])
    recs = eda.get("recommendations", [])
    target_info = eda.get("target_info")

    n_rows = ov.get("rows", 0)
    n_cols = ov.get("columns", 0)
    p = pal()

    h = ""
    h += f'<p style="font-family:\'JetBrains Mono\',monospace;color:{p["text"]};">Dataset shape: {n_rows:,} rows × {n_cols} columns</p>'

    num_cols = [c for c, colp in cols_prof.items() if colp.get("dtype_group") == "numeric"]
    cat_cols = [c for c, colp in cols_prof.items() if colp.get("dtype_group") == "categorical"]
    h += f'<p><strong>Numeric columns:</strong> {", ".join(num_cols) if num_cols else "none"}</p>'
    h += f'<p><strong>Categorical columns:</strong> {", ".join(cat_cols) if cat_cols else "none"}</p>'
    h += f'<p><strong>Duplicate rows:</strong> {ov.get("duplicate_rows", 0)}</p>'

    if miss.get("by_column"):
        rows = []
        for col, info in miss["by_column"].items():
            pct = info.get("pct", 0)
            row_style = f"color:{p['red']}" if pct > 30 else f"color:{p['amber']}" if pct > 10 else ""
            rows.append(f"<tr><td>{esc(col)}</td><td>{info.get('count', 0)}</td><td style='{row_style}'>{pct:.1f}%</td></tr>")
        h += "<table><thead><tr><th>Column</th><th>Missing count</th><th>Missing %</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"

    if target_info and target_info.get("inferred_task") == "classification":
        dist = target_info.get("class_distribution", {})
        h += "<p><strong>Class distribution</strong></p><table><thead><tr><th>Class</th><th>Count</th></tr></thead><tbody>"
        for cls, cnt in dist.items():
            h += f"<tr><td>{esc(cls)}</td><td>{cnt}</td></tr>"
        h += "</tbody></table>"
        if "imbalance_ratio" in target_info:
            h += f"<p><strong>Imbalance ratio:</strong> {target_info['imbalance_ratio']}:1</p>"

    skewed = [
        (c, colp.get("skewness"))
        for c, colp in cols_prof.items()
        if colp.get("dtype_group") == "numeric"
        and colp.get("skewness") is not None
        and abs(float(colp["skewness"])) > 2.0
    ]
    if skewed:
        h += "<p><strong>Skewed columns (|skewness| &gt; 2.0):</strong></p><ul>"
        for c, s in skewed:
            h += f"<li>{esc(c)}: skewness = {float(s):.4f}</li>"
        h += "</ul>"

    if flags:
        h += "<p><strong>Quality flags</strong></p><ul>"
        for f in flags:
            h += f"<li>{esc(f)}</li>"
        h += "</ul>"
    if recs:
        h += "<p><strong>Preprocessing recommendations</strong></p><ul>"
        for r in recs:
            h += f"<li>{esc(r)}</li>"
        h += "</ul>"

    st.markdown(f'<div class="step-card-body">{h}</div>', unsafe_allow_html=True)

    if n_rows < 500:
        st.warning("Small dataset detected — model performance may be unreliable", icon="⚠️")
    if target_info and target_info.get("inferred_task") == "classification":
        ratio = target_info.get("imbalance_ratio", 0)
        if ratio and ratio > 5:
            st.error("Severe class imbalance detected — consider SMOTE or class_weight=balanced", icon="🚨")
    high_missing = [col for col, info in (miss.get("by_column") or {}).items() if info.get("pct", 0) > 30]
    if high_missing:
        st.warning("High missing rate detected in one or more columns", icon="⚠️")


def render_step_2b_domain_research(r: dict) -> None:
    dr = r.get("domain_research") or {}
    msg = r.get(
        "message",
        "The agent searched the web to better understand your dataset. Here is what it found:",
    )
    p = pal()
    st.markdown(
        f'<p style="font-family:\'DM Sans\',sans-serif;color:{p["text"]};line-height:1.5;">'
        f"{esc(msg)}</p>",
        unsafe_allow_html=True,
    )
    q = esc(str(dr.get("query", "")))
    st.markdown(
        f'<p style="color:{p["muted"]};font-size:12px;"><strong>Query:</strong> {q}</p>',
        unsafe_allow_html=True,
    )
    for i, row in enumerate(dr.get("results") or [], 1):
        if isinstance(row, dict) and row.get("error"):
            st.markdown(
                f'<p style="color:{p["amber"]};">{esc(str(row["error"]))}</p>',
                unsafe_allow_html=True,
            )
            break
        if not isinstance(row, dict):
            continue
        title = esc(str(row.get("title", "")))
        url = str(row.get("url", "") or "")
        sn = esc((row.get("snippet") or "")[:600])
        link = esc(url)
        st.markdown(
            f'<div style="margin:12px 0;padding:10px;border-left:3px solid {p["accent"]};'
            f'background:#141416;border-radius:4px;">'
            f"<strong>{i}. {title}</strong><br/>"
            f'<a href="{link}" target="_blank" rel="noopener noreferrer">{link}</a><br/>'
            f'<span style="color:{p["muted"]};font-size:13px;">{sn}</span></div>',
            unsafe_allow_html=True,
        )


def render_step_2_task(task: dict) -> None:
    h = (
        f"<p><strong>Target column:</strong> <code>{esc(task.get('target_col', '—'))}</code></p>"
        f"<p><strong>Task type:</strong> {esc(task.get('task_type', '—'))}</p>"
        f"<p><strong>Confidence:</strong> {esc(task.get('confidence', '—'))}</p>"
        f"<p><strong>Reasoning:</strong> {esc(task.get('reasoning', '—'))}</p>"
    )
    if task.get("alternatives"):
        h += f"<p><strong>Alternative candidate columns:</strong> {esc(', '.join(str(x) for x in task['alternatives']))}</p>"
    st.markdown(f'<div class="step-card-body">{h}</div>', unsafe_allow_html=True)


def render_step_3_prep(prep: dict) -> None:
    html_b = f"<p><strong>Numeric columns used:</strong> {', '.join(prep.get('num_cols', [])) or 'none'}</p>"
    html_b += f"<p><strong>Categorical columns used:</strong> {', '.join(prep.get('cat_cols', [])) or 'none'}</p>"
    enc = prep.get("encoding_summary", {})
    html_b += "<p><strong>Encoding strategy:</strong></p><ul>"
    for col, strat in list(enc.items())[:40]:
        html_b += f"<li>{esc(col)}: {esc(strat)}</li>"
    html_b += "</ul>"
    if prep.get("dropped_cols"):
        html_b += "<p><strong>Columns dropped:</strong></p><ul>"
        for c in prep["dropped_cols"]:
            html_b += f"<li>{esc(c)}</li>"
        html_b += "</ul>"
    log = prep.get("preprocessing_log", [])
    html_b += "<p><strong>Preprocessing log</strong></p>"
    for line in log:
        html_b += f"<p style='font-family:JetBrains Mono,monospace;font-size:12px;'>{esc(line)}</p>"
    n_feat = prep.get("final_feature_count")
    if n_feat is None and prep.get("feature_names") is not None:
        n_feat = len(prep["feature_names"])
    html_b += f"<p><strong>Final feature count (after encoding):</strong> {esc(n_feat)}</p>"
    ts = prep.get("train_size")
    te = prep.get("test_size")
    if ts is not None and te is not None:
        html_b += f"<p><strong>Train size:</strong> {ts} rows · <strong>Test size:</strong> {te} rows</p>"
    elif prep.get("X_train") is not None:
        html_b += (
            f"<p><strong>Train size:</strong> {prep['X_train'].shape[0]} rows · "
            f"<strong>Test size:</strong> {prep['X_test'].shape[0]} rows</p>"
        )
    st.markdown(f'<div class="step-card-body">{html_b}</div>', unsafe_allow_html=True)
    sm = prep.get("smote_applied")
    lt = prep.get("log_transformed_cols") or []
    st.caption(f"SMOTE applied: **{sm}** · Log1p columns: {lt or 'none'}")
    leak = prep.get("target_leakage_suspicion")
    if leak:
        m = re.search(r"Column '([^']+)'", str(leak)) or re.search(r"column ([A-Za-z0-9_]+)", str(leak), re.I)
        col = m.group(1) if m else "unknown"
        st.error(
            f"Potential target leakage detected in column {col} — this may inflate your metrics",
            icon="🚨",
        )


def render_step_4_plan_training(plan: dict) -> None:
    p = pal()
    dp = plan.get("dataset_profile") or {}
    n_rows = int(dp.get("n_rows", 0))
    n_features = int(dp.get("n_features", 0))
    is_reg = plan.get("primary_metric") == "r2"
    ir = float(dp.get("imbalance_ratio", 1.0) or 1.0)
    smote = bool(dp.get("smote_applied"))

    sec = (
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{p["accent_soft"]};'
        'margin:0 0 10px 0;">SECTION 1 — Dataset profile</p>'
        '<table><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>'
        f"<tr><td>Rows</td><td>{esc(n_rows)}</td></tr>"
        f"<tr><td>Features</td><td>{esc(n_features)}</td></tr>"
        f"<tr><td>Dataset size</td><td>{esc(plan_dataset_size_label(n_rows))}</td></tr>"
        f"<tr><td>Task type</td><td>{'regression' if is_reg else 'classification'}</td></tr>"
    )
    if not is_reg:
        sec += (
            f"<tr><td>Class imbalance ratio</td><td>{esc(f'{ir:.2f}')}</td></tr>"
            f"<tr><td>SMOTE applied</td><td>{'Yes' if smote else 'No'}</td></tr>"
        )
    sec += "</tbody></table>"

    sec += (
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{p["accent_soft"]};'
        'margin:20px 0 10px 0;">SECTION 2 — Model selection</p>'
    )

    rec = plan.get("recommended_models") or []
    adj_all = plan.get("adjusted_params") or {}
    for m in rec:
        params = adj_all.get(m) or {}
        param_str = ", ".join(f"{k}={v!r}" for k, v in sorted(params.items())) if params else "defaults"
        reason = plan_why_included(m, plan)
        sec += (
            f'<p style="font-family:\'DM Sans\',sans-serif;margin:10px 0 4px 0;">'
            f'<span style="color:{p["green"]};font-size:16px;">✓</span> '
            f'<strong style="font-family:\'JetBrains Mono\',monospace;">{esc(m)}</strong></p>'
            f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:{p["muted"]};margin:0 0 4px 0;">'
            f"Parameters: {esc(param_str)}</p>"
            f'<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;color:{p["text"]};margin:0 0 12px 0;">'
            f"{esc(reason)}</p>"
        )

    skip = plan.get("skip_models") or []
    reasons = plan.get("skip_reasons") or {}
    hook = plan_skip_dataset_hook(plan)
    for m in skip:
        r = reasons.get(m, "Excluded by training plan rules.")
        sec += (
            f'<p style="font-family:\'DM Sans\',sans-serif;margin:10px 0 4px 0;">'
            f'<span style="color:{p["red"]};font-size:16px;">✗</span> '
            f'<strong style="font-family:\'JetBrains Mono\',monospace;">{esc(m)}</strong></p>'
            f'<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;color:{p["text"]};margin:0 0 4px 0;">'
            f"{esc(r)}</p>"
            f'<p style="font-family:\'DM Sans\',sans-serif;font-size:12px;color:{p["muted"]};margin:0 0 12px 0;">'
            f"{esc(hook)}</p>"
        )

    pm = plan.get("primary_metric") or "—"
    mr = plan.get("metric_reasoning") or ""
    sec += (
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{p["accent_soft"]};'
        'margin:20px 0 10px 0;">SECTION 3 — Evaluation metric</p>'
        f'<p style="font-family:\'JetBrains Mono\',monospace;color:{p["text"]};">'
        f"Primary metric: <strong>{esc(pm)}</strong></p>"
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;">{esc(mr)}</p>'
    )

    nt = plan.get("n_trials", "—")
    to = plan.get("timeout", "—")
    sec += (
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{p["accent_soft"]};'
        'margin:20px 0 10px 0;">SECTION 4 — Tuning budget</p>'
        f'<p style="font-family:\'JetBrains Mono\',monospace;color:{p["text"]};">'
        f"Optuna trials: <strong>{esc(nt)}</strong> · "
        f"Timeout: <strong>{esc(to)}</strong> seconds</p>"
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;">'
        f"{esc(plan_tuning_budget_why(n_rows))}</p>"
    )

    warns = plan.get("warnings") or []
    notes = plan.get("notes") or []
    sec += (
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{p["accent_soft"]};'
        'margin:20px 0 10px 0;">SECTION 5 — Warnings and notes</p>'
    )
    for w in warns:
        sec += f'<div style="margin-bottom:8px;background:rgba(251,191,36,0.1);padding:8px;border-radius:6px;">⚠ {esc(w)}</div>'
    if notes:
        sec += f'<ul style="font-family:\'DM Sans\',sans-serif;color:{p["text"]};">'
        for note in notes:
            sec += f"<li>{esc(note)}</li>"
        sec += "</ul>"
    elif not warns:
        sec += f'<p style="font-family:\'DM Sans\',sans-serif;color:{p["muted"]};">No additional warnings or notes for this plan.</p>'

    summary = plan.get("plan_summary") or ""
    sec += (
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{p["accent_soft"]};'
        'margin:20px 0 10px 0;">SECTION 6 — Plan summary</p>'
        f'<div style="background:{p["summary_bg"]};border:1px solid {p["summary_border"]};border-radius:8px;padding:16px 18px;">'
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:14px;color:{p["text"]};margin:0;line-height:1.55;">'
        f"{esc(summary)}</p>"
        "</div>"
    )

    st.markdown(f'<div class="step-card-body">{sec}</div>', unsafe_allow_html=True)


def render_step_4_train(train: dict) -> None:
    results = train.get("results", [])
    comp_df = train.get("comparison_df")
    best_name = train.get("best_name", "")
    primary = train.get("metric_name", "roc_auc")
    task_type = train.get("task_type") or (
        "classification"
        if primary in ("roc_auc", "f1", "f1_weighted", "accuracy")
        else "regression"
    )
    gap_thr = gap_threshold_for_task(task_type)
    overfit_warnings = train.get("overfitting_warnings", [])
    severe_note = any("All models showed severe overfitting" in w for w in overfit_warnings)

    if comp_df is not None and not comp_df.empty:
        try:
            if "Gap" in comp_df.columns:
                p = pal()

                def _gap_style(s: pd.Series):
                    hi_bg = "rgba(251,113,133,0.25)"
                    hi_fg = p["red"]
                    return [
                        f"background-color: {hi_bg}; color: {hi_fg}; font-weight: 600;"
                        if isinstance(v, (int, float)) and not pd.isna(v) and float(v) > gap_thr
                        else ""
                        for v in s
                    ]

                st.dataframe(
                    comp_df.style.apply(_gap_style, subset=["Gap"]),
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
    else:
        st.info("No comparison table available.")

    st.caption(
        "Cross-validation splits the training data into several folds, trains on all but one each time, "
        "and scores on the held-out fold. **CV Mean** is the average of those scores — often more reliable "
        "than a single train/test split because every row is used for validation once."
    )

    if best_name:
        base = best_name.replace(" (tuned)", "").strip()
        best_r = next((r for r in results if r.get("name") == base), None)
        cv_std = best_r.get("cv_std") if best_r else None
        lbl, col = cv_reliability_label(cv_std)
        reason = (
            "Least overfit model (all others exceeded severe gap threshold)."
            if severe_note
            else (
                "Highest CV mean on the primary metric (when CV ran); otherwise ranked by held-out test score."
            )
        )
        st.markdown(
            f'<p style="font-family:\'JetBrains Mono\',monospace;color:{pal()["text"]};">'
            f"<strong>Selected best model:</strong> {esc(best_name)} "
            f'<span style="margin-left:10px;padding:2px 8px;border-radius:4px;font-size:12px;background:rgba(128,128,128,0.15);color:{col};border:1px solid {col};">{esc(lbl)}</span>'
            f" — {esc(reason)}</p>",
            unsafe_allow_html=True,
        )

    if results and any(r.get("cv_scores") for r in results):
        st.markdown("**Cross-validation details**")
        for r in results:
            scores = r.get("cv_scores")
            if not scores:
                continue
            st.markdown(f"*{esc(r.get('name', 'Model'))}* — scores per fold")
            chart_df = pd.DataFrame(
                {"Score": [float(s) for s in scores]},
                index=[f"Fold {i + 1}" for i in range(len(scores))],
            )
            st.bar_chart(chart_df)

    for r in results:
        m = r.get("metrics", {})
        if m.get("overfit"):
            ts = m.get("train_score", 0)
            tss = m.get("test_score", 0)
            gap = m.get("generalization_gap", 0)
            label = "ROC-AUC" if task_type == "classification" else "R²"
            st.error(
                f"Overfitting detected in {r['name']} — train {label} {ts:.4f} vs test {label} {tss:.4f}, gap {gap:.4f}",
                icon="🚨",
            )


def render_step_4b_tune(tune: dict) -> None:
    model = tune.get("model_name") or "—"
    if not tune.get("success", False):
        err = tune.get("error", "Unknown error")
        st.markdown(
            f'<p style="font-family:\'JetBrains Mono\',monospace;color:{pal()["text"]};">'
            f"<strong>Model:</strong> {esc(model)}</p>",
            unsafe_allow_html=True,
        )
        st.error(f"Hyperparameter tuning failed: {err}")
        return

    bp = tune.get("best_params") or {}
    h = (
        f"<p><strong>Model tuned:</strong> {esc(model)}</p>"
        f"<p><strong>Baseline score (test):</strong> {float(tune.get('baseline_score', 0)):.4f} · "
        f"<strong>After tuning:</strong> {float(tune.get('best_score', 0)):.4f} · "
        f"<strong>Improvement:</strong> {float(tune.get('improvement', 0)):+.4f}</p>"
        f"<p><strong>Optuna trials:</strong> {tune.get('n_trials_run', 0)} · "
        f"<strong>Time:</strong> {float(tune.get('tuning_time_s', 0)):.1f}s · "
        f"<strong>Train–test gap:</strong> {float(tune.get('generalization_gap', 0)):.4f}</p>"
    )
    if bp:
        h += "<p><strong>Best hyperparameters</strong></p><table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>"
        for k, v in sorted(bp.items())[:48]:
            h += f"<tr><td><code>{esc(k)}</code></td><td>{esc(v)}</td></tr>"
        h += "</tbody></table>"
    st.markdown(f'<div class="step-card-body">{h}</div>', unsafe_allow_html=True)
    if tune.get("overfit"):
        st.warning(
            "Tuned model still shows an elevated train–test gap — consider more data or stronger regularization.",
            icon="⚠️",
        )


def is_shap_plot_key(k: str) -> bool:
    return k in ("shap_bar", "shap_summary", "shap_waterfall") or k.startswith("shap_dependence_")


def render_shap_dependence_deep_dive(plot_paths: dict) -> None:
    dep_keys = sorted(k for k in plot_paths if str(k).startswith("shap_dependence_"))
    if not dep_keys:
        return
    st.markdown("---")
    st.markdown("### SHAP Feature Deep Dive")
    st.caption(
        "These plots show how each top feature individually drives the model's predictions across all test samples."
    )
    for key in dep_keys:
        path = plot_paths.get(key)
        if not path or not Path(path).exists():
            continue
        feat_label = str(key).replace("shap_dependence_", "", 1).replace("_", " ")
        st.image(path, use_container_width=True)
        st.caption(
            f"Feature: {feat_label} — each point is one test sample. "
            "Color indicates the interacting feature value."
        )


def render_shap_bar_and_summary(plot_paths: dict) -> None:
    if not any(plot_paths.get(k) for k in ("shap_bar", "shap_summary")):
        return

    st.markdown("---")
    st.markdown("### SHAP explainability")
    pbar = plot_paths.get("shap_bar")
    if pbar and Path(pbar).exists():
        st.markdown("#### Feature importance (SHAP)")
        st.caption(
            "Mean absolute SHAP values show which features push predictions the most, on average."
        )
        st.image(pbar, use_container_width=True)
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    psum = plot_paths.get("shap_summary")
    if psum and Path(psum).exists():
        st.markdown("#### Feature impact distribution")
        st.caption(
            "Each dot represents a data point. Red = high feature value, Blue = low. "
            "Position shows impact on prediction."
        )
        st.image(psum, use_container_width=True)


def render_shap_waterfall_only(plot_paths: dict) -> None:
    pw = plot_paths.get("shap_waterfall")
    if not pw or not Path(pw).exists():
        return
    st.markdown("#### Local explanation (one example row)")
    st.image(pw, use_container_width=True)


def render_shap_ui(plot_paths: dict) -> None:
    if not any(is_shap_plot_key(k) and plot_paths.get(k) for k in plot_paths):
        return
    render_shap_bar_and_summary(plot_paths)
    render_shap_dependence_deep_dive(plot_paths)
    render_shap_waterfall_only(plot_paths)


def render_step_5_eval(eval_data: dict) -> None:
    metrics = eval_data.get("metrics", {})
    plot_paths = eval_data.get("plot_paths", {})
    task = eval_data.get("task_type", "classification")

    st.markdown(
        f'<p style="font-family:\'JetBrains Mono\',monospace;color:{pal()["muted"]};">Evaluation metrics</p>',
        unsafe_allow_html=True,
    )
    if task == "classification":
        c1, c2, c3 = st.columns(3)
        if metrics.get("accuracy") is not None:
            c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        if metrics.get("f1") is not None:
            c2.metric("F1 (weighted)", f"{metrics['f1']:.4f}")
        if metrics.get("roc_auc") is not None:
            c3.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    else:
        c1, c2, c3, c4 = st.columns(4)
        if metrics.get("r2") is not None:
            c1.metric("R²", f"{metrics['r2']:.4f}")
        if metrics.get("rmse") is not None:
            c2.metric("RMSE", f"{metrics['rmse']:.4f}")
        if metrics.get("mae") is not None:
            c3.metric("MAE", f"{metrics['mae']:.4f}")
        if metrics.get("mape") is not None:
            c4.metric("MAPE", f"{metrics['mape']:.2f}%")

    if task == "classification":
        auc = metrics.get("roc_auc")
        if auc is not None and auc < 0.6:
            st.warning(
                "Model performance is near random — review your features and target column",
                icon="⚠️",
            )
    else:
        r2 = metrics.get("r2")
        if r2 is not None and r2 < 0.3:
            st.warning(
                "Model performance is near random — review your features and target column",
                icon="⚠️",
            )

    for plot_name in [
        "confusion_matrix",
        "roc_curve",
        "actual_vs_predicted",
        "residuals",
        "feature_importance",
    ]:
        path = plot_paths.get(plot_name)
        if path and Path(path).exists():
            st.image(path, caption=plot_name.replace("_", " ").title(), use_container_width=True)

    render_shap_ui(plot_paths)


def feature_interpretation_sentence(feat: str, rank: int, task_type: str, has_shap: bool) -> str:
    strong = rank == 0
    if task_type == "classification":
        return (
            f"{feat} was {'the strongest predictor' if strong else 'among the strongest predictors'} — "
            f"higher values tend to push the prediction toward the positive class "
            f"({'SHAP and importance agree' if has_shap else 'per model feature importance'})."
        )
    return (
        f"{feat} was {'the strongest driver' if strong else 'among the strongest drivers'} of predicted values — "
        f"larger values are associated with higher predicted outcomes "
        f"({'consistent with SHAP analysis' if has_shap else 'per feature importance'})."
    )


def render_what_model_learned(result: dict) -> None:
    fi = result.get("feature_importances") or {}
    if not fi:
        return
    ev = result.get("eval") or {}
    has_shap = bool(ev.get("has_shap"))
    task = str(result.get("task_type", "classification") or "classification")
    top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]
    if not top5:
        return

    st.markdown("**What the model learned**")
    names = [t[0] for t in top5]
    vals = [float(t[1]) for t in top5]
    chart_df = pd.DataFrame({"importance": vals}, index=names)
    st.bar_chart(chart_df)
    for i, (feat, _) in enumerate(top5):
        st.caption(feature_interpretation_sentence(str(feat), i, task, has_shap))


def render_step_8_final(result: dict, llm_summary: str | None = None) -> None:
    """Final recommendation card (no export buttons — those live in app.py)."""
    best = result.get("best_model_name", "—")
    metrics = result.get("best_metrics", {})
    task = result.get("task_type", "")
    primary = "roc_auc" if task == "classification" else "r2"
    train_data = result.get("train", {})
    overfit_warnings = list(train_data.get("overfitting_warnings", []))

    st.markdown(
        f'<div class="metric-card" style="margin-bottom:16px;background:#141416;border:1px solid #2a2a2e;border-radius:8px;padding:16px;">'
        f'<div style="font-size:11px;color:#888780;text-transform:uppercase;">Best model</div>'
        f'<div style="font-size:22px;font-weight:600;color:#7c3aed;">{esc(best)}</div>'
        f'<p style="font-family:\'JetBrains Mono\',monospace;margin-top:12px;color:#888780;">'
        f'Primary metric ({primary}): {float(metrics.get(primary, 0) or 0):.4f}</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "This model was chosen because it achieved the best test score on the primary metric among models "
        "that were not excluded for severe overfitting (generalization gap above 0.25). "
        "If every model was severely overfit, the least overfit model was selected.",
    )

    st.markdown("**Active warnings from this run**")
    warnings_list: list[str] = []
    eda = result.get("eda", {})
    if eda.get("overview", {}).get("rows", 0) < 500:
        warnings_list.append("Small dataset (<500 rows) — unreliable metrics.")
    ti = eda.get("target_info") or {}
    if ti.get("inferred_task") == "classification" and (ti.get("imbalance_ratio") or 0) > 5:
        warnings_list.append("Severe class imbalance (>5:1).")
    miss = eda.get("missing", {}).get("by_column", {}) or {}
    if any(info.get("pct", 0) > 30 for info in miss.values()):
        warnings_list.append("High missing rate (>30%) in one or more columns.")
    prep = result.get("prep") or {}
    if prep.get("target_leakage_suspicion"):
        warnings_list.append("Possible target leakage (high correlation with target).")
    if overfit_warnings:
        warnings_list.extend(overfit_warnings)
    if not warnings_list:
        st.caption("No critical warnings.")
    else:
        for w in warnings_list:
            st.warning(w, icon="⚠️")

    render_what_model_learned(result)

    st.markdown("**Recommended next actions**")
    for a in _generate_next_steps(result):
        st.markdown(f"- {a}")

    if llm_summary:
        st.markdown("**Executive summary (model-generated)**")
        st.write(llm_summary)


def _inference_empty_state() -> None:
    p = pal()
    st.markdown(
        f'<p style="color:{p["muted"]};line-height:1.6;font-size:15px;">'
        "Train a model in the <strong style=\"color:#e2e0d8;\">Pipeline</strong> tab first, "
        "then click <strong style=\"color:#e2e0d8;\">Save Model</strong> to use it here. "
        "Or upload a <code>.pkl</code> file produced by this app.</p>",
        unsafe_allow_html=True,
    )


def _inference_load_bundle(path: str) -> bool:
    try:
        b = load_model(path)
        st.session_state["oss_inference_bundle"] = b
        st.session_state["oss_inference_predictions"] = None
        return True
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return False


def _confidence_for_classification(bundle: dict, df_in: pd.DataFrame) -> np.ndarray | None:
    """Max class probability per row for display (app-side; does not change predict.py)."""
    model = bundle.get("model")
    task = bundle.get("task_type")
    if task != "classification" or model is None or not hasattr(model, "predict_proba"):
        return None
    df2 = df_in.drop(
        columns=[c for c in ("prediction", "probability") if c in df_in.columns],
        errors="ignore",
    )
    try:
        X_t, _ = prepare_transformed_features(bundle, df2)
        proba = model.predict_proba(X_t)
        if proba.ndim == 1:
            return np.asarray(proba).ravel() * 100.0
        return np.max(proba, axis=1) * 100.0
    except Exception:
        return None


def _class_color_style(pred: object, le_classes: np.ndarray | None) -> str:
    """Green / red heuristic for binary-ish class names."""
    p = pal()
    s = str(pred).strip().lower()
    pos = ("1", "true", "yes", "pos", "surviv", "good", "approved", "positive")
    neg = ("0", "false", "no", "neg", "death", "bad", "denied", "negative")
    if any(x in s for x in pos):
        return p["green"]
    if any(x in s for x in neg):
        return p["red"]
    if le_classes is not None and len(le_classes) == 2:
        try:
            idx = list(le_classes).index(pred)
            return p["green"] if idx == 0 else p["red"]
        except ValueError:
            pass
    return p["text"]


def render_inference_tab() -> None:
    p = pal()
    st.markdown(
        f'<p class="section-head" style="margin-top:0;">Inference</p>',
        unsafe_allow_html=True,
    )

    bundle = st.session_state.get("oss_inference_bundle")

    st.markdown(f'### <span style="color:{p["text"]};">Load a trained model</span>', unsafe_allow_html=True)
    if bundle is None:
        _inference_empty_state()

    c1, c2 = st.columns(2)
    with c1:
        sess_path = st.session_state.get("oss_saved_model_path") or st.session_state.get("saved_model_path")
        if sess_path and Path(sess_path).exists():
            if st.button("Use current session model", key="oss_inf_use_session", use_container_width=True):
                if _inference_load_bundle(str(sess_path)):
                    st.rerun()
        else:
            st.caption("No model saved in this session yet.")
    with c2:
        up_m = st.file_uploader("Upload a saved model (.pkl)", type=["pkl"], key="oss_inf_model_upload")
        if up_m is not None:
            if st.button("Load uploaded model", key="oss_inf_load_upload"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
                        tmp.write(up_m.getvalue())
                        tmp_path = tmp.name
                    if _inference_load_bundle(tmp_path):
                        st.rerun()
                except Exception as e:
                    st.error(str(e))

    if bundle is None:
        return

    summary = get_model_summary(bundle)
    st.markdown("---")
    st.markdown("#### Model info")
    badge = (
        f'<span style="background:#1e3a2a;color:#4ade80;padding:4px 10px;border-radius:6px;'
        f'font-size:12px;border:1px solid #2d5e3e;font-family:JetBrains Mono,monospace;">'
        f"Model loaded ✓</span>"
    )
    st.markdown(badge, unsafe_allow_html=True)
    mcols = st.columns(2)
    with mcols[0]:
        st.markdown(f"**Model name:** {esc(summary.get('model_name', '—'))}")
        st.markdown(f"**Task type:** {esc(summary.get('task_type', '—'))}")
        st.markdown(f"**Target column:** `{esc(summary.get('target_col', '—'))}`")
    with mcols[1]:
        st.markdown("**Key metrics (test)**")
        met = summary.get("metrics") or {}
        if met:
            for k, v in list(met.items())[:8]:
                st.caption(f"{k}: {v}")
        else:
            st.caption("—")
    feats = summary.get("expected_input_columns") or []
    st.markdown("**Feature columns expected:**")
    st.code(", ".join(str(x) for x in feats) if feats else "—", language=None)

    st.markdown("---")
    st.markdown(f'### <span style="color:{p["text"]};">Upload new data for prediction</span>', unsafe_allow_html=True)

    inf_csv = st.file_uploader("CSV with rows to score", type=["csv"], key="oss_inf_csv")
    pred_df: pd.DataFrame | None = None
    if inf_csv is not None:
        try:
            pred_df = pd.read_csv(inf_csv)
            st.dataframe(pred_df.head(5), use_container_width=True)
            exp_cols = set(feats)
            have = set(pred_df.columns)
            missing = [c for c in feats if c not in have]
            if missing:
                st.warning(f"Missing feature columns (will be imputed like predict.py): {missing}")
            tc = bundle.get("target_col")
            if tc and tc in pred_df.columns:
                st.warning(
                    f"Target column `{tc}` is present — it will be ignored for prediction "
                    "(dropped before scoring)."
                )
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            pred_df = None

    csv_btn = st.button("Predict on uploaded CSV", key="oss_inf_predict_csv", disabled=pred_df is None)
    if csv_btn and pred_df is not None:
        try:
            out, _fill = predict(bundle, pred_df)
            st.session_state["oss_inference_predictions"] = {
                "kind": "csv",
                "result_df": out,
                "task_type": bundle.get("task_type"),
                "bundle": bundle,
            }
            st.rerun()
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    n_feat = len(feats)
    manual_vals: dict[str, Any] = {}
    if n_feat > 0 and n_feat <= 10:
        st.markdown("##### Manual input (single row)")
        means = bundle.get("feature_means") or {}
        modes = bundle.get("feature_modes") or {}
        cat_u = bundle.get("categorical_uniques") or {}
        num_cols = set(bundle.get("num_cols") or [])
        cat_cols = set(bundle.get("cat_cols") or [])
        for fi, col in enumerate(feats):
            key = f"oss_inf_f_{fi}"
            choices = cat_u.get(col)
            if choices:
                default = modes.get(col, choices[0])
                try:
                    idx = int(list(choices).index(default)) if default in list(choices) else 0
                except ValueError:
                    idx = 0
                manual_vals[col] = st.selectbox(
                    str(col),
                    options=list(choices),
                    index=min(idx, len(choices) - 1),
                    key=key,
                )
            elif col in num_cols or (col in means and col not in cat_cols):
                default = float(means.get(col, 0.0))
                manual_vals[col] = st.number_input(
                    str(col),
                    value=default,
                    key=key,
                    format="%.6f",
                )
            elif col in cat_cols and not choices:
                default = modes.get(col, "")
                manual_vals[col] = st.text_input(str(col), value=str(default), key=key)
            else:
                default = float(means.get(col, 0.0)) if col in means else 0.0
                manual_vals[col] = st.number_input(
                    str(col),
                    value=default,
                    key=key,
                    format="%.6f",
                )

        manual_btn = st.button("Predict (manual row)", key="oss_inf_predict_manual")
        if manual_btn:
            try:
                row_df = pd.DataFrame([manual_vals])
                out, _fill = predict(bundle, row_df)
                st.session_state["oss_inference_predictions"] = {
                    "kind": "manual",
                    "result_df": out,
                    "task_type": bundle.get("task_type"),
                    "bundle": bundle,
                }
                st.rerun()
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("---")
    st.markdown(f'### <span style="color:{p["text"]};">Prediction results</span>', unsafe_allow_html=True)

    pred_pack = st.session_state.get("oss_inference_predictions")
    if not pred_pack:
        st.info("Run a prediction from CSV or manual input to see results here.")
        return

    result_df: pd.DataFrame = pred_pack["result_df"]
    ttype = pred_pack.get("task_type") or bundle.get("task_type")
    le = bundle.get("label_encoder")
    classes = le.classes_ if le is not None else None

    if ttype == "classification":
        conf_pct: np.ndarray | None = None
        if "probability" in result_df.columns:
            conf_pct = (pd.to_numeric(result_df["probability"], errors="coerce").fillna(0) * 100).values
        else:
            base_df = result_df.drop(
                columns=[c for c in ("prediction", "probability") if c in result_df.columns],
                errors="ignore",
            )
            conf_pct = _confidence_for_classification(bundle, base_df)

        if len(result_df) == 1:
            pred0 = result_df["prediction"].iloc[0]
            col = _class_color_style(pred0, classes)
            st.markdown(
                f'<div style="font-size:36px;font-weight:700;color:{col};line-height:1.2;">'
                f"{esc(pred0)}</div>",
                unsafe_allow_html=True,
            )
            if conf_pct is not None and len(conf_pct) > 0:
                cp = float(np.clip(conf_pct[0], 0, 100))
                st.progress(cp / 100.0)
                st.caption(f"Confidence: **{cp:.1f}%**")
        else:
            disp = result_df.copy()
            if conf_pct is not None and len(conf_pct) == len(disp):
                disp["Confidence %"] = np.round(conf_pct, 2)
            rows_html = []
            for row_num, (_, row) in enumerate(disp.iterrows(), start=1):
                pr = row.get("prediction")
                col = _class_color_style(pr, classes)
                conf_cell = ""
                if "Confidence %" in disp.columns:
                    conf_cell = esc(row.get("Confidence %"))
                rows_html.append(
                    f"<tr><td>{row_num}</td>"
                    f"<td style='color:{col};font-weight:600;'>{esc(pr)}</td>"
                    f"<td>{conf_cell}</td></tr>"
                )
            st.markdown(
                f'<table style="width:100%;border-collapse:collapse;font-family:JetBrains Mono,monospace;font-size:13px;">'
                f"<thead><tr><th>Row</th><th>Predicted class</th><th>Confidence %</th></tr></thead>"
                f"<tbody>{''.join(rows_html)}</tbody></table>",
                unsafe_allow_html=True,
            )
    else:
        preds = pd.to_numeric(result_df["prediction"], errors="coerce")
        if len(result_df) == 1:
            v = float(preds.iloc[0])
            st.markdown(
                f'<div style="font-size:36px;font-weight:700;color:{p["accent_soft"]};line-height:1.2;">'
                f"{v:.6f}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="font-size:28px;font-weight:600;color:{p["accent_soft"]};">'
                f"Predicted values ({len(result_df)} rows)</div>",
                unsafe_allow_html=True,
            )
            tbl = pd.DataFrame(
                {
                    "Row": np.arange(1, len(result_df) + 1),
                    "Predicted value": preds.values,
                }
            )
            st.dataframe(tbl, use_container_width=True, hide_index=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Min", f"{float(preds.min()):.6f}")
            with m2:
                st.metric("Mean", f"{float(preds.mean()):.6f}")
            with m3:
                st.metric("Max", f"{float(preds.max()):.6f}")

    st.download_button(
        label="⬇ Download Predictions as CSV",
        data=result_df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv",
        key="download_predictions_csv",
    )


# ── Page chrome ───────────────────────────────────────────────────────────────
_init_state()

st.markdown(
    r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0e0e10; color: #e2e0d8; }
[data-testid="stSidebar"] { background-color: #141416; border-right: 1px solid #2a2a2e; }
.step-card { background: #141416; border: 1px solid #2a2a2e; border-radius: 8px; padding: 16px 20px; margin-bottom: 16px; }
.hf-banner { background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%); border: 1px solid #2a2a3e; border-radius: 8px; padding: 12px 16px; margin-bottom: 20px; color: #c4c2bc; font-size: 14px; }
.log-container { background: #0a0a0c; border: 1px solid #2a2a2e; border-radius: 8px; padding: 16px; font-family: 'JetBrains Mono', monospace; font-size: 12px; max-height: 380px; overflow-y: auto; }
.log-line { padding: 3px 0; line-height: 1.6; }
.log-ts { color: #444441; margin-right: 8px; }
.log-text { color: #a8a49c; }
.section-head { font-family: 'JetBrains Mono', monospace; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: #444441; margin: 24px 0 12px 0; }
.stButton > button { background: #1e1e22; color: #e2e0d8; border: 1px solid #3a3a3e; border-radius: 6px; font-family: 'JetBrains Mono', monospace; }
.stButton > button:hover { background: #7c3aed; border-color: #7c3aed; color: #fff; }
[data-testid="stFileUploader"] { background: #141416; border: 1px dashed #3a3a3e; border-radius: 8px; }
.stTextArea textarea { background: #141416 !important; border-color: #2a2a2e !important; color: #e2e0d8 !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hf-banner"><strong>Powered by Qwen2.5 7B Instruct</strong> — runs free on Hugging Face ZeroGPU. '
    "No API key needed.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div style="display:flex;align-items:baseline;gap:12px;padding:8px 0 16px 0;border-bottom:1px solid #2a2a2e;margin-bottom:16px;">'
    '<span style="font-family:JetBrains Mono,monospace;font-size:22px;font-weight:600;color:#e2e0d8;">Open Source AutoML Agent</span>'
    '<span style="font-family:JetBrains Mono,monospace;font-size:11px;background:#2e1065;color:#c084fc;padding:2px 8px;border-radius:4px;">OSS</span>'
    "</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Data")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        try:
            st.session_state.df = pd.read_csv(up)
            st.session_state.filename = up.name
            st.session_state.load_error = None
        except Exception as e:
            st.session_state.load_error = str(e)
    st.markdown("**Sample datasets**")
    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("Healthcare", use_container_width=True):
            p = _sample_path("sample_healthcare_classification.csv")
            if p.is_file():
                st.session_state.df = pd.read_csv(p)
                st.session_state.filename = p.name
    with sc2:
        if st.button("Titanic", use_container_width=True):
            p = _sample_path("titanic_demo_synth.csv")
            if p.is_file():
                st.session_state.df = pd.read_csv(p)
                st.session_state.filename = p.name
    sc3, sc4 = st.columns(2)
    with sc3:
        if st.button("Housing", use_container_width=True):
            p = _sample_path("sample_housing_regression.csv")
            if p.is_file():
                st.session_state.df = pd.read_csv(p)
                st.session_state.filename = p.name
    with sc4:
        if st.button("Diabetes", use_container_width=True):
            p = _sample_path("diabetes_sklearn_demo.csv")
            if p.is_file():
                st.session_state.df = pd.read_csv(p)
                st.session_state.filename = p.name
    st.markdown("---")
    st.markdown("### Goal")
    goal = st.text_area(
        "What do you want to predict?",
        value=st.session_state.get("goal", ""),
        height=100,
        placeholder="e.g. predict readmission risk from patient features",
        key="goal_area",
    )
    st.session_state.goal = goal
    run = st.button("Run pipeline", type="primary", use_container_width=True)

if st.session_state.load_error:
    st.error(st.session_state.load_error)

df = st.session_state.df

if df is None:
    st.markdown("### Open Source AutoML Agent")
    st.caption("Powered by Qwen2.5 · No API key needed · Runs free on Hugging Face")
    st.markdown(
        """
1. **Upload any CSV** or use a sample dataset from the sidebar  
2. **Describe what you want to predict**  
3. **Click Run** — the agent runs the full AutoML pipeline  
"""
    )
    if st.button("Try it now — load Healthcare sample"):
        p = _sample_path("sample_healthcare_classification.csv")
        if p.is_file():
            st.session_state.df = pd.read_csv(p)
            st.session_state.filename = p.name
            st.rerun()
    st.stop()

st.caption(f"Dataset: **{st.session_state.filename}** · {len(df):,} rows × {df.shape[1]} columns")

tab_pipeline, tab_inference = st.tabs(["Pipeline", "Inference"])

with tab_pipeline:
    if run:
        st.session_state.run_error = None
        st.session_state.pipeline_events = []
        st.session_state.final_result = None
        st.session_state.log_lines_oss = []
        st.session_state.report_export = None
        st.session_state.oss_model_pkl_bytes = None
        try:
            with st.spinner("Loading Qwen2.5 — this may take ~30s on first run..."):
                pipe = get_llm_pipeline()
        except Exception as e:
            st.session_state.run_error = f"Failed to load model: {e}"
            st.error(st.session_state.run_error)
            st.stop()

        agent = OssAutoMLAgent(df, st.session_state.goal, pipe)
        all_events: list = []
        log_lines: list[str] = []
        pipe_ph = st.empty()
        log_ph = st.empty()
        step_dones: list[dict] = []

        try:
            for ev in agent.run():
                all_events.append(ev)
                et = ev.get("type")
                if et == "log":
                    ts = datetime.now().strftime("%H:%M:%S")
                    line = f"[{ts}] {ev.get('content', '')}"
                    log_lines.append(line)
                    log_ph.markdown(_log_html(log_lines), unsafe_allow_html=True)
                elif et == "step_done":
                    step_dones.append(ev)
                    with pipe_ph.container():
                        for s in step_dones:
                            _step_card_header(s.get("name", ""), s.get("step", ""))
                            _render_step_body(s)
                elif et == "done":
                    st.session_state.final_result = ev.get("result")

            st.session_state.pipeline_events = all_events
            st.session_state.log_lines_oss = log_lines
        except Exception as e:
            st.session_state.run_error = str(e)
            st.error(f"Pipeline error: {e}")

        st.rerun()

    if st.session_state.run_error and not run:
        st.error(st.session_state.run_error)

    st.markdown(
        '<p class="section-head">Pipeline steps</p>',
        unsafe_allow_html=True,
    )

    for ev in st.session_state.get("pipeline_events") or []:
        if ev.get("type") == "step_done":
            _step_card_header(ev.get("name", ""), ev.get("step", ""))
            _render_step_body(ev)

    log_lines = st.session_state.get("log_lines_oss") or []
    if log_lines:
        with st.expander("Activity log", expanded=False):
            st.markdown(_log_html(log_lines), unsafe_allow_html=True)

    fr = st.session_state.get("final_result")
    if fr and fr.get("status") == "complete":
        st.markdown("---")
        st.markdown("### Results")
        result = fr
        metrics = result.get("metrics", {})
        task = result.get("task_type", "")
        best = result.get("best_model_name", "—")

        cards_html = (
            '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin:16px 0;">'
            f'<div style="background:#141416;border:1px solid #2a2a2e;border-radius:8px;padding:16px;">'
            f'<div style="font-size:10px;color:#888780;">Best model</div>'
            f'<div style="font-size:20px;color:#7c3aed;font-weight:600;">{html_mod.escape(str(best))}</div></div>'
            f'<div style="background:#141416;border:1px solid #2a2a2e;border-radius:8px;padding:16px;">'
            f'<div style="font-size:10px;color:#888780;">Task</div>'
            f'<div style="font-size:18px;">{html_mod.escape(str(task).capitalize())}</div></div>'
            f'<div style="background:#141416;border:1px solid #2a2a2e;border-radius:8px;padding:16px;">'
            f'<div style="font-size:10px;color:#888780;">Target</div>'
            f'<div style="font-size:18px;">{html_mod.escape(str(result.get("target_col", "—")))}</div></div>'
            "</div>"
        )
        st.markdown(cards_html, unsafe_allow_html=True)

        tab_model, tab_plots, tab_feat, tab_data = st.tabs(
            ["Model comparison", "Plots", "Features", "Data profile"]
        )

        with tab_model:
            comp_df = result.get("comparison_df")
            if comp_df is not None:
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
            tr = result.get("train") or {}
            if tr.get("training_log"):
                with st.expander("Training log", expanded=False):
                    st.code("\n".join(tr["training_log"]), language=None)

        with tab_plots:
            pp = result.get("plot_paths") or {}
            base_order = [
                "confusion_matrix",
                "roc_curve",
                "actual_vs_predicted",
                "residuals",
                "feature_importance",
            ]
            ordered = [p for p in base_order if p in pp]
            ordered += [p for p in pp if p not in ordered and not str(p).startswith("shap_")]
            cols = st.columns(2)
            for i, name in enumerate(ordered):
                path = pp.get(name)
                if path and Path(path).exists():
                    cols[i % 2].image(str(path), caption=name.replace("_", " ").title(), use_container_width=True)
            render_shap_ui(pp)

        with tab_feat:
            fi = result.get("feature_importances") or {}
            if fi:
                top = sorted(fi.items(), key=lambda x: -x[1])[:15]
                chart_df = pd.DataFrame(
                    {"importance": [float(t[1]) for t in top]},
                    index=[str(t[0]) for t in top],
                )
                st.bar_chart(chart_df)
            else:
                st.info("No feature importance available.")

        with tab_data:
            eda_full = run_eda(df, target_col=result.get("target_col"))
            render_step_1_eda(eda_full)

        st.markdown("### Export")
        ex1, ex2 = st.columns(2)
        with ex1:
            if st.button("Generate Report", key="oss_gen_report", use_container_width=True):
                try:
                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                    md_text = _build_markdown(result)
                    html_text = _build_html(result)
                    html_path = OUTPUT_DIR / "automl_report.html"
                    md_path = OUTPUT_DIR / "automl_report.md"
                    html_path.write_text(html_text, encoding="utf-8")
                    md_path.write_text(md_text, encoding="utf-8")
                    generate_report(
                        result.get("goal", ""),
                        st.session_state.get("filename", "dataset"),
                        result.get("metrics", {}),
                        str(result.get("best_model_name", "")),
                        output_dir=OUTPUT_DIR,
                    )
                    st.session_state.report_export = {
                        "html_path": str(html_path.resolve()),
                        "md_path": str(md_path.resolve()),
                        "size_kb": html_path.stat().st_size / 1024.0,
                        "n_plots": count_embedded_plots_html(html_text),
                        "html_content": html_text,
                        "md_content": md_text,
                    }
                except Exception as ex:
                    st.error(str(ex))
        with ex2:
            if st.button("Save Model", key="oss_save_model", use_container_width=True):
                try:
                    prep_raw = result.get("prep_raw")
                    tr = result.get("train")
                    if not prep_raw or not tr:
                        st.error("Pipeline artifacts missing.")
                    else:
                        run_id = datetime.now().strftime("oss_%Y%m%d_%H%M%S")
                        fn = result.get("feature_names") or prep_raw.get("feature_names")
                        X_raw = _original_feature_frame(df, prep_raw, result["target_col"])
                        path = save_model(
                            pipeline=prep_raw["pipeline"],
                            model=result.get("model_for_inference"),
                            label_encoder=prep_raw.get("label_encoder"),
                            feature_names=list(fn) if fn is not None else [],
                            task_type=result["task_type"],
                            target_col=result["target_col"],
                            best_metrics=result.get("best_metrics") or {},
                            model_name=str(result.get("best_model_name", "model")),
                            run_id=run_id,
                            X_train=X_raw,
                            num_cols=prep_raw.get("num_cols"),
                            cat_cols=prep_raw.get("cat_cols"),
                            n_training_rows=(result.get("prep") or {}).get("train_size"),
                        )
                        st.session_state.saved_model_path = path
                        st.session_state["oss_saved_model_path"] = path
                        with open(path, "rb") as mf:
                            st.session_state.oss_model_pkl_bytes = mf.read()
                        st.success(f"Model saved to `{path}`. Use the download button below to copy it to your computer.")
                except Exception as ex:
                    st.error(str(ex))

        exp = st.session_state.get("report_export")
        if exp and exp.get("html_content"):
            html_content = exp["html_content"]
            md_content = exp.get("md_content") or ""
            html_size_kb = len(html_content) / 1024
            st.success(
                f"Report generated successfully — "
                f"{html_size_kb:.1f} KB with all plots embedded. "
                f"Click the buttons below to download."
            )
            dh, dm = st.columns(2)
            with dh:
                st.download_button(
                    label="⬇ Download HTML Report",
                    data=html_content,
                    file_name="automl_report.html",
                    mime="text/html",
                    key="download_html_report",
                )
            with dm:
                st.download_button(
                    label="⬇ Download Markdown Report",
                    data=md_content,
                    file_name="automl_report.md",
                    mime="text/markdown",
                    key="download_md_report",
                )

        mb = st.session_state.get("oss_model_pkl_bytes")
        if mb:
            st.download_button(
                label="⬇ Download Model (.pkl)",
                data=mb,
                file_name="automl_model.pkl",
                mime="application/octet-stream",
                key="download_model_pkl",
            )


with tab_inference:
    render_inference_tab()

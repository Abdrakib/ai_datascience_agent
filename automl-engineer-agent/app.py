"""
app.py — Explainable ML Pipeline Agent · Streamlit UI

Run: streamlit run app.py
"""

from __future__ import annotations

import copy
import html
import io
import json
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from agent.report import (
    _build_html,
    _build_markdown,
    _generate_next_steps,
    count_embedded_plots_html,
)
from config import OUTPUT_DIR


def _init_state() -> None:
    defaults = {
        "df":             None,
        "filename":       "",
        "result":         None,
        "log_lines":      [],
        "step_cards":     [],
        "pipeline_track": [],
        "running":        False,
        "error":          None,
        "theme":          "dark",
        "report_export":  None,
        "agent":          None,
        "saved_model_path": None,
        "inference_bundle": None,
        "inference_predictions": None,
        "demo_dataset":     "healthcare",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


APP_ROOT = Path(__file__).parent.resolve()


def _load_demo_json_file() -> dict | None:
    """Load the full demo snapshot JSON (version, demo_dataset_path, result, pipeline_track, log_lines)."""
    dataset = st.session_state.get("demo_dataset", "healthcare")
    candidates = [
        APP_ROOT / f"demo_result_{dataset}.json",
        APP_ROOT.parent / f"demo_result_{dataset}.json",
        APP_ROOT / "demo_result.json",
        APP_ROOT.parent / "demo_result.json",
    ]
    for path in candidates:
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


def _load_demo_result() -> dict | None:
    """Return the nested pipeline ``result`` dict from the demo JSON file."""
    raw = _load_demo_json_file()
    if raw is None:
        return None
    if isinstance(raw, dict) and "result" in raw:
        return raw["result"]
    return raw


def _hydrate_comparison_dfs(obj: Any) -> None:
    """Turn JSON list-of-rows into DataFrames wherever comparison_df appears."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if k == "comparison_df" and isinstance(v, list):
                obj[k] = pd.DataFrame(v)
            else:
                _hydrate_comparison_dfs(v)
    elif isinstance(obj, list):
        for item in obj:
            _hydrate_comparison_dfs(item)


def _resolve_plot_paths_relative(obj: Any, base: Path) -> None:
    """Resolve relative plot_paths entries against base (APP_ROOT)."""
    if isinstance(obj, dict):
        pp = obj.get("plot_paths")
        if isinstance(pp, dict):
            for pk, pv in list(pp.items()):
                if isinstance(pv, str) and pv.strip():
                    pth = Path(pv)
                    if not pth.is_absolute():
                        cand = (base / pv).resolve()
                        if cand.is_file():
                            pp[pk] = str(cand)
        for v in obj.values():
            _resolve_plot_paths_relative(v, base)
    elif isinstance(obj, list):
        for item in obj:
            _resolve_plot_paths_relative(item, base)


def _apply_demo_payload(data: dict) -> None:
    raw_res = copy.deepcopy(data["result"])
    _hydrate_comparison_dfs(raw_res)
    _resolve_plot_paths_relative(raw_res, APP_ROOT)
    st.session_state.result = raw_res

    track = copy.deepcopy(data.get("pipeline_track", []))
    for s in track:
        d = s.get("data")
        if isinstance(d, dict):
            _hydrate_comparison_dfs(d)
            _resolve_plot_paths_relative(d, APP_ROOT)
    st.session_state.pipeline_track = track

    st.session_state.log_lines = list(data.get("log_lines", []))
    st.session_state.step_cards = []
    st.session_state.error = None
    st.session_state.report_export = None
    st.session_state["agent"] = None
    st.session_state.running = False

    dp = data.get("demo_dataset_path")
    if dp:
        p = APP_ROOT / dp
        if p.is_file():
            st.session_state.df = pd.read_csv(p)
            st.session_state.filename = p.name
    goal = data.get("demo_goal", "")
    if goal:
        st.session_state["user_goal_input"] = goal


def _on_demo_mode_change() -> None:
    if st.session_state.get("demo_mode_toggle"):
        snap = _load_demo_json_file()
        if not snap:
            st.session_state["_demo_data_missing"] = True
        else:
            st.session_state["_demo_data_missing"] = False
            st.session_state.pop("_demo_snapshot_error", None)
            _apply_demo_payload(snap)
    else:
        st.session_state.pop("_demo_snapshot_error", None)
        st.session_state.pop("_demo_data_missing", None)
        st.session_state.result = None
        st.session_state.log_lines = []
        st.session_state.pipeline_track = []
        st.session_state.error = None
        st.session_state.report_export = None
        st.session_state["agent"] = None


def _pal() -> dict[str, str]:
    """UI colors for inline HTML; matches dark/light theme."""
    if st.session_state.get("theme", "dark") == "light":
        return {
            "text": "#111111",
            "muted": "#525252",
            "muted2": "#737373",
            "accent": "#6d28d9",
            "accent_soft": "#7c3aed",
            "blue": "#2563eb",
            "green": "#15803d",
            "red": "#dc2626",
            "amber": "#b45309",
            "border": "#d4d4d4",
            "card_bg": "#ffffff",
            "pre_bg": "#f5f5f5",
            "section": "#737373",
            "summary_bg": "#f4f4f5",
            "summary_border": "#e4e4e7",
            "empty_icon": "#d4d4d4",
            "empty_sub": "#6b7280",
            "empty_body": "#374151",
        }
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
        "empty_icon": "#2a2a2e",
        "empty_sub": "#444441",
        "empty_body": "#333330",
    }


# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Explainable ML Pipeline Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_state()
if "theme_light_toggle" not in st.session_state:
    st.session_state.theme_light_toggle = st.session_state.get("theme", "dark") == "light"
st.session_state.theme = "light" if st.session_state.theme_light_toggle else "dark"

# ── Custom CSS (dark base + optional light overrides) ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark theme overrides */
.stApp {
    background-color: #0e0e10;
    color: #e2e0d8;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #141416;
    border-right: 1px solid #2a2a2e;
}

/* Header strip */
.agent-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    padding: 8px 0 20px 0;
    border-bottom: 1px solid #2a2a2e;
    margin-bottom: 24px;
}
.agent-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: #e2e0d8;
    letter-spacing: -0.5px;
}
.agent-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    background: #1e3a2a;
    color: #4ade80;
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid #2d5e3e;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.metric-card {
    background: #141416;
    border: 1px solid #2a2a2e;
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
}
.metric-label {
    font-size: 10px;
    color: #888780;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 26px;
    font-weight: 600;
    color: #7c3aed;
}
.metric-value.green  { color: #4ade80; }
.metric-value.amber  { color: #fbbf24; }
.metric-value.coral  { color: #fb7185; }

/* Activity log */
.log-container {
    background: #0a0a0c;
    border: 1px solid #2a2a2e;
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    max-height: 420px;
    overflow-y: auto;
}
.log-line { padding: 3px 0; line-height: 1.6; }
.log-text  { color: #a8a49c; }
.log-tool-run  { color: #60a5fa; }
.log-tool-done { color: #4ade80; }
.log-error     { color: #fb7185; }
.log-ts { color: #444441; margin-right: 8px; }

/* Tool chip */
.tool-chip {
    display: inline-block;
    background: #1a1a1e;
    border: 1px solid #3a3a3e;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: #c084fc;
    margin: 0 2px;
}

/* Section heading */
.section-head {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #444441;
    margin: 28px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #2a2a2e;
}

/* Feature importance bar */
.fi-bar-bg {
    background: #1a1a1e;
    border-radius: 3px;
    height: 8px;
    margin-top: 4px;
}
.fi-bar-fill {
    height: 8px;
    border-radius: 3px;
    background: #7c3aed;
}

/* Dataframe override */
[data-testid="stDataFrame"] {
    background: #141416;
}

/* Buttons — keep label horizontal in narrow columns */
.stButton > button {
    background: #1e1e22;
    color: #e2e0d8;
    border: 1px solid #3a3a3e;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    padding: 8px 20px;
    transition: all 0.15s;
    white-space: nowrap;
}
.stButton > button:hover {
    background: #7c3aed;
    border-color: #7c3aed;
    color: #fff;
}
.stButton > button[kind="primary"] {
    background: #7c3aed;
    border-color: #7c3aed;
    color: #fff;
}
.stButton > button[kind="primary"]:hover {
    background: #6d28d9;
    border-color: #6d28d9;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #141416;
    border: 1px dashed #3a3a3e;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #141416 !important;
    border-bottom: 2px solid #3a3a3e !important;
    padding: 4px 0 0 0 !important;
    margin-bottom: 16px !important;
}

.stTabs [data-baseweb="tab"] {
    color: #aaa8a0 !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 10px 28px !important;
    background-color: transparent !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
}

.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 3px solid #7c3aed !important;
    background-color: transparent !important;
}

.stTabs [data-baseweb="tab"]:hover {
    color: #ffffff !important;
    background-color: #1e1e22 !important;
    border-radius: 4px 4px 0 0 !important;
}

/* Input */
.stTextArea textarea, .stTextInput input, .stSelectbox select {
    background: #141416 !important;
    border-color: #2a2a2e !important;
    color: #e2e0d8 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Step cards */
.step-card {
    background: #141416;
    border: 1px solid #2a2a2e;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 20px;
    font-family: 'DM Sans', sans-serif;
}
.step-card-failed {
    background: rgba(251, 113, 133, 0.08);
    border-color: #fb7185;
}
.step-card-waiting {
    opacity: 0.65;
}
.step-card table {
    border-collapse: collapse;
    margin: 12px 0;
    font-size: 13px;
}
.step-card th, .step-card td {
    padding: 8px 12px;
    border: 1px solid #2a2a2e;
    text-align: left;
}
.step-card th { background: #1a1a1e; color: #888780; }
.step-card td { color: #e2e0d8; }
.step-card-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid #2a2a2e;
}
.step-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 600;
}
.step-badge-waiting { background: #2a2a2e; color: #888780; border: 1px solid #3a3a3e; }
.step-badge-running { background: #1e3a5f; color: #60a5fa; border: 1px solid #2d5e8e; }
.step-badge-done { background: #1e3a2a; color: #4ade80; border: 1px solid #2d5e3e; }
.step-badge-failed { background: #4a1e1e; color: #fb7185; border: 1px solid #6e2d2d; }
.warning-banner {
    padding: 12px 16px;
    border-radius: 6px;
    margin: 12px 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
}
.warning-banner-amber { background: rgba(251, 191, 36, 0.15); border: 1px solid #fbbf24; color: #fbbf24; }
.warning-banner-red { background: rgba(251, 113, 133, 0.15); border: 1px solid #fb7185; color: #fb7185; }

/* Hide menu/footer only — do NOT hide the top header or the sidebar toggle becomes inaccessible */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding-top: 24px; padding-bottom: 40px; }
</style>
""", unsafe_allow_html=True)

if st.session_state.get("theme", "dark") == "light":
    st.markdown(
        """
<style>
/* Light theme — professional, high contrast */
.stApp { background-color: #ffffff !important; color: #111111 !important; }
html, body, [class*="css"] { color: #111111 !important; }
[data-testid="stSidebar"] { background-color: #fafafa !important; border-right: 1px solid #d4d4d4 !important; }
[data-testid="stSidebar"] * { color: #111111 !important; }
.agent-header { border-bottom-color: #d4d4d4 !important; }
.agent-title { color: #111111 !important; }
.agent-badge { background: #ecfdf5 !important; color: #15803d !important; border-color: #bbf7d0 !important; }
.metric-card { background: #ffffff !important; border: 1px solid #d4d4d4 !important; }
.metric-label { color: #525252 !important; }
.metric-value { color: #6d28d9 !important; }
.metric-value.green { color: #15803d !important; }
.metric-value.amber { color: #b45309 !important; }
.metric-value.coral { color: #dc2626 !important; }
.log-container { background: #f5f5f5 !important; border: 1px solid #d4d4d4 !important; }
.log-text { color: #525252 !important; }
.log-tool-run { color: #2563eb !important; }
.log-tool-done { color: #15803d !important; }
.log-error { color: #dc2626 !important; }
.log-ts { color: #a3a3a3 !important; }
.tool-chip { background: #f4f4f5 !important; border-color: #d4d4d4 !important; color: #6d28d9 !important; }
.section-head { color: #737373 !important; border-bottom-color: #d4d4d4 !important; }
.fi-bar-bg { background: #e5e5e5 !important; }
.fi-bar-fill { background: #6d28d9 !important; }
[data-testid="stDataFrame"] { background: #ffffff !important; }
.stButton > button { background: #f4f4f5 !important; color: #111111 !important; border: 1px solid #d4d4d4 !important; white-space: nowrap !important; }
.stButton > button:hover { background: #6d28d9 !important; border-color: #6d28d9 !important; color: #fff !important; }
.stButton > button[kind="primary"] { background: #6d28d9 !important; border-color: #6d28d9 !important; color: #fff !important; }
[data-testid="stFileUploader"] { background: #ffffff !important; border: 1px dashed #d4d4d4 !important; }
.stTabs [data-baseweb="tab-list"] { background: #fafafa !important; border-bottom: 1px solid #d4d4d4 !important; }
.stTabs [data-baseweb="tab"] { color: #525252 !important; }
.stTabs [aria-selected="true"] { color: #6d28d9 !important; border-bottom-color: #6d28d9 !important; }
.stTextArea textarea, .stTextInput input, .stSelectbox select {
    background: #ffffff !important; border-color: #d4d4d4 !important; color: #111111 !important;
}
.step-card { background: #ffffff !important; border: 1px solid #d4d4d4 !important; color: #111111 !important; }
.step-card-failed { background: rgba(220, 38, 38, 0.06) !important; border-color: #f87171 !important; }
.step-card th { background: #f4f4f5 !important; color: #525252 !important; }
.step-card td { color: #111111 !important; border-color: #d4d4d4 !important; }
.step-card-header { border-bottom-color: #d4d4d4 !important; }
.step-badge-waiting { background: #e5e5e5 !important; color: #525252 !important; border-color: #d4d4d4 !important; }
.step-badge-running { background: #dbeafe !important; color: #1d4ed8 !important; border-color: #93c5fd !important; }
.step-badge-done { background: #dcfce7 !important; color: #15803d !important; border-color: #86efac !important; }
.step-badge-failed { background: #fee2e2 !important; color: #dc2626 !important; border-color: #fca5a5 !important; }
.warning-banner-amber { background: rgba(180, 83, 9, 0.1) !important; border-color: #d97706 !important; color: #b45309 !important; }
.warning-banner-red { background: rgba(220, 38, 38, 0.08) !important; border-color: #f87171 !important; color: #dc2626 !important; }
[data-testid="stMetric"] { background: transparent !important; }
[data-testid="stMetricValue"] { color: #111111 !important; }
[data-testid="stMetricLabel"] { color: #525252 !important; }
[data-testid="stExpander"] { background: #fafafa !important; border: 1px solid #d4d4d4 !important; }
.stAlert { color: #111111 !important; }
[data-testid="stToggle"] label { color: #111111 !important; }
[data-testid="stToggle"] label p { color: #111111 !important; }
</style>
""",
        unsafe_allow_html=True,
    )

STEP_NAMES = {
    "run_eda":        (1, "EDA"),
    "detect_task":    (2, "Task detection"),
    "preprocess":     (3, "Preprocessing"),
    "plan_training":  (4, "Training Plan"),
    "train_models":   (5, "Model training"),
    "tune_model":     (6, "Hyperparameter tuning"),
    "evaluate_model": (7, "Evaluation"),
    "final":          (8, "Final recommendation"),
}


def _new_pipeline_track() -> list[dict]:
    """Eight steps: seven tools + final recommendation (waiting until pipeline completes)."""
    return [
        {"step": 1, "name": "run_eda", "label": "EDA", "status": "waiting",
         "data": None, "error": None},
        {"step": 2, "name": "detect_task", "label": "Task detection", "status": "waiting",
         "data": None, "error": None},
        {"step": 3, "name": "preprocess", "label": "Preprocessing", "status": "waiting",
         "data": None, "error": None},
        {"step": 4, "name": "plan_training", "label": "Training Plan", "status": "waiting",
         "data": None, "error": None},
        {"step": 5, "name": "train_models", "label": "Model training", "status": "waiting",
         "data": None, "error": None},
        {"step": 6, "name": "tune_model", "label": "Hyperparameter tuning", "status": "waiting",
         "data": None, "error": None},
        {"step": 7, "name": "evaluate_model", "label": "Evaluation", "status": "waiting",
         "data": None, "error": None},
        {"step": 8, "name": "final", "label": "Final recommendation", "status": "waiting",
         "data": None, "error": None},
    ]


def _pipeline_track_update_running(name: str, tune_model_name: str | None = None):
    for s in st.session_state.pipeline_track:
        if s["name"] == name:
            s["status"] = "running"
            if name == "tune_model" and tune_model_name:
                s["running_detail"] = tune_model_name
            else:
                s["running_detail"] = None
            break


def _pipeline_track_update_done(name: str, step_data: dict | None):
    for s in st.session_state.pipeline_track:
        if s["name"] == name:
            s["status"] = "done"
            s["data"] = step_data
            s.pop("running_detail", None)
            break


def _pipeline_track_fail_running(err: str):
    for s in reversed(st.session_state.pipeline_track):
        if s["status"] == "running":
            s["status"] = "failed"
            s["error"] = err
            return


def _pipeline_track_finalize(result: dict):
    for s in st.session_state.pipeline_track:
        if s["name"] == "final":
            s["status"] = "done"
            s["data"] = result
            break


def _load_csv_from_upload(uploaded_file) -> None:
    """Load a new CSV from Streamlit's UploadedFile into session state."""
    if uploaded_file is None:
        return
    try:
        if uploaded_file.name != st.session_state.get("filename", ""):
            st.session_state.df = pd.read_csv(uploaded_file)
            st.session_state.filename = uploaded_file.name
            st.session_state.result = None
            st.session_state.log_lines = []
            st.session_state.step_cards = []
            st.session_state.pipeline_track = []
            st.session_state.error = None
    except Exception as e:
        st.error(f"Could not read CSV: {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ts() -> str:
    return time.strftime("%H:%M:%S")

def _log(html: str):
    st.session_state.log_lines.append(html)

def _log_text(msg: str):
    _log(f'<div class="log-line log-text"><span class="log-ts">{_ts()}</span>{msg}</div>')

def _log_tool(name: str, status: str, output: str = ""):
    if status == "running":
        _log(f'<div class="log-line log-tool-run"><span class="log-ts">{_ts()}</span>'
             f'▶ running <span class="tool-chip">{name}</span>...</div>')
    else:
        short = output[:120].replace("\n", " ") + ("…" if len(output) > 120 else "")
        _log(f'<div class="log-line log-tool-done"><span class="log-ts">{_ts()}</span>'
             f'✓ <span class="tool-chip">{name}</span> — {short}</div>')

def _log_error(msg: str):
    _log(f'<div class="log-line log-error"><span class="log-ts">{_ts()}</span>✗ {msg}</div>')

def _metric_card(label: str, value: str, cls: str = "") -> str:
    return (f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value {cls}">{value}</div>'
            f'</div>')

def _color_for_metric(name: str, val: float) -> str:
    name = name.lower()
    if "r2" in name or "auc" in name or "accuracy" in name:
        return "green" if val >= 0.8 else "amber" if val >= 0.6 else "coral"
    if "rmse" in name or "mae" in name or "mape" in name:
        return "amber"
    return ""


def _running_step_ui(name: str, running_detail: str | None) -> tuple[str, str]:
    """Return (spinner label, HTML paragraph) while a pipeline step is running."""
    p = _pal()
    p_style = (
        f"color:{p['blue']};font-family:'JetBrains Mono',monospace;font-size:13px;line-height:1.5;"
    )
    if name == "run_eda":
        return (
            "Running exploratory data analysis…",
            f'<p style="{p_style}">EDA: profiling columns, dtypes, missing values, and distributions…</p>',
        )
    if name == "detect_task":
        return (
            "Detecting task and target column…",
            f'<p style="{p_style}">Task detection: inferring classification vs regression and the target column…</p>',
        )
    if name == "preprocess":
        return (
            "Building preprocessing pipeline…",
            f'<p style="{p_style}">Preprocessing: encoding, scaling, and train/test split…</p>',
        )
    if name == "plan_training":
        return (
            "Building training plan…",
            f'<p style="{p_style}">Training plan: sizing models, metrics, and Optuna budget to your dataset…</p>',
        )
    if name == "train_models":
        return (
            "Training and comparing models…",
            f'<p style="{p_style}">Training: fitting and comparing multiple models on the training set…</p>',
        )
    if name == "tune_model":
        model = running_detail or "the best model"
        return (
            f"Hyperparameter tuning (Optuna): {model}…",
            f'<p style="{p_style}">Hyperparameter tuning · Optuna is optimizing <strong>{model}</strong> '
            f"(Bayesian search; multiple trials — may take a minute)…</p>",
        )
    if name == "evaluate_model":
        return (
            "Evaluating model and generating plots…",
            f'<p style="{p_style}">Evaluation: test-set metrics, confusion matrix / curves, and SHAP…</p>',
        )
    return (
        "Running pipeline step…",
        f'<p style="{p_style}">Executing…</p>',
    )


def _step_header_html(step: int, label: str, status: str) -> str:
    p = _pal()
    if status == "waiting":
        badge_cls, badge_text = "step-badge-waiting", "Waiting"
    elif status == "running":
        badge_cls, badge_text = "step-badge-running", "Running…"
    elif status == "done":
        badge_cls, badge_text = "step-badge-done", "✓ Done"
    else:
        badge_cls, badge_text = "step-badge-failed", "✗ Failed"
    return (
        f'<div class="step-card-header">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:14px;color:{p["muted"]};">Step {step}</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:16px;font-weight:600;color:{p["text"]};">{label}</span>'
        f'<span class="step-badge {badge_cls}">{badge_text}</span></div>'
    )


def _render_pipeline_step(s: dict):
    """Render one pipeline step from pipeline_track (waiting | running | done | failed)."""
    step, name, label, status = s["step"], s["name"], s["label"], s["status"]
    data, error = s.get("data"), s.get("error")
    header = _step_header_html(step, label, status)
    card_cls = "step-card step-card-failed" if status == "failed" else "step-card"
    if status == "waiting":
        pm = _pal()["muted"]
        st.markdown(f'<div class="step-card step-card-waiting">{header}'
                    f'<p style="color:{pm};font-family:\'DM Sans\',sans-serif;">Waiting for previous steps…</p></div>',
                    unsafe_allow_html=True)
        return
    if status == "running":
        spin_msg, run_html = _running_step_ui(name, s.get("running_detail"))
        st.markdown(f'<div class="{card_cls}">{header}</div>', unsafe_allow_html=True)
        with st.spinner(spin_msg):
            st.markdown(run_html, unsafe_allow_html=True)
        return
    if status == "failed" and error:
        pr = _pal()["red"]
        st.markdown(
            f'<div class="{card_cls}">{header}'
            f'<pre style="color:{pr};font-family:\'JetBrains Mono\',monospace;font-size:12px;white-space:pre-wrap;">'
            f'{str(error)[:4000]}</pre></div>',
            unsafe_allow_html=True,
        )
        return
    if status == "done" and data is not None:
        _render_step_content(step, name, data, header, card_cls)
        return
    if status == "done" and name == "plan_training" and (data is None or not data.get("plan")):
        st.markdown(f'<div class="{card_cls}">{header}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="step-card-body">'
            f'<p style="font-family:\'DM Sans\',sans-serif;color:{_pal()["amber"]};">'
            "Training plan could not be generated (see activity log). "
            "Training will continue with default model set and settings."
            "</p></div>",
            unsafe_allow_html=True,
        )
        return
    st.markdown(f'<div class="{card_cls}">{header}</div>', unsafe_allow_html=True)


def _render_step_content(step: int, name: str, data: dict, header: str, card_cls: str = "step-card"):
    st.markdown(f'<div class="{card_cls}">{header}</div>', unsafe_allow_html=True)
    if name == "run_eda" and "eda" in data:
        _render_step_1_eda(data["eda"])
    elif name == "detect_task" and "task" in data:
        _render_step_2_task(data["task"], data.get("domain_research"))
    elif name == "preprocess" and "prep" in data:
        _render_step_3_prep(data["prep"])
    elif name == "plan_training" and data.get("plan"):
        _render_step_4_plan_training(data["plan"])
    elif name == "train_models" and "train" in data:
        _render_step_4_train(data["train"])
    elif name == "tune_model" and "tune" in data:
        _render_step_4b_tune(data["tune"])
    elif name == "evaluate_model" and "eval" in data:
        _render_step_5_eval(data["eval"])
    elif name == "final":
        _render_step_6_final(data)


def _render_step_1_eda(eda: dict):
    ov = eda.get("overview", {})
    miss = eda.get("missing", {})
    cols_prof = eda.get("columns", {})
    flags = eda.get("quality_flags", [])
    recs = eda.get("recommendations", [])
    target_info = eda.get("target_info")

    n_rows = ov.get("rows", 0)
    n_cols = ov.get("columns", 0)
    p = _pal()

    html = ""
    html += f'<p style="font-family:\'JetBrains Mono\',monospace;color:{p["text"]};">Dataset shape: {n_rows:,} rows × {n_cols} columns</p>'

    num_cols = [c for c, p in cols_prof.items() if p.get("dtype_group") == "numeric"]
    cat_cols = [c for c, p in cols_prof.items() if p.get("dtype_group") == "categorical"]
    html += f'<p><strong>Numeric columns:</strong> {", ".join(num_cols) if num_cols else "none"}</p>'
    html += f'<p><strong>Categorical columns:</strong> {", ".join(cat_cols) if cat_cols else "none"}</p>'
    html += f'<p><strong>Duplicate rows:</strong> {ov.get("duplicate_rows", 0)}</p>'

    if miss.get("by_column"):
        rows = []
        for col, info in miss["by_column"].items():
            pct = info.get("pct", 0)
            row_style = f"color:{p['red']}" if pct > 30 else f"color:{p['amber']}" if pct > 10 else ""
            rows.append(f"<tr><td>{col}</td><td>{info.get('count', 0)}</td><td style='{row_style}'>{pct:.1f}%</td></tr>")
        html += "<table><thead><tr><th>Column</th><th>Missing count</th><th>Missing %</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"

    if target_info and target_info.get("inferred_task") == "classification":
        dist = target_info.get("class_distribution", {})
        html += "<p><strong>Class distribution</strong></p><table><thead><tr><th>Class</th><th>Count</th></tr></thead><tbody>"
        for cls, cnt in dist.items():
            html += f"<tr><td>{cls}</td><td>{cnt}</td></tr>"
        html += "</tbody></table>"
        if "imbalance_ratio" in target_info:
            html += f"<p><strong>Imbalance ratio:</strong> {target_info['imbalance_ratio']}:1</p>"

    skewed = [(c, p.get("skewness")) for c, p in cols_prof.items()
              if p.get("dtype_group") == "numeric" and p.get("skewness") is not None
              and abs(float(p["skewness"])) > 2.0]
    if skewed:
        html += "<p><strong>Skewed columns (|skewness| &gt; 2.0):</strong></p><ul>"
        for c, s in skewed:
            html += f"<li>{c}: skewness = {s:.4f}</li>"
        html += "</ul>"

    if flags:
        html += "<p><strong>Quality flags</strong></p><ul>"
        for f in flags:
            html += f"<li>{f}</li>"
        html += "</ul>"
    if recs:
        html += "<p><strong>Preprocessing recommendations</strong></p><ul>"
        for r in recs:
            html += f"<li>{r}</li>"
        html += "</ul>"

    st.markdown(f'<div class="step-card-body">{html}</div>', unsafe_allow_html=True)

    if n_rows < 500:
        st.warning("Small dataset detected — model performance may be unreliable", icon="⚠️")
    if target_info and target_info.get("inferred_task") == "classification":
        ratio = target_info.get("imbalance_ratio", 0)
        if ratio and ratio > 5:
            st.error("Severe class imbalance detected — consider SMOTE or class_weight=balanced", icon="🚨")
    high_missing = [col for col, info in (miss.get("by_column") or {}).items() if info.get("pct", 0) > 30]
    if high_missing:
        st.warning("High missing rate detected in one or more columns", icon="⚠️")


def _render_step_2b_domain_research(domain_research: dict):
    """Separate card after task detection when confidence was low/medium."""
    p = _pal()
    header = (
        f'<div class="step-card-header">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:14px;color:{p["muted"]};">Step 2b</span>'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:16px;font-weight:600;color:{p["text"]};">'
        f'Domain Research</span>'
        f'<span class="step-badge step-badge-done">✓ Done</span></div>'
    )
    intro = (
        f'<p style="font-family:\'DM Sans\',sans-serif;color:{p["text"]};">'
        "The agent searched the web to better understand your dataset. "
        "Here is what it found:</p>"
    )
    q = html.escape(str(domain_research.get("query", "")))
    body = f'<p style="color:{p["muted"]};font-size:12px;"><strong>Query:</strong> {q}</p>'
    for i, r in enumerate(domain_research.get("results") or [], 1):
        if isinstance(r, dict) and r.get("error"):
            body += f'<p style="color:{p["amber"]};">{html.escape(str(r["error"]))}</p>'
            break
        if not isinstance(r, dict):
            continue
        title = html.escape(str(r.get("title", "")))
        url = html.escape(str(r.get("url", "")))
        sn = html.escape((r.get("snippet") or "")[:600])
        uhref = r.get("url") or ""
        body += (
            f'<div style="margin:12px 0;padding:10px;border-left:3px solid {p["accent"]};'
            f'background:#141416;border-radius:4px;">'
            f"<strong>{i}. {title}</strong><br/>"
            f'<a href="{html.escape(str(uhref), quote=True)}" target="_blank" rel="noopener noreferrer">'
            f"{url}</a><br/>"
            f'<span style="color:{p["muted"]};font-size:13px;">{sn}</span></div>'
        )
    st.markdown(
        f'<div class="step-card">{header}<div class="step-card-body">{intro}{body}</div></div>',
        unsafe_allow_html=True,
    )


def _render_step_2_task(task: dict, domain_research: dict | None = None):
    html = (
        f"<p><strong>Target column:</strong> <code>{task.get('target_col', '—')}</code></p>"
        f"<p><strong>Task type:</strong> {task.get('task_type', '—')}</p>"
        f"<p><strong>Confidence:</strong> {task.get('confidence', '—')}</p>"
        f"<p><strong>Reasoning:</strong> {task.get('reasoning', '—')}</p>"
    )
    if task.get("alternatives"):
        html += f"<p><strong>Alternative candidate columns:</strong> {', '.join(str(x) for x in task['alternatives'])}</p>"
    st.markdown(f'<div class="step-card-body">{html}</div>', unsafe_allow_html=True)
    if domain_research:
        _render_step_2b_domain_research(domain_research)


def _render_step_3_prep(prep: dict):
    html = f"<p><strong>Numeric columns used:</strong> {', '.join(prep.get('num_cols', [])) or 'none'}</p>"
    html += f"<p><strong>Categorical columns used:</strong> {', '.join(prep.get('cat_cols', [])) or 'none'}</p>"
    enc = prep.get("encoding_summary", {})
    html += "<p><strong>Encoding strategy:</strong></p><ul>"
    for col, strat in list(enc.items())[:20]:
        html += f"<li>{col}: {strat}</li>"
    html += "</ul>"
    if prep.get("dropped_cols"):
        html += "<p><strong>Columns dropped (see log for reasons):</strong></p><ul>"
        for c in prep["dropped_cols"]:
            html += f"<li>{c}</li>"
        html += "</ul>"
    log = prep.get("preprocessing_log", [])
    html += "<p><strong>Preprocessing log</strong></p>"
    for line in log:
        html += f"<p style='font-family:JetBrains Mono,monospace;font-size:12px;'>{line}</p>"
    n_feat = prep.get("final_feature_count")
    html += f"<p><strong>Final feature count (after encoding):</strong> {n_feat}</p>"
    if prep.get("train_size") is not None:
        html += (
            f"<p><strong>Train size:</strong> {prep['train_size']} rows · "
            f"<strong>Test size:</strong> {prep['test_size']} rows</p>"
        )
    st.markdown(f'<div class="step-card-body">{html}</div>', unsafe_allow_html=True)
    leak = prep.get("target_leakage_suspicion")
    if leak:
        m = re.search(r"Column '([^']+)'", leak) or re.search(r"column ([A-Za-z0-9_]+)", leak, re.I)
        col = m.group(1) if m else "unknown"
        st.error(
            f"Potential target leakage detected in column {col} — this may inflate your metrics",
            icon="🚨",
        )


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True) if s is not None else ""


def _plan_dataset_size_label(n_rows: int) -> str:
    if n_rows < 1000:
        return "Small (< 1000 rows)"
    if n_rows <= 10000:
        return "Medium"
    return "Large (> 10000 rows)"


def _plan_tuning_budget_why(n_rows: int) -> str:
    if n_rows < 1000:
        return (
            f"Only {n_rows} rows — fewer Optuna trials and a short timeout to limit "
            "overfitting risk and keep the UI responsive."
        )
    if n_rows <= 10000:
        return (
            f"At {n_rows} rows, a mid-sized budget balances search quality with runtime."
        )
    return (
        f"Large dataset ({n_rows:,} rows) — a higher trial count and longer timeout "
        "let Optuna explore the hyperparameter space properly."
    )


def _plan_why_included(model_name: str, plan: dict) -> str:
    """One-line, dataset-specific rationale for including a model."""
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
    return (
        f"Included for this run ({n:,} rows, {nf} features, primary metric {pm})."
    )


def _plan_skip_dataset_hook(plan: dict) -> str:
    dp = plan.get("dataset_profile") or {}
    n = int(dp.get("n_rows", 0))
    nf = int(dp.get("n_features", 0))
    return f"Context: your processed data has {n:,} rows and {nf} features after preprocessing."


def _render_step_4_plan_training(plan: dict) -> None:
    """Step 4 — Training Plan card (after preprocess, before model training)."""
    p = _pal()
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
        f"<tr><td>Rows</td><td>{_esc(n_rows)}</td></tr>"
        f"<tr><td>Features</td><td>{_esc(n_features)}</td></tr>"
        f"<tr><td>Dataset size</td><td>{_esc(_plan_dataset_size_label(n_rows))}</td></tr>"
        f"<tr><td>Task type</td><td>{'regression' if is_reg else 'classification'}</td></tr>"
    )
    if not is_reg:
        sec += (
            f"<tr><td>Class imbalance ratio</td><td>{_esc(f'{ir:.2f}')}</td></tr>"
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
        reason = _plan_why_included(m, plan)
        sec += (
            f'<p style="font-family:\'DM Sans\',sans-serif;margin:10px 0 4px 0;">'
            f'<span style="color:{p["green"]};font-size:16px;">✓</span> '
            f'<strong style="font-family:\'JetBrains Mono\',monospace;">{_esc(m)}</strong></p>'
            f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:{p["muted"]};margin:0 0 4px 0;">'
            f"Parameters: {_esc(param_str)}</p>"
            f'<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;color:{p["text"]};margin:0 0 12px 0;">'
            f"{_esc(reason)}</p>"
        )

    skip = plan.get("skip_models") or []
    reasons = plan.get("skip_reasons") or {}
    hook = _plan_skip_dataset_hook(plan)
    for m in skip:
        r = reasons.get(m, "Excluded by training plan rules.")
        sec += (
            f'<p style="font-family:\'DM Sans\',sans-serif;margin:10px 0 4px 0;">'
            f'<span style="color:{p["red"]};font-size:16px;">✗</span> '
            f'<strong style="font-family:\'JetBrains Mono\',monospace;">{_esc(m)}</strong></p>'
            f'<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;color:{p["text"]};margin:0 0 4px 0;">'
            f"{_esc(r)}</p>"
            f'<p style="font-family:\'DM Sans\',sans-serif;font-size:12px;color:{p["muted"]};margin:0 0 12px 0;">'
            f"{_esc(hook)}</p>"
        )

    pm = plan.get("primary_metric") or "—"
    mr = plan.get("metric_reasoning") or ""
    sec += (
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{p["accent_soft"]};'
        'margin:20px 0 10px 0;">SECTION 3 — Evaluation metric</p>'
        f'<p style="font-family:\'JetBrains Mono\',monospace;color:{p["text"]};">'
        f"Primary metric: <strong>{_esc(pm)}</strong></p>"
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;">{_esc(mr)}</p>'
    )

    nt = plan.get("n_trials", "—")
    to = plan.get("timeout", "—")
    sec += (
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{p["accent_soft"]};'
        'margin:20px 0 10px 0;">SECTION 4 — Tuning budget</p>'
        f'<p style="font-family:\'JetBrains Mono\',monospace;color:{p["text"]};">'
        f"Optuna trials: <strong>{_esc(nt)}</strong> · "
        f"Timeout: <strong>{_esc(to)}</strong> seconds</p>"
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;">'
        f"{_esc(_plan_tuning_budget_why(n_rows))}</p>"
    )

    warns = plan.get("warnings") or []
    notes = plan.get("notes") or []
    sec += (
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{p["accent_soft"]};'
        'margin:20px 0 10px 0;">SECTION 5 — Warnings and notes</p>'
    )
    for w in warns:
        sec += (
            f'<div class="warning-banner warning-banner-amber" style="margin-bottom:8px;">'
            f"⚠ {_esc(w)}</div>"
        )
    if notes:
        sec += f'<ul style="font-family:\'DM Sans\',sans-serif;color:{p["text"]};">'
        for note in notes:
            sec += f"<li>{_esc(note)}</li>"
        sec += "</ul>"
    elif not warns:
        sec += (
            f'<p style="font-family:\'DM Sans\',sans-serif;color:{p["muted"]};">'
            "No additional warnings or notes for this plan.</p>"
        )

    summary = plan.get("plan_summary") or ""
    sec += (
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{p["accent_soft"]};'
        'margin:20px 0 10px 0;">SECTION 6 — Plan summary</p>'
        f'<div style="background:{p["summary_bg"]};border:1px solid {p["summary_border"]};border-radius:8px;padding:16px 18px;">'
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:{p["muted"]};margin:0 0 8px 0;">'
        "Agent reasoning</p>"
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:14px;color:{p["text"]};margin:0;line-height:1.55;">'
        f"{_esc(summary)}</p>"
        "</div>"
    )

    st.markdown(f'<div class="step-card-body">{sec}</div>', unsafe_allow_html=True)


def _gap_threshold_for_task(task_type: str) -> float:
    return 0.15 if task_type == "classification" else 0.20


def _cv_reliability_label(cv_std: float | None) -> tuple[str, str]:
    if cv_std is None:
        return "—", _pal()["muted"]
    if cv_std < 0.02:
        return "Reliable", _pal()["green"]
    if cv_std <= 0.05:
        return "Moderate", _pal()["amber"]
    return "Unstable", _pal()["red"]


def _render_step_4_train(train: dict):
    results = train.get("results", [])
    comp_df = train.get("comparison_df")
    best_name = train.get("best_name", "")
    primary = train.get("metric_name", "roc_auc")
    task_type = train.get("task_type") or (
        "classification"
        if primary in ("roc_auc", "f1", "f1_weighted", "accuracy")
        else "regression"
    )
    gap_thr = _gap_threshold_for_task(task_type)
    overfit_warnings = train.get("overfitting_warnings", [])
    severe_note = any("All models showed severe overfitting" in w for w in overfit_warnings)

    if comp_df is not None and not comp_df.empty:
        try:
            if "Gap" in comp_df.columns:
                gp = _pal()

                def _gap_style(s: pd.Series):
                    hi_bg = "rgba(251,113,133,0.25)" if st.session_state.get("theme", "dark") == "dark" else "rgba(220,38,38,0.12)"
                    hi_fg = gp["red"]
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
        lbl, col = _cv_reliability_label(cv_std)
        reason = (
            "Least overfit model (all others exceeded severe gap threshold)."
            if severe_note
            else (
                "Highest CV mean on the primary metric (when CV ran); otherwise ranked by held-out test score."
            )
        )
        st.markdown(
            f'<p style="font-family:\'JetBrains Mono\',monospace;color:{_pal()["text"]};">'
            f"<strong>Selected best model:</strong> {best_name} "
            f'<span style="margin-left:10px;padding:2px 8px;border-radius:4px;font-size:12px;background:rgba(128,128,128,0.15);color:{col};border:1px solid {col};">{lbl}</span>'
            f" — {reason}</p>",
            unsafe_allow_html=True,
        )

    if results and any(r.get("cv_scores") for r in results):
        st.markdown("**Cross-validation details**")
        for r in results:
            scores = r.get("cv_scores")
            if not scores:
                continue
            st.markdown(f"*{r.get('name', 'Model')}* — scores per fold")
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


def _render_step_4b_tune(tune: dict):
    model = tune.get("model_name") or "—"
    if not tune.get("success", False):
        err = tune.get("error", "Unknown error")
        st.markdown(
            f'<p style="font-family:\'JetBrains Mono\',monospace;color:{_pal()["text"]};">'
            f"<strong>Model:</strong> {model}</p>",
            unsafe_allow_html=True,
        )
        st.error(f"Hyperparameter tuning failed: {err}")
        return

    bp = tune.get("best_params") or {}
    html = (
        f"<p><strong>Model tuned:</strong> {model}</p>"
        f"<p><strong>Baseline score (test):</strong> {tune.get('baseline_score', 0):.4f} · "
        f"<strong>After tuning:</strong> {tune.get('best_score', 0):.4f} · "
        f"<strong>Improvement:</strong> {tune.get('improvement', 0):+.4f}</p>"
        f"<p><strong>Optuna trials:</strong> {tune.get('n_trials_run', 0)} · "
        f"<strong>Time:</strong> {tune.get('tuning_time_s', 0):.1f}s · "
        f"<strong>Train–test gap:</strong> {tune.get('generalization_gap', 0):.4f}</p>"
    )
    if bp:
        html += "<p><strong>Best hyperparameters</strong></p><table><thead><tr><th>Parameter</th><th>Value</th></tr></thead><tbody>"
        for k, v in sorted(bp.items())[:24]:
            html += f"<tr><td><code>{k}</code></td><td>{v}</td></tr>"
        html += "</tbody></table>"
    st.markdown(f'<div class="step-card-body">{html}</div>', unsafe_allow_html=True)
    if tune.get("overfit"):
        st.warning(
            "Tuned model still shows an elevated train–test gap — consider more data or stronger regularization.",
            icon="⚠️",
        )


def _is_shap_plot_key(k: str) -> bool:
    return k in ("shap_bar", "shap_summary", "shap_waterfall") or k.startswith("shap_dependence_")


def _render_shap_dependence_deep_dive(plot_paths: dict) -> None:
    """Full-width SHAP dependence plots (dynamic keys shap_dependence_*)."""
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


def _render_shap_bar_and_summary(plot_paths: dict) -> None:
    """Mean |SHAP| bar and beeswarm summary."""
    if not any(plot_paths.get(k) for k in ("shap_bar", "shap_summary")):
        return

    st.markdown("---")
    st.markdown(
        '<p style="margin:24px 0 16px 0;">SHAP explainability</p>',
        unsafe_allow_html=True,
    )

    pbar = plot_paths.get("shap_bar")
    if pbar and Path(pbar).exists():
        st.markdown("### Feature Importance")
        st.caption(
            "Mean absolute SHAP values show which features push predictions the most, on average, "
            "across the explained samples."
        )
        st.image(pbar, use_container_width=True)
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    psum = plot_paths.get("shap_summary")
    if psum and Path(psum).exists():
        st.markdown("### Feature Impact Distribution")
        st.caption(
            "Each dot represents a data point. Red = high feature value, Blue = low. "
            "Position shows impact on prediction."
        )
        st.image(psum, use_container_width=True)
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)


def _render_shap_waterfall_only(plot_paths: dict) -> None:
    pw = plot_paths.get("shap_waterfall")
    if not pw or not Path(pw).exists():
        return
    st.markdown("#### Local explanation (one example row)")
    st.image(pw, use_container_width=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


def _render_shap_ui(plot_paths: dict) -> None:
    """SHAP: bar & summary → dependence deep dive → waterfall."""
    if not any(_is_shap_plot_key(k) and plot_paths.get(k) for k in plot_paths):
        return
    _render_shap_bar_and_summary(plot_paths)
    _render_shap_dependence_deep_dive(plot_paths)
    _render_shap_waterfall_only(plot_paths)


def _render_step_5_eval(eval_data: dict):
    metrics = eval_data.get("metrics", {})
    plot_paths = eval_data.get("plot_paths", {})
    task = eval_data.get("task_type", "classification")

    st.markdown(
        f'<p style="font-family:\'JetBrains Mono\',monospace;color:{_pal()["muted"]};">Evaluation metrics</p>',
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

    _render_shap_ui(plot_paths)


def _feature_interpretation_sentence(
    feat: str,
    rank: int,
    task_type: str,
    has_shap: bool,
) -> str:
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


def _render_what_model_learned(result: dict) -> None:
    """Top-5 importances + plain-English lines (uses feature_importances + eval SHAP flag)."""
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
        st.caption(_feature_interpretation_sentence(feat, i, task, has_shap))


def _render_step_6_final(result: dict):
    best = result.get("best_model_name", "—")
    metrics = result.get("best_metrics", {})
    task = result.get("task_type", "")
    primary = "roc_auc" if task == "classification" else "r2"
    train_data = result.get("train", {})
    overfit_warnings = list(train_data.get("overfitting_warnings", []))
    search_results = result.get("overfitting_search_results") or train_data.get("overfitting_search_results", [])

    st.markdown(
        f'<div class="metric-card" style="margin-bottom:16px;">'
        f'<div class="metric-label">Best model</div>'
        f'<div class="metric-value" style="font-size:22px;">{best}</div>'
        f'<p style="font-family:\'JetBrains Mono\',monospace;margin-top:12px;color:{_pal()["muted"]};">'
        f'Primary metric ({primary}): {metrics.get(primary, 0):.4f}</p></div>',
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

    if search_results and overfit_warnings:
        st.markdown("**Suggestions for fixing overfitting (web search)**")
        for r in search_results[:8]:
            if isinstance(r, dict) and "error" not in r:
                st.markdown(
                    f"- **{r.get('title', '')}**  \n  {r.get('snippet', '')[:400]}",
                )

    _render_what_model_learned(result)

    st.markdown("**Recommended next actions**")
    for a in _generate_next_steps(result):
        st.markdown(f"- {a}")

    md = _build_markdown(result)
    html = _build_html(result)
    n_embedded = count_embedded_plots_html(html)
    html_path = OUTPUT_DIR / "automl_report.html"
    md_path = OUTPUT_DIR / "automl_report.md"

    st.markdown("**Export**")
    ex1, ex2 = st.columns(2)
    with ex1:
        if st.button("Generate & save report to disk", key="btn_save_pipeline_reports", use_container_width=True):
            try:
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                html_path.write_text(html, encoding="utf-8")
                md_path.write_text(md, encoding="utf-8")
                sz_kb = html_path.stat().st_size / 1024.0
                st.session_state.report_export = {
                    "html_path": str(html_path.resolve()),
                    "size_kb": sz_kb,
                    "n_plots": n_embedded,
                }
            except Exception as ex:
                st.error(str(ex))
                st.session_state.report_export = None
    with ex2:
        if st.button("Save model", key="btn_save_model_bundle", use_container_width=True):
            from predict import save_model

            agent = st.session_state.get("agent")
            r = st.session_state.get("result")
            if agent is None or r is None or r.get("status") != "complete":
                st.error("Run a complete pipeline first, then save the model.")
            else:
                try:
                    prep = agent._prep_result
                    tr = agent._train_result
                    if (
                        prep is None
                        or tr is None
                        or prep.get("pipeline") is None
                        or tr.get("best_model") is None
                    ):
                        st.error("Model artifacts are not available. Re-run the pipeline.")
                    else:
                        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        fn = (
                            (r.get("prep") or {}).get("feature_names")
                            or prep.get("feature_names")
                        )
                        X_raw = _original_feature_frame_from_agent(agent)
                        train_size = (r.get("prep") or {}).get("train_size")
                        path = save_model(
                            pipeline=prep["pipeline"],
                            model=tr["best_model"],
                            label_encoder=prep.get("label_encoder"),
                            feature_names=fn,
                            task_type=r["task_type"],
                            target_col=r["target_col"],
                            best_metrics=r["best_metrics"],
                            model_name=r["best_model_name"],
                            run_id=run_id,
                            X_train=X_raw,
                            num_cols=prep.get("num_cols"),
                            cat_cols=prep.get("cat_cols"),
                            n_training_rows=int(train_size) if train_size is not None else None,
                        )
                        st.session_state["saved_model_path"] = path
                        st.success(f"Model saved to outputs/{run_id}_model.pkl")
                except Exception as ex:
                    st.error(str(ex))
    exp = st.session_state.get("report_export")
    if exp:
        st.success(
            f"Saved HTML report to `{exp['html_path']}` — **{exp['size_kb']:.1f} KB**, "
            f"**{exp['n_plots']}** plot(s) embedded. "
            "The HTML report is fully self-contained — all plots are embedded. Safe to email or share."
        )

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download HTML",
            data=html,
            file_name="automl_report.html",
            mime="text/html",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download Markdown",
            data=md,
            file_name="automl_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    preview_lines = md.splitlines()[:50]
    with st.expander("Preview report summary", expanded=False):
        st.code("\n".join(preview_lines) + ("\n…" if len(md.splitlines()) > 50 else ""), language="markdown")


def _original_feature_frame_from_agent(agent) -> pd.DataFrame | None:
    """Raw feature columns as used before the ColumnTransformer (for save_model stats)."""
    prep = getattr(agent, "_prep_result", None)
    task = getattr(agent, "_task_result", None)
    if prep is None or task is None:
        return None
    target = task["target_col"]
    num = prep.get("num_cols") or []
    cat = prep.get("cat_cols") or []
    cols = list(num) + list(cat)
    if not cols:
        return None
    X = agent.df.drop(columns=[target], errors="ignore")
    use = [c for c in cols if c in X.columns]
    if not use:
        return None
    return X[use].copy()


def _render_inference_tab() -> None:
    """Inference UI: load model, predict, explain."""
    p = _pal()
    st.markdown(
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:18px;color:{p["accent"]};">'
        f"Inference</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:14px;color:{p["muted"]};">'
        f"Load a saved model and run predictions on new data.</p>",
        unsafe_allow_html=True,
    )

    bundle = st.session_state.get("inference_bundle")

    st.markdown("### Load a trained model")
    saved = st.session_state.get("saved_model_path")
    if saved and Path(saved).exists():
        mname = "session model"
        ag = st.session_state.get("agent")
        r = st.session_state.get("result") or {}
        if r.get("best_model_name"):
            mname = str(r["best_model_name"])
        if st.button(f"Use current session model: {mname}", key="inf_use_session_pkl"):
            try:
                from predict import load_model

                st.session_state["inference_bundle"] = load_model(saved)
                st.session_state["inference_predictions"] = None
                st.success("Model loaded from this session.")
                st.rerun()
            except Exception as ex:
                st.error(str(ex))

    up = st.file_uploader("Upload a saved model (.pkl)", type=["pkl"], key="inf_model_upload")
    if up is not None:
        try:
            from predict import load_model

            tmp = Path(tempfile.gettempdir()) / f"inf_model_{up.name}"
            tmp.write_bytes(up.getvalue())
            st.session_state["inference_bundle"] = load_model(str(tmp))
            st.session_state["inference_predictions"] = None
            st.success("Model loaded from file.")
        except Exception as ex:
            st.error(str(ex))

    if bundle is None:
        st.info(
            "Train a model in the Pipeline tab first, then save it to use here — "
            "or upload a `.pkl` produced by this app."
        )
        return

    from predict import get_model_summary, predict

    summary = get_model_summary(bundle)
    st.markdown(
        f'<div class="metric-card" style="margin:16px 0;">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:{p["green"]};">'
        f"● Model loaded</span>"
        f'<p style="font-family:\'JetBrains Mono\',monospace;margin:12px 0 4px 0;color:{p["text"]};">'
        f'<strong>{summary["model_name"]}</strong></p>'
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;color:{p["muted"]};">'
        f"Task: {summary['task_type']} · Target: {summary['target_col']} · "
        f"Encoded features: {summary['n_features']} · "
        f"Input columns: {len(summary['original_features'])}</p>"
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:12px;color:{p["muted"]};">'
        f"Trained: {summary['training_date']} · Rows in training: {summary['n_training_rows']}</p>"
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:{p["muted"]};">'
        f"Metrics: {', '.join(f'{k}={v}' for k, v in list(summary['metrics'].items())[:8])}</p>"
        f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:11px;color:{p["muted"]};">'
        f"Expected columns: {', '.join(summary['expected_input_columns'][:24])}"
        f"{'…' if len(summary['expected_input_columns']) > 24 else ''}</p></div>",
        unsafe_allow_html=True,
    )

    expected = summary["expected_input_columns"]
    target_col = bundle["target_col"]
    task_type = bundle["task_type"]
    num_cols = bundle.get("num_cols") or []
    cat_cols = bundle.get("cat_cols") or []
    means = bundle.get("feature_means") or {}
    modes = bundle.get("feature_modes") or {}
    cats = bundle.get("categorical_uniques") or {}

    st.markdown("### Upload new data for prediction")
    pred_df: pd.DataFrame | None = None
    csv_up = st.file_uploader("Upload CSV for prediction", type=["csv"], key="inf_csv_pred")
    csv_ready: pd.DataFrame | None = None
    if csv_up is not None:
        csv_ready = pd.read_csv(csv_up)
        st.dataframe(csv_ready.head(5), use_container_width=True)
        miss = [c for c in expected if c not in csv_ready.columns]
        if miss:
            st.warning(f"Missing expected feature columns (will be imputed): {miss}")
        if target_col in csv_ready.columns:
            st.warning(
                f"Target column **{target_col}** is present — it will be ignored for prediction."
            )
        if st.button("Run prediction on CSV", key="inf_run_csv_pred"):
            pred_df = csv_ready

    n_orig = len(num_cols) + len(cat_cols) if (num_cols or cat_cols) else len(expected)
    if csv_up is None and n_orig <= 10:
        st.markdown("**Or enter a single row manually**")
        vals: dict[str, Any] = {}
        with st.form("inf_manual_form"):
            for col in num_cols:
                if col not in expected:
                    continue
                default = float(means.get(col, 0.0))
                vals[col] = st.number_input(
                    col,
                    value=default,
                    format="%.6f",
                    key=f"inf_num_{col}",
                )
            for col in cat_cols:
                if col not in expected:
                    continue
                opts = cats.get(col) or [str(modes.get(col, ""))]
                vals[col] = st.selectbox(col, opts, key=f"inf_cat_{col}")
            submitted = st.form_submit_button("Predict")
        if submitted:
            pred_df = pd.DataFrame([vals])

    if pred_df is None:
        return

    st.markdown("### Prediction results")
    try:
        out_df, fill_log = predict(bundle, pred_df)
        st.session_state["inference_predictions"] = out_df
        for line in fill_log:
            st.caption(line)
    except Exception as ex:
        st.error(f"Prediction failed: {ex}")
        return

    out_df = st.session_state["inference_predictions"]
    if out_df is None:
        return

    pred_col = out_df["prediction"]
    proba = out_df["probability"] if "probability" in out_df.columns else None

    if task_type == "classification":
        le = bundle.get("label_encoder")
        classes = list(le.classes_) if le is not None else []
        pos = classes[1] if len(classes) >= 2 else None
        st.markdown(
            f'<div class="metric-card" style="border-color:{p["accent"]};">'
            f'<div class="metric-label">Prediction</div>'
            f'<div class="metric-value" style="font-size:32px;">{pred_col.iloc[0]}</div></div>',
            unsafe_allow_html=True,
        )
        if proba is not None:
            conf = float(proba.iloc[0]) * 100.0
            st.progress(min(conf / 100.0, 1.0))
            st.markdown(f"**Confidence:** {conf:.1f}%")
        if len(out_df) > 1:
            tbl = pd.DataFrame(
                {
                    "Row": range(1, len(out_df) + 1),
                    "Predicted Class": out_df["prediction"].astype(str),
                }
            )
            if proba is not None:
                tbl["Confidence %"] = (proba * 100.0).round(2)
            st.dataframe(tbl, use_container_width=True, hide_index=True)
            vc = out_df["prediction"].astype(str).value_counts()
            st.bar_chart(vc)
        if len(out_df) == 1 and pos is not None:
            c1 = "#4ade80" if str(pred_col.iloc[0]) == str(pos) else "#fb7185"
            st.markdown(
                f'<div style="padding:12px;border-radius:8px;border:2px solid {c1};'
                f'font-family:\'JetBrains Mono\',monospace;">'
                f"Class highlighted: <strong>{pred_col.iloc[0]}</strong></div>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div class="metric-card" style="border-color:{p["accent"]};">'
            f'<div class="metric-label">{target_col} (predicted)</div>'
            f'<div class="metric-value" style="font-size:32px;">{float(pred_col.iloc[0]):,.4f}</div></div>',
            unsafe_allow_html=True,
        )
        if len(out_df) > 1:
            st.dataframe(
                pd.DataFrame(
                    {
                        "Row": range(1, len(out_df) + 1),
                        "Predicted Value": out_df["prediction"],
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.bar_chart(out_df["prediction"])
            c1, c2, c3 = st.columns(3)
            c1.metric("Min", f"{out_df['prediction'].min():.4f}")
            c2.metric("Mean", f"{out_df['prediction'].mean():.4f}")
            c3.metric("Max", f"{out_df['prediction'].max():.4f}")

    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        data=csv_bytes,
        file_name="predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    model = bundle["model"]
    feat_names = bundle.get("feature_names") or []
    pred0 = out_df["prediction"].iloc[0]
    st.markdown(
        f'<p style="font-family:\'DM Sans\',sans-serif;font-size:15px;color:{p["text"]};">'
        f"Why did the model predict <strong>{html.escape(str(pred0))}</strong> for row 1? "
        f"The chart below shows which encoded features pushed the score up or down.</p>",
        unsafe_allow_html=True,
    )
    shap_ok = False
    try:
        import matplotlib.pyplot as plt
        import shap
        from predict import prepare_transformed_features

        Xt, _ = prepare_transformed_features(bundle, pred_df)
        n = min(len(Xt), 50)
        Xt = Xt[:n]
        explainer = shap.Explainer(model)
        sv = explainer(Xt)
        shap.plots.waterfall(sv[0], max_display=12, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        plt.close("all")
        names = feat_names[: sv[0].values.shape[0]] if feat_names else []
        vals = np.asarray(sv[0].values).ravel()
        if len(names) >= len(vals):
            order = np.argsort(np.abs(vals))[-5:][::-1]
            bits = [f"{names[i]} ({vals[i]:+.4f})" for i in order if i < len(vals)]
            if bits:
                st.caption("Top 5 drivers: " + " · ".join(bits))
        shap_ok = True
    except Exception:
        shap_ok = False

    if not shap_ok:
        st.markdown(
            f'<p style="font-family:\'JetBrains Mono\',monospace;font-size:12px;color:{p["muted"]};">'
            f"Feature influence</p>",
            unsafe_allow_html=True,
        )
        bfi = bundle.get("bundle_feature_importances") or {}
        if bfi:
            top = sorted(bfi.items(), key=lambda x: -x[1])[:12]
            for feat, imp in top:
                st.markdown(f"- **{feat}** — {imp:.4f}")
        elif hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            order = np.argsort(imp)[-5:][::-1]
            for i in order:
                if i < len(feat_names):
                    st.markdown(f"- **{feat_names[i]}** — {float(imp[i]):.4f}")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:13px;'
        f'color:{_pal()["accent_soft"]};padding:12px 0 20px 0;letter-spacing:1px;">⚡ Explainable ML Pipeline Agent</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-head">Demo</div>', unsafe_allow_html=True)
    st.toggle(
        "Demo Mode",
        key="demo_mode_toggle",
        help="Browse a pre-computed example without an API key.",
        on_change=_on_demo_mode_change,
    )
    if not st.session_state.get("demo_mode_toggle", False):
        st.markdown('<div class="section-head">API key</div>', unsafe_allow_html=True)
        _env_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
        if _env_key:
            st.caption("Using ANTHROPIC_API_KEY from environment (.env).")
        else:
            st.text_input(
                "ANTHROPIC_API_KEY",
                type="password",
                key="anthropic_api_key_input",
                placeholder="sk-ant-...",
            )
            st.caption("Your key is never stored or logged.")

    st.markdown('<div class="section-head">Dataset</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload CSV", type=["csv"],
        label_visibility="collapsed",
        key="sidebar_csv_uploader",
    )
    _load_csv_from_upload(uploaded)

    # Sample datasets
    st.markdown('<div class="section-head">Or use a sample</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Titanic", use_container_width=True):
            st.session_state["demo_dataset"] = "titanic"
            p = Path("datasets/titanic_demo_synth.csv")
            if not p.exists():
                p = Path("datasets/titanic.csv")
            if p.exists():
                st.session_state.df = pd.read_csv(p)
                st.session_state.filename = p.name
                st.session_state.result = None
                st.session_state.log_lines = []
                st.session_state.step_cards = []
                st.session_state.pipeline_track = []
                st.session_state.error = None
                if st.session_state.get("demo_mode_toggle"):
                    _ds = _load_demo_json_file()
                    if _ds:
                        st.session_state["_demo_data_missing"] = False
                        _apply_demo_payload(_ds)
                    else:
                        st.session_state["_demo_data_missing"] = True
            else:
                st.warning("datasets/titanic_demo_synth.csv or datasets/titanic.csv not found.")
        if st.button("Healthcare", use_container_width=True):
            st.session_state["demo_dataset"] = "healthcare"
            p = Path("datasets/healthcare_demo_synth.csv")
            if not p.exists():
                p = Path("datasets/sample_healthcare_classification.csv")
            if p.exists():
                st.session_state.df = pd.read_csv(p)
                st.session_state.filename = p.name
                st.session_state.result = None
                st.session_state.log_lines = []
                st.session_state.step_cards = []
                st.session_state.pipeline_track = []
                st.session_state.error = None
                if st.session_state.get("demo_mode_toggle"):
                    _ds = _load_demo_json_file()
                    if _ds:
                        st.session_state["_demo_data_missing"] = False
                        _apply_demo_payload(_ds)
                    else:
                        st.session_state["_demo_data_missing"] = True
            else:
                st.warning("Run generate_samples.py first.")
    with col2:
        if st.button("Diabetes", use_container_width=True):
            st.session_state["demo_dataset"] = "diabetes"
            p = Path("datasets/diabetes_sklearn_demo.csv")
            if not p.exists():
                p = Path("datasets/diabetes.csv")
            if p.exists():
                st.session_state.df = pd.read_csv(p)
                st.session_state.filename = p.name
                st.session_state.result = None
                st.session_state.log_lines = []
                st.session_state.step_cards = []
                st.session_state.pipeline_track = []
                st.session_state.error = None
                if st.session_state.get("demo_mode_toggle"):
                    _ds = _load_demo_json_file()
                    if _ds:
                        st.session_state["_demo_data_missing"] = False
                        _apply_demo_payload(_ds)
                    else:
                        st.session_state["_demo_data_missing"] = True
            else:
                st.warning("datasets/diabetes_sklearn_demo.csv or datasets/diabetes.csv not found.")
        if st.button("Housing", use_container_width=True):
            st.session_state["demo_dataset"] = "housing"
            p = Path("datasets/sample_housing_regression.csv")
            if p.exists():
                st.session_state.df = pd.read_csv(p)
                st.session_state.filename = p.name
                st.session_state.result = None
                st.session_state.log_lines = []
                st.session_state.step_cards = []
                st.session_state.pipeline_track = []
                st.session_state.error = None
                if st.session_state.get("demo_mode_toggle"):
                    _ds = _load_demo_json_file()
                    if _ds:
                        st.session_state["_demo_data_missing"] = False
                        _apply_demo_payload(_ds)
                    else:
                        st.session_state["_demo_data_missing"] = True
            else:
                st.warning("Run generate_samples.py first.")

    # Dataset preview
    if st.session_state.df is not None:
        df = st.session_state.df
        st.markdown('<div class="section-head">Preview</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:11px;'
            f'color:{_pal()["muted"]};margin-bottom:8px;">'
            f'{len(df):,} rows · {len(df.columns)} cols · '
            f'{df.isnull().sum().sum()} nulls</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(df.head(6), use_container_width=True, height=180)

    st.markdown('<div class="section-head">Settings</div>', unsafe_allow_html=True)
    _icon = "☀️" if st.session_state.get("theme", "dark") == "light" else "🌙"
    st.toggle(
        _icon,
        key="theme_light_toggle",
        help="Switch between light and dark theme",
    )


if st.session_state.get("demo_mode_toggle") and st.session_state.get("result") is None:
    _snap = _load_demo_json_file()
    if _snap:
        st.session_state["_demo_data_missing"] = False
        _apply_demo_payload(_snap)
    else:
        st.session_state["_demo_data_missing"] = True


tab_pipeline, tab_inference = st.tabs(["Pipeline", "Inference"])
with tab_pipeline:
    _demo_on = st.session_state.get("demo_mode_toggle", False)
    if _demo_on:
        _dname = st.session_state.get("demo_dataset", "healthcare")
        _label = str(_dname).replace("_", " ").title()
        st.info(
            f"Demo Mode — showing pre-computed results for the {_label} dataset. "
            "Toggle off to use your own API key."
        )
        if st.session_state.get("_demo_data_missing"):
            st.warning(
                "Demo data not found. Run `python generate_all_demos.py` locally "
                "and push the demo_result_*.json files to GitHub."
            )

    goal_col, btn_col = st.columns([5, 1])
    with goal_col:
        user_goal = st.text_input(
            "What do you want to predict?",
            placeholder='e.g. "predict whether a patient will be readmitted"',
            key="user_goal_input",
        )
    with btn_col:
        st.write("")  # spacing
        run_clicked = st.button(
            "▶ Run",
            type="primary",
            use_container_width=True,
            key="run_button",
            disabled=st.session_state.running or _demo_on,
        )
        if _demo_on:
            st.caption("Add your ANTHROPIC_API_KEY to run on your own data")

    # ── Main-area dataset upload (sidebar stays available via top-left «) ────────
    if st.session_state.df is None:
        st.markdown('<div class="section-head">Upload dataset</div>', unsafe_allow_html=True)
        st.caption(
            "Drag and drop a CSV here, or use the file browser. "
            "You can also upload from the left sidebar, or load a sample dataset there."
        )
        main_upload = st.file_uploader(
            "Choose a CSV file",
            type=["csv"],
            accept_multiple_files=False,
            key="main_csv_uploader",
            label_visibility="collapsed",
        )
        _load_csv_from_upload(main_upload)
    
    # ── Run the agent ─────────────────────────────────────────────────────────────
    if run_clicked:
        if st.session_state.df is None:
            st.error("Upload a CSV first (or load a sample dataset from the sidebar).")
        elif not user_goal.strip():
            st.error("Describe your goal so the agent knows what to predict.")
        else:
            _env_k = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
            _paste_k = (st.session_state.get("anthropic_api_key_input") or "").strip()
            _effective_key = _env_k or _paste_k
            if not _effective_key:
                st.error("Add ANTHROPIC_API_KEY to your .env file or paste it in the sidebar.")
            else:
                os.environ["ANTHROPIC_API_KEY"] = _effective_key
                st.session_state.running   = True
                st.session_state.result    = None
                st.session_state.log_lines = []
                st.session_state.step_cards = []
                st.session_state.pipeline_track = _new_pipeline_track()
                st.session_state.error     = None
                st.session_state.report_export = None

                df = st.session_state.df.copy()
                log_placeholder = st.empty()
                status_placeholder = st.empty()
                step_cards_placeholder = st.empty()
                agent_holder: dict = {"agent": None}

                def run_agent_events():
                    from agent.core import AutoMLAgent

                    agent = AutoMLAgent(df, user_goal)
                    agent_holder["agent"] = agent
                    yield from agent.run()

                try:
                    status_placeholder.info("▶ Pipeline running… (this may take a minute)")
                    pipeline_failed = False
                    for event in run_agent_events():
                        etype = event["type"]
                        if etype == "text":
                            _log_text(event["content"])
                        elif etype == "tool":
                            name = event["name"]
                            status = event["status"]
                            _log_tool(name, status, event.get("output", ""))
                            if status == "running":
                                _pipeline_track_update_running(
                                    name, event.get("tune_model_name"),
                                )
                            elif status == "done":
                                _pipeline_track_update_done(name, event.get("step_data"))
                        elif etype == "error":
                            _log_error(event["content"])
                            st.session_state.error = event["content"]
                            _pipeline_track_fail_running(event["content"])
                            pipeline_failed = True
                        elif etype == "done":
                            st.session_state.result = event["result"]
                            _log_text("Pipeline complete.")
                            _pipeline_track_finalize(event["result"])

                        log_html = "".join(st.session_state.log_lines)
                        log_placeholder.markdown(
                            f'<div class="log-container">{log_html}</div>',
                            unsafe_allow_html=True,
                        )
                        with step_cards_placeholder.container():
                            for s in st.session_state.pipeline_track:
                                _render_pipeline_step(s)

                        if pipeline_failed:
                            break

                    if not pipeline_failed:
                        st.session_state["agent"] = agent_holder.get("agent")

                    status_placeholder.empty()
                except Exception as e:
                    err_msg = str(e)
                    st.session_state.error = err_msg
                    _log_error(err_msg)
                    status_placeholder.error(f"Pipeline failed: {err_msg}")
                    st.error(f"Pipeline failed: {err_msg}")

                st.session_state.running = False
                st.rerun()
    
    
    # ── Pipeline steps (persistent after run) ────────────────────────────────────
    if st.session_state.pipeline_track:
        st.markdown('<div class="section-head">Pipeline steps</div>', unsafe_allow_html=True)
        for s in st.session_state.pipeline_track:
            _render_pipeline_step(s)
    
    # ── Activity log (persistent after run) ──────────────────────────────────────
    if st.session_state.log_lines:
        st.markdown('<div class="section-head">Activity log</div>', unsafe_allow_html=True)
        log_html = "".join(st.session_state.log_lines)
        st.markdown(
            f'<div class="log-container">{log_html}</div>',
            unsafe_allow_html=True,
        )
    
    if st.session_state.error:
        st.error(st.session_state.error)
    
    
    # ── Debug section (temporary) ─────────────────────────────────────────────────
    with st.expander("🔧 Debug", expanded=False):
        st.write("**Button:**", "Run clicked" if run_clicked else "Not clicked")
        result_debug = st.session_state.get("result")
        st.write("**Result exists:**", result_debug is not None)
        if result_debug is not None:
            keys = list(result_debug.keys()) if isinstance(result_debug, dict) else str(type(result_debug))
            st.write("**Result keys:**", keys)
            st.write("**Result status:**", result_debug.get("status") if isinstance(result_debug, dict) else "N/A")
        st.write("**Running:**", st.session_state.get("running", False))
        st.write("**Log lines count:**", len(st.session_state.get("log_lines", [])))
    
    
    # ── Results dashboard ─────────────────────────────────────────────────────────
    result = st.session_state.get("result")
    if result and result.get("status") == "complete":
    
        st.markdown('<div class="section-head">Results</div>', unsafe_allow_html=True)
    
        # ── Top metric cards ──────────────────────────────────────────────────────
        metrics = result.get("metrics", {})
        task    = result.get("task_type", "")
        best    = result.get("best_model_name", "—")
    
        cards_html = '<div class="metric-grid">'
        cards_html += _metric_card("Best model", best)
        cards_html += _metric_card("Task", task.capitalize())
        cards_html += _metric_card("Target", result.get("target_col", "—"))
    
        if task == "classification":
            for key in ("accuracy", "f1", "roc_auc"):
                val = metrics.get(key)
                if val is not None:
                    label = {"accuracy": "Accuracy", "f1": "F1 (weighted)",
                             "roc_auc": "ROC-AUC"}[key]
                    cls   = _color_for_metric(key, val)
                    cards_html += _metric_card(label, f"{val:.3f}", cls)
        else:
            for key in ("r2", "rmse", "mape"):
                val = metrics.get(key)
                if val is not None:
                    label = {"r2": "R²", "rmse": "RMSE", "mape": "MAPE %"}[key]
                    suffix = "%" if key == "mape" else ""
                    cls    = _color_for_metric(key, val)
                    fmt    = f"{val:.2f}{suffix}" if key in ("rmse", "mape") else f"{val:.4f}"
                    cards_html += _metric_card(label, fmt, cls)
    
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)
    
        # ── Tabs ──────────────────────────────────────────────────────────────────
        tab_model, tab_plots, tab_features, tab_data = st.tabs(
            ["Model comparison", "Plots", "Features", "Data profile"]
        )
    
        # ── Tab 1: Model comparison ───────────────────────────────────────────────
        with tab_model:
            comp_df = result.get("comparison_df")
            if comp_df is not None:
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
    
            train_info = result.get("train", {})
            log_lines  = train_info.get("training_log", [])
            if log_lines:
                with st.expander("Training log", expanded=False):
                    st.code("\n".join(log_lines), language=None)
    
        # ── Tab 2: Plots ──────────────────────────────────────────────────────────
        with tab_plots:
            plot_paths = result.get("plot_paths", {})
            if not plot_paths:
                st.info("No plots generated.")
            else:
                base_order = [
                    "confusion_matrix",
                    "roc_curve",
                    "actual_vs_predicted",
                    "residuals",
                    "feature_importance",
                ]
                ordered = [p for p in base_order if p in plot_paths]
                ordered += [
                    p for p in plot_paths
                    if p not in ordered and not _is_shap_plot_key(p)
                ]
    
                cols = st.columns(2)
                for i, name in enumerate(ordered):
                    path = plot_paths[name]
                    col = cols[i % 2]
                    if Path(path).exists():
                        col.image(
                            path,
                            caption=name.replace("_", " ").title(),
                            use_container_width=True,
                        )
                    else:
                        col.warning(f"Plot not found: {path}")
    
                _render_shap_ui(plot_paths)
    
        # ── Tab 3: Feature importances ────────────────────────────────────────────
        with tab_features:
            fi = result.get("feature_importances", {})
            if not fi:
                st.info("Feature importances not available for this model.")
            else:
                top = list(fi.items())[:20]
                max_val = top[0][1] if top else 1.0
    
                fp = _pal()
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace;'
                    f'font-size:11px;color:{fp["muted"]};margin-bottom:16px;">'
                    'Normalized importance scores</div>',
                    unsafe_allow_html=True,
                )
                for feat, imp in top:
                    bar_pct = int((imp / max_val) * 100)
                    color   = fp["accent_soft"] if imp == max_val else fp["green"]
                    st.markdown(
                        f'<div style="margin-bottom:10px;">'
                        f'<div style="display:flex;justify-content:space-between;'
                        f'font-family:\'JetBrains Mono\',monospace;font-size:12px;'
                        f'color:{fp["text"]};margin-bottom:4px;">'
                        f'<span>{feat}</span>'
                        f'<span style="color:{fp["muted"]}">{imp:.4f}</span></div>'
                        f'<div class="fi-bar-bg"><div class="fi-bar-fill" '
                        f'style="width:{bar_pct}%;background:{color};"></div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    
        # ── Tab 4: Data profile ───────────────────────────────────────────────────
        with tab_data:
            eda = result.get("eda", {})
            if not eda:
                st.info("EDA data not available.")
            else:
                ov = eda.get("overview", {})
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows",      f"{ov.get('rows', 0):,}")
                c2.metric("Columns",   ov.get("columns", 0))
                c3.metric("Numeric",   ov.get("numeric_cols", 0))
                c4.metric("Nulls",     eda.get("missing", {}).get("total_missing", 0))
    
                # Column type summary
                col_profiles = eda.get("columns", {})
                if col_profiles:
                    rows = []
                    for col_name, prof in col_profiles.items():
                        miss   = prof.get("missing", 0)
                        n_rows = ov.get("rows", 1)
                        miss_p = f"{miss / n_rows * 100:.1f}%" if n_rows else "—"
                        if prof["dtype_group"] == "numeric":
                            rows.append({
                                "Column":   col_name,
                                "Type":     "numeric",
                                "Missing":  miss_p,
                                "Mean":     f"{prof.get('mean', 0):.2f}" if prof.get("mean") is not None else "—",
                                "Std":      f"{prof.get('std', 0):.2f}"  if prof.get("std")  is not None else "—",
                                "Min":      f"{prof.get('min', 0):.2f}"  if prof.get("min")  is not None else "—",
                                "Max":      f"{prof.get('max', 0):.2f}"  if prof.get("max")  is not None else "—",
                            })
                        else:
                            rows.append({
                                "Column":   col_name,
                                "Type":     "categorical",
                                "Missing":  miss_p,
                                "Mean":     "—",
                                "Std":      "—",
                                "Min":      f"{prof.get('n_unique', 0)} unique",
                                "Max":      f"top: {prof.get('top_value', '—')}",
                            })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
                # Quality flags
                flags = eda.get("quality_flags", [])
                if flags:
                    st.markdown("**Quality flags**")
                    for f in flags:
                        st.warning(f, icon="⚠️")
    
                # Preprocessing recommendations
                recs = eda.get("recommendations", [])
                if recs:
                    st.markdown("**Preprocessing applied**")
                    for r in recs:
                        st.markdown(f"- {r}")
    
    
    # ── Empty state ───────────────────────────────────────────────────────────────
    elif not st.session_state.get("running", False) and not st.session_state.get("log_lines", []):
        ep = _pal()
        st.markdown(
            f"""
        <div style="text-align:center;padding:60px 0 40px 0;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:48px;
                        color:{ep['empty_icon']};margin-bottom:16px;">⚡</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:15px;
                        color:{ep['empty_sub']};margin-bottom:8px;">Drop a CSV. Describe your goal. Run.</div>
            <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:{ep['empty_body']};">
                Upload a CSV above (or in the sidebar) → describe your goal → Run
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

with tab_inference:
    _render_inference_tab()

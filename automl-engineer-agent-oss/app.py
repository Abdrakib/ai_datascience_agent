"""
Streamlit UI — Open Source AutoML Agent (Llama 3.1, no API key).
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from agent.core import OssAutoMLAgent, load_llm_pipeline  # noqa: E402

st.set_page_config(
    page_title="AutoML Engineer OSS",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    defaults = {
        "df": None,
        "filename": "",
        "goal": "",
        "pipeline_events": [],
        "final_result": None,
        "load_error": None,
        "run_error": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource(show_spinner=False)
def get_llm_pipeline():
    return load_llm_pipeline()


def _sample_path(name: str) -> Path:
    return APP_ROOT / "datasets" / name


def _render_step_done(ev: dict) -> None:
    name = ev.get("name", "Step")
    step = ev.get("step", "")
    res = ev.get("result") or {}
    expl = res.get("explanation", "")
    badge = (
        '<span style="background:#1e3a2a;color:#4ade80;padding:2px 8px;border-radius:4px;'
        'font-size:11px;border:1px solid #2d5e3e;">Done</span>'
    )
    st.markdown(
        f'<div class="step-card"><div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
        f'<strong style="color:#e2e0d8;">{html.escape(str(name))}</strong> {badge}'
        f'<span style="color:#444441;font-size:12px;">Step {html.escape(str(step))}</span></div>',
        unsafe_allow_html=True,
    )
    if expl:
        st.markdown(
            f'<p style="color:#a8a49c;line-height:1.5;margin:0 0 12px 0;">{html.escape(str(expl))}</p>',
            unsafe_allow_html=True,
        )

    # Step-specific metrics
    if name == "EDA":
        ov = res.get("overview") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{ov.get('rows', 0):,}")
        c2.metric("Columns", ov.get("columns", "—"))
        c3.metric("Memory (MB)", f"{float(ov.get('memory_mb', 0)):.2f}")
    elif name == "Task detection":
        st.caption(f"Target: `{res.get('target_col', '')}` · Type: **{res.get('task_type', '')}**")
    elif name == "Preprocessing":
        st.caption(
            f"Train {res.get('train_shape')} · Test {res.get('test_shape')} · "
            f"Features: {res.get('feature_count', 0)}"
        )
    elif name == "Training plan":
        st.caption(res.get("plan_summary", "")[:500] + ("…" if len(str(res.get("plan_summary", ""))) > 500 else ""))
    elif name == "Training":
        bm = res.get("best_metrics") or {}
        mn = str(res.get("metric_name") or "")
        if mn in bm:
            pv = bm[mn]
        elif "roc_auc" in bm:
            pv = bm["roc_auc"]
        elif "f1" in bm:
            pv = bm["f1"]
        elif "r2" in bm:
            pv = bm["r2"]
        else:
            pv = next(iter(bm.values()), "—") if bm else "—"
        st.caption(f"**{res.get('best_name', '')}** — {mn or 'metric'}: {pv}")
    elif name == "Tuning":
        if res.get("success") is False:
            st.warning(res.get("error", "Tuning failed."))
        else:
            st.caption(
                f"Best score: {res.get('best_score')} "
                f"(Δ vs baseline: {res.get('improvement')})"
            )
    elif name == "Evaluation":
        m = res.get("metrics") or {}
        if "accuracy" in m:
            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{float(m.get('accuracy', 0)):.4f}")
            c2.metric("F1 (weighted)", f"{float(m.get('f1', 0)):.4f}")
        elif "r2" in m:
            c1, c2, c3 = st.columns(3)
            c1.metric("R²", f"{float(m.get('r2', 0)):.4f}")
            c2.metric("RMSE", f"{float(m.get('rmse', 0)):.4f}")
            c3.metric("MAE", f"{float(m.get('mae', 0)):.4f}")
        pp = res.get("plot_paths") or {}
        if pp:
            keys = list(pp.keys())[:4]
            cols = st.columns(min(4, len(keys)))
            for i, k in enumerate(keys):
                p = pp[k]
                if p and Path(p).is_file():
                    try:
                        cols[i].image(str(p), caption=k.replace("_", " "))
                    except Exception:
                        pass

    st.markdown("</div>", unsafe_allow_html=True)


_init_state()

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #0e0e10; color: #e2e0d8; }
[data-testid="stSidebar"] { background-color: #141416; border-right: 1px solid #2a2a2e; }
.step-card {
    background: #141416;
    border: 1px solid #2a2a2e;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 16px;
}
.hf-banner {
    background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 20px;
    color: #c4c2bc;
    font-size: 14px;
}
.stButton > button {
    background: #1e1e22;
    color: #e2e0d8;
    border: 1px solid #3a3a3e;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.stButton > button:hover { background: #7c3aed; border-color: #7c3aed; color: #fff; }
[data-testid="stFileUploader"] { background: #141416; border: 1px dashed #3a3a3e; border-radius: 8px; }
.stTextArea textarea { background: #141416 !important; border-color: #2a2a2e !important; color: #e2e0d8 !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hf-banner"><strong>Powered by Llama 3.1 8B</strong> — runs free on Hugging Face ZeroGPU. '
    "No API key needed.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="agent-header" style="display:flex;align-items:baseline;gap:12px;padding:8px 0 16px 0;border-bottom:1px solid #2a2a2e;margin-bottom:16px;">'
    '<span class="agent-title" style="font-family:JetBrains Mono,monospace;font-size:22px;font-weight:600;color:#e2e0d8;">Open Source AutoML Agent</span>'
    '<span style="font-family:JetBrains Mono,monospace;font-size:11px;background:#2e1065;color:#c084fc;padding:2px 8px;border-radius:4px;">OSS</span>'
    "</div>",
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
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

# ── Main area ─────────────────────────────────────────────────────────────────
df = st.session_state.df

if df is None:
    st.markdown("### Open Source AutoML Agent")
    st.caption("Powered by Llama 3.1 · No API key needed · Runs free on Hugging Face")
    st.markdown(
        """
1. **Upload any CSV** or use a sample dataset from the sidebar  
2. **Describe what you want to predict**  
3. **Click Run** — the agent runs EDA, preprocessing, training, tuning, and evaluation in order  
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

if run:
    st.session_state.run_error = None
    st.session_state.pipeline_events = []
    st.session_state.final_result = None
    try:
        with st.spinner("Loading Llama 3.1 — this takes about 30 seconds on first run..."):
            pipe = get_llm_pipeline()
    except Exception as e:
        st.session_state.run_error = f"Failed to load model: {e}"
        st.error(st.session_state.run_error)
        st.stop()

    agent = OssAutoMLAgent(df, st.session_state.goal, pipe)
    events: list = []
    try:
        with st.status("Running AutoML pipeline…", expanded=True) as status:
            for ev in agent.run():
                events.append(ev)
                if ev["type"] == "step_start":
                    status.write(f"Starting **{ev.get('name', '')}**…")
                elif ev["type"] == "step_done":
                    status.write(f"✓ **{ev.get('name', '')}** completed")
                elif ev["type"] == "done":
                    status.write("**All steps complete.**")
        st.session_state.pipeline_events = events
        for ev in events:
            if ev.get("type") == "done":
                st.session_state.final_result = ev.get("result")
    except Exception as e:
        st.session_state.run_error = str(e)
        st.error(f"Pipeline error: {e}")

if st.session_state.run_error and not run:
    st.error(st.session_state.run_error)

for ev in st.session_state.get("pipeline_events") or []:
    if ev.get("type") == "step_done":
        _render_step_done(ev)

fr = st.session_state.get("final_result")
if fr:
    st.markdown("---")
    st.markdown("### Final summary")
    st.write(fr.get("final_summary", ""))
    with st.expander("Technical details"):
        st.json(
            {
                "target_col": fr.get("target_col"),
                "task_type": fr.get("task_type"),
                "best_model": fr.get("best_model_name"),
                "status": fr.get("status"),
            }
        )

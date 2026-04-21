"""
Explainable ML Pipeline Agent — Gradio UI with native tabs and file uploads.
Pipeline step cards are rendered as HTML inside dedicated output panels.
"""

import gradio as gr
import pandas as pd
import numpy as np
import os
import json
import inspect
import tempfile
import traceback
import datetime
from pathlib import Path

# ── Agent imports (graceful fallback) ───────────────────────────
try:
    from agent.core import OssAutoMLAgent, load_llm_pipeline
    from agent.report import _build_html, _build_markdown
    from predict import save_model, load_model, predict as run_predict
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

APP_ROOT = Path(__file__).resolve().parent
DATASETS_DIR = APP_ROOT / "datasets"
OUTPUT_DIR = APP_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SAMPLE_FILES = {
    "titanic":    "titanic_demo_synth.csv",
    "healthcare": "sample_healthcare_classification.csv",
    "housing":    "sample_housing_regression.csv",
    "diabetes":   "diabetes_sklearn_demo.csv",
}

light_theme = gr.themes.Default(
    primary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.gray,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
)

dark_theme = gr.themes.Default(
    primary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    block_border_color="rgba(148, 163, 184, 0.14)",
    block_border_color_dark="rgba(148, 163, 184, 0.14)",
    block_label_background_fill="#1e293b",
    block_label_background_fill_dark="#1e293b",
    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    body_text_color="#f8fafc",
    body_text_color_dark="#f8fafc",
    body_text_color_subdued="#94a3b8",
    body_text_color_subdued_dark="#94a3b8",
    button_primary_background_fill="#6366f1",
    button_primary_background_fill_dark="#6366f1",
    button_primary_background_fill_hover="#4f46e5",
    button_primary_background_fill_hover_dark="#4f46e5",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
)

# ════════════════════════════════════════════════════════════════
# HTML HELPERS
# ════════════════════════════════════════════════════════════════

def _badge(status):
    if status == "done":    return '<span class="badge badge-done">Done</span>'
    if status == "running": return '<span class="badge badge-run">Running</span>'
    if status == "warning": return '<span class="badge badge-warn">Warning</span>'
    return '<span class="badge badge-fail">Failed</span>'

def _explain(text):
    if not text: return ""
    return (
        '<div class="explain-box">'
        '<span class="explain-label">Agent explanation</span>'
        f'{text}</div>'
    )

def _stat_row(stats):
    cells = "".join(
        f'<div class="stat-cell"><span class="stat-val">{v}</span>'
        f'<span class="stat-lbl">{k}</span></div>'
        for k, v in stats.items()
    )
    return f'<div class="stat-row">{cells}</div>'

def _metric_row(metrics, task_type="classification"):
    if task_type == "classification":
        keys   = ["accuracy","f1","roc_auc"]
        labels = {"accuracy":"Accuracy","f1":"F1 score","roc_auc":"ROC-AUC"}
    else:
        keys   = ["r2","rmse","mae","mape"]
        labels = {"r2":"R²","rmse":"RMSE","mae":"MAE","mape":"MAPE %"}
    cells = ""
    for k in keys:
        v = metrics.get(k)
        if v is None: continue
        display = f"{v:.4f}" if isinstance(v,float) and k!="mape" else (f"{v:.2f}%" if k=="mape" else str(v))
        cells += (f'<div class="metric-cell"><span class="metric-val">{display}</span>'
                  f'<span class="metric-lbl">{labels.get(k,k)}</span></div>')
    return f'<div class="metric-row">{cells}</div>'

def _alert(kind, text):
    return f'<div class="alert alert-{kind}">{text}</div>'

def _comp_table(comp_df):
    if comp_df is None: return ""
    if isinstance(comp_df, list):
        try: comp_df = pd.DataFrame(comp_df)
        except: return ""
    best_idx = None
    for c in ["CV Mean","Test Score","roc_auc","r2"]:
        if c in comp_df.columns:
            try: best_idx = comp_df[c].astype(float).idxmax()
            except: pass
            break
    heads = "".join(f"<th>{c}</th>" for c in comp_df.columns)
    rows  = ""
    for i, row in comp_df.iterrows():
        cls   = "row-best" if i==best_idx else ""
        cells = ""
        for col in comp_df.columns:
            val = row[col]
            if col.lower() in ("overfit","cv overfit"):
                tag = "tag-bad" if str(val).lower()=="true" else "tag-good"
                lbl = "Yes"     if str(val).lower()=="true" else "No"
                cells += f'<td><span class="{tag}">{lbl}</span></td>'
            elif isinstance(val, float):
                cells += f'<td>{val:.4f}</td>'
            else:
                cells += f'<td>{val}</td>'
        rows += f'<tr class="{cls}">{cells}</tr>'
    return (f'<div class="tbl-wrap"><table class="data-tbl">'
            f'<thead><tr>{heads}</tr></thead><tbody>{rows}</tbody></table></div>')

def _log_html(lines):
    items = ""
    for line in lines[-80:]:
        l = line.lower()
        if any(x in l for x in ["done","saved","complete","✓"]): cls = "ll-ok"
        elif any(x in l for x in ["warning","overfit","⚠"]):     cls = "ll-warn"
        elif any(x in l for x in ["error","failed","✗"]):        cls = "ll-err"
        else:                                                      cls = "ll-info"
        items += f'<div class="{cls}">{line}</div>'
    return f'<div class="log-box">{items or "<span class=ll-info>No log lines yet</span>"}</div>'

def _ds_info(df):
    if df is None: return ""
    nulls = int(df.isnull().sum().sum())
    null_str = f" · <b>{nulls}</b> nulls" if nulls else ""
    cols = " · ".join(df.columns[:5].tolist())
    if len(df.columns)>5: cols += f" +{len(df.columns)-5}"
    return (f'<div class="ds-info"><div><b>{len(df)}</b> rows · '
            f'<b>{len(df.columns)}</b> cols{null_str}</div>'
            f'<div class="ds-cols">{cols}</div></div>')

# ════════════════════════════════════════════════════════════════
# STEP RENDERERS
# ════════════════════════════════════════════════════════════════

def _card(num, name, status="done"):
    cls = " card-running" if status=="running" else ""
    return (f'<div class="step-card{cls}"><div class="card-header">'
            f'<div class="step-num">{num}</div>'
            f'<div class="step-name">{name}</div>'
            f'{_badge(status)}</div>')

def render_eda(r,e):
    ov  = r.get("eda",{}).get("overview",{})
    rows= ov.get("n_rows", r.get("n_rows","—"))
    cols= ov.get("n_cols", r.get("n_cols","—"))
    nul = ov.get("total_nulls",0)
    num = ", ".join((ov.get("numeric_cols") or [])[:5]) or "—"
    cat = ", ".join((ov.get("categorical_cols") or [])[:5]) or "—"
    h   = _card(1,"Data analysis")
    h  += _stat_row({"rows":rows,"columns":cols,"nulls":nul})
    h  += (f'<div class="kv-list">'
           f'<div><span class="kv-key">Numeric</span><span class="kv-val">{num}</span></div>'
           f'<div><span class="kv-key">Categorical</span><span class="kv-val">{cat}</span></div>'
           f'</div>')
    h  += _explain(e)
    return h + '</div>'

def render_task(r,e):
    t   = r.get("task",{})
    tgt = t.get("target_col",  r.get("target_col","—"))
    typ = t.get("task_type",   r.get("task_type","—"))
    con = t.get("confidence","—")
    rsn = t.get("reasoning","")
    h   = _card(2,"Task detection")
    h  += _stat_row({"target":tgt,"type":typ,"confidence":con})
    if rsn: h += f'<div class="reasoning">{rsn}</div>'
    h  += _explain(e)
    return h + '</div>'

def render_domain(r,e):
    dr  = r.get("domain_research",r)
    q   = dr.get("query","—")
    sns = dr.get("snippets",[])
    h   = _card("2b","Domain research")
    h  += f'<div class="query-box">Search: <code>{q}</code></div>'
    for s in sns[:3]:
        tit = s.get("title","") or s.get("url","")
        snp = s.get("body","")  or s.get("snippet","")
        h  += (f'<div class="snippet-card"><div class="snippet-title">{tit}</div>'
               f'<div class="snippet-body">{snp[:140]}…</div></div>')
    h  += _explain(e)
    return h + '</div>'

def render_prep(r,e):
    p   = r.get("prep",r)
    num = ", ".join((p.get("numeric_cols_used")    or [])[:5]) or "—"
    cat = ", ".join((p.get("categorical_cols_used") or [])[:5]) or "—"
    smt = p.get("smote_applied",False)
    h   = _card(3,"Data preparation")
    h  += (f'<div class="kv-list">'
           f'<div><span class="kv-key">Numeric</span><span class="kv-val">{num}</span></div>'
           f'<div><span class="kv-key">Categorical</span><span class="kv-val">{cat}</span></div>'
           f'</div>')
    if smt: h += _alert("ok","SMOTE applied — class imbalance corrected")
    h  += _explain(e)
    return h + '</div>'

def render_plan(r,e):
    p   = r.get("plan",r)
    ms  = p.get("models_to_train",[])
    bud = p.get("tuning_budget","—")
    h   = _card(4,"Training plan")
    if ms:
        tags = "".join(f'<span class="model-tag">{m}</span>' for m in ms[:6])
        h   += f'<div class="model-tags">{tags}</div>'
    h  += (f'<div class="kv-list"><div><span class="kv-key">Budget</span>'
           f'<span class="kv-val">{bud} trials</span></div></div>')
    h  += _explain(e)
    return h + '</div>'

def render_train(r,e):
    t   = r.get("train",r)
    bn  = t.get("best_name",  r.get("best_model_name","—"))
    bs  = t.get("best_score","—")
    if isinstance(bs,float): bs = f"{bs:.4f}"
    ws  = t.get("overfitting_warnings",[])
    cdf = t.get("comparison_df")
    h   = _card(5,"Model training")
    h  += _stat_row({"best model":bn,"score":bs})
    for w in (ws or []): h += _alert("warn",w)
    if cdf is not None: h += _comp_table(cdf)
    h  += _explain(e)
    return h + '</div>'

def render_tune(r,e):
    t   = r.get("tune",r)
    nm  = t.get("model_name","—")
    bl  = t.get("baseline_score")
    tu  = t.get("tuned_score")
    tr  = t.get("n_trials","—")
    imp = t.get("improvement")
    st  = {"model":nm,"trials":str(tr)}
    if bl is not None: st["before"] = f"{bl:.4f}"
    if tu is not None: st["after"]  = f"{tu:.4f}"
    h   = _card(6,"Hyperparameter tuning")
    h  += _stat_row(st)
    if imp and float(imp)>0: h += _alert("ok",f"Improvement: +{float(imp):.4f}")
    h  += _explain(e)
    return h + '</div>'

def render_eval(r,e):
    mets = r.get("metrics",r.get("best_metrics",{}))
    typ  = r.get("task_type","classification")
    plts = r.get("plot_paths",{})
    h    = _card(7,"Evaluation")
    if mets: h += _metric_row(mets,typ)
    if plts:
        imgs = ""
        for nm,path in list((plts or {}).items())[:4]:
            if path and Path(path).exists():
                imgs += (f'<div class="plot-cell"><div class="plot-label">{nm}</div>'
                         f'<img src="file={path}" class="plot-img"/></div>')
        if imgs: h += f'<div class="plot-grid">{imgs}</div>'
    h  += _explain(e)
    return h + '</div>'

def render_final(r,e):
    bn  = r.get("best_model_name","—")
    met = r.get("best_metrics",r.get("metrics",{}))
    typ = r.get("task_type","classification")
    pk  = "roc_auc" if typ=="classification" else "r2"
    pv  = met.get(pk)
    pvs = f"{pv:.4f}" if isinstance(pv,float) else "—"
    lbl = "ROC-AUC" if typ=="classification" else "R²"
    return (f'<div class="final-card"><span class="final-label">Best model</span>'
            f'<div class="final-model">{bn}</div>'
            f'<div class="final-metric">{lbl}: {pvs}</div>'
            f'<div class="final-explain">{e or ""}</div></div>')

def build_pipeline_html(events):
    html = ""
    fin  = ""
    for ev in events:
        t = ev.get("type")
        r = ev.get("result",{}) or {}
        e = r.get("explanation","") if r else ""
        if t=="step_start":
            nm = ev.get("name","Step")
            st = ev.get("step","")
            html += (_card(st,nm,"running") +
                     '<div class="running-msg">Processing…</div>'
                     '<div class="progress-bar"><div class="progress-fill"></div></div></div>')
        elif t=="step_done":
            nm = ev.get("name","")
            st = str(ev.get("step",""))
            if   st=="1"  or "eda"    in nm.lower() or "data an" in nm.lower(): html += render_eda(r,e)
            elif st=="2"  or "task"   in nm.lower():                             html += render_task(r,e)
            elif st=="2b" or "domain" in nm.lower():                             html += render_domain(r,e)
            elif st=="3"  or "prep"   in nm.lower():                             html += render_prep(r,e)
            elif st=="4"  and "plan"  in nm.lower():                             html += render_plan(r,e)
            elif st=="5"  or ("train" in nm.lower() and "plan" not in nm.lower()):html += render_train(r,e)
            elif st=="6"  or "tun"    in nm.lower():                             html += render_tune(r,e)
            elif st=="7"  or "eval"   in nm.lower():                             html += render_eval(r,e)
            elif st=="8"  or "final"  in nm.lower() or "recommend" in nm.lower():fin   = render_final(r,e)
            else:                                                                 html += render_eda(r,e)
    return html + fin

# ════════════════════════════════════════════════════════════════
# UI CSS + empty state (pipeline HTML uses #pipeline-output wrapper in DOM)
# ════════════════════════════════════════════════════════════════

EMPTY_PIPE_HTML = """
<div id="empty-state">
  <div class="empty-icon">⚡</div>
  <div class="empty-title">Ready to analyze your data</div>
  <div class="empty-sub">Upload a CSV and describe your prediction goal, then click Run.</div>
</div>
"""


def _export_html(show: bool, html_p, md_p, pkl_p) -> str:
    if not show:
        return ""
    dl_h = f'<a class="export-btn" href="file={html_p}" download>⬇ HTML report</a>' if html_p else ""
    dl_m = f'<a class="export-btn" href="file={md_p}" download>⬇ Markdown</a>' if md_p else ""
    dl_p = f'<a class="export-btn" href="file={pkl_p}" download>⬇ Model .pkl</a>' if pkl_p else ""
    return (
        f'<div class="export-section"><div class="export-title">Export results</div>'
        f'<div class="export-btns">{dl_h}{dl_m}{dl_p}</div></div>'
    )


def _fmt_pipeline_html(ph: str) -> str:
    body = (ph or "").strip()
    if not body:
        body = EMPTY_PIPE_HTML.strip()
    return body


APP_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* —— Dashboard tokens (consistent dark; no mixed light panels) —— */
:root {
  --dash-bg: #0f172a;
  --dash-card: #1e293b;
  --dash-elevated: #273549;
  --dash-border: rgba(148, 163, 184, 0.14);
  --dash-input: #0f172a;
  --dash-text: #f8fafc;
  --dash-muted: #94a3b8;
  --dash-faint: #64748b;
  --dash-accent: #8b5cf6;
  --dash-accent-2: #6366f1;
}

body, html {
  background-color: var(--dash-bg) !important;
  margin: 0 !important;
  padding: 0 !important;
  color: var(--dash-text) !important;
}

.gradio-container {
  background-color: var(--dash-bg) !important;
  min-height: 100vh !important;
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
  color: var(--dash-text) !important;
}

.main, .wrap, .contain {
  background-color: var(--dash-bg) !important;
  max-width: 100% !important;
}

.block, .form, .panel {
  background-color: var(--dash-card) !important;
  border: 1px solid var(--dash-border) !important;
  border-radius: 12px !important;
}

input, textarea, select {
  background-color: var(--dash-input) !important;
  color: var(--dash-text) !important;
  border: 1px solid var(--dash-border) !important;
  border-radius: 8px !important;
}

.gradio-container label,
.gradio-container .label-wrap span,
.gradio-container .caption {
  color: var(--dash-muted) !important;
}

.gradio-container .markdown,
.gradio-container .markdown p,
.gradio-container .markdown h1,
.gradio-container .prose {
  color: var(--dash-text) !important;
}

.tabs { background: var(--dash-card) !important; border-radius: 12px 12px 0 0 !important; }
.tab-nav {
  background: var(--dash-card) !important;
  border-bottom: 1px solid var(--dash-border) !important;
}
.tab-nav button { color: var(--dash-muted) !important; background: transparent !important; }
.tab-nav button.selected {
  color: #c4b5fd !important;
  border-bottom: 2px solid var(--dash-accent) !important;
}

button:not(.primary) {
  background: var(--dash-elevated) !important;
  color: var(--dash-text) !important;
  border: 1px solid var(--dash-border) !important;
}
button.primary {
  background: linear-gradient(135deg, var(--dash-accent-2) 0%, var(--dash-accent) 100%) !important;
  color: #fff !important;
  border: none !important;
}

footer { display: none !important; }

.primary, button.primary, .gr-button-primary {
  background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
  border: none !important;
  color: #fff !important;
}
.primary:hover, button.primary:hover, .gr-button-primary:hover {
  filter: brightness(1.08) !important;
  box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
}

.gr-column {
  background: transparent !important;
}

/* Sidebar / file upload surfaces */
.gr-file,
.upload-container,
[data-testid="file-upload"] {
  background-color: var(--dash-card) !important;
  border: 1px dashed rgba(139, 92, 246, 0.35) !important;
  border-radius: 12px !important;
  color: var(--dash-text) !important;
}
.gr-file:hover,
.upload-container:hover {
  border-color: rgba(139, 92, 246, 0.55) !important;
  background-color: var(--dash-elevated) !important;
}

/* Dropdowns, number fields, and nested Gradio panels */
.gr-box,
.gr-input,
.gr-dropdown,
.gr-number,
.gr-checkbox-group,
.gr-radio-group {
  background-color: var(--dash-card) !important;
  border-color: var(--dash-border) !important;
  color: var(--dash-text) !important;
}
.gr-dropdown ul,
.gr-dropdown option {
  background-color: var(--dash-card) !important;
  color: var(--dash-text) !important;
}

.ds-info {
  background: var(--dash-card);
  border: 1px solid var(--dash-border);
  border-radius: 8px;
  padding: 9px 11px;
  font-size: 11px;
  color: var(--dash-muted);
  line-height: 1.8;
}
.ds-info b { color: var(--dash-text); font-weight: 500; }
.ds-cols { font-size: 10px; color: var(--dash-faint); font-family: 'JetBrains Mono', monospace; margin-top: 2px; }

#pipeline-output {
  --purple: #8b5cf6;
  --purple-lt: #c4b5fd;
  --purple-dim: rgba(139, 92, 246, 0.14);
  --purple-bdr: rgba(139, 92, 246, 0.35);
  --success: #34d399;
  --success-dim: rgba(52, 211, 153, 0.12);
  --warning: #fbbf24;
  --warning-dim: rgba(251, 191, 36, 0.12);
  --error: #f87171;
  --error-dim: rgba(248, 113, 113, 0.12);
  --mono: 'JetBrains Mono', monospace;
  --bg: #0f172a;
  --surface: #1e293b;
  --border: rgba(148, 163, 184, 0.14);
  --text: #f8fafc;
  --text2: #94a3b8;
  --text3: #64748b;
  font-family: 'Inter', sans-serif;
  background: var(--surface) !important;
  border: 1px solid var(--border);
  padding: 16px;
  border-radius: 12px;
  min-height: 220px;
  color: var(--text);
}
#pipeline-output #empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  min-height: 360px;
  color: var(--text2);
}
#pipeline-output .empty-icon { font-size: 38px; opacity: 0.35; margin-bottom: 14px; }
#pipeline-output .empty-title { font-size: 18px; font-weight: 500; color: var(--text); margin-bottom: 8px; }
#pipeline-output .empty-sub { font-size: 13px; line-height: 1.65; max-width: 300px; }
#pipeline-output .step-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px 18px;
  animation: ph-fadein 0.25s ease;
  margin-bottom: 12px;
}
#pipeline-output .step-card.card-running { border-color: var(--purple-bdr); }
@keyframes ph-fadein { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
#pipeline-output .card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
#pipeline-output .step-num {
  width: 24px; height: 24px; border-radius: 50%; background: var(--purple); color: #fff;
  font-size: 10px; font-weight: 600; display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; font-family: var(--mono);
}
#pipeline-output .step-name { font-size: 13px; font-weight: 500; color: var(--text); flex: 1; }
#pipeline-output .badge { font-size: 10px; font-weight: 500; border-radius: 20px; padding: 2px 9px; font-family: var(--mono); }
#pipeline-output .badge-done { background: var(--success-dim); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); }
#pipeline-output .badge-run { background: var(--purple-dim); color: var(--purple-lt); border: 1px solid var(--purple-bdr); animation: ph-pulse 1.4s infinite; }
#pipeline-output .badge-warn { background: var(--warning-dim); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.3); }
#pipeline-output .badge-fail { background: var(--error-dim); color: var(--error); border: 1px solid rgba(239, 68, 68, 0.3); }
@keyframes ph-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }
#pipeline-output .running-msg { font-size: 12px; color: var(--text2); }
#pipeline-output .progress-bar { height: 3px; background: var(--bg); border-radius: 2px; overflow: hidden; margin-top: 10px; }
#pipeline-output .progress-fill { height: 100%; background: var(--purple); border-radius: 2px; animation: ph-progress 2s ease-in-out infinite; width: 30%; }
@keyframes ph-progress { 0% { width: 5%; margin-left: 0; } 50% { width: 50%; margin-left: 20%; } 100% { width: 5%; margin-left: 90%; } }
#pipeline-output .stat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(90px, 1fr)); gap: 8px; margin-bottom: 12px; }
#pipeline-output .stat-cell { background: var(--bg); border-radius: 7px; padding: 8px 10px; text-align: center; }
#pipeline-output .stat-val { display: block; font-size: 18px; font-weight: 500; color: var(--purple); font-family: var(--mono); }
#pipeline-output .stat-lbl { display: block; font-size: 10px; color: var(--text2); margin-top: 2px; }
#pipeline-output .metric-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px; margin: 12px 0; }
#pipeline-output .metric-cell { background: var(--bg); border-radius: 7px; padding: 11px 12px; text-align: center; }
#pipeline-output .metric-val { display: block; font-size: 24px; font-weight: 500; color: var(--purple); font-family: var(--mono); }
#pipeline-output .metric-lbl { display: block; font-size: 10px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 3px; }
#pipeline-output .explain-box {
  border-left: 2px solid var(--purple); background: var(--purple-dim); border-radius: 0 8px 8px 0;
  padding: 10px 14px; margin-top: 10px; font-size: 12px; color: var(--text2); line-height: 1.65;
}
#pipeline-output .explain-label {
  display: block; font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--purple-lt); font-family: var(--mono); margin-bottom: 5px;
}
#pipeline-output .alert { border-radius: 7px; padding: 8px 11px; font-size: 11px; margin: 5px 0; }
#pipeline-output .alert-ok { background: var(--success-dim); color: #6ee7b7; border: 1px solid rgba(52, 211, 153, 0.28); }
#pipeline-output .alert-warn { background: var(--warning-dim); color: #fde68a; border: 1px solid rgba(251, 191, 36, 0.28); display: flex; gap: 6px; }
#pipeline-output .alert-err { background: var(--error-dim); color: #fca5a5; border: 1px solid rgba(248, 113, 113, 0.28); }
#pipeline-output .kv-list { font-size: 11px; line-height: 1.9; margin-bottom: 10px; }
#pipeline-output .kv-key { color: var(--text2); margin-right: 6px; }
#pipeline-output .kv-val { color: var(--text); font-weight: 500; font-family: var(--mono); font-size: 10px; }
#pipeline-output .model-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px; }
#pipeline-output .model-tag {
  background: var(--purple-dim); color: var(--purple); border: 1px solid var(--purple-bdr);
  border-radius: 5px; padding: 2px 8px; font-size: 11px; font-weight: 500; font-family: var(--mono);
}
#pipeline-output .tbl-wrap { overflow-x: auto; margin: 10px 0; }
#pipeline-output .data-tbl { width: 100%; border-collapse: collapse; font-size: 11px; font-family: var(--mono); }
#pipeline-output .data-tbl th {
  background: var(--bg); color: var(--text2); padding: 6px 10px; text-align: left; font-size: 10px;
  text-transform: uppercase; letter-spacing: 0.05em; border-bottom: 1px solid var(--border); font-weight: 500;
}
#pipeline-output .data-tbl td { padding: 6px 10px; border-bottom: 1px solid var(--border); color: var(--text); }
#pipeline-output .data-tbl tr.row-best td { background: var(--purple-dim); }
#pipeline-output .data-tbl tr:hover td { background: rgba(15, 23, 42, 0.65); }
#pipeline-output .tag-good { background: var(--success-dim); color: var(--success); border-radius: 4px; padding: 1px 6px; font-size: 10px; }
#pipeline-output .tag-bad { background: var(--error-dim); color: var(--error); border-radius: 4px; padding: 1px 6px; font-size: 10px; }
#pipeline-output .plot-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
#pipeline-output .plot-cell { text-align: center; }
#pipeline-output .plot-label { font-size: 10px; color: var(--text2); margin-bottom: 4px; }
#pipeline-output .plot-img { width: 100%; border-radius: 6px; border: 1px solid var(--border); }
#pipeline-output .reasoning { font-size: 11px; color: var(--text2); margin-bottom: 8px; line-height: 1.6; }
#pipeline-output .query-box { font-size: 11px; color: var(--text2); margin-bottom: 8px; }
#pipeline-output .query-box code { background: var(--bg); padding: 1px 5px; border-radius: 4px; font-family: var(--mono); font-size: 11px; color: var(--purple); }
#pipeline-output .snippet-card { background: var(--bg); border-radius: 6px; padding: 7px 9px; margin-bottom: 6px; }
#pipeline-output .snippet-title { font-size: 11px; font-weight: 500; color: var(--text); }
#pipeline-output .snippet-body { font-size: 11px; color: var(--text2); margin-top: 2px; }
#pipeline-output .final-card { background: var(--purple-dim); border: 1px solid var(--purple-bdr); border-radius: 12px; padding: 18px 20px; }
#pipeline-output .final-label {
  display: block; font-size: 9px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;
  color: var(--purple-lt); font-family: var(--mono); margin-bottom: 6px;
}
#pipeline-output .final-model { font-size: 22px; font-weight: 500; color: var(--purple); font-family: var(--mono); }
#pipeline-output .final-metric { font-size: 13px; color: var(--text2); margin-top: 4px; }
#pipeline-output .final-explain { font-size: 12px; color: var(--text2); line-height: 1.65; margin-top: 10px; padding-top: 10px; border-top: 1px solid var(--purple-bdr); }

#log-output .log-box {
  background: #0f172a;
  border: 1px solid rgba(148, 163, 184, 0.14);
  border-radius: 8px;
  padding: 10px 14px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  max-height: 200px;
  overflow-y: auto;
  color: #94a3b8;
  margin-top: 6px;
}
#log-output .ll-ok { color: #34d399; }
#log-output .ll-warn { color: #fbbf24; }
#log-output .ll-err { color: #f87171; }
#log-output .ll-info { color: #a5b4fc; }

#export-output .export-section { background: #1e293b; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 12px; padding: 14px 16px; }
#export-output .export-title { font-size: 12px; font-weight: 500; color: #f8fafc; margin-bottom: 10px; }
#export-output .export-btns { display: flex; gap: 8px; flex-wrap: wrap; }
#export-output .export-btn {
  flex: 1; min-width: 120px; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 7px; padding: 7px 10px; font-size: 11px;
  font-weight: 500; color: #f8fafc; background: #273549; cursor: pointer; text-align: center;
  text-decoration: none; display: inline-block;
}
#export-output .export-btn:hover { background: rgba(99, 102, 241, 0.2); border-color: rgba(139, 92, 246, 0.45); color: #c4b5fd; }

#infer-pred-wrap .model-info-card {
  background: #1e293b; border: 1px solid rgba(148, 163, 184, 0.14); border-radius: 8px; padding: 11px 13px;
  font-size: 11px; color: #94a3b8; line-height: 1.8; margin-bottom: 10px;
}
#infer-pred-wrap .model-info-card b { color: #f8fafc; font-weight: 500; }
#infer-pred-wrap .pred-metric-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px; margin: 10px 0; }
#infer-pred-wrap .pred-metric { background: #0f172a; border: 1px solid rgba(148, 163, 184, 0.1); border-radius: 7px; padding: 12px; text-align: center; }
#infer-pred-wrap .pred-val { display: block; font-size: 26px; font-weight: 500; color: #a5b4fc; font-family: 'JetBrains Mono', monospace; }
#infer-pred-wrap .pred-lbl { display: block; font-size: 10px; color: #94a3b8; margin-top: 3px; }
#infer-pred-wrap .alert { border-radius: 7px; padding: 8px 11px; font-size: 11px; margin: 5px 0; }
#infer-pred-wrap .alert-err { background: rgba(248, 113, 113, 0.12); color: #fca5a5; border: 1px solid rgba(248, 113, 113, 0.28); }
#infer-pred-wrap .alert-ok { background: rgba(52, 211, 153, 0.12); color: #6ee7b7; border: 1px solid rgba(52, 211, 153, 0.28); }
"""

js = """
function toggleTheme() {
    const body = document.body;
    const isDark = body.getAttribute('data-theme') === 'dark';
    const gc = document.querySelector('.gradio-container');
    const borderSubtle = 'rgba(148, 163, 184, 0.14)';
    const borderLift = 'rgba(148, 163, 184, 0.2)';

    if (isDark) {
        /* Softer slate (still dark — no white cards) */
        body.setAttribute('data-theme', 'light');
        body.style.background = '#1e293b';
        if (gc) gc.style.background = '#1e293b';
        document.querySelectorAll('.block,.form,.panel').forEach(el => {
            el.style.background = '#334155';
            el.style.borderColor = borderLift;
        });
        document.querySelectorAll('input,textarea,select').forEach(el => {
            el.style.background = '#0f172a';
            el.style.color = '#f8fafc';
            el.style.borderColor = borderSubtle;
        });
        const btn = document.querySelector('#theme-toggle-btn button');
        if (btn) btn.textContent = '☾ Dark mode';
    } else {
        body.setAttribute('data-theme', 'dark');
        body.style.background = '#0f172a';
        if (gc) gc.style.background = '#0f172a';
        document.querySelectorAll('.block,.form,.panel').forEach(el => {
            el.style.background = '#1e293b';
            el.style.borderColor = borderSubtle;
        });
        document.querySelectorAll('input,textarea,select').forEach(el => {
            el.style.background = '#0f172a';
            el.style.color = '#f8fafc';
            el.style.borderColor = borderSubtle;
        });
        const btn = document.querySelector('#theme-toggle-btn button');
        if (btn) btn.textContent = '☀ Softer contrast';
    }
}

setTimeout(() => {
    document.body.setAttribute('data-theme', 'dark');
    document.body.style.background = '#0f172a';
    const gc = document.querySelector('.gradio-container');
    if (gc) gc.style.background = '#0f172a';
    document.querySelectorAll('.block,.form,.panel').forEach(el => {
        el.style.background = '#1e293b';
        el.style.borderColor = 'rgba(148, 163, 184, 0.14)';
    });
    document.querySelectorAll('input,textarea,select').forEach(el => {
        el.style.background = '#0f172a';
        el.style.color = '#f8fafc';
        el.style.borderColor = 'rgba(148, 163, 184, 0.14)';
    });
    const btn = document.querySelector('#theme-toggle-btn button');
    if (btn) btn.textContent = '☀ Softer contrast';
}, 800);
"""


def _no_public_api() -> dict:
    """
    Keep handlers off the public REST/OpenAPI schema.

    Gradio 6.x on ZeroGPU can emit invalid JSON Schema for some components
    (e.g. DownloadButton), which crashes /info (TypeError: 'bool' is not iterable
    in gradio_client). Gradio 4.x skips schema when api_name=False; 5+ uses
    api_visibility='private'.
    """
    sig = inspect.signature(gr.Button.click)
    if "api_visibility" in sig.parameters:
        return {"api_visibility": "private"}
    return {"api_name": False}


# ════════════════════════════════════════════════════════════════
# GRADIO BLOCKS
# ════════════════════════════════════════════════════════════════

with gr.Blocks(
    title="Explainable ML Pipeline Agent",
    theme=dark_theme,
    css=APP_CSS,
    js=js,
) as demo:

    df_state = gr.State(None)
    events_st = gr.State([])
    logs_st = gr.State([])
    result_st = gr.State(None)
    bundle_st = gr.State(None)
    html_p_st = gr.State(None)
    md_p_st = gr.State(None)
    pkl_p_st = gr.State(None)
    pred_csv_st = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1, min_width=0):
            gr.Markdown("# Explainable ML Pipeline Agent")
        with gr.Column(scale=0, min_width=140):
            theme_btn = gr.Button(
                "☀ Softer contrast",
                size="sm",
                elem_id="theme-toggle-btn",
            )

    theme_btn.click(None, js="toggleTheme")

    with gr.Tabs():
        with gr.Tab("⚡ Pipeline"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=0, min_width=276):
                    gr_csv = gr.File(
                        label="Upload CSV dataset",
                        file_types=[".csv"],
                        type="filepath",
                    )
                    gr.Markdown("**Or try a sample:**")
                    with gr.Row():
                        btn_titanic = gr.Button("Titanic", size="sm")
                        btn_healthcare = gr.Button("Healthcare", size="sm")
                    with gr.Row():
                        btn_housing = gr.Button("Housing", size="sm")
                        btn_diabetes = gr.Button("Diabetes", size="sm")
                    preview_out = gr.HTML(value="<div class='ds-info' style='color:#94a3b8'>No dataset loaded</div>")
                    goal_input = gr.Textbox(
                        label="What do you want to predict?",
                        placeholder='e.g. "predict whether a patient will be readmitted"',
                        lines=3,
                    )
                    run_btn = gr.Button("▶ Run pipeline", variant="primary")

                with gr.Column(scale=1):
                    pipeline_out = gr.HTML(value=EMPTY_PIPE_HTML, elem_id="pipeline-output")
                    log_out = gr.HTML(elem_id="log-output")
                    export_out = gr.HTML(elem_id="export-output")

        with gr.Tab("🔮 Inference"):
            with gr.Row():
                model_upload = gr.File(
                    label="Upload model (.pkl)",
                    file_types=[".pkl"],
                    type="filepath",
                )
            model_info_out = gr.HTML()
            infer_csv = gr.File(
                label="Upload CSV for prediction",
                file_types=[".csv"],
                type="filepath",
            )
            predict_btn = gr.Button("🔮 Predict", variant="primary")
            with gr.Column(elem_id="infer-pred-wrap"):
                pred_out = gr.HTML()
            pred_dl = gr.DownloadButton("⬇ Download predictions CSV", visible=False)

    def _views_from_data(df, info_html, ev, lg, res, hp, mp, pp):
        pipe = _fmt_pipeline_html(build_pipeline_html(ev))
        log_v = _log_html(lg) if lg else ""
        exp_v = _export_html(res is not None, hp, mp, pp)
        return (
            gr.update(value=info_html),
            gr.update(value=pipe),
            gr.update(value=log_v),
            gr.update(value=exp_v),
        )

    def on_csv(path, ev, lg, res, hp, mp, pp):
        if path is None:
            return (
                gr.update(value="<div class='ds-info' style='color:#94a3b8'>No dataset loaded</div>"),
                gr.update(value=_fmt_pipeline_html(build_pipeline_html(ev))),
                gr.update(value=_log_html(lg) if lg else ""),
                gr.update(value=_export_html(res is not None, hp, mp, pp)),
                None,
            )
        try:
            df = pd.read_csv(path)
            info = _ds_info(df)
            return (*_views_from_data(df, info, ev, lg, res, hp, mp, pp), df)
        except Exception as e:
            return (
                gr.update(value=_alert("err", str(e))),
                gr.update(value=EMPTY_PIPE_HTML),
                gr.update(value=""),
                gr.update(value=""),
                None,
            )

    gr_csv.change(
        on_csv,
        inputs=[gr_csv, events_st, logs_st, result_st, html_p_st, md_p_st, pkl_p_st],
        outputs=[preview_out, pipeline_out, log_out, export_out, df_state],
        **_no_public_api(),
    )

    def on_sample(name, ev, lg, res, hp, mp, pp):
        path = DATASETS_DIR / SAMPLE_FILES.get(name, "")
        if not path.exists():
            return (
                gr.update(value=_alert("err", f"Sample '{name}' not found.")),
                gr.update(value=EMPTY_PIPE_HTML),
                gr.update(value=""),
                gr.update(value=""),
                None,
            )
        try:
            df = pd.read_csv(path)
            info = _ds_info(df)
            return (*_views_from_data(df, info, ev, lg, res, hp, mp, pp), df)
        except Exception as e:
            return (
                gr.update(value=_alert("err", str(e))),
                gr.update(value=EMPTY_PIPE_HTML),
                gr.update(value=""),
                gr.update(value=""),
                None,
            )

    btn_titanic.click(
        lambda e, l, r, h, m, p: on_sample("titanic", e, l, r, h, m, p),
        inputs=[events_st, logs_st, result_st, html_p_st, md_p_st, pkl_p_st],
        outputs=[preview_out, pipeline_out, log_out, export_out, df_state],
        **_no_public_api(),
    )
    btn_healthcare.click(
        lambda e, l, r, h, m, p: on_sample("healthcare", e, l, r, h, m, p),
        inputs=[events_st, logs_st, result_st, html_p_st, md_p_st, pkl_p_st],
        outputs=[preview_out, pipeline_out, log_out, export_out, df_state],
        **_no_public_api(),
    )
    btn_housing.click(
        lambda e, l, r, h, m, p: on_sample("housing", e, l, r, h, m, p),
        inputs=[events_st, logs_st, result_st, html_p_st, md_p_st, pkl_p_st],
        outputs=[preview_out, pipeline_out, log_out, export_out, df_state],
        **_no_public_api(),
    )
    btn_diabetes.click(
        lambda e, l, r, h, m, p: on_sample("diabetes", e, l, r, h, m, p),
        inputs=[events_st, logs_st, result_st, html_p_st, md_p_st, pkl_p_st],
        outputs=[preview_out, pipeline_out, log_out, export_out, df_state],
        **_no_public_api(),
    )

    def run_pipe(df, goal, _ev, _lg):
        events = []
        logs = []

        def _out(ph, lg, exp=False, hp=None, mp=None, pp=None, res=None, bundle=None, pc=None):
            return (
                gr.update(value=_fmt_pipeline_html(ph)),
                gr.update(value=_log_html(lg) if lg else ""),
                gr.update(value=_export_html(exp, hp, mp, pp)),
                list(events),
                list(logs),
                res,
                bundle,
                hp,
                mp,
                pp,
                pc,
            )

        if df is None:
            yield _out(_alert("err", "Please upload a dataset or pick a sample first."), logs)
            return
        if not goal or not goal.strip():
            yield _out(_alert("err", "Please describe what you want to predict."), logs)
            return
        if not AGENT_AVAILABLE:
            yield _out(_alert("err", "Agent modules not found."), logs)
            return

        load_html = (
            _card("…", "Loading Qwen2.5", "running")
            + '<div class="running-msg">Loading model — first run ~30 s…</div>'
            + '<div class="progress-bar"><div class="progress-fill"></div></div></div>'
        )
        yield _out(load_html, logs)

        try:
            pipe = load_llm_pipeline()
            agent = OssAutoMLAgent(df, goal.strip(), pipe)
            final = None

            for ev in agent.run():
                t = ev.get("type")
                if t == "log":
                    logs.append(ev.get("content", ""))
                    yield _out(build_pipeline_html(events), logs)
                elif t in ("step_start", "step_done"):
                    events.append(ev)
                    yield _out(build_pipeline_html(events), logs)
                elif t == "done":
                    final = ev.get("result", {})
                    events.append(ev)
                    html_p = md_p = pkl_p = None
                    try:
                        hc = _build_html(final)
                        mc = _build_markdown(final)
                        html_p = str(OUTPUT_DIR / "automl_report.html")
                        md_p = str(OUTPUT_DIR / "automl_report.md")
                        Path(html_p).write_text(hc, encoding="utf-8")
                        Path(md_p).write_text(mc, encoding="utf-8")
                    except Exception:
                        pass
                    try:
                        rid = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
                        pkl_p = save_model(
                            pipeline=final.get("prep", {}).get("pipeline"),
                            model=final.get("best_model"),
                            label_encoder=final.get("prep", {}).get("label_encoder"),
                            feature_names=final.get("prep", {}).get("feature_names", []),
                            task_type=final.get("task_type", "classification"),
                            target_col=final.get("target_col", ""),
                            best_metrics=final.get("best_metrics", {}),
                            model_name=final.get("best_model_name", "model"),
                            run_id=rid,
                        )
                    except Exception:
                        pkl_p = None
                    yield (
                        gr.update(value=_fmt_pipeline_html(build_pipeline_html(events))),
                        gr.update(value=_log_html(logs)),
                        gr.update(value=_export_html(True, html_p, md_p, pkl_p)),
                        list(events),
                        list(logs),
                        final,
                        None,
                        html_p,
                        md_p,
                        pkl_p,
                        None,
                    )
                    return
        except Exception as ex:
            logs.append(f"✗ {ex}")
            yield _out(build_pipeline_html(events) + _alert("err", str(ex)), logs)

    run_btn.click(
        run_pipe,
        inputs=[df_state, goal_input, events_st, logs_st],
        outputs=[
            pipeline_out,
            log_out,
            export_out,
            events_st,
            logs_st,
            result_st,
            bundle_st,
            html_p_st,
            md_p_st,
            pkl_p_st,
            pred_csv_st,
        ],
        **_no_public_api(),
    )

    def on_model(path):
        if path is None:
            return gr.update(
                value="<div class='model-info-card'>No model loaded</div>"
            ), None
        try:
            b = load_model(path)
            info = (
                f'<div class="model-info-card">'
                f'<div><b>Model:</b> {b.get("model_name", "—")}</div>'
                f'<div><b>Task:</b> {b.get("task_type", "—")}</div>'
                f'<div><b>Target:</b> {b.get("target_col", "—")}</div>'
                f'{_alert("ok", "Model loaded ✓")}</div>'
            )
            return gr.update(value=info), b
        except Exception as e:
            return gr.update(value=_alert("err", f"Could not load model: {e}")), None

    model_upload.change(
        on_model,
        inputs=[model_upload],
        outputs=[model_info_out, bundle_st],
        **_no_public_api(),
    )

    def on_predict(bundle, csv_path):
        if bundle is None:
            return (
                gr.update(value=_alert("err", "Load a model first.")),
                gr.update(visible=False, value=None),
                None,
            )
        if csv_path is None:
            return (
                gr.update(value=_alert("err", "Upload a CSV first.")),
                gr.update(visible=False, value=None),
                None,
            )
        try:
            df = pd.read_csv(csv_path)
            rdf, _fill = run_predict(bundle, df)
            task = bundle.get("task_type", "classification")
            if task == "classification":
                vc = rdf["prediction"].value_counts()
                cells = "".join(
                    f'<div class="pred-metric"><span class="pred-val">{cnt / len(rdf) * 100:.1f}%</span>'
                    f'<span class="pred-lbl">Class {cls}</span></div>'
                    for cls, cnt in vc.items()
                )
                ph = f'<div class="pred-metric-row">{cells}</div>'
            else:
                preds = pd.to_numeric(rdf["prediction"], errors="coerce")
                mn = float(np.mean(preds))
                ph = (
                    f'<div class="pred-metric-row">'
                    f'<div class="pred-metric"><span class="pred-val">{mn:.2f}</span><span class="pred-lbl">Mean</span></div>'
                    f'<div class="pred-metric"><span class="pred-val">{float(np.min(preds)):.2f}</span><span class="pred-lbl">Min</span></div>'
                    f'<div class="pred-metric"><span class="pred-val">{float(np.max(preds)):.2f}</span><span class="pred-lbl">Max</span></div>'
                    f"</div>"
                )
            pc = str(OUTPUT_DIR / "predictions.csv")
            rdf.to_csv(pc, index=False)
            return gr.update(value=ph), gr.update(visible=True, value=pc), pc
        except Exception as e:
            return (
                gr.update(value=_alert("err", f"Prediction error: {e}")),
                gr.update(visible=False, value=None),
                None,
            )

    predict_btn.click(
        on_predict,
        inputs=[bundle_st, infer_csv],
        outputs=[pred_out, pred_dl, pred_csv_st],
        **_no_public_api(),
    )


if __name__ == "__main__":
    # Hugging Face Spaces / ZeroGPU sets SPACE_ID. Gradio verifies local_url with httpx;
    # HTTP(S)_PROXY without a proper bypass often makes that probe fail on the Space image.
    _hf_space = bool(os.environ.get("SPACE_ID"))
    if _hf_space:
        _loop = "localhost,127.0.0.1,127.0.0.1/8,::1"
        for _key in ("NO_PROXY", "no_proxy"):
            _cur = os.environ.get(_key, "").strip()
            if _cur and _loop.split(",")[0] not in _cur:
                os.environ[_key] = f"{_loop},{_cur}"
            elif not _cur:
                os.environ[_key] = _loop
        for _pk in (
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
        ):
            os.environ.pop(_pk, None)

    demo.queue()

    _launch_kw: dict = {"show_error": True}
    if _hf_space:
        if "_frontend" in inspect.signature(demo.launch).parameters:
            _launch_kw["_frontend"] = False
    else:
        _launch_kw["server_name"] = "0.0.0.0"
        _launch_kw["server_port"] = int(os.environ.get("PORT", "7860"))

    demo.launch(**_launch_kw)

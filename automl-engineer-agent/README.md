---
title: Explainable ML Pipeline Agent
emoji: ⚡
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: 1.55.0
app_file: app.py
pinned: true
short_description: Explainable ML pipeline agent (Claude) for any CSV
license: mit
---

<div align="center">

# Explainable ML Pipeline Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Powered by Claude API](https://img.shields.io/badge/Powered%20by-Claude%20API-8B5CF6.svg?logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Models](https://img.shields.io/badge/Models-XGBoost%20%2B%20LightGBM-orange.svg)](https://xgboost.readthedocs.io/)
[![Explainability](https://img.shields.io/badge/Explainability-SHAP-14B8A6.svg)](https://shap.readthedocs.io/)

**A Claude-powered AI agent that acts as a senior ML engineer.**  
Drop in any CSV, describe your goal in plain English — it runs the full ML pipeline autonomously from EDA to inference.

```
🔍 Web Search → 📊 EDA → 🎯 Task Detection → ⚙️ Preprocessing → 📋 Training Plan
    → 🏋️ Model Training → 🎛️ Hyperparameter Tuning → 📈 Evaluation → 🔬 SHAP
    → 📄 Report Export → 🔮 Inference
```

**Live demos:** [Streamlit Cloud](https://ai-data-science-agent.streamlit.app/) · [Hugging Face Space](https://huggingface.co/spaces/Abdourakib/ai-data-science-agent)

**Repository:** [github.com/Abdrakib/ai_datascience_agent](https://github.com/Abdrakib/ai_datascience_agent) · `git clone https://github.com/Abdrakib/ai_datascience_agent.git` · **Local package name:** `automl-engineer-agent`

</div>

---

## 📌 What it does

**Explainable ML Pipeline Agent** is an end-to-end machine learning system that combines **Anthropic Claude** (tool use) with **scikit-learn**, **XGBoost**, **LightGBM**, **Optuna**, and **SHAP**. You upload a tabular dataset, describe what you want to predict in natural language, and the agent orchestrates EDA, preprocessing, model comparison with cross-validation, optional tuning, evaluation, explainability, and export — with a **Streamlit** notebook-style UI and a dedicated **Inference** tab for saved models.

| Feature | Description |
|--------|-------------|
| **Step-by-step UI** | Jupyter-like flow: each pipeline step renders results before the next runs. |
| **Web search** | Looks up domain context for unfamiliar column names before modeling (DuckDuckGo via `duckduckgo-search`; Tavily optional if configured). |
| **Intelligent training plan** | Reasons about dataset size, imbalance, and dimensionality to select and configure models. |
| **Automatic task detection** | Infers classification vs regression and suggests a target column from your goal text. |
| **Smart preprocessing** | Missing-value imputation, one-hot encoding, scaling, automatic log transforms for skewed numeric columns. |
| **SMOTE class balancing** | Applies SMOTE or SMOTEENN when class imbalance exceeds thresholds. |
| **Multi-model training** | Trains Logistic/Linear Regression, Random Forest, XGBoost, and LightGBM in parallel. |
| **5-fold cross-validation** | Reports CV mean ± std; **CV mean** is used to rank models for more reliable selection. |
| **Overfitting detection** | Computes train/test gap per model, flags severe overfitting, can surface web search suggestions for fixes. |
| **Hyperparameter tuning (Optuna)** | Bayesian optimization on the best model; trial budget scales with dataset size. |
| **SHAP explainability** | Summary plot, dependence plots for top features, waterfall-style explanations in the UI. |
| **Sandboxed code execution** | Claude can propose and run constrained Python for custom transforms or plots mid-pipeline. |
| **Self-contained HTML report** | All figures embedded — safe to email or host as a single file. |
| **Model persistence** | Save pipeline + estimator + encoders as a single `.pkl` bundle under `outputs/`. |
| **Inference tab** | Load a saved model, predict on CSV or manual form (≤10 original features), optional SHAP-style explanation. |
| **CLI inference** | `predict.py` for batch predictions and bundle inspection from the command line. |
| **Warning banners** | Surfaces small data, imbalance, missingness, leakage suspicion, overfitting, and weak metrics. |
| **Any CSV domain** | Healthcare, finance, housing, churn, fraud, sports — any tabular CSV the pipeline can ingest. |

---

## 🖼️ Demo / screenshots

> **Note:** Screenshots will be added after final testing. Place images under `assets/` and replace paths below.

| | |
|:--|:--|
| ![Pipeline UI](assets/pipeline_screenshot.png) | ![Training Plan](assets/training_plan_screenshot.png) |
| ![Evaluation](assets/evaluation_screenshot.png) | ![Inference Tab](assets/inference_screenshot.png) |

---

## 🚀 Quickstart

### Step 1 — Clone and install

```bash
git clone https://github.com/Abdrakib/ai_datascience_agent.git
cd ai_datascience_agent   # or your local folder name (e.g. automl-engineer-agent)
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

### Step 2 — API key

```bash
copy .env.example .env   # Windows; use `cp` on Unix
```

Edit `.env` and set:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

The app reads keys via `python-dotenv` (`config.py`).

### Step 3 — Web search (optional)

- **DuckDuckGo (default path in code):** install `duckduckgo-search` (see Step 4). No API key required.
- **Tavily (optional):** add `TAVILY_API_KEY` to `.env` if you integrate Tavily in your deployment; the bundled search helper primarily uses DuckDuckGo.

### Step 4 — Recommended extras

```bash
pip install shap xgboost lightgbm optuna imbalanced-learn duckduckgo-search
```

### Step 5 — Sample datasets

```bash
python generate_samples.py
```

### Step 6 — Launch the UI

```bash
streamlit run app.py
```

Open the **Pipeline** tab to run the agent, and **Inference** to load a saved `.pkl` and predict.

---

## 📁 Project layout

```
automl-engineer-agent/
├── agent/
│   ├── core.py                 # Claude tool-use loop, tool dispatch, result builder
│   ├── report.py               # Markdown + HTML report generation from pipeline result
│   └── tools/
│       ├── eda.py              # Dataset profiling, quality flags, recommendations
│       ├── preprocess.py       # ColumnTransformer pipeline, SMOTE, splits, feature names
│       ├── task_detector.py    # Target + task type inference from data + user hint
│       ├── train.py            # Multi-model training, CV, comparison DataFrame, best pick
│       ├── evaluate.py         # Metrics, plots, SHAP integration for best model
│       ├── tune.py             # Optuna hyperparameter search on tuned estimator
│       ├── plan_training.py    # JSON training plan from dataset heuristics
│       ├── code_exec.py        # Sandboxed execution of model-generated Python
│       └── search.py           # Web search wrapper (DuckDuckGo)
├── app.py                      # Streamlit UI: Pipeline + Inference tabs, export, save model
├── predict.py                  # CLI: save/load bundle, predict, print bundle info
├── config.py                   # Central env, paths, ML defaults, model lists
├── generate_samples.py         # Writes sample CSVs under datasets/
├── verify_pipeline.py          # End-to-end wiring test without live Claude API
├── test_step2.py               # EDA / preprocessing smoke tests
├── test_step3.py               # Task detection + training smoke tests
├── test_step4.py               # Evaluation + plots smoke tests
├── test_step5.py               # Full agent integration (optional LIVE_TEST with API)
├── datasets/                   # Sample and user CSVs (generated + uploaded)
├── outputs/                    # Models, plots, predictions (default OUTPUT_DIR)
├── reports/                    # Optional extra report path (REPORTS_DIR)
├── assets/                     # README screenshots (add your images here)
├── .streamlit/
│   └── config.toml             # Streamlit theme / server options
├── .env.example                # Template for environment variables
├── .gitignore                  # Ignores venv, .env, caches, large outputs
├── requirements.txt          # Pinned / core dependencies
├── LICENSE                     # MIT
└── README.md                   # This file
```

---

## 🧠 How the agent works

The **Claude** model receives your goal and a tool schema. It calls tools in a loop until the pipeline is complete. Typical order:

1. **`web_search`** — Optional context on column semantics or domain terms.
2. **`run_eda`** — Profiles shape, dtypes, missing values, skewness, class balance, quality flags.
3. **`detect_task`** — Locks target column and classification vs regression.
4. **`plan_training`** — Produces a structured plan (models, CV folds, tuning budget hints).
5. **`preprocess`** — Fits imputers, encoders, scalers, optional log1p, SMOTE when needed.
6. **`train_models`** — Fits all candidate models with **k-fold CV**; ranks by CV mean (fallbacks if CV fails).
7. **`tune_model`** — Optuna search on the winning architecture.
8. **`evaluate_model`** — Hold-out metrics, confusion matrix / ROC / residual plots, SHAP plots.
9. **Report + inference** — Markdown/HTML export, optional `.pkl` save, Streamlit **Inference** tab.

### Programmatic usage

```python
from agent.core import AutoMLAgent
import pandas as pd

df = pd.read_csv("your_dataset.csv")
agent = AutoMLAgent(df, "predict whether a customer will churn")

for event in agent.run():
    if event["type"] == "text":
        print(event["content"])
    elif event["type"] == "tool":
        print(f"Tool: {event.get('name')} — {event.get('status', '')}")
    elif event["type"] == "done":
        result = event["result"]
        print(f"Best model: {result['best_model_name']}")
        print(f"Metrics: {result['best_metrics']}")
```

---

## 🛠️ Tool reference

| Tool | File | When it runs | What it does |
|------|------|--------------|--------------|
| `web_search` | `search.py` | Claude requests context | Runs DuckDuckGo text search; returns titles, URLs, snippets for domain-aware reasoning. |
| `run_eda` | `eda.py` | Early in pipeline | Profiles shape, dtypes, missing %, numeric stats, categorical cardinality, skewness hints, target-oriented flags, and human-readable recommendations. |
| `detect_task` | `task_detector.py` | After EDA | Scores candidate target columns, infers classification vs regression, returns confidence and alternatives. |
| `plan_training` | `plan_training.py` | After preprocess | Builds JSON plan: model list, class weights, tuning trials, CV folds from dataset size and imbalance. |
| `preprocess` | `preprocess.py` | Before training | Drops bad columns, builds `ColumnTransformer` (numeric + categorical pipes), SMOTE/SMOTEENN when needed, train/test split, label encoding for classification. |
| `train_models` | `train.py` | Core training step | Fits LogReg/LinReg, RF, XGBoost, LGBM; stratified **KFold** or **KFold** CV; primary metric by task; overfitting warnings. |
| `tune_model` | `tune.py` | After training | Optuna study on hyperparameters of the best model; reports improvement vs baseline. |
| `evaluate_model` | `evaluate.py` | After tuning | Computes task metrics, saves plots (confusion, ROC, residuals, feature importance, SHAP), writes structured eval dict. |
| `code_exec` | `code_exec.py` | On demand | Executes sandboxed Python snippets (limited builtins) for custom analysis Claude proposes. |

---

## 🤖 Supported models

| Task | Models | Primary selection metric (CV / hold-out) |
|------|--------|------------------------------------------|
| **Classification** | Logistic Regression, Random Forest, XGBoost, LightGBM | ROC-AUC (binary), F1 weighted (multiclass), aligned with `train_models` |
| **Regression** | Linear Regression, Random Forest, XGBoost, LightGBM | R² (primary), plus RMSE / MAE in reports |

The **training plan** adapts which models are emphasized, class weights, and Optuna trial counts from row counts, feature counts, and imbalance — so small or noisy datasets get a conservative plan automatically.

---

## 📤 Generated outputs

| Artifact | Description |
|----------|-------------|
| `confusion_matrix.png` | Classification: predicted vs actual classes. |
| `roc_curve.png` | Binary classification ROC. |
| `actual_vs_predicted.png` | Regression: fit quality scatter. |
| `residuals.png` | Regression: residual distribution / plot. |
| `feature_importance.png` | Model-native importances or coefficients. |
| `shap_summary.png` | SHAP beeswarm-style summary for explained samples. |
| `shap_dependence_<feature>.png` | SHAP dependence for top features. |
| `shap_bar.png`, `shap_waterfall.png` | Additional SHAP views when enabled. |
| `*_model.pkl` | Joblib bundle: pipeline, model, encoders, metrics, training stats (`predict.py`). |
| `automl_report.html` | Self-contained HTML report with embedded images. |
| `automl_report.md` | Markdown twin of the report. |

Default directories: `outputs/` (and optional `reports/` per `config.py`).

---

## 🔮 Inference

### Way 1 — Streamlit Inference tab

1. **Load model** — Use the session’s saved `.pkl` after **Save model**, or upload a bundle file.  
2. **Data** — Upload a CSV **or** fill the manual form (shown when ≤10 original feature columns).  
3. **Results** — Predictions, optional probability for binary classification, download CSV, explanation section (SHAP when available, else feature influence).

### Way 2 — Command line (`predict.py`)

```bash
python predict.py predict --model outputs/run_YYYYMMDD_HHMMSS_model.pkl --input new_data.csv

python predict.py info --model outputs/run_YYYYMMDD_HHMMSS_model.pkl
```

---

## 🧪 Tests & verification

| Command | Purpose |
|---------|---------|
| `python verify_pipeline.py` | Full pipeline wiring (classification + regression); **no live Claude API** required (uses stubs). |
| `python test_step2.py` | EDA + preprocessor checks. |
| `python test_step3.py` | Task detector + trainer checks. |
| `python test_step4.py` | Evaluator + plotting checks. |
| `LIVE_TEST=1 python test_step5.py` | Full agent with real **Anthropic** API (`ANTHROPIC_API_KEY` in `.env`). |

> **Note:** A dedicated `test_step8.py` is not included in this repository; add your own integration tests for search/code paths if needed.

---

## ⚙️ Configuration (`config.py`)

| Setting | Description |
|---------|-------------|
| `ANTHROPIC_MODEL` | Default: `claude-sonnet-4-20250514` (override via `.env`). |
| `ANTHROPIC_API_KEY` | Required for live agent runs; loaded from `.env`. |
| `TAVILY_API_KEY` | Optional; use if you wire Tavily in custom search code. |
| `TEST_SIZE` | Hold-out fraction (default `0.2`). |
| `CV_FOLDS` | Cross-validation folds (default `5`). |
| `RANDOM_STATE` | Global seed (default `42`). |
| `MAX_AGENT_ITERATIONS` | Safety cap on tool loops (default `20`). |
| `MAX_TOKENS` | Max tokens per Claude completion (default `4096`). |
| `OUTPUT_DIR` | Where models and plots are written (default `outputs`). |
| `REPORTS_DIR` | Secondary report directory (default `reports`). |

---

## 🧱 Tech stack

| Layer | Library | Purpose |
|-------|---------|---------|
| LLM | `anthropic` | Claude Sonnet 4 tool use |
| UI | `streamlit` | Pipeline + Inference app |
| ML | `scikit-learn`, `xgboost`, `lightgbm` | Models + preprocessing |
| Tuning | `optuna` | Hyperparameter optimization |
| Explainability | `shap` | Global and local explanations |
| Class balancing | `imbalanced-learn` | SMOTE / SMOTEENN |
| Visualization | `matplotlib`, `seaborn` | Plots for eval + reports |
| Web search | `duckduckgo-search` (optional: Tavily client) | Column / domain lookup |
| Persistence | `joblib` | Model bundles |
| Data | `pandas`, `numpy` | Tabular ML |
| Config | `python-dotenv` | Environment loading |

---

## 🗺️ Roadmap

**Done**

- [x] 9-tool Claude agent with tool-use API  
- [x] Step-by-step Jupyter-style UI  
- [x] Intelligent training plan  
- [x] 5-fold cross-validation for model selection  
- [x] Optuna hyperparameter tuning  
- [x] Overfitting detection + web search fix suggestions  
- [x] SMOTE / SMOTEENN balancing  
- [x] Automatic log transforms for skewed features  
- [x] SHAP summary + dependence plots  
- [x] Sandboxed code execution  
- [x] Web search for domain context  
- [x] Self-contained HTML report export  
- [x] Model persistence + CLI inference  
- [x] Streamlit Inference tab  

**Planned (v2)**

- [ ] Time-series aware train/test split  
- [ ] JSON, Excel, SQL data sources  
- [ ] Image datasets (CNN feature pipelines)  
- [ ] Audio datasets (MFCC features)  
- [ ] Docker deployment  
- [ ] Hugging Face Hub publishing integration  
- [ ] Hyperparameter tuning with nested cross-validation  

---

## 👤 About the author

- **GitHub:** [@Abdourakib](https://github.com/Abdourakib)  
- **Hugging Face:** [@Abdourakib](https://huggingface.co/Abdourakib)  

Built as a **portfolio project** showcasing AI agent design, LLM tool use, ML engineering, and a production-minded end-to-end system.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).

---

<p align="center">
  <b>Explainable ML Pipeline Agent</b> · Claude · Streamlit · scikit-learn · XGBoost · LightGBM · SHAP
</p>

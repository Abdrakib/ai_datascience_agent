---
title: Explainable ML Pipeline Agent
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.12.0"
app_file: app.py
pinned: true
short_description: ML pipeline that explains every step in plain English
license: mit
tags:
  - machine-learning
  - automl
  - explainable-ai
  - gradio
  - qwen
  - python
  - pytorch
  - transformers
  - huggingface
  - scikit-learn
  - pandas
  - numpy
  - streamlit
  - fastapi
  - xgboost
  - lightgbm
  - optuna
  - shap
  - plotly
---

# Explainable ML Pipeline Agent

**Explainable ML Pipeline Agent** runs a full ML pipeline on any CSV dataset and explains every step in plain English. It uses **Qwen2.5** via `transformers` — **no API key** needed for inference in the hosted app.

## Live demo (Hugging Face Spaces)

- **[Explainable ML Pipeline — Analysis Agent](https://huggingface.co/spaces/Abdourakib/explainable-ml-pipeline-analysis-agent)** (primary Space)
- **[Explainable ML Pipeline Agent](https://huggingface.co/spaces/Abdourakib/explainable-ml-pipeline-agent)**

## Tech stack

| Area | Tools |
|------|--------|
| **Language** | Python 3 |
| **UI** | [Gradio](https://gradio.app/), [Streamlit](https://streamlit.io/) (sibling package) |
| **LLM** | [Hugging Face Transformers](https://huggingface.co/docs/transformers), [Qwen2.5](https://huggingface.co/Qwen), `accelerate`, `huggingface_hub` |
| **ML / AutoML** | [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), [Optuna](https://optuna.org/), [imbalanced-learn](https://imbalanced-learn.org/) |
| **Explainability** | [SHAP](https://shap.readthedocs.io/) |
| **Data** | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Joblib](https://joblib.readthedocs.io/) |
| **Viz & reports** | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), [Plotly](https://plotly.com/python/), [Tabulate](https://github.com/astanin/python-tabulate) |
| **Web / API** | [FastAPI](https://fastapi.tiangolo.com/) |
| **Search** | [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/) (domain research) |
| **Other** | [python-dotenv](https://pypi.org/project/python-dotenv/), [pydub](https://github.com/jiaaro/pydub) |

**Suggested GitHub topics:** `python`, `gradio`, `transformers`, `pytorch`, `scikit-learn`, `automl`, `explainable-ai`, `qwen`, `huggingface`, `machine-learning`, `pandas`, `xgboost`, `lightgbm`, `optuna`, `shap`.

- **Gradio UI:** [`app.py`](app.py) at repo root; full monorepo source is [`automl-engineer-agent-oss/app.py`](https://github.com/Abdrakib/ai_datascience_agent/blob/main/automl-engineer-agent-oss/app.py).
- **Streamlit UI:** [`automl-engineer-agent`](https://github.com/Abdrakib/ai_datascience_agent/tree/main/automl-engineer-agent) (`streamlit run app.py` there).
- **Hosting:** set `HF_TOKEN` in Space **Secrets** if model weights are gated.

For the **Claude API** tool-calling variant, see [`automl-engineer-agent`](https://github.com/Abdrakib/ai_datascience_agent/tree/main/automl-engineer-agent).

## Run locally (Gradio)

**Space / flat checkout** (this folder is repo root):

```bash
pip install -r requirements.txt
python app.py
```

**GitHub monorepo** (from repository root):

```bash
cd automl-engineer-agent-oss
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860`. A GPU is recommended for Qwen2.5.

## License

MIT.

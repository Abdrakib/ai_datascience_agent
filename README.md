---
title: Explainable ML Pipeline Agent
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.12.0"
app_file: automl-engineer-agent-oss/app.py
pinned: false
short_description: ML pipeline that explains every step in plain English
license: mit
tags:
  - machine-learning
  - automl
  - explainable-ai
  - gradio
  - qwen
---

# Explainable ML Pipeline Agent

**Explainable ML Pipeline Agent** runs a full ML pipeline on any CSV dataset and explains every step in plain English. It uses **Qwen2.5** via `transformers` — **no API key** needed for inference in the hosted app.

- **Gradio UI (source):** [`automl-engineer-agent-oss/app.py`](automl-engineer-agent-oss/app.py) — root [`app.py`](app.py) is kept in sync for the Hugging Face Space flat layout.
- **Streamlit UI:** sibling package [`automl-engineer-agent`](automl-engineer-agent/) (`streamlit run app.py` there).
- **Hosting:** [Hugging Face Spaces](https://huggingface.co/spaces); set `HF_TOKEN` in **Secrets** if the model weights are gated.

For the **Claude API** tool-calling variant, see [`automl-engineer-agent`](automl-engineer-agent/).

## Run locally (Gradio)

```bash
cd automl-engineer-agent-oss
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860`. A GPU is recommended for Qwen2.5.

## License

MIT.

---
title: Explainable ML Pipeline Agent
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.25.2
app_file: gradio_app.py
pinned: true
short_description: ML pipeline that explains every step in plain English
license: mit
tags:
  - machine-learning
  - automl
  - explainable-ai
  - gradio
  - qwen
  - zerogpu
---

# Explainable ML Pipeline Agent

**Explainable ML Pipeline Agent** runs a full ML pipeline on any CSV dataset and explains every step in plain English. It uses **Qwen2.5** via `transformers` — **no API key** needed for inference in the hosted app.

- **Gradio UI:** `gradio_app.py` (this Space defaults to it when using the header above).
- **Streamlit UI:** `app.py` — same agent stack; run locally with `streamlit run app.py`.
- **Hosting:** [Hugging Face Spaces](https://huggingface.co/spaces) with **ZeroGPU** is supported; set `HF_TOKEN` in **Secrets** if the model weights are gated.

For the **Claude API** tool-calling variant, see [`automl-engineer-agent`](../automl-engineer-agent/).

## Run locally (Gradio)

```bash
cd automl-engineer-agent-oss
pip install -r requirements.txt
python gradio_app.py
```

Open `http://127.0.0.1:7860`. A GPU is recommended for Qwen2.5. The `spaces` package is optional locally.

## Run locally (Streamlit)

```bash
streamlit run app.py
```

## License

MIT.

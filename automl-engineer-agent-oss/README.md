---
title: Explainable ML Pipeline Agent
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
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
  - zerogpu
---

# Explainable ML Pipeline Agent

**Explainable ML Pipeline Agent** runs a full ML pipeline on any CSV dataset and explains every step in plain English. It uses **Qwen2.5** via `transformers` — **no API key** needed for inference in the hosted app.

- **Gradio UI:** `app.py` (this Space defaults to it when using the header above).
- **Streamlit UI:** use the sibling package [`automl-engineer-agent`](../automl-engineer-agent/) (`streamlit run app.py` there).
- **Hosting:** [Hugging Face Spaces](https://huggingface.co/spaces) with **ZeroGPU** is supported; set `HF_TOKEN` in **Secrets** if the model weights are gated.

For the **Claude API** tool-calling variant, see [`automl-engineer-agent`](../automl-engineer-agent/).

## Run locally (Gradio)

```bash
cd automl-engineer-agent-oss
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:7860`. A GPU is recommended for Qwen2.5. The `spaces` package is optional locally.

## License

MIT.

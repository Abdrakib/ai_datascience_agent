---
title: AutoML Engineer OSS
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.55.0
app_file: app.py
pinned: true
short_description: Open source AutoML agent powered by Llama 3.1
license: mit
---

# AutoML Engineer (open source)

This is the **open source** variant of the [AutoML Engineer Agent](https://github.com/huggingface/spaces) — the same ML toolchain (EDA → preprocess → train → tune → evaluate), but **without the Claude API**.

- **LLM:** [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) via `transformers`
- **Hosting:** intended for [Hugging Face Spaces](https://huggingface.co/spaces) with **ZeroGPU** — no API key for inference in the app (the Space runtime provides the GPU).
- **Gated weights:** If the base model is gated on Hugging Face, set a `HF_TOKEN` with access in the Space **Secrets** (repository settings), not in user-facing UI.

For the **Claude API** interactive agent version (tool-calling), see the sibling project folder [`automl-engineer-agent`](../automl-engineer-agent/).

## Run locally

```bash
cd automl-engineer-agent-oss
pip install -r requirements.txt
streamlit run app.py
```

Local runs do not use `spaces.ZeroGPU`; the `spaces` package is optional. A GPU is recommended for Llama 3.1 8B.

## License

MIT.

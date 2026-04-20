# Explainable ML Pipeline Agent

Monorepo with two agents (**Explainable ML Pipeline Agent** — Claude and open-source variants) built on the same ML toolchain (EDA → preprocess → train → tune → evaluate).

## Projects

| Folder | Description |
|--------|-------------|
| [**automl-engineer-agent**](automl-engineer-agent/) | **Claude API** — interactive agent with tool use, Streamlit UI, optional demo mode. |
| [**automl-engineer-agent-oss**](automl-engineer-agent-oss/) | **Open source** — **Llama 3.1 8B** via `transformers`, fixed pipeline, no user API key; aimed at **Hugging Face Spaces** with ZeroGPU. |

## Quick start

**Claude (API key required):**

```bash
cd automl-engineer-agent
pip install -r requirements.txt
streamlit run app.py
```

**Open source (local GPU recommended):**

```bash
cd automl-engineer-agent-oss
pip install -r requirements.txt
streamlit run app.py
```

## License

See each subfolder; the original Claude app includes `LICENSE` under `automl-engineer-agent/`.

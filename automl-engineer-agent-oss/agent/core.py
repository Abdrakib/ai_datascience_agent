"""
LLM orchestrator: fixed sequential ML pipeline + local LLM for plain-English blurbs.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Generator

import numpy as np
import pandas as pd
import torch
from transformers import pipeline

from config import MAX_NEW_TOKENS, MODEL_ID

from agent.tools.eda import run_eda
from agent.tools.evaluate import evaluate_model
from agent.tools.plan_training import plan_training
from agent.tools.preprocess import build_preprocessing_pipeline
from agent.tools.task_detector import detect_task
from agent.tools.train import _get_primary_score, train_and_compare
from agent.tools.tune import tune_best_model
from agent.tools.search import web_search

DEFAULT_EXPLANATION = (
    "Explanation unavailable — the model did not return text, but the pipeline step completed successfully."
)


def _domain_research_log_lines(domain_research: dict[str, Any]) -> list[str]:
    """Lines prepended to preprocessing_log from web search context."""
    q = str(domain_research.get("query", ""))
    lines: list[str] = [f"Domain research query: {q}"]
    rows = domain_research.get("results") or []
    if rows and isinstance(rows[0], dict) and rows[0].get("error"):
        lines.append(f"Web search: {rows[0]['error']}")
        return lines
    for i, r in enumerate(rows[:5], 1):
        if not isinstance(r, dict):
            continue
        title = (r.get("title") or "")[:200]
        sn = (r.get("snippet") or "")[:500]
        lines.append(f"[{i}] {title} — {sn}")
    return lines


def _safe_json_snippet(obj: Any, max_len: int = 2000) -> str:
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except TypeError:
        s = str(obj)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


_EXPL_SYSTEM_PROMPT = (
    "You are a friendly data science assistant explaining "
    "results to someone who has never studied machine learning. "
    "Use simple everyday language. Avoid technical jargon. "
    "If you must use a technical term, explain it immediately "
    "in parentheses using a simple analogy. "
    "Keep explanations to 3-4 short sentences maximum. "
    "Write like you are talking to a smart friend who is "
    "not a data scientist. "
    "Never use bullet points or markdown. Just plain "
    "conversational sentences."
)

# Terms that confuse beginners if used without a parenthetical gloss in the same text.
_JARGON_TERMS: list[tuple[str, str]] = [
    ("gradient boosting", "building many small corrective models one after another to get a strong final guess"),
    ("cross-validation", "testing the model on several slices of the data so the score is more trustworthy"),
    ("ROC-AUC", "a score from 0 to 1 that says how well the model ranks positives versus negatives"),
    ("hyperparameter", "a dial or setting on the model that changes how it learns"),
    ("imputation", "filling in missing values with guesses from the rest of the data"),
    ("encoding", "turning categories like colors or city names into numbers the model can use"),
    ("normalization", "scaling numbers so they sit on a similar range so one huge column does not dominate"),
    ("SMOTE", "a trick that creates extra examples for rare categories so the model sees them more often"),
    ("F1", "one number that blends correctness and completeness for class predictions"),
    ("precision", "out of everything the model labeled positive, how many were truly positive"),
    ("recall", "out of all real positives, how many the model actually caught"),
    ("overfitting", "when the model memorizes training data and does worse on new data"),
    ("ensemble", "combining several models so mistakes cancel out a bit"),
]


def _truncate_to_four_sentences(text: str) -> str:
    """Keep at most four sentences (split on . ? ! followed by space or end)."""
    text = (text or "").strip()
    if not text:
        return text
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p for p in parts if p.strip()]
    if len(parts) <= 4:
        return text
    return " ".join(parts[:4]).strip()


def _append_jargon_glossary(text: str) -> str:
    """If jargon appears without a following '(...)' explanation, append plain notes."""
    out = text.rstrip()
    notes: list[str] = []
    seen_gloss: set[str] = set()
    for term, plain in _JARGON_TERMS:
        if term == "ROC-AUC":
            pat = r"(?i)(?<![a-z0-9_])roc[-_ ]?auc(?![a-z0-9])"
        elif " " in term:
            pat = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
        else:
            pat = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
        if not re.search(pat, text):
            continue
        if re.search(rf"(?i){re.escape(term)}\s*\(", text):
            continue
        gloss_key = "roc-auc" if term == "ROC-AUC" else term.lower()
        if gloss_key in seen_gloss:
            continue
        seen_gloss.add(gloss_key)
        note = f"(Note: {term} means {plain}.)"
        notes.append(note)
    if notes:
        sep = " " if out and not out.endswith(" ") else ""
        out = out + sep + " ".join(notes)
    return out


def generate_explanation(
    pipe: Any,
    prompt: str,
    max_tokens: int | None = None,
) -> str:
    """Run chat-style generation; never raises — returns default on failure."""
    mt = max_tokens if max_tokens is not None else MAX_NEW_TOKENS
    messages = [
        {"role": "system", "content": _EXPL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            pad_id = getattr(tokenizer, "eos_token_id", None)
            out = pipe(
                prompt_text,
                max_new_tokens=mt,
                do_sample=False,
                temperature=1.0,
                pad_token_id=pad_id,
            )
            raw = out[0].get("generated_text", "") if isinstance(out[0], dict) else str(out[0])
            if isinstance(raw, str) and raw.startswith(prompt_text):
                gen = raw[len(prompt_text) :].strip() or DEFAULT_EXPLANATION
            elif isinstance(raw, list) and raw and isinstance(raw[-1], dict):
                gen = str(raw[-1].get("content", "")).strip() or DEFAULT_EXPLANATION
            else:
                gen = str(raw).strip() or DEFAULT_EXPLANATION
            gen = _truncate_to_four_sentences(gen)
            return _append_jargon_glossary(gen)

        out = pipe(
            messages,
            max_new_tokens=mt,
            do_sample=False,
            temperature=1.0,
            pad_token_id=getattr(pipe, "tokenizer", None)
            and getattr(pipe.tokenizer, "eos_token_id", None),
        )
        o0 = out[0] if out else {}
        if isinstance(o0, dict):
            gt = o0.get("generated_text")
            if isinstance(gt, list) and gt:
                last = gt[-1]
                if isinstance(last, dict) and last.get("content"):
                    gen = str(last["content"]).strip()
                    gen = _truncate_to_four_sentences(gen)
                    return _append_jargon_glossary(gen)
            if isinstance(gt, str):
                gen = gt.strip()
                gen = _truncate_to_four_sentences(gen)
                return _append_jargon_glossary(gen)
        return DEFAULT_EXPLANATION
    except Exception:
        return DEFAULT_EXPLANATION


def load_llm_pipeline():
    """Load the instruction-tuned model for text generation (GPU recommended)."""
    return pipeline(
        "text-generation",
        model=MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )


def _sanitize_metrics(m: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (m or {}).items():
        if k in ("y_prob", "classification_report"):
            continue
        if isinstance(v, (np.floating, np.integer)):
            out[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)[:500]
    return out


def _prep_for_report(prep: dict[str, Any]) -> dict[str, Any]:
    """Serializable prep dict + fields expected by agent/report.py."""
    out = {k: v for k, v in prep.items() if k not in ("X_train", "X_test", "y_train", "y_test", "pipeline")}
    fn = prep.get("feature_names") or []
    out["final_feature_count"] = len(fn)
    out["train_size"] = int(prep["X_train"].shape[0])
    out["test_size"] = int(prep["X_test"].shape[0])
    return out


class OssAutoMLAgent:
    """Fixed-order pipeline; local LLM explains EDA, model choice, SHAP, and final summary."""

    def __init__(self, df: pd.DataFrame, goal: str, pipe: Any) -> None:
        self.df = df
        self.goal = (goal or "").strip()
        self.pipe = pipe

    def run(self) -> Generator[dict[str, Any], None, None]:
        yield {"type": "log", "content": "Pipeline run started — executing fixed ML sequence."}

        yield {"type": "step_start", "name": "EDA", "step": 1}
        t0 = time.perf_counter()
        eda_result = run_eda(self.df)
        eda_summary = _safe_json_snippet(
            {
                "overview": eda_result.get("overview"),
                "quality_flags": eda_result.get("quality_flags"),
                "missing": eda_result.get("missing"),
            }
        )
        eda_prompt = (
            "Explain these dataset facts to a complete beginner "
            "in 3 simple sentences. Tell them: how big the dataset "
            "is, what kind of information it contains, and whether "
            "there are any obvious problems they should know about.\n"
            f"Dataset info: {eda_summary}"
        )
        eda_result = dict(eda_result)
        eda_result["explanation"] = generate_explanation(self.pipe, eda_prompt)
        elapsed = time.perf_counter() - t0
        yield {"type": "step_done", "name": "EDA", "step": 1, "result": eda_result}
        yield {"type": "log", "content": f"✓ EDA completed in {elapsed:.1f}s"}

        yield {"type": "step_start", "name": "Task detection", "step": 2}
        t0 = time.perf_counter()
        task_info = detect_task(self.df, user_hint=self.goal)
        elapsed = time.perf_counter() - t0
        task_info = dict(task_info)
        task_summary = _safe_json_snippet(task_info)
        task_info["explanation"] = generate_explanation(
            self.pipe,
            "In 2-3 simple sentences, explain what the computer "
            "is trying to learn from this data. "
            "Use an analogy if possible. "
            'For example if it is classification say something like '
            '"The agent is trying to learn how to sort things into '
            'categories, like sorting emails into spam and not spam."\n'
            f"Task info: {task_summary}",
        )
        yield {"type": "step_done", "name": "Task detection", "step": 2, "result": task_info}
        yield {"type": "log", "content": f"✓ Task detection completed in {elapsed:.1f}s"}

        domain_research: dict[str, Any] | None = None
        conf = str(task_info.get("confidence", "")).lower()
        if conf in ("low", "medium"):
            col_names = list(self.df.columns)[:5]
            query = f"{self.goal} dataset {' '.join(col_names)} machine learning features"
            try:
                domain_research = {"query": query, "results": web_search(query, num_results=5)}
            except Exception as e:
                domain_research = {
                    "query": query,
                    "results": [{"error": f"{type(e).__name__}: {e}"}],
                }
            search_summary = _safe_json_snippet(domain_research)
            domain_expl = generate_explanation(
                self.pipe,
                "In 2 simple sentences, explain why the agent searched "
                "the internet and what useful information it found that "
                "will help it understand this dataset better.\n"
                f"Search results: {search_summary}",
            )
            dr_result = {
                "domain_research": domain_research,
                "message": (
                    "The agent searched the web to better understand your dataset. "
                    "Here is what it found:"
                ),
                "explanation": domain_expl,
            }
            yield {
                "type": "step_done",
                "name": "Step 2b: Domain Research",
                "step": "2b",
                "result": dr_result,
            }
            yield {"type": "log", "content": "✓ Domain research (web) completed"}

        yield {"type": "step_start", "name": "Preprocessing", "step": 3}
        t0 = time.perf_counter()
        prep = build_preprocessing_pipeline(
            self.df,
            target_col=task_info["target_col"],
            task_type=task_info["task_type"],
        )
        if domain_research:
            extra = ["--- Domain research context (for preprocessing) ---"] + _domain_research_log_lines(
                domain_research
            )
            log = list(prep.get("preprocessing_log") or [])
            prep["preprocessing_log"] = extra + log
            prep["domain_research"] = domain_research
        prep_summary = _safe_json_snippet(
            {
                "prep": _prep_for_report(prep),
                "domain_research": domain_research,
            }
        )
        prep_explain = generate_explanation(
            self.pipe,
            "In 3 simple sentences, explain what the agent did "
            "to clean and prepare the data before training. "
            "Use everyday language — for example instead of "
            "'imputed missing values' say 'filled in the blank "
            "spots in the data with reasonable guesses'.\n"
            f"Preprocessing info: {prep_summary}",
        )
        elapsed = time.perf_counter() - t0
        yield {
            "type": "step_done",
            "name": "Preprocessing",
            "step": 3,
            "result": {"prep": prep, "explanation": prep_explain},
        }
        yield {"type": "log", "content": f"✓ Preprocessing completed in {elapsed:.1f}s"}

        yield {"type": "step_start", "name": "Training plan", "step": 4}
        t0 = time.perf_counter()
        plan_result = plan_training(eda_result, task_info, prep)
        plan_summary = _safe_json_snippet(plan_result.get("plan_summary"))
        plan_explain = generate_explanation(
            self.pipe,
            "In 2-3 simple sentences, explain which machine "
            "learning methods the agent decided to try and why. "
            "Think of it like choosing which tools to use for "
            "a job — explain it that way.\n"
            f"Plan info: {plan_summary}",
        )
        plan_out = dict(plan_result)
        plan_out["explanation"] = plan_explain
        elapsed = time.perf_counter() - t0
        yield {"type": "step_done", "name": "Training plan", "step": 4, "result": plan_out}
        yield {"type": "log", "content": f"✓ Training plan completed in {elapsed:.1f}s"}

        X_train = prep["X_train"]
        X_test = prep["X_test"]
        y_train = prep["y_train"]
        y_test = prep["y_test"]
        task_type = str(prep["task_type"])
        if task_type == "classification":
            n_classes = int(prep["n_classes"]) if prep.get("n_classes") is not None else 2
        else:
            n_classes = 2
        feature_names = prep["feature_names"]
        label_encoder = prep.get("label_encoder")

        yield {"type": "step_start", "name": "Training", "step": 5}
        t0 = time.perf_counter()
        train_result = train_and_compare(
            X_train,
            X_test,
            y_train,
            y_test,
            task_type,
            feature_names=feature_names,
            n_classes=n_classes,
            adjusted_params=plan_result.get("adjusted_params"),
            skip_models=plan_result.get("skip_models"),
            skip_reasons=plan_result.get("skip_reasons"),
            primary_metric=plan_result.get("primary_metric"),
        )
        _cdf = train_result.get("comparison_df")
        _comp_sample = ""
        if _cdf is not None and hasattr(_cdf, "head"):
            try:
                _comp_sample = _cdf.head(12).to_string()
            except Exception:
                _comp_sample = str(_cdf)[:1500]
        train_summary = _safe_json_snippet(
            {
                "best_model": train_result.get("best_name"),
                "metric_focus": train_result.get("metric_name"),
                "best_metrics": _sanitize_metrics(train_result.get("best_metrics") or {}),
                "comparison_sample": _comp_sample[:2000],
            }
        )
        train_expl = generate_explanation(
            self.pipe,
            "In 3 simple sentences, explain how the different "
            "methods performed. Tell the reader which method won "
            "and give them a sense of how good the score is — "
            "for example 'a score of 0.85 out of 1.0 means the "
            "model gets it right about 85% of the time'.\n"
            f"Training results: {train_summary}",
        )
        elapsed = time.perf_counter() - t0
        yield {
            "type": "step_done",
            "name": "Training",
            "step": 5,
            "result": {"train": train_result, "explanation": train_expl},
        }
        yield {"type": "log", "content": f"✓ Training completed in {elapsed:.1f}s"}

        for w in train_result.get("overfitting_warnings") or []:
            yield {"type": "log", "content": f"⚠ Overfitting warning: {w}"}
        for r in train_result.get("results") or []:
            if r.get("metrics", {}).get("overfit"):
                nm = r.get("name", "model")
                yield {"type": "log", "content": f"⚠ {nm}: elevated train vs test gap (overfit flag)."}

        yield {"type": "step_start", "name": "Tuning", "step": 6}
        t0 = time.perf_counter()
        baseline_score = float(
            _get_primary_score(
                train_result.get("best_metrics") or {},
                task_type,
                train_result.get("metric_name"),
            )
        )

        tune_result = tune_best_model(
            str(train_result["best_name"]),
            task_type,
            X_train,
            y_train,
            X_test,
            y_test,
            baseline_score,
            n_trials=int(plan_result.get("n_trials") or 50),
            timeout=int(plan_result.get("timeout") or 120),
        )
        tune_result = dict(tune_result)
        tune_result["model_name"] = str(train_result["best_name"])
        tune_summary = _safe_json_snippet(
            {k: tune_result.get(k) for k in ("success", "best_score", "improvement", "error")}
        )
        tune_expl = generate_explanation(
            self.pipe,
            "In 2-3 simple sentences, explain what hyperparameter "
            "tuning means in plain English and whether it helped. "
            "A good analogy: tuning is like adjusting the settings "
            "on a machine to make it work better.\n"
            f"Tuning results: {tune_summary}",
        )
        elapsed = time.perf_counter() - t0
        yield {
            "type": "step_done",
            "name": "Tuning",
            "step": 6,
            "result": {"tune": tune_result, "explanation": tune_expl},
        }
        yield {"type": "log", "content": f"✓ Tuning completed in {elapsed:.1f}s"}
        if tune_result.get("success") and float(tune_result.get("improvement") or 0) > 1e-6:
            yield {
                "type": "log",
                "content": (
                    f"Tuning improved primary metric by {float(tune_result.get('improvement')):+.4f} "
                    f"vs baseline."
                ),
            }

        model_for_eval = train_result["best_model"]
        if tune_result.get("success") and tune_result.get("tuned_model") is not None:
            model_for_eval = tune_result["tuned_model"]

        yield {"type": "step_start", "name": "Evaluation", "step": 7}
        t0 = time.perf_counter()
        eval_result = evaluate_model(
            model_for_eval,
            X_test,
            y_test,
            X_train,
            y_train,
            task_type,
            feature_names=feature_names,
            label_encoder=label_encoder,
            run_id="oss_run",
            n_classes=n_classes,
        )
        shap_txt = eval_result.get("shap_explanation_text") or ""
        eval_summary = _safe_json_snippet(
            {
                "metrics": _sanitize_metrics(eval_result.get("metrics") or {}),
                "explainability_notes": (shap_txt or "")[:1200],
            }
        )
        eval_expl = generate_explanation(
            self.pipe,
            "In 3-4 simple sentences, explain how well the final "
            "model performs. Make the numbers meaningful — "
            "for example if accuracy is 0.78 say "
            "'the model makes the right prediction about 78 times "
            "out of every 100 — which is pretty good for this "
            "kind of problem'.\n"
            f"Evaluation results: {eval_summary}",
        )
        elapsed = time.perf_counter() - t0
        yield {
            "type": "step_done",
            "name": "Evaluation",
            "step": 7,
            "result": {"eval": eval_result, "explanation": eval_expl},
        }
        yield {"type": "log", "content": f"✓ Evaluation completed in {elapsed:.1f}s — metrics and plots ready."}

        eda_rich = run_eda(self.df, target_col=task_info["target_col"])

        final_summary_payload = _safe_json_snippet(
            {
                "goal": self.goal,
                "best_model": train_result.get("best_name"),
                "metrics": _sanitize_metrics(eval_result.get("metrics") or {}),
                "tuning": {k: tune_result.get(k) for k in ("success", "improvement", "best_score")},
            }
        )
        final_summary = generate_explanation(
            self.pipe,
            "In 3-4 simple sentences, give a plain English "
            "summary of what was accomplished. Mention which "
            "method worked best and why, how confident they "
            "should feel about the results, and one practical "
            "thing they could do next with this model. "
            "Keep it encouraging and clear.\n"
            f"Final results: {final_summary_payload}",
            max_tokens=400,
        )

        prep_report = _prep_for_report(prep)

        final_result: dict[str, Any] = {
            "status": "complete",
            "goal": self.goal,
            "target_col": task_info["target_col"],
            "task_type": task_type,
            "eda": eda_rich,
            "task": task_info,
            "domain_research": domain_research,
            "prep": prep_report,
            "prep_raw": prep,
            "plan": plan_result,
            "train": train_result,
            "tune": tune_result,
            "eval": eval_result,
            "final_summary": final_summary,
            "best_model_name": train_result.get("best_name"),
            "best_metrics": train_result.get("best_metrics"),
            "feature_importances": train_result.get("feature_importances"),
            "comparison_df": train_result.get("comparison_df"),
            "metrics": eval_result.get("metrics"),
            "plot_paths": eval_result.get("plot_paths"),
            "model_for_inference": model_for_eval,
            "prep_pipeline": prep.get("pipeline"),
            "label_encoder": label_encoder,
            "feature_names": feature_names,
            "overfitting_search_results": [],
        }

        yield {
            "type": "step_done",
            "name": "Final summary",
            "step": 8,
            "result": {"explanation": final_summary, "full": final_result},
        }
        yield {"type": "log", "content": "Pipeline finished successfully."}
        yield {"type": "done", "result": final_result}


def run_pipeline(pipe: Any, df: pd.DataFrame, goal: str) -> list[dict[str, Any]]:
    """Optional: materialize the full event stream as a list."""
    return list(OssAutoMLAgent(df, goal, pipe).run())

"""
LLM orchestrator: fixed sequential ML pipeline + Llama for plain-English blurbs only.
"""

from __future__ import annotations

import json
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

try:
    import spaces  # type: ignore

    USING_ZEROGPU = True
except ImportError:
    USING_ZEROGPU = False

    def _gpu_noop(func):
        return func

    spaces = type("spaces", (), {"GPU": staticmethod(_gpu_noop)})()

DEFAULT_EXPLANATION = (
    "Explanation unavailable — the model did not return text, but the pipeline step completed successfully."
)


def _safe_json_snippet(obj: Any, max_len: int = 2000) -> str:
    try:
        s = json.dumps(obj, default=str, ensure_ascii=False)
    except TypeError:
        s = str(obj)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def generate_explanation(
    pipe: Any,
    prompt: str,
    max_tokens: int | None = None,
) -> str:
    """Run chat-style generation; never raises — returns default on failure."""
    mt = max_tokens if max_tokens is not None else MAX_NEW_TOKENS
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful data science assistant. "
                "Give clear, concise explanations in 2-3 sentences. "
                "No markdown, no bullet points, just plain English."
            ),
        },
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
                return raw[len(prompt_text) :].strip() or DEFAULT_EXPLANATION
            if isinstance(raw, list) and raw and isinstance(raw[-1], dict):
                return str(raw[-1].get("content", "")).strip() or DEFAULT_EXPLANATION
            return str(raw).strip() or DEFAULT_EXPLANATION

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
                    return str(last["content"]).strip()
            if isinstance(gt, str):
                return gt.strip()
        return DEFAULT_EXPLANATION
    except Exception:
        return DEFAULT_EXPLANATION


@spaces.GPU
def load_llm_pipeline():
    """Load Llama for text generation (ZeroGPU on Hugging Face Spaces)."""
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


def _prep_for_step_ui(prep: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_type": prep.get("task_type"),
        "feature_count": len(prep.get("feature_names") or []),
        "n_classes": prep.get("n_classes"),
        "dropped_cols": prep.get("dropped_cols"),
        "smote_applied": prep.get("smote_applied"),
        "preprocessing_log": prep.get("preprocessing_log"),
        "train_shape": tuple(prep["X_train"].shape),
        "test_shape": tuple(prep["X_test"].shape),
    }


def _train_for_step_ui(tr: dict[str, Any]) -> dict[str, Any]:
    cdf = tr.get("comparison_df")
    cdf_md = ""
    try:
        if cdf is not None and hasattr(cdf, "to_markdown"):
            cdf_md = cdf.to_markdown(index=False)
    except Exception:
        cdf_md = str(cdf)
    return {
        "best_name": tr.get("best_name"),
        "metric_name": tr.get("metric_name"),
        "best_metrics": _sanitize_metrics(tr.get("best_metrics") or {}),
        "comparison_md": cdf_md[:8000],
        "training_log": tr.get("training_log"),
        "overfitting_warnings": tr.get("overfitting_warnings"),
        "feature_importances": tr.get("feature_importances"),
    }


def _tune_for_step_ui(tu: dict[str, Any]) -> dict[str, Any]:
    if not tu.get("success"):
        return {"success": False, "error": tu.get("error", "Tuning failed.")}
    return {
        "success": True,
        "best_score": tu.get("best_score"),
        "baseline_score": tu.get("baseline_score"),
        "improvement": tu.get("improvement"),
        "n_trials_run": tu.get("n_trials_run"),
        "tuning_log": tu.get("tuning_log"),
        "overfit": tu.get("overfit"),
    }


def _eval_for_step_ui(ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "metrics": _sanitize_metrics(ev.get("metrics") or {}),
        "plot_paths": ev.get("plot_paths"),
        "has_shap": ev.get("has_shap"),
        "eval_log": ev.get("eval_log"),
        "shap_explanation_text": (ev.get("shap_explanation_text") or "")[:4000],
    }


class OssAutoMLAgent:
    """Fixed-order pipeline; Llama explains EDA, model choice, SHAP, and final summary."""

    def __init__(self, df: pd.DataFrame, goal: str, pipe: Any) -> None:
        self.df = df
        self.goal = (goal or "").strip()
        self.pipe = pipe

    def run(self) -> Generator[dict[str, Any], None, None]:
        yield {"type": "step_start", "name": "EDA", "step": 1}
        eda_result = run_eda(self.df)
        eda_prompt = (
            "Explain these dataset statistics in plain English: "
            + _safe_json_snippet(eda_result.get("overview"))
        )
        eda_result = dict(eda_result)
        eda_result["explanation"] = generate_explanation(self.pipe, eda_prompt)
        yield {"type": "step_done", "name": "EDA", "step": 1, "result": eda_result}

        yield {"type": "step_start", "name": "Task detection", "step": 2}
        task_info = detect_task(self.df, user_hint=self.goal)
        yield {"type": "step_done", "name": "Task detection", "step": 2, "result": task_info}

        yield {"type": "step_start", "name": "Preprocessing", "step": 3}
        prep = build_preprocessing_pipeline(
            self.df,
            target_col=task_info["target_col"],
            task_type=task_info["task_type"],
        )
        prep_summary = _prep_for_step_ui(prep)
        prep_summary["explanation"] = generate_explanation(
            self.pipe,
            "Summarize what preprocessing did and why it matters for modeling, in plain English: "
            + _safe_json_snippet(prep_summary),
        )
        yield {"type": "step_done", "name": "Preprocessing", "step": 3, "result": prep_summary}

        yield {"type": "step_start", "name": "Training plan", "step": 4}
        plan_result = plan_training(eda_result, task_info, prep)
        plan_explain = generate_explanation(
            self.pipe,
            "Explain this training plan briefly for a practitioner: "
            + _safe_json_snippet(plan_result.get("plan_summary")),
        )
        plan_out = dict(plan_result)
        plan_out["explanation"] = plan_explain
        yield {"type": "step_done", "name": "Training plan", "step": 4, "result": plan_out}

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
        train_ui = _train_for_step_ui(train_result)
        train_ui["explanation"] = generate_explanation(
            self.pipe,
            "Explain briefly why this model may be a reasonable choice given these results "
            f"(best: {train_result.get('best_name')}): "
            + _safe_json_snippet(train_ui.get("best_metrics")),
        )
        yield {"type": "step_done", "name": "Training", "step": 5, "result": train_ui}

        yield {"type": "step_start", "name": "Tuning", "step": 6}
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
        tune_ui = _tune_for_step_ui(tune_result)
        tune_ui["explanation"] = generate_explanation(
            self.pipe,
            "In one short paragraph, describe hyperparameter tuning outcome: "
            + _safe_json_snippet(tune_ui),
        )
        yield {"type": "step_done", "name": "Tuning", "step": 6, "result": tune_ui}

        model_for_eval = train_result["best_model"]
        if tune_result.get("success") and tune_result.get("tuned_model") is not None:
            model_for_eval = tune_result["tuned_model"]

        yield {"type": "step_start", "name": "Evaluation", "step": 7}
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
        eval_ui = _eval_for_step_ui(eval_result)
        shap_txt = eval_result.get("shap_explanation_text") or ""
        eval_ui["explanation"] = generate_explanation(
            self.pipe,
            "Interpret these SHAP / explainability notes in plain English: " + (shap_txt or "No SHAP text."),
        )
        yield {"type": "step_done", "name": "Evaluation", "step": 7, "result": eval_ui}

        final_summary = generate_explanation(
            self.pipe,
            "Give a short executive summary of this AutoML run: goal, best model, and whether metrics look solid. "
            + _safe_json_snippet(
                {
                    "goal": self.goal,
                    "best_model": train_result.get("best_name"),
                    "metrics": eval_ui.get("metrics"),
                    "tuning": tune_ui,
                }
            ),
            max_tokens=400,
        )

        final_result: dict[str, Any] = {
            "status": "success",
            "goal": self.goal,
            "target_col": task_info["target_col"],
            "task_type": task_type,
            "eda": eda_result,
            "task": task_info,
            "prep": prep,
            "plan": plan_result,
            "train": train_result,
            "tune": tune_result,
            "eval": eval_result,
            "final_summary": final_summary,
            "best_model_name": train_result.get("best_name"),
            "model_for_inference": model_for_eval,
            "prep_pipeline": prep.get("pipeline"),
            "label_encoder": label_encoder,
            "feature_names": feature_names,
        }
        yield {"type": "done", "result": final_result}


@spaces.GPU
def run_pipeline(pipe: Any, df: pd.DataFrame, goal: str) -> list[dict[str, Any]]:
    """
    Optional single-GPU entry for Spaces: same steps as OssAutoMLAgent.run(), materialized as a list.
    The Streamlit app uses OssAutoMLAgent.run() directly for progressive yields.
    """
    return list(OssAutoMLAgent(df, goal, pipe).run())

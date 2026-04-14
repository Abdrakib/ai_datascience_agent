"""
agent/core.py — Claude agent orchestrator.

The agent receives a user message + a dataframe, then autonomously:
  1. Runs EDA
  2. Detects the ML task and target column
  3. Preprocesses the data
  4. Trains and compares models
  5. Evaluates the best model + generates plots
  6. Returns a structured result the UI can render

It uses Claude's tool-use API so the LLM decides the order of operations,
can ask clarifying questions, and explains its reasoning at each step.
"""

from __future__ import annotations

import json
import traceback
from typing import Any, Generator

import anthropic
import numpy as np
import pandas as pd

import os

from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, CV_FOLDS, MAX_AGENT_ITERATIONS, MAX_TOKENS
from agent.tools.eda import run_eda, eda_to_markdown
from agent.tools.task_detector import detect_task, task_detection_to_markdown
from agent.tools.preprocess import build_preprocessing_pipeline, preprocessing_log_to_markdown
from agent.tools.plan_training import plan_training, plan_to_markdown
from agent.tools.train import (
    OVERFIT_GAP_THRESHOLD_CLF,
    OVERFIT_GAP_THRESHOLD_REG,
    train_and_compare,
    training_results_to_markdown,
    tuning_result_to_markdown,
    _get_feature_importances,
    _get_primary_score,
)
from agent.tools.tune import tune_best_model
from agent.tools.evaluate import evaluate_model, evaluation_to_markdown
from agent.tools.search import web_search


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


# ── Tool schemas (passed to Claude API) ──────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "run_eda",
        "description": (
            "Run exploratory data analysis on the loaded dataset. "
            "Returns shape, dtypes, missing values, stats, quality flags, "
            "and preprocessing recommendations. Always run this first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_col": {
                    "type": "string",
                    "description": "Target column name if already known. Leave empty to auto-detect.",
                }
            },
            "required": [],
        },
    },
    {
        "name": "detect_task",
        "description": (
            "Detect the ML task type (classification or regression) and identify "
            "the target column. Run after EDA. Pass the user's goal as user_hint."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "user_hint": {
                    "type": "string",
                    "description": "User's description of what they want to predict.",
                },
                "target_col": {
                    "type": "string",
                    "description": "Override target column if user specified it explicitly.",
                },
            },
            "required": ["user_hint"],
        },
    },
    {
        "name": "preprocess",
        "description": (
            "Build and fit the preprocessing pipeline. Automatically handles missing "
            "values, encoding, scaling, log transforms for skewed columns, and SMOTE "
            "resampling when class imbalance is detected. Run after detect_task."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_col": {
                    "type": "string",
                    "description": "Target column name (from detect_task result).",
                },
                "task_type": {
                    "type": "string",
                    "enum": ["classification", "regression"],
                    "description": "Task type (from detect_task result).",
                },
                "scaler_type": {
                    "type": "string",
                    "enum": ["standard", "minmax", "none"],
                    "description": "Scaling strategy. Default: standard.",
                },
            },
            "required": ["target_col", "task_type"],
        },
    },
    {
        "name": "plan_training",
        "description": (
            "Analyze the dataset characteristics from EDA, task detection, and "
            "preprocessing results to build an intelligent training plan. Determines "
            "which models to use, adjusts hyperparameters based on dataset size and "
            "imbalance, selects the right evaluation metric, and sets the Optuna "
            "tuning budget. Always call this after preprocess and before train_models."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "train_models",
        "description": (
            "Train and compare multiple ML models (Logistic/Linear Regression, "
            "Random Forest, XGBoost, LightGBM). Returns ranked results and the best model. "
            "Run after plan_training (or after preprocess if no plan step)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "tune_model",
        "description": (
            "Tune the best model found by train_models using Optuna Bayesian optimization. "
            "Run this after train_models to find the optimal hyperparameters. "
            "Returns improved metrics and the best parameter configuration."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "n_trials": {
                    "type": "integer",
                    "description": (
                        "Number of Optuna trials. Use 20 for small datasets under 1000 rows, "
                        "50 for medium, 100 for large datasets above 10000 rows."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum tuning time in seconds.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "evaluate_model",
        "description": (
            "Evaluate the best model on the test set. Generates metrics, confusion matrix, "
            "ROC curve, actual-vs-predicted, residuals, feature importance, and SHAP plots. "
            "Run after tune_model (or train_models if tuning was skipped)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": "string",
                    "description": "Short identifier for naming saved plot files.",
                }
            },
            "required": [],
        },
    },
]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are AutoML Engineer, an expert ML engineering agent.

Your job is to take any CSV dataset and a user goal, then autonomously run the
full ML pipeline: EDA → task detection → preprocessing → training plan → model training → tuning → evaluation.

## Rules
- Always start with run_eda to understand the data before anything else.
- Always call detect_task after EDA, passing the user's goal as user_hint.
- Always call preprocess after detect_task.
- plan_training — always call after preprocess and before train_models.
  It analyzes dataset size, imbalance, dimensionality and produces an
  intelligent training configuration. Read the plan summary carefully
  and explain it to the user before training starts.
- Always call train_models after plan_training (or after preprocess if you skip planning).
- train_models — trains all planned models with k-fold cross-validation (default 5 folds). Reports both CV mean scores and single held-out test scores; CV mean is used to pick the best model for more reliable comparison.
- After train_models completes, always call tune_model to optimize the best model's hyperparameters. Respect the Optuna budget from plan_training when present; otherwise use fewer trials (20) for small datasets and more (50-100) for large ones.
- Always call evaluate_model last.
- Between tool calls, briefly explain what you found and what you're doing next.
- Be specific: mention actual column names, actual metric values, actual findings.
- If something looks wrong (e.g. very high missing rate, severe class imbalance),
  flag it clearly and explain how you handled it.
- Keep explanations concise — the user sees them as a live activity log.
- After evaluate_model, write a final summary: best model, key metrics,
  top 3 features, and 2-3 concrete next steps to improve the model further.

## Tone
Act like a senior ML engineer walking a colleague through the analysis.
Be direct, specific, and technically precise.
"""


# ── Agent class ───────────────────────────────────────────────────────────────

class AutoMLAgent:
    """
    Stateful agent that runs the full ML pipeline for a single dataset.

    Usage
    -----
    agent = AutoMLAgent(df, user_message="predict whether patient survived")
    for event in agent.run():
        print(event)          # stream progress to UI
    result = agent.result     # final structured output
    """

    def __init__(self, df: pd.DataFrame, user_message: str):
        self.df = df
        self.user_message = user_message
        _key = (os.getenv("ANTHROPIC_API_KEY", "") or "").strip() or ANTHROPIC_API_KEY
        self.client = anthropic.Anthropic(api_key=_key) if _key else None

        # Shared state populated by tool calls
        self._eda_report: dict | None = None
        self._task_result: dict | None = None
        self._domain_research: dict[str, Any] | None = None
        self._prep_result: dict | None = None
        self._plan_result: dict | None = None
        self._train_result: dict | None = None
        self._tune_result: dict | None = None
        self._eval_result: dict | None = None

        # Final structured result (set when pipeline completes)
        self.result: dict[str, Any] = {}

    # ── Public streaming interface ────────────────────────────────────────────

    def run(self) -> Generator[dict, None, None]:
        """
        Stream agent events as dicts:
          {"type": "text",    "content": str}          — Claude narration
          {"type": "tool",    "name": str, "status": "running"|"done", "output": str}
          {"type": "error",   "content": str}
          {"type": "done",    "result": dict}           — final structured result
        """
        if self.client is None:
            yield {
                "type": "error",
                "content": (
                    "ANTHROPIC_API_KEY is not set. Add it to your .env file or paste it in the "
                    "Streamlit sidebar, then run again."
                ),
            }
            return

        messages = [
            {
                "role": "user",
                "content": (
                    f"Dataset info: {len(self.df)} rows × {len(self.df.columns)} columns. "
                    f"Columns: {', '.join(self.df.columns.tolist())}.\n\n"
                    f"User goal: {self.user_message}\n\n"
                    "Please run the full ML pipeline."
                ),
            }
        ]

        for iteration in range(MAX_AGENT_ITERATIONS):
            response = self.client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            # Collect text and tool_use blocks
            tool_calls = []
            for block in response.content:
                if block.type == "text" and block.text.strip():
                    yield {"type": "text", "content": block.text.strip()}
                elif block.type == "tool_use":
                    tool_calls.append(block)

            # If no tool calls → Claude is done
            if response.stop_reason == "end_turn" or not tool_calls:
                break

            # Append assistant message
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call
            tool_results = []
            for tool_block in tool_calls:
                tool_name = tool_block.name
                tool_input = tool_block.input

                running_evt: dict[str, Any] = {
                    "type": "tool",
                    "name": tool_name,
                    "status": "running",
                }
                if tool_name == "tune_model" and self._train_result:
                    running_evt["tune_model_name"] = self._train_result.get(
                        "best_name", "best model"
                    )
                yield running_evt

                try:
                    output = self._dispatch(tool_name, tool_input)
                    step_data = self._get_step_data(tool_name)
                    yield {"type": "tool", "name": tool_name,
                           "status": "done", "output": output, "step_data": step_data}
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": output,
                    })
                except Exception as e:
                    err = f"Tool '{tool_name}' failed: {e}\n{traceback.format_exc()}"
                    yield {"type": "error", "content": err}
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": f"ERROR: {e}",
                        "is_error": True,
                    })

            messages.append({"role": "user", "content": tool_results})

        # Build and emit final result
        self.result = self._build_result()
        yield {"type": "done", "result": self.result}

    # ── Tool dispatcher ───────────────────────────────────────────────────────

    def _dispatch(self, name: str, inputs: dict) -> str:
        if name == "run_eda":
            return self._tool_eda(inputs)
        elif name == "detect_task":
            return self._tool_detect_task(inputs)
        elif name == "preprocess":
            return self._tool_preprocess(inputs)
        elif name == "plan_training":
            return self._tool_plan_training(inputs)
        elif name == "train_models":
            return self._tool_train(inputs)
        elif name == "tune_model":
            return self._tool_tune_model(inputs)
        elif name == "evaluate_model":
            return self._tool_evaluate(inputs)
        else:
            raise ValueError(f"Unknown tool: {name}")

    def _get_step_data(self, tool_name: str) -> dict | None:
        """Return structured data for the UI step card."""
        if tool_name == "run_eda" and self._eda_report:
            return {"eda": self._eda_report}
        if tool_name == "detect_task" and self._task_result:
            out: dict[str, Any] = {"task": self._task_result}
            if self._domain_research:
                out["domain_research"] = self._domain_research
            return out
        if tool_name == "plan_training" and getattr(self, "_plan_result", None):
            return {"plan": self._plan_result}
        if tool_name == "preprocess" and self._prep_result:
            p = self._prep_result
            prep = {k: v for k, v in p.items()
                    if k not in ("X_train", "X_test", "y_train", "y_test", "pipeline", "label_encoder")}
            if "X_train" in p:
                prep["train_size"] = p["X_train"].shape[0]
                prep["test_size"] = p["X_test"].shape[0]
                prep["final_feature_count"] = p["X_train"].shape[1]
            # Target leakage check: correlation > 0.95 between target and any non-target
            target_col = self._task_result["target_col"] if self._task_result else None
            if target_col and target_col in self.df.columns:
                y = self.df[target_col]
                if pd.api.types.is_numeric_dtype(y):
                    X_num = self.df.drop(columns=[target_col]).select_dtypes(include="number")
                    for c in X_num.columns:
                        corr = self.df[[c, target_col]].corr().iloc[0, 1]
                        if not np.isnan(corr) and abs(corr) > 0.95:
                            prep["target_leakage_suspicion"] = f"Column '{c}' has correlation {corr:.3f} with target."
                            break
            return {"prep": prep}
        if tool_name == "train_models" and self._train_result:
            t = self._train_result
            train = {k: v for k, v in t.items() if k not in ("best_model", "results")}
            train["results"] = []
            for r in t.get("results", []):
                train["results"].append({
                    "name":               r["name"],
                    "metrics":            r["metrics"],
                    "cv_scores":          r.get("cv_scores"),
                    "cv_mean":            r.get("cv_mean"),
                    "cv_std":             r.get("cv_std"),
                    "cv_train_scores":    r.get("cv_train_scores"),
                    "cv_train_mean":      r.get("cv_train_mean"),
                    "cv_overfit":         r.get("cv_overfit"),
                    "train_score":        r.get("train_score"),
                    "generalization_gap": r.get("generalization_gap"),
                    "overfit":            r.get("overfit"),
                })
            return {"train": train}
        if tool_name == "tune_model" and self._tune_result:
            tr = {k: v for k, v in self._tune_result.items() if k != "tuned_model"}
            if self._train_result:
                raw = str(self._train_result.get("best_name", "")).replace(" (tuned)", "").strip()
                if raw:
                    tr["model_name"] = raw
            return {"tune": tr}
        if tool_name == "evaluate_model" and self._eval_result:
            return {"eval": {k: v for k, v in self._eval_result.items()
                     if k not in ("shap_values", "y_pred")}}
        return None

    # ── Tool implementations ──────────────────────────────────────────────────

    def _tool_eda(self, inputs: dict) -> str:
        target_hint = inputs.get("target_col")
        self._eda_report = run_eda(self.df, target_col=target_hint)
        return eda_to_markdown(self._eda_report)

    def _tool_detect_task(self, inputs: dict) -> str:
        user_hint  = inputs.get("user_hint", self.user_message)
        target_override = inputs.get("target_col")

        if target_override:
            # User specified target explicitly — skip scoring
            from agent.tools.task_detector import _infer_task_type
            task_type, reason = _infer_task_type(self.df[target_override], user_hint.lower())
            self._task_result = {
                "target_col":   target_override,
                "task_type":    task_type,
                "confidence":   "high",
                "reasoning":    f"Target specified by user. {reason}",
                "alternatives": [],
                "n_classes":    int(self.df[target_override].nunique()) if task_type == "classification" else None,
                "target_dtype": str(self.df[target_override].dtype),
            }
        else:
            self._task_result = detect_task(self.df, user_hint=user_hint)

        self._domain_research = None
        conf = str(self._task_result.get("confidence", "")).lower()
        if conf in ("low", "medium"):
            goal = (user_hint or self.user_message or "").strip()
            col_names = list(self.df.columns)[:5]
            query = f"{goal} dataset {' '.join(col_names)} machine learning features"
            try:
                results = web_search(query, num_results=5)
                self._domain_research = {"query": query, "results": results}
            except Exception as e:
                self._domain_research = {
                    "query": query,
                    "results": [{"error": f"{type(e).__name__}: {e}"}],
                }

        md = task_detection_to_markdown(self._task_result)
        if self._domain_research:
            md += "\n\n## Domain research (web)\n"
            for i, r in enumerate((self._domain_research.get("results") or [])[:5], 1):
                if isinstance(r, dict) and r.get("error"):
                    md += f"\n*Search note:* {r['error']}\n"
                    break
                if not isinstance(r, dict):
                    continue
                title = r.get("title", "")
                snippet = (r.get("snippet", "") or "")[:240]
                md += f"\n{i}. **{title}** — {snippet}...\n"
        return md

    def _tool_preprocess(self, inputs: dict) -> str:
        target_col  = inputs["target_col"]
        task_type   = inputs["task_type"]
        scaler_type = inputs.get("scaler_type", "standard")

        self._prep_result = build_preprocessing_pipeline(
            self.df,
            target_col=target_col,
            task_type=task_type,
            scaler_type=scaler_type,
        )
        if self._domain_research:
            dr = self._domain_research
            extra = ["--- Domain research context (for preprocessing) ---"] + _domain_research_log_lines(dr)
            log = list(self._prep_result.get("preprocessing_log") or [])
            self._prep_result["preprocessing_log"] = extra + log
            self._prep_result["domain_research"] = dr
        return preprocessing_log_to_markdown(self._prep_result)

    def _tool_plan_training(self, _inputs: dict) -> str:
        if self._eda_report is None or self._task_result is None or self._prep_result is None:
            raise RuntimeError(
                "run_eda, detect_task, and preprocess must complete before plan_training."
            )
        try:
            self._plan_result = plan_training(
                self._eda_report,
                self._task_result,
                self._prep_result,
            )
        except Exception as e:
            self._plan_result = None
            return (
                f"Training plan could not be generated ({e!s}). "
                "Continuing without a plan."
            )
        return plan_to_markdown(self._plan_result)

    def _tool_train(self, _inputs: dict) -> str:
        if self._prep_result is None:
            raise RuntimeError("preprocess must be called before train_models.")

        p = self._prep_result
        train_kw: dict[str, Any] = {}
        pl = getattr(self, "_plan_result", None)
        if pl is not None:
            for key in ("adjusted_params", "skip_models", "skip_reasons", "primary_metric"):
                val = pl.get(key)
                if val is not None:
                    train_kw[key] = val

        n_train = len(p["X_train"])
        cv_folds = CV_FOLDS
        if pl is not None and pl.get("cv_folds") is not None:
            try:
                cv_folds = int(pl["cv_folds"])
            except (TypeError, ValueError):
                cv_folds = CV_FOLDS
        if n_train < 20:
            cv_folds = 0
        elif n_train < 50:
            cv_folds = min(cv_folds, 3)

        self._train_result = train_and_compare(
            p["X_train"], p["X_test"],
            p["y_train"], p["y_test"],
            task_type=p["task_type"],
            feature_names=p["feature_names"],
            n_classes=p.get("n_classes", 2),
            cv_folds=cv_folds,
            **train_kw,
        )
        if n_train < 20:
            self._train_result["training_log"].insert(
                0,
                "Dataset too small for cross-validation — using single split only",
            )
        out = training_results_to_markdown(self._train_result)

        # If overfitting detected, search for fixes and append to output
        warnings = self._train_result.get("overfitting_warnings", [])
        if warnings:
            best_name = self._train_result.get("best_name", "model")
            model_name = str(best_name).replace(" (tuned)", "").strip()
            query = f"overfitting {model_name} regularization techniques machine learning"
            try:
                search_results = web_search(query, num_results=5)
                if search_results and "error" not in search_results[0]:
                    out += "\n\n## Overfitting fix suggestions (web search)\n"
                    for i, r in enumerate(search_results[:5], 1):
                        title = r.get("title", "")
                        snippet = r.get("snippet", "")[:200]
                        out += f"\n{i}. **{title}**\n   {snippet}...\n"
                    self._train_result["overfitting_search_results"] = search_results
            except Exception:
                pass

        return out

    def _tool_tune_model(self, inputs: dict) -> str:
        if self._train_result is None or self._prep_result is None:
            raise RuntimeError("train_models and preprocess must run before tune_model.")

        p = self._prep_result
        t = self._train_result
        pl = getattr(self, "_plan_result", None)
        default_trials = pl.get("n_trials", 50) if pl else 50
        default_timeout = pl.get("timeout", 120) if pl else 120
        n_trials = int(inputs.get("n_trials", default_trials))
        timeout = int(inputs.get("timeout", default_timeout))
        if pl is None and len(p["X_train"]) < 1000:
            n_trials = min(n_trials, 20)

        task_type = p["task_type"]
        raw_name = str(t["best_name"]).replace(" (tuned)", "").strip()
        plan_pm = pl.get("primary_metric") if pl else None
        baseline_score = _get_primary_score(t["best_metrics"], task_type, plan_pm)

        res = tune_best_model(
            model_name=raw_name,
            task_type=task_type,
            X_train=p["X_train"],
            y_train=p["y_train"],
            X_test=p["X_test"],
            y_test=p["y_test"],
            baseline_score=baseline_score,
            n_trials=n_trials,
            timeout=timeout,
        )
        self._tune_result = res

        if res.get("success"):
            self._train_result["best_model"] = res["tuned_model"]
            bm_te = dict(res["test_metrics_full"])
            bm_tr = dict(res["train_metrics_full"])
            ts = _get_primary_score(bm_tr, task_type, plan_pm)
            tss = _get_primary_score(bm_te, task_type, plan_pm)
            gap = ts - tss
            thresh = OVERFIT_GAP_THRESHOLD_CLF if task_type == "classification" else OVERFIT_GAP_THRESHOLD_REG
            overfit = gap > thresh
            bm_te["train_score"] = ts
            bm_te["test_score"] = tss
            bm_te["generalization_gap"] = gap
            bm_te["overfit"] = overfit
            self._train_result["best_metrics"] = bm_te
            self._train_result["feature_importances"] = _get_feature_importances(
                res["tuned_model"], p["feature_names"]
            )
            if res.get("improvement", 0) > 0:
                self._train_result["best_name"] = f"{raw_name} (tuned)"
            else:
                self._train_result["best_name"] = raw_name

        return tuning_result_to_markdown(res)

    def _tool_evaluate(self, inputs: dict) -> str:
        if self._train_result is None:
            raise RuntimeError("train_models must be called before evaluate_model.")

        p   = self._prep_result
        t   = self._train_result
        run_id = inputs.get("run_id", "agent_run")

        self._eval_result = evaluate_model(
            model=t["best_model"],
            X_test=p["X_test"],
            y_test=p["y_test"],
            X_train=p["X_train"],
            y_train=p["y_train"],
            task_type=p["task_type"],
            feature_names=p["feature_names"],
            label_encoder=p.get("label_encoder"),
            run_id=run_id,
            n_classes=p.get("n_classes", 2),
        )
        return evaluation_to_markdown(self._eval_result, t["best_name"], self._train_result)

    # ── Result builder ────────────────────────────────────────────────────────

    def _build_result(self) -> dict[str, Any]:
        prep_dict = {
            k: v for k, v in (self._prep_result or {}).items()
            if k not in ("X_train", "X_test", "y_train", "y_test",
                         "pipeline", "label_encoder")
        }
        # Target leakage flag for UI (same logic as _get_step_data)
        if self._prep_result and self._task_result:
            target_col = self._task_result["target_col"]
            if target_col and target_col in self.df.columns:
                y = self.df[target_col]
                if pd.api.types.is_numeric_dtype(y):
                    for c in self.df.drop(columns=[target_col]).select_dtypes(include="number").columns:
                        corr = self.df[[c, target_col]].corr().iloc[0, 1]
                        if not np.isnan(corr) and abs(corr) > 0.95:
                            prep_dict["target_leakage_suspicion"] = (
                                f"Column '{c}' has correlation {corr:.3f} with target."
                            )
                            break

        train_block: dict[str, Any] = {}
        if self._train_result:
            train_block = {
                k: v for k, v in self._train_result.items()
                if k not in ("best_model", "results")
            }
            train_block["results"] = [
                {kk: vv for kk, vv in r.items() if kk != "model"}
                for r in self._train_result.get("results", [])
            ]

        result: dict[str, Any] = {
            "status": "complete",
            "eda":    self._eda_report,
            "task":   self._task_result,
            "prep":   prep_dict,
            "train":  train_block,
            "eval":   {
                k: v for k, v in (self._eval_result or {}).items()
                if k not in ("shap_values", "y_pred")
            },
        }

        # Convenience top-level keys the UI uses directly
        if self._task_result:
            result["target_col"] = self._task_result["target_col"]
            result["task_type"]  = self._task_result["task_type"]
        if self._train_result:
            result["best_model_name"]    = self._train_result["best_name"]
            result["best_metrics"]       = self._train_result["best_metrics"]
            result["comparison_df"]      = self._train_result["comparison_df"]
            result["feature_importances"]= self._train_result.get("feature_importances", {})
        if self._eval_result:
            result["plot_paths"] = self._eval_result["plot_paths"]
            result["metrics"]    = self._eval_result["metrics"]

        if self._train_result:
            sr = self._train_result.get("overfitting_search_results")
            if sr:
                result["overfitting_search_results"] = sr

        if self._tune_result:
            result["tune"] = {
                k: v for k, v in self._tune_result.items()
                if k != "tuned_model"
            }

        pr = getattr(self, "_plan_result", None)
        if pr is not None:
            result["plan"] = pr

        if self._domain_research:
            result["domain_research"] = self._domain_research

        return result

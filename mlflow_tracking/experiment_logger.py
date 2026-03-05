"""
mlflow_tracking/experiment_logger.py

MLflow experiment logger for the Multimodal RAG Platform.

Logs:
  - Parameters: embedding model, top_k, chunk_size, session_id
  - Metrics:    MRR@K, Hit@K, Precision@K, faithfulness, latency
  - Artifacts:  query text, retrieved chunks JSON

Usage:
    from mlflow_tracking.experiment_logger import log_rag_experiment, log_eval_report
"""

import os
import json
import mlflow
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_EXPERIMENT_NAME  = "Multimodal-RAG-Platform"
_TRACKING_URI     = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(_TRACKING_URI)

try:
    mlflow.set_experiment(_EXPERIMENT_NAME)
except Exception:
    pass   # MLflow server might not be running yet — that's OK


def log_rag_experiment(
    session_id:        str,
    query:             str,
    retrieved_chunks:  list[dict],
    metrics:           dict,
    run_name:          str | None = None,
) -> str | None:
    """
    Log a single RAG query + retrieval metrics to MLflow.

    Args:
        session_id:       The session being queried.
        query:            The user's question.
        retrieved_chunks: List of chunk dicts from retrieval_service.retrieve().
        metrics:          {"mrr": float, "hit": float, "precision": float,
                           "faithfulness": float, "latency_ms": float}
        run_name:         Optional human-readable MLflow run name.

    Returns:
        MLflow run_id or None if tracking failed.
    """
    _name = run_name or f"query_{session_id[:8]}"
    try:
        with mlflow.start_run(run_name=_name) as run:
            # ── Parameters (what we configured) ──────────────
            mlflow.log_params({
                "embedding_model": "intfloat/e5-base-v2",
                "llm_model":       os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                "session_id":      session_id,
                "top_k":           len(retrieved_chunks),
                "chunk_size":      1000,
                "chunk_overlap":   100,
            })

            # ── Metrics (what we measured) ────────────────────
            mlflow.log_metrics({
                "mrr_at_k":      metrics.get("mrr",         0.0),
                "hit_at_k":      metrics.get("hit",         0.0),
                "precision_at_k":metrics.get("precision",   0.0),
                "faithfulness":  metrics.get("faithfulness",0.0),
                "latency_ms":    metrics.get("latency_ms",  0.0),
            })

            # ── Artifacts (what was retrieved) ────────────────
            mlflow.log_text(query, "query.txt")
            mlflow.log_dict(
                {"chunks": retrieved_chunks},
                "retrieved_chunks.json"
            )

            return run.info.run_id

    except Exception as exc:
        print(f"[MLflow] Warning: could not log experiment: {exc}")
        return None


def log_eval_report(report: dict) -> str | None:
    """
    Log a full evaluation report (from eval/run_eval.py) as an MLflow run.

    Args:
        report: The dict returned by run_evaluation().

    Returns:
        MLflow run_id or None.
    """
    session_id = report.get("session_id", "unknown")
    summary    = report.get("summary", {})

    try:
        with mlflow.start_run(run_name=f"eval_{session_id[:8]}") as run:
            mlflow.log_params({
                "session_id": session_id,
                "top_k":      report.get("top_k", 5),
                "timestamp":  report.get("timestamp", ""),
            })

            mlflow.log_metrics({
                "mrr_at_k":       summary.get("mrr_at_k",       0.0),
                "hit_at_k":       summary.get("hit_at_k",       0.0),
                "precision_at_k": summary.get("precision_at_k", 0.0),
                "faithfulness":   summary.get("faithfulness",   0.0),
                "avg_latency_ms": summary.get("avg_latency_ms", 0.0),
                "num_queries":    summary.get("num_queries",    0),
            })

            mlflow.log_dict(report, "full_eval_report.json")

            return run.info.run_id

    except Exception as exc:
        print(f"[MLflow] Warning: could not log eval report: {exc}")
        return None

"""
mlflow_tracking/tracker.py

Unified MLflow tracking facade for the Multimodal RAG Platform.

Design principles:
  - ALL tracking calls are wrapped in try/except — the app NEVER crashes
    because MLflow is unavailable.
  - When MLflow server is unreachable, falls back to a local SQLite store
    at mlruns/ (MLflow's default). No extra config needed.
  - Provides a clean singleton so models are only initialised once.

Tracked events:
  ┌─────────────────────┬──────────────────────────────────────────────────┐
  │ Event               │ Logged data                                      │
  ├─────────────────────┼──────────────────────────────────────────────────┤
  │ Ingestion job       │ modality, num_chunks, embed_time_ms, file_name   │
  │ Single RAG query    │ mrr, hit, precision, faithfulness, latency_ms    │
  │ Full eval report    │ all summary metrics + per-query artifact         │
  └─────────────────────┴──────────────────────────────────────────────────┘

Usage:
    from mlflow_tracking.tracker import tracker
    tracker.log_ingestion(session_id, file_name, modality, num_chunks, ms)
    tracker.log_query(session_id, query, chunks, metrics)
    tracker.log_eval_report(report)
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT     = Path(__file__).parent.parent
_EXPERIMENT_NAME  = os.getenv("MLFLOW_EXPERIMENT", "Multimodal-RAG-Platform")
_TRACKING_URI     = os.getenv("MLFLOW_TRACKING_URI", "")   # empty → local mlruns/
_LOCAL_MLRUNS     = str(_PROJECT_ROOT / "mlruns")

# ─────────────────────────────────────────────────
# Lazy MLflow import (app works without it)
# ─────────────────────────────────────────────────
_mlflow     = None
_HAS_MLFLOW = False


def _get_mlflow():
    global _mlflow, _HAS_MLFLOW
    if _mlflow is not None:
        return _mlflow
    try:
        import mlflow as _m
        _mlflow = _m
        _HAS_MLFLOW = True
        # Prefer explicit URI; fall back to local SQLite store
        uri = _TRACKING_URI or f"file://{_LOCAL_MLRUNS}"
        _mlflow.set_tracking_uri(uri)
        _mlflow.set_experiment(_EXPERIMENT_NAME)
        print(f"[MLflow] Tracking to: {uri}  |  experiment: {_EXPERIMENT_NAME}")
    except Exception as exc:
        print(f"[MLflow] Not available ({exc}). Tracking disabled.")
        _HAS_MLFLOW = False
    return _mlflow


# ─────────────────────────────────────────────────
# Model config helpers (read from env / defaults)
# ─────────────────────────────────────────────────
def _base_params() -> dict:
    return {
        "embedding_model": "intfloat/e5-small-v2",
        "llm_model":       os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        "chunk_size":      1000,
        "chunk_overlap":   100,
        "index_type":      "FAISS-L2",
    }


# ─────────────────────────────────────────────────
# Main tracker class
# ─────────────────────────────────────────────────
class RAGTracker:
    """
    Thread-safe, fault-tolerant MLflow tracker for the RAG pipeline.
    Every public method is silent on failure — tracking must never break
    the application.
    """

    # ── Ingestion tracking ────────────────────────────────────────────────
    def log_ingestion(
        self,
        session_id:  str,
        file_name:   str,
        modality:    str,
        num_chunks:  int,
        elapsed_ms:  float,
        *,
        num_visuals: int = 0,
        num_pages:   int = 0,
    ) -> Optional[str]:
        """
        Log a completed ingestion job.

        Args:
            session_id:  RAG session identifier.
            file_name:   Uploaded file name.
            modality:    'text', 'image', or 'audio'.
            num_chunks:  Number of embedded chunks produced.
            elapsed_ms:  Total ingestion wall-clock time in ms.
            num_visuals: Number of figure/table crops processed (PDFs).
            num_pages:   Number of PDF pages processed.

        Returns:
            MLflow run_id or None.
        """
        ml = _get_mlflow()
        if not _HAS_MLFLOW or ml is None:
            return None
        try:
            with ml.start_run(run_name=f"ingest_{modality}_{file_name[:20]}") as run:
                ml.set_tags({
                    "event":      "ingestion",
                    "session_id": session_id,
                    "file_name":  file_name,
                    "modality":   modality,
                })
                ml.log_params({
                    **_base_params(),
                    "session_id": session_id,
                    "file_name":  file_name,
                    "modality":   modality,
                })
                ml.log_metrics({
                    "num_chunks":      float(num_chunks),
                    "ingestion_ms":    elapsed_ms,
                    "num_visuals":     float(num_visuals),
                    "num_pages":       float(num_pages),
                    "chunks_per_sec":  (num_chunks / (elapsed_ms / 1000))
                                       if elapsed_ms > 0 else 0.0,
                })
                return run.info.run_id
        except Exception as exc:
            print(f"[MLflow] log_ingestion failed: {exc}")
            return None

    # ── Single-query tracking ─────────────────────────────────────────────
    def log_query(
        self,
        session_id:       str,
        query:            str,
        retrieved_chunks: list,
        metrics:          dict,
        *,
        run_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Log a single RAG query + retrieval metrics run.

        metrics should contain: mrr, hit, precision, faithfulness, latency_ms
        """
        ml = _get_mlflow()
        if not _HAS_MLFLOW or ml is None:
            return None
        try:
            name = run_name or f"query_{session_id[:8]}_{int(time.time())}"
            with ml.start_run(run_name=name) as run:
                ml.set_tags({
                    "event":      "query",
                    "session_id": session_id,
                })
                ml.log_params({
                    **_base_params(),
                    "session_id": session_id,
                    "top_k":      len(retrieved_chunks),
                })
                ml.log_metrics({
                    "mrr_at_k":       metrics.get("mrr",         0.0),
                    "hit_at_k":       metrics.get("hit",         0.0),
                    "precision_at_k": metrics.get("precision",   0.0),
                    "recall_at_k":    metrics.get("recall",      0.0),
                    "faithfulness":   metrics.get("faithfulness",0.0),
                    "latency_ms":     metrics.get("latency_ms",  0.0),
                    "keyword_score":  metrics.get("keyword_score",0.0),
                })
                ml.log_text(query, "query.txt")
                ml.log_dict(
                    {"chunks": [
                        {"source": c.get("source"), "modality": c.get("modality"),
                         "score": c.get("score"), "text_preview": (c.get("text","")[:200])}
                        for c in retrieved_chunks
                    ]},
                    "retrieved_chunks.json",
                )
                return run.info.run_id
        except Exception as exc:
            print(f"[MLflow] log_query failed: {exc}")
            return None

    # ── Full eval report tracking ─────────────────────────────────────────
    def log_eval_report(self, report: dict) -> Optional[str]:
        """
        Log a complete eval report (from eval/run_eval.py) as one MLflow run.

        report schema:
            {session_id, timestamp, top_k, summary: {...}, per_query: [...]}
        """
        ml = _get_mlflow()
        if not _HAS_MLFLOW or ml is None:
            return None
        try:
            session_id = report.get("session_id", "unknown")
            summary    = report.get("summary", {})
            ts         = report.get("timestamp", "")

            with ml.start_run(run_name=f"eval_{session_id[:12]}_{ts}") as run:
                ml.set_tags({
                    "event":      "evaluation",
                    "session_id": session_id,
                    "timestamp":  ts,
                })
                ml.log_params({
                    **_base_params(),
                    "session_id":  session_id,
                    "top_k":       report.get("top_k", 5),
                    "num_queries": summary.get("num_queries", 0),
                })
                ml.log_metrics({
                    "mrr_at_k":         summary.get("mrr_at_k",       0.0),
                    "hit_at_k":         summary.get("hit_at_k",       0.0),
                    "precision_at_k":   summary.get("precision_at_k", 0.0),
                    "recall_at_k":      summary.get("recall_at_k",    0.0),
                    "faithfulness":     summary.get("faithfulness",   0.0),
                    "keyword_score":    summary.get("keyword_score",  0.0),
                    "avg_latency_ms":   summary.get("avg_latency_ms", 0.0),
                    "num_queries":      float(summary.get("num_queries", 0)),
                })

                # Log full report as JSON artifact
                ml.log_dict(report, "full_eval_report.json")

                # Log per-query CSV for easy analysis in MLflow UI
                pq = report.get("per_query", [])
                if pq:
                    import csv, io
                    buf = io.StringIO()
                    fields = ["query", "mrr", "hit", "precision",
                              "recall", "faithfulness", "keyword_score", "latency_ms"]
                    w = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(pq)
                    ml.log_text(buf.getvalue(), "per_query_metrics.csv")

                return run.info.run_id

        except Exception as exc:
            print(f"[MLflow] log_eval_report failed: {exc}")
            return None

    # ── System info snapshot ──────────────────────────────────────────────
    def log_system_info(self) -> Optional[str]:
        """Log GPU/CPU device info as a one-off MLflow run at startup."""
        ml = _get_mlflow()
        if not _HAS_MLFLOW or ml is None:
            return None
        try:
            with ml.start_run(run_name="system_info") as run:
                ml.set_tags({"event": "system_info"})
                params = {"os": os.name}
                try:
                    import torch
                    params["torch_version"] = torch.__version__
                    if torch.cuda.is_available():
                        params["device"]   = "cuda"
                        params["gpu_name"] = torch.cuda.get_device_name(0)
                        params["gpu_mem_gb"] = round(
                            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
                        )
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        params["device"] = "mps"
                    else:
                        params["device"] = "cpu"
                except Exception:
                    params["device"] = "unknown"
                ml.log_params(params)
                return run.info.run_id
        except Exception as exc:
            print(f"[MLflow] log_system_info failed: {exc}")
            return None


# ─────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────
tracker = RAGTracker()

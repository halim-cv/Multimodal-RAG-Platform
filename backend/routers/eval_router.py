"""
backend/routers/eval_router.py
POST /api/evaluate/{session_id}  — run retrieval eval metrics on a session
"""

import re
import json
import time
import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.services import retrieval_service, generation_service
from backend.models.schemas import EvalRequest, EvalResponse, EvalMetrics
from eval.metrics import mrr_at_k, hit_at_k, precision_at_k

router = APIRouter(prefix="/api/evaluate", tags=["evaluation"])

_PROJECT_ROOT      = Path(__file__).parent.parent.parent
_BENCHMARK_PATH    = _PROJECT_ROOT / "eval" / "benchmark_dataset.json"


@router.post("/{session_id}", response_model=EvalResponse)
async def evaluate_session(session_id: str, request: EvalRequest = None):
    """
    Run the benchmark dataset against this session and return retrieval metrics.
    Also computes faithfulness via LLM-as-judge on a subset of queries.
    """
    if not _BENCHMARK_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Benchmark dataset not found at eval/benchmark_dataset.json"
        )

    with open(_BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    top_k   = (request.top_k if request else 5)
    mrr_sum = hit_sum = prec_sum = faith_sum = 0.0
    latencies = []

    for item in benchmark:
        query          = item["query"]
        relevant_srcs  = item.get("relevant_sources", [])

        # Retrieval
        t0 = time.time()
        try:
            chunks = retrieval_service.retrieve(session_id, query, top_k=top_k)
        except Exception:
            continue
        latencies.append((time.time() - t0) * 1000)

        retrieved_srcs = [c["source"] for c in chunks]
        mrr_sum  += mrr_at_k(retrieved_srcs, relevant_srcs, top_k)
        hit_sum  += hit_at_k(retrieved_srcs, relevant_srcs, top_k)
        prec_sum += precision_at_k(retrieved_srcs, relevant_srcs, top_k)

        # Faithfulness (LLM-as-judge) — only for first 3 queries to save API calls
        if benchmark.index(item) < 3:
            try:
                answer       = await generation_service.generate_answer(query, chunks)
                faith_prompt = (
                    f"Rate how faithful this answer is to the context on a scale 0.0-1.0.\n"
                    f"Context: {[c['text'][:300] for c in chunks[:3]]}\n"
                    f"Answer: {answer[:500]}\n"
                    f"Return ONLY a JSON: {{\"score\": <float>}}"
                )
                faith_resp = await generation_service.generate_answer(faith_prompt, [])
                match = re.search(r'"score"\s*:\s*([\d.]+)', faith_resp)
                faith_sum += float(match.group(1)) if match else 0.5
            except Exception:
                faith_sum += 0.5   # neutral default on error

    n = max(len(benchmark), 1)
    metrics = EvalMetrics(
        mrr_at_k       = round(mrr_sum  / n, 4),
        hit_at_k       = round(hit_sum  / n, 4),
        precision_at_k = round(prec_sum / n, 4),
        faithfulness   = round(faith_sum / min(3, n), 4),
        avg_latency_ms = round(sum(latencies) / max(len(latencies), 1), 1),
        num_queries    = n,
    )

    return EvalResponse(session_id=session_id, metrics=metrics)

"""
backend/routers/chat.py
POST /api/chat  — SSE-streaming RAG endpoint

Flow per request:
  1. Retrieve top-K chunks from FAISS (multimodal, session-scoped)
  2. Emit an SSE event with the retrieved sources (for citation UI)
  3. Stream Gemini tokens one-by-one as SSE events
  4. Emit a 'done' event with latency metadata
"""

import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.services import retrieval_service, generation_service
from backend.models.schemas import ChatRequest

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("")
async def chat_endpoint(request: ChatRequest):
    """
    Server-Sent Events streaming endpoint.

    Event types emitted:
      {"type": "sources",  "chunks": [...]}          — retrieved context (first)
      {"type": "token",    "content": "word "}       — LLM token stream
      {"type": "done",     "latency_ms": 1234.5}     — completion signal
      {"type": "error",    "message": "..."}          — on failure
    """
    modalities = [m.value for m in request.modalities] if request.modalities else None

    # --- Retrieve chunks synchronously (fast FAISS search) ---
    try:
        start_retrieve = time.time()
        chunks = retrieval_service.retrieve(
            session_id = request.session_id,
            query      = request.query,
            top_k      = request.top_k,
            modalities = modalities,
        )
        retrieve_ms = (time.time() - start_retrieve) * 1000
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")

    # --- Async SSE generator ---
    async def event_stream():
        start_gen = time.time()

        # Event 1: send retrieved sources immediately (for UI citation rendering)
        yield _sse({"type": "sources", "chunks": chunks, "retrieve_ms": round(retrieve_ms, 1)})

        # Events 2..N: stream Gemini tokens
        try:
            async for token in generation_service.stream_answer(request.query, chunks):
                yield _sse({"type": "token", "content": token})
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
            return

        # Final event: latency summary
        total_ms = (time.time() - start_gen) * 1000
        yield _sse({"type": "done", "latency_ms": round(total_ms, 1)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(payload: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(payload)}\n\n"

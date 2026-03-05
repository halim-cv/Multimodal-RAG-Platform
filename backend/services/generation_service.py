"""
backend/services/generation_service.py

Streaming LLM answer generation using the google-genai SDK.

Uses native SDK streaming (generate_content_stream) for true token-by-token
delivery, with retry logic for transient errors.
"""

import os
import asyncio
from typing import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────
# Client initialisation  (google.genai)
# ─────────────────────────────────────────────────

_API_KEY  = os.getenv("GEMINI_API_KEY", "")
_MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

_client   = None
_HAS_GENAI = False

try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except ImportError:
    genai = None   # type: ignore
    types = None   # type: ignore


def _get_client():
    """Lazily initialise the Gemini client on first use."""
    global _client
    if _client is not None:
        return _client

    if not _HAS_GENAI:
        raise RuntimeError(
            "google-genai SDK not installed. Install with: pip install google-genai"
        )

    if not _API_KEY:
        raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")

    _client = genai.Client(api_key=_API_KEY)
    print(f"[GenerationService] google-genai client ready — model: {_MODEL_ID}")
    return _client


# ─────────────────────────────────────────────────
# Prompt template & builder
# ─────────────────────────────────────────────────

_MODALITY_LABELS = {
    "text":  "📄 Document Excerpt",
    "image": "🖼️ Image Description / OCR",
    "audio": "🎵 Audio Transcript",
}

PROMPT_TEMPLATE = """
You are a helpful assistant for a multimodal knowledge base.
Use the provided context (text excerpts, image descriptions, audio transcripts).
Answer concisely and cite source filenames in [brackets].
If the context is insufficient, say so explicitly — do not hallucinate.

Context:
{formatted_chunks}

Question: {query}

Answer:
""".strip()


def _format_chunks(chunks: list[dict]) -> str:
    """Format retrieved multimodal chunks with modality labels and source metadata."""
    if not chunks:
        return "(No context retrieved — answer from general knowledge only)"

    blocks = []
    for i, chunk in enumerate(chunks, 1):
        modality = chunk.get("modality", "text")
        source   = chunk.get("source", "unknown")
        text     = chunk.get("text", "").strip()
        score    = chunk.get("score", 0.0)
        label    = _MODALITY_LABELS.get(modality, "📄 Document Excerpt")

        ts_hint = ""
        if modality == "audio" and chunk.get("timestamp"):
            ts = chunk["timestamp"]
            ts_hint = f" [{ts[0]:.1f}s – {ts[1]:.1f}s]"

        blocks.append(
            f"[{i}] {label} | '{source}'{ts_hint} | relevance {score:.3f}\n{text}"
        )

    return "\n\n---\n\n".join(blocks)


def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Format retrieved multimodal chunks into a grounded RAG prompt."""
    formatted = _format_chunks(chunks)
    return PROMPT_TEMPLATE.format(formatted_chunks=formatted, query=query)


# ─────────────────────────────────────────────────
# Core generation  (sync SDK → run in thread for async compatibility)
# ─────────────────────────────────────────────────

def _generate_sync(prompt: str) -> str:
    """
    Call Gemini synchronously (non-streaming).
    Returns the full response text.
    """
    client = _get_client()
    response = client.models.generate_content(
        model    = _MODEL_ID,
        contents = prompt,
    )
    return response.text or ""


def _generate_stream_sync(prompt: str):
    """
    Call Gemini with native streaming.
    Yields text chunks as they arrive from the API.
    """
    client = _get_client()
    for chunk in client.models.generate_content_stream(
        model    = _MODEL_ID,
        contents = prompt,
    ):
        if chunk.text:
            yield chunk.text


# ─────────────────────────────────────────────────
# Async streaming generator (native SDK streaming)
# ─────────────────────────────────────────────────

async def generate_stream(query: str, chunks: list[dict]) -> AsyncGenerator[str, None]:
    """
    Generate a Gemini answer and stream it token-by-token as an async generator.

    Uses the native `generate_content_stream` SDK method for true streaming
    (real tokens as they're generated, not post-hoc word splitting).

    Includes retry logic for transient 503/429 errors.
    """
    prompt = build_rag_prompt(query, chunks)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Run the synchronous streaming generator in a thread and
            # collect chunks via a queue for async yielding.
            queue: asyncio.Queue[str | None] = asyncio.Queue()

            def _stream_to_queue():
                try:
                    for text_chunk in _generate_stream_sync(prompt):
                        queue.put_nowait(text_chunk)
                finally:
                    queue.put_nowait(None)  # sentinel

            # Start streaming in a background thread
            loop = asyncio.get_event_loop()
            task = loop.run_in_executor(None, _stream_to_queue)

            # Yield chunks as they arrive
            while True:
                try:
                    item = queue.get_nowait()
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.02)
                    if task.done() and queue.empty():
                        break
                    continue

                if item is None:
                    break
                yield item

            # Ensure the thread has finished
            await task
            return  # Success

        except Exception as exc:
            err_str = str(exc)
            is_retryable = any(
                code in err_str
                for code in ["503", "429", "UNAVAILABLE", "overloaded", "high demand"]
            )

            if is_retryable and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)   # 2s → 4s
                print(
                    f"[GenerationService] Retryable error "
                    f"(attempt {attempt+1}/{max_retries}), waiting {wait}s: "
                    f"{err_str[:120]}"
                )
                yield f"\n⏳ Model busy, retrying in {wait}s…\n"
                await asyncio.sleep(wait)
                continue
            else:
                yield f"\n\n⚠️ Generation error: {exc}"
                return


# Alias used by chat router
stream_answer = generate_stream


# ─────────────────────────────────────────────────
# Non-streaming fallback (used by eval pipeline)
# ─────────────────────────────────────────────────

async def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Non-streaming fallback — returns the complete answer as a string.
    Used by the evaluation pipeline.
    """
    prompt = build_rag_prompt(query, chunks)
    return await asyncio.to_thread(_generate_sync, prompt)

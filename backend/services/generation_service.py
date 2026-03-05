"""
backend/services/generation_service.py

Streaming LLM answer generation using the new google-genai SDK.
Uses gemini-3-flash-preview (free tier) with native async streaming via SSE.
"""

import os
import asyncio
import threading
from typing import AsyncGenerator

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────
# Client initialisation  (new SDK: google.genai)
# ─────────────────────────────────────────────────

_API_KEY  = os.getenv("GEMINI_API_KEY", "")
_MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-preview-05-20")

if not _API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")

_client = genai.Client(api_key=_API_KEY)

print(f"[GenerationService] google-genai client ready — model: {_MODEL_ID}")


# ─────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────

_MODALITY_LABELS = {
    "text":  "📄 Document Excerpt",
    "image": "🖼️  Image Description / OCR",
    "audio": "🎵 Audio Transcript",
}

def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Format retrieved multimodal chunks into a grounded RAG prompt."""
    context_blocks = []
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

        context_blocks.append(
            f"[Source {i} | {label} | '{source}'{ts_hint} | score {score:.3f}]\n{text}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are a helpful AI research assistant for a multimodal knowledge base.

The knowledge base contains text documents, images (with auto-generated captions and OCR), and audio transcripts.
Use ONLY the retrieved context below to answer the user's question.
Cite your sources inline with [1], [2], etc. at the end of relevant sentences.
If the context is insufficient, say so explicitly — do not hallucinate.

===================== RETRIEVED CONTEXT =====================

{context}

===================== USER QUESTION ========================

{query}

===================== YOUR ANSWER ==========================
"""


# ─────────────────────────────────────────────────
# Async streaming generator  (new SDK pattern)
# ─────────────────────────────────────────────────

async def stream_answer(query: str, chunks: list[dict]) -> AsyncGenerator[str, None]:
    """
    Stream the Gemini response token-by-token as an async generator.

    Uses google-genai's native async streaming:
        client.aio.models.generate_content_stream(...)

    Yields:
        str: Individual text tokens/chunks from the model.
    """
    prompt = build_rag_prompt(query, chunks)

    try:
        async for chunk in await _client.aio.models.generate_content_stream(
            model    = _MODEL_ID,
            contents = prompt,
            config   = types.GenerateContentConfig(
                temperature      = 0.2,
                max_output_tokens= 2048,
            ),
        ):
            if chunk.text:
                yield chunk.text
    except Exception as exc:
        yield f"\n\n⚠️ Generation error: {exc}"


async def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Non-streaming fallback — returns the complete answer as a string.
    Used by the evaluation pipeline.
    """
    prompt = build_rag_prompt(query, chunks)

    response = await _client.aio.models.generate_content(
        model    = _MODEL_ID,
        contents = prompt,
        config   = types.GenerateContentConfig(
            temperature       = 0.2,
            max_output_tokens = 2048,
        ),
    )
    return response.text or ""

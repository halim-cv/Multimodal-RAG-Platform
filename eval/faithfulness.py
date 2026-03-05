"""
eval/faithfulness.py

LLM-as-judge faithfulness scorer using the new google-genai SDK.
"""

import os
import re
import json
import asyncio
from typing import Optional

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_client   = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
_MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")


_FAITHFULNESS_PROMPT = """
You are an expert evaluator for AI-generated answers in a Retrieval-Augmented Generation (RAG) system.

Your task: Rate how FAITHFUL the given answer is to the provided context.

Faithfulness means every claim in the answer must be supported by the context.
Penalise answers that introduce facts not present in the context (hallucinations).

Scale:
  1.0 — Fully faithful: every statement is directly supported by the context.
  0.7 — Mostly faithful: minor details not in context but core claims are supported.
  0.5 — Partially faithful: some supported, some hallucinated.
  0.3 — Mostly hallucinated: most claims are NOT in the context.
  0.0 — Completely hallucinated: nothing in the answer is grounded in the context.

CONTEXT (retrieved chunks):
{context}

GENERATED ANSWER:
{answer}

Return ONLY valid JSON, no explanation:
{{"score": <float 0.0-1.0>, "reason": "<one sentence explanation>"}}
"""


def faithfulness_score(
    answer: str,
    context_chunks: list[dict],
    max_ctx_chars: int = 2000,
) -> dict:
    """Synchronous faithfulness scoring."""
    ctx_parts, total = [], 0
    for i, chunk in enumerate(context_chunks, 1):
        text  = chunk.get("text", "")[:500]
        src   = chunk.get("source", "?")
        entry = f"[{i}] ({src}) {text}"
        if total + len(entry) > max_ctx_chars:
            break
        ctx_parts.append(entry)
        total += len(entry)

    prompt = _FAITHFULNESS_PROMPT.format(
        context = "\n\n".join(ctx_parts),
        answer  = answer[:1000],
    )

    try:
        response = _client.models.generate_content(
            model    = _MODEL_ID,
            contents = prompt,
            config   = types.GenerateContentConfig(temperature=0.0),
        )
        raw   = (response.text or "").strip()
        match = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "score":  float(data.get("score", 0.5)),
                "reason": str(data.get("reason", "")),
            }
    except Exception as exc:
        return {"score": 0.5, "reason": f"Scoring failed: {exc}"}

    return {"score": 0.5, "reason": "Could not parse model response."}


async def faithfulness_score_async(answer: str, context_chunks: list[dict]) -> dict:
    """Async wrapper for use in async evaluation contexts."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, faithfulness_score, answer, context_chunks)

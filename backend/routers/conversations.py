"""
backend/routers/conversations.py
GET    /api/sessions/{session_id}/conversations  — load chat history
POST   /api/sessions/{session_id}/conversations  — append a message pair
DELETE /api/sessions/{session_id}/conversations  — clear history
"""

import json
import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/api/sessions", tags=["conversations"])

_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_SESSIONS_ROOT = _PROJECT_ROOT / "Text-encoding" / "sessions"


# ─────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────

class ConversationMessage(BaseModel):
    role: str                          # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None
    chunks: Optional[list] = None      # retrieved chunks (for assistant messages)

class AppendConversationRequest(BaseModel):
    user_message: str
    assistant_message: str
    chunks: Optional[list] = None      # retrieved source chunks

class ConversationHistory(BaseModel):
    session_id: str
    messages: list[ConversationMessage]


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────

def _history_path(session_id: str) -> Path:
    return _SESSIONS_ROOT / session_id / "_conversations" / "history.json"


def _load_history(session_id: str) -> list[dict]:
    path = _history_path(session_id)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _save_history(session_id: str, messages: list[dict]):
    path = _history_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────

@router.get("/{session_id}/conversations", response_model=ConversationHistory)
def get_conversations(session_id: str):
    """Load the full chat history for a session."""
    session_dir = _SESSIONS_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    messages = _load_history(session_id)
    return ConversationHistory(
        session_id=session_id,
        messages=[ConversationMessage(**m) for m in messages],
    )


@router.post("/{session_id}/conversations")
def append_conversation(session_id: str, body: AppendConversationRequest):
    """Append a user+assistant message pair to the session history."""
    session_dir = _SESSIONS_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    now = datetime.datetime.now().isoformat()
    messages = _load_history(session_id)

    messages.append({
        "role": "user",
        "content": body.user_message,
        "timestamp": now,
    })
    messages.append({
        "role": "assistant",
        "content": body.assistant_message,
        "timestamp": now,
        "chunks": body.chunks,
    })

    _save_history(session_id, messages)
    return {"detail": "Conversation saved", "total_messages": len(messages)}


@router.delete("/{session_id}/conversations")
def clear_conversations(session_id: str):
    """Clear all chat history for a session."""
    session_dir = _SESSIONS_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

    path = _history_path(session_id)
    if path.exists():
        path.unlink()
    return {"detail": "Conversation history cleared"}

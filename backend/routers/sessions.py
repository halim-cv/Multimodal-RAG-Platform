"""
backend/routers/sessions.py
GET  /api/sessions           — list all sessions
POST /api/sessions           — create a new session
GET  /api/sessions/{id}      — get session detail + source list
DELETE /api/sessions/{id}    — delete a session
"""

import os
import sys
import pickle
import shutil
import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from backend.models.schemas import (
    SessionCreate, SessionInfo, SessionListResponse, SourceFile,
    Modality, JobStatus,
)

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_SESSIONS_ROOT = _PROJECT_ROOT / "Text-encoding" / "sessions"
_SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

# Map folder-category names → Modality
_CAT_TO_MODALITY = {
    "docs": Modality.TEXT,
    "txt":  Modality.TEXT,
    "pdf":  Modality.TEXT,
    "image": Modality.IMAGE,
    "img":   Modality.IMAGE,
    "audio": Modality.AUDIO,
    "_overall": Modality.TEXT,
}


def _read_session(session_id: str) -> SessionInfo:
    session_dir = _SESSIONS_ROOT / session_id
    sources: list[SourceFile] = []

    for cat_dir in session_dir.iterdir():
        if not cat_dir.is_dir() or cat_dir.name.startswith("_"):
            continue
        modality = _CAT_TO_MODALITY.get(cat_dir.name, Modality.TEXT)

        for file_dir in cat_dir.iterdir():
            if not file_dir.is_dir():
                continue
            emb_path = file_dir / "embeddings.pkl"
            embedded = emb_path.exists()
            sources.append(SourceFile(
                file_name  = file_dir.name,
                category   = cat_dir.name,
                modality   = modality,
                embedded   = embedded,
                job_status = JobStatus.DONE if embedded else JobStatus.QUEUED,
            ))

    return SessionInfo(
        session_id   = session_id,
        created_at   = datetime.datetime.fromtimestamp(
            session_dir.stat().st_ctime
        ).isoformat(),
        source_count = len(sources),
        sources      = sources,
    )


@router.get("", response_model=SessionListResponse)
def list_sessions():
    """List all available sessions."""
    sessions = []
    if _SESSIONS_ROOT.exists():
        for d in sorted(_SESSIONS_ROOT.iterdir(), key=lambda p: p.stat().st_ctime, reverse=True):
            if d.is_dir() and not d.name.startswith("_"):
                try:
                    sessions.append(_read_session(d.name))
                except Exception:
                    pass
    return SessionListResponse(sessions=sessions)


@router.post("", response_model=SessionInfo)
def create_session(body: SessionCreate):
    """Create a new empty session."""
    ts         = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{ts}"
    (_SESSIONS_ROOT / session_id).mkdir(parents=True, exist_ok=True)
    return SessionInfo(
        session_id   = session_id,
        created_at   = datetime.datetime.now().isoformat(),
        source_count = 0,
        sources      = [],
    )


@router.get("/{session_id}", response_model=SessionInfo)
def get_session(session_id: str):
    """Get full session info including all sources and their embedding status."""
    session_dir = _SESSIONS_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return _read_session(session_id)


@router.delete("/{session_id}")
def delete_session(session_id: str):
    """Delete a session and all its data."""
    session_dir = _SESSIONS_ROOT / session_id
    if not session_dir.exists():
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    shutil.rmtree(session_dir)
    return {"detail": f"Session '{session_id}' deleted."}

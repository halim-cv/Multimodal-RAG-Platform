"""
backend/services/ingestion_service.py

Unified multimodal ingestion dispatcher.

Given any uploaded file, this service:
  1. Detects the modality (text / image / audio) from the file extension.
  2. Routes to the correct encoder pipeline.
  3. Chunks the extracted representation into fixed-size windows.
  4. Embeds all chunks with the shared E5-base-v2 model.
  5. Saves `embeddings.pkl` into the session directory.
  6. Tracks job status in an in-memory dict (queryable via the /upload/status endpoint).

All modalities end up embedded in the **same vector space** (E5 text embeddings),
enabling true cross-modal retrieval:
  - Text   → extract raw text, chunk, embed
  - Image  → Florence-2 caption + OCR → concatenate, chunk, embed
  - Audio  → Whisper timestamped segments → chunk by segment, embed
"""

import os
import sys
import pickle
import uuid
import shutil
import asyncio
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# ─────────────────────────────────────────────────
# Path bootstrap
# ─────────────────────────────────────────────────
_PROJECT_ROOT  = Path(__file__).parent.parent.parent
_TEXT_CODE     = _PROJECT_ROOT / "Text-encoding" / "model" / "code"
_IMAGE_ROOT    = _PROJECT_ROOT / "image-encoder"
_AUDIO_ROOT    = _PROJECT_ROOT / "audio-encoder"
_SESSIONS_ROOT = _PROJECT_ROOT / "Text-encoding" / "sessions"

for _p in [str(_TEXT_CODE), str(_IMAGE_ROOT), str(_AUDIO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Text-encoding imports
from embeddings_utils import (          # type: ignore
    extract_text,
    get_local_model,
    _ensure_embedding_fn,
    _l2_normalize_rows,
)

# ─────────────────────────────────────────────────
# File-type → modality mapping
# ─────────────────────────────────────────────────
_TEXT_EXTS  = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}


def detect_modality(file_name: str) -> str:
    ext = Path(file_name).suffix.lower()
    if ext in _TEXT_EXTS:
        return "text"
    if ext in _IMAGE_EXTS:
        return "image"
    if ext in _AUDIO_EXTS:
        return "audio"
    return "text"   # fallback


# ─────────────────────────────────────────────────
# In-memory job status registry
# ─────────────────────────────────────────────────
_jobs: dict[str, dict] = {}

def get_job_status(job_id: str) -> Optional[dict]:
    return _jobs.get(job_id)

def _update_job(job_id: str, **kwargs) -> None:
    _jobs[job_id].update(kwargs)


# ─────────────────────────────────────────────────
# Lazy model singletons (loaded on first use)
# ─────────────────────────────────────────────────
_embed_model     = None
_scene_engine    = None
_voice_engine    = None
_models_lock     = threading.Lock()


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        with _models_lock:
            if _embed_model is None:
                _embed_model = get_local_model()
    return _embed_model


def _get_scene_engine():
    global _scene_engine
    if _scene_engine is None:
        with _models_lock:
            if _scene_engine is None:
                from scene_understanding.scene_understanding_engine import (  # type: ignore
                    SceneUnderstandingEngine,
                )
                _scene_engine = SceneUnderstandingEngine()
                _scene_engine.load()
    return _scene_engine


def _get_voice_engine():
    global _voice_engine
    if _voice_engine is None:
        with _models_lock:
            if _voice_engine is None:
                from audio_understanding.audio_understanding_engine import (  # type: ignore
                    VoiceUnderstandingEngine,
                )
                _voice_engine = VoiceUnderstandingEngine(
                    model_name="openai/whisper-tiny"
                )
    return _voice_engine


# ─────────────────────────────────────────────────
# Core embedding helper
# ─────────────────────────────────────────────────
_CHUNK_SIZE    = 1000
_CHUNK_OVERLAP = 100


def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap if end - overlap > start else end
    return chunks or [""]


def _embed_and_save(
    texts:      list[str],
    metadata:   list[dict],
    out_path:   Path,
    normalize:  bool = True,
) -> None:
    """Embed a list of text chunks with E5 and pickle the result."""
    model  = _get_embed_model()
    emb_fn = _ensure_embedding_fn(model)

    batch_size = 64
    emb_parts  = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb   = emb_fn(batch).astype(np.float32)
        emb_parts.append(emb)

    embeddings = np.vstack(emb_parts).astype(np.float32)
    if normalize:
        embeddings = _l2_normalize_rows(embeddings)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "texts": texts, "metadata": metadata}, f)


# ─────────────────────────────────────────────────
# Modality-specific processors
# ─────────────────────────────────────────────────

def _process_text(
    file_path:  Path,
    session_id: str,
    job_id:     str,
) -> None:
    file_name  = file_path.name
    file_base  = file_path.stem
    ext        = file_path.suffix.lower()

    # Choose session sub-category matching existing text-encoding conventions
    category = "docs" if ext in {".pdf", ".docx", ".doc"} else "txt"
    out_dir  = _SESSIONS_ROOT / session_id / category / file_base
    out_path = out_dir / "embeddings.pkl"

    _update_job(job_id, progress="Extracting text…")
    raw_text = extract_text(str(file_path))

    _update_job(job_id, progress="Chunking and embedding…")
    chunks = _chunk_text(raw_text)
    metas  = [
        {
            "session_uid": session_id,
            "file_name":   file_name,
            "chunk_idx":   i,
            "modality":    "text",
        }
        for i in range(len(chunks))
    ]
    _embed_and_save(chunks, metas, out_path)


def _process_image(
    file_path:  Path,
    session_id: str,
    job_id:     str,
) -> None:
    from PIL import Image  # type: ignore

    file_name = file_path.name
    file_base = file_path.stem
    out_dir   = _SESSIONS_ROOT / session_id / "image" / file_base
    out_path  = out_dir / "embeddings.pkl"

    _update_job(job_id, progress="Loading image encoder (Florence-2)…")
    engine = _get_scene_engine()

    _update_job(job_id, progress="Generating caption + OCR…")
    img     = Image.open(str(file_path)).convert("RGB")
    caption = engine.more_detailed_caption(img)
    ocr     = engine.ocr(img) or ""

    combined_text = f"Image Caption: {caption}\n\nOCR Extracted Text: {ocr}"

    _update_job(job_id, progress="Embedding image description…")
    chunks = _chunk_text(combined_text)
    metas  = [
        {
            "session_uid": session_id,
            "file_name":   file_name,
            "chunk_idx":   i,
            "modality":    "image",
        }
        for i in range(len(chunks))
    ]
    _embed_and_save(chunks, metas, out_path)


def _process_audio(
    file_path:  Path,
    session_id: str,
    job_id:     str,
) -> None:
    file_name = file_path.name
    file_base = file_path.stem
    out_dir   = _SESSIONS_ROOT / session_id / "audio" / file_base
    out_path  = out_dir / "embeddings.pkl"

    _update_job(job_id, progress="Loading audio encoder (Whisper)…")
    engine = _get_voice_engine()

    _update_job(job_id, progress="Transcribing audio with timestamps…")
    result = engine.transcribe_with_timestamps(str(file_path))

    _update_job(job_id, progress="Embedding transcript segments…")
    chunks_raw = result.get("chunks", [])

    if chunks_raw:
        # Use Whisper's natural segment boundaries as chunks
        texts = [c["text"] for c in chunks_raw]
        metas = [
            {
                "session_uid": session_id,
                "file_name":   file_name,
                "chunk_idx":   i,
                "modality":    "audio",
                "timestamp":   c.get("timestamp", []),
                "language":    result.get("language", "auto"),
            }
            for i, c in enumerate(chunks_raw)
        ]
    else:
        # Fallback: embed full transcription as single chunk
        full_text = result.get("text", "")
        texts = _chunk_text(full_text)
        metas = [
            {
                "session_uid": session_id,
                "file_name":   file_name,
                "chunk_idx":   i,
                "modality":    "audio",
                "language":    result.get("language", "auto"),
            }
            for i in range(len(texts))
        ]

    _embed_and_save(texts, metas, out_path)


# ─────────────────────────────────────────────────
# Main dispatcher (runs in background thread)
# ─────────────────────────────────────────────────

_PROCESSORS = {
    "text":  _process_text,
    "image": _process_image,
    "audio": _process_audio,
}


def _run_ingestion(
    file_path:  Path,
    session_id: str,
    job_id:     str,
    modality:   str,
) -> None:
    """Entry point executed in a background thread."""
    try:
        _update_job(job_id, status="processing", progress="Starting…")
        processor = _PROCESSORS[modality]
        processor(file_path, session_id, job_id)
        _update_job(job_id, status="done", progress="Complete ✅")

        # Invalidate retrieval cache for this session
        from backend.services import retrieval_service  # lazy import avoids circular
        retrieval_service.invalidate_cache(session_id)

    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc), progress="Failed ❌")
        print(f"[IngestionService] ERROR job={job_id}: {exc}")
    finally:
        # Clean up the temp file
        try:
            file_path.unlink(missing_ok=True)
        except Exception:
            pass


def ingest_file(
    tmp_file_path: Path,
    session_id:    str,
    file_name:     str,
) -> tuple[str, str]:
    """
    Register a new ingestion job and start it asynchronously.

    Args:
        tmp_file_path: Path to the already-saved temp file.
        session_id:    Target session.
        file_name:     Original filename (used for extension detection).

    Returns:
        (job_id, modality)
    """
    job_id   = str(uuid.uuid4())
    modality = detect_modality(file_name)

    # Rename temp file to the real name in a temp job dir
    job_dir = _SESSIONS_ROOT / session_id / "_tmp" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    dest = job_dir / file_name
    shutil.move(str(tmp_file_path), str(dest))

    # Register job
    _jobs[job_id] = {
        "job_id":     job_id,
        "session_id": session_id,
        "file_name":  file_name,
        "modality":   modality,
        "status":     "queued",
        "progress":   "Queued",
        "error":      None,
    }

    # Launch in background thread (FastAPI BackgroundTasks can't run blocking code)
    t = threading.Thread(
        target=_run_ingestion,
        args=(dest, session_id, job_id, modality),
        daemon=True,
    )
    t.start()

    return job_id, modality

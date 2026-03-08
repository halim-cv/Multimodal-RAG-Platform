"""
backend/services/ingestion_service.py

Unified multimodal ingestion dispatcher.

Given any uploaded file, this service:
  1. Detects the modality (text / image / audio) from the file extension.
  2. Routes to the correct encoder pipeline.
  3. Chunks the extracted representation into fixed-size windows.
  4. Embeds all chunks with the shared E5-small-v2 model.
  5. Saves `embeddings.pkl` into the session directory.
  6. Tracks job status in an in-memory dict (queryable via the /upload/status endpoint).

PDF pipeline (full structured):
  PDF → DocLayout-YOLO (layout detection)
      → PyMuPDF OCR per text region
      → Florence-2 caption + OCR per figure/table crop
      → Merge all chunks → E5 embed

Other text files (.docx, .txt, .md): fast extract_text() → chunk → embed
Image files:  Florence-2 caption + OCR → embed
Audio files:  Whisper timestamped segments → 30s windows → embed
"""

import os
import sys
import gc
import pickle
import uuid
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

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

# ─────────────────────────────────────────────────
# Device auto-detection (once at import time, shared by all engines)
# ─────────────────────────────────────────────────
def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[IngestionService] GPU detected: {name} — using CUDA")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[IngestionService] Apple Silicon GPU — using MPS")
            return "mps"
    except Exception:
        pass
    print("[IngestionService] No GPU — using CPU")
    return "cpu"

DEVICE = _detect_device()

# Graceful imports — the server can START even if heavy ML libs aren't installed
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    from embeddings_utils import (          # type: ignore
        extract_text,
        get_local_model,
        _ensure_embedding_fn,
        _l2_normalize_rows,
    )
    _HAS_EMBEDDINGS = True
except Exception:
    _HAS_EMBEDDINGS = False
    extract_text = None
    get_local_model = None
    _ensure_embedding_fn = None
    _l2_normalize_rows = None

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
_doc_engine      = None   # DocLayout-YOLO for PDF structure detection
_voice_engine    = None
_models_lock     = threading.Lock()


def _get_embed_model():
    if not _HAS_EMBEDDINGS:
        raise RuntimeError(
            "Embedding utilities not available. "
            "Install sentence-transformers, faiss-cpu, and run download_model.py."
        )
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
                _scene_engine = SceneUnderstandingEngine(device=DEVICE)
                _scene_engine.load()
    return _scene_engine


def _get_doc_engine():
    """Lazy-load the DocLayout-YOLO document understanding engine."""
    global _doc_engine
    if _doc_engine is None:
        with _models_lock:
            if _doc_engine is None:
                _doc_dir = _IMAGE_ROOT / "document_understanding"
                _doc_dir_str = str(_doc_dir)
                if _doc_dir_str not in sys.path:
                    sys.path.insert(0, _doc_dir_str)
                from document_understanding_engine import (  # type: ignore
                    DocumentUnderstandingEngine,
                )
                _doc_engine = DocumentUnderstandingEngine()
                _doc_engine.load()
    return _doc_engine


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


def _offload_processing_models() -> None:
    """
    Free heavy processing models (Florence-2, DocLayout-YOLO, Whisper)
    from GPU/CPU memory after an ingestion job finishes.
    Only the E5 embedding model is kept alive for fast retrieval.
    """
    global _scene_engine, _doc_engine, _voice_engine

    with _models_lock:
        if _scene_engine is not None:
            try:
                _scene_engine.unload()
            except Exception as e:
                print(f"[IngestionService] Florence-2 offload warning: {e}")
            _scene_engine = None

        if _doc_engine is not None:
            try:
                # DocLayout-YOLO model object
                _doc_engine.model = None
                _doc_engine._is_loaded = False
            except Exception as e:
                print(f"[IngestionService] DocLayout offload warning: {e}")
            _doc_engine = None

        if _voice_engine is not None:
            try:
                _voice_engine.model = None
            except Exception as e:
                print(f"[IngestionService] Whisper offload warning: {e}")
            _voice_engine = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    print("[IngestionService] ✅ Processing models offloaded. E5 remains loaded.")


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
    if not _HAS_NUMPY or not _HAS_EMBEDDINGS:
        raise RuntimeError("ML dependencies not available for embedding.")

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
    file_name = file_path.name
    file_base = file_path.stem
    ext       = file_path.suffix.lower()
    category  = "docs" if ext in {".pdf", ".docx", ".doc"} else "txt"
    out_dir   = _SESSIONS_ROOT / session_id / category / file_base
    out_path  = out_dir / "embeddings.pkl"

    # ── PDF: full structured pipeline ──────────────────────────────────────
    if ext == ".pdf":
        _process_pdf(file_path, session_id, job_id, file_name, file_base, out_path)
        return

    # ── Non-PDF (docx, txt, md, csv): fast text-extraction path ────────────
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


def _process_pdf(
    file_path:  Path,
    session_id: str,
    job_id:     str,
    file_name:  str,
    file_base:  str,
    out_path:   Path,
) -> None:
    """
    Full PDF ingestion pipeline (mirrors image_pipeline.py):
      Phase 1 — DocLayout-YOLO: extract_all_pairs() → crop_and_combine()
                produces combined figure+caption and table+caption images
      Phase 2 — Florence-2: more_detailed_caption() + ocr() on each combined image
      Text    — PyMuPDF: full-page text extraction (no bbox clipping needed)
      Final   — Merge all text + visual chunks → E5 embed → save .pkl
    """
    import fitz  # PyMuPDF
    from PIL import Image as PILImage

    all_texts: list[str] = []
    all_metas: list[dict] = []
    chunk_idx = 0

    # ── Phase 1: DocLayout-YOLO — extract figure/table + caption pairs ───
    extracted_pairs: dict = {"figures": [], "tables": [], "total_pairs": 0}

    _update_job(job_id, progress="Loading document layout engine…")
    try:
        doc_engine = _get_doc_engine()
        _update_job(job_id, progress="Extracting figures & tables (DocLayout-YOLO)…")
        # extract_all_pairs does: process_pdf → associate → crop_and_combine
        # Returns: {"figures": [...], "tables": [...], "total_pairs": N}
        # Each item has: combined_image, pair_type, page_number, main_bbox, caption_bbox
        extracted_pairs = doc_engine.extract_all_pairs(str(file_path), dpi=200)
        print(f"[IngestionService] DocLayout extracted {extracted_pairs['total_pairs']} "
              f"visual pairs ({len(extracted_pairs['figures'])} figures, "
              f"{len(extracted_pairs['tables'])} tables)")
    except Exception as exc:
        print(f"[IngestionService] DocLayout-YOLO failed ({exc}), skipping visual extraction.")

    # ── Text extraction via PyMuPDF (all pages, full text) ───────────────
    _update_job(job_id, progress="Extracting text from document pages…")
    doc = fitz.open(str(file_path))

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        page_text = page.get_text("text").strip()

        if page_text:
            for chunk in _chunk_text(page_text):
                all_texts.append(chunk)
                all_metas.append({
                    "session_uid": session_id,
                    "file_name":   file_name,
                    "chunk_idx":   chunk_idx,
                    "modality":    "text",
                    "chunk_type":  "page_text",
                    "page":        page_num,
                })
                chunk_idx += 1

    doc.close()

    # ── Phase 2: Florence-2 — caption + OCR on each combined image ───────
    all_visual_items = extracted_pairs["figures"] + extracted_pairs["tables"]

    if all_visual_items:
        _update_job(job_id, progress="Loading scene understanding engine (Florence-2)…")
        try:
            scene_engine = _get_scene_engine()

            for vis_idx, pair in enumerate(all_visual_items, 1):
                pair_type   = pair["pair_type"]       # "figure-caption" or "table-caption"
                page_number = pair["page_number"]
                combined_img_np = pair["combined_image"]  # numpy array from crop_and_combine

                _update_job(
                    job_id,
                    progress=f"Captioning {pair_type} {vis_idx}/{len(all_visual_items)} "
                             f"(page {page_number + 1})…"
                )

                # Convert to PIL for Florence-2
                pil_img = PILImage.fromarray(combined_img_np)

                # Generate caption + OCR (matching image_pipeline.py Phase 2)
                detailed_caption = scene_engine.more_detailed_caption(pil_img) or ""
                extracted_ocr    = scene_engine.ocr(pil_img) or ""

                combined_text = (
                    f"[{pair_type.upper()}] Page {page_number + 1}\n"
                    f"Visual description: {detailed_caption}\n"
                    f"OCR text: {extracted_ocr}"
                ).strip()

                if combined_text:
                    for chunk in _chunk_text(combined_text):
                        all_texts.append(chunk)
                        all_metas.append({
                            "session_uid": session_id,
                            "file_name":   file_name,
                            "chunk_idx":   chunk_idx,
                            "modality":    "image",
                            "chunk_type":  pair_type,
                            "page":        page_number,
                        })
                        chunk_idx += 1

                print(f"  ✓ {pair_type} p{page_number+1}: "
                      f"{detailed_caption[:80]}…")

        except Exception as exc:
            print(f"[IngestionService] Florence-2 visual captioning failed ({exc}), "
                  f"skipping visuals.")

    # ── Final: embed all chunks together ─────────────────────────────────
    if not all_texts:
        # Last-resort fallback: plain pypdf extraction
        _update_job(job_id, progress="Fallback: extracting text…")
        raw_text = extract_text(str(file_path))
        all_texts = _chunk_text(raw_text)
        all_metas = [
            {"session_uid": session_id, "file_name": file_name,
             "chunk_idx": i, "modality": "text"}
            for i in range(len(all_texts))
        ]

    _update_job(job_id, progress=f"Embedding {len(all_texts)} chunks…")
    _embed_and_save(all_texts, all_metas, out_path)


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

    # Spec format: "Caption: {caption}\nOCR: {text}"
    combined_text = f"Caption: {caption}\nOCR: {ocr}"

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


# ─────────────────────────────────────────────────
# Audio: 30-second timestamp window chunking
# ─────────────────────────────────────────────────
_AUDIO_WINDOW_SEC  = 30.0   # 30-second windows
_AUDIO_OVERLAP_SEC = 5.0    # 5-second overlap between windows


def _chunk_audio_by_windows(
    segments: list[dict],
    window_sec:  float = _AUDIO_WINDOW_SEC,
    overlap_sec: float = _AUDIO_OVERLAP_SEC,
) -> list[dict]:
    """
    Group Whisper timestamped segments into fixed-duration windows.

    Each Whisper segment has:
        {"text": "...", "timestamp": [start, end]}

    Returns a list of dicts:
        {"text": "merged text", "timestamp": [window_start, window_end]}

    Strategy:
      - Slide a window of `window_sec` seconds across the timeline.
      - Collect all segments whose timestamp overlaps the window.
      - Advance by (window_sec - overlap_sec) seconds.
    """
    if not segments:
        return []

    # Determine the full audio duration from the last segment
    max_end = max(
        (seg["timestamp"][1] for seg in segments if seg.get("timestamp")),
        default=0.0,
    )

    if max_end <= 0:
        # No valid timestamps — return all segments merged as one chunk
        full_text = " ".join(s["text"].strip() for s in segments if s.get("text"))
        return [{"text": full_text, "timestamp": [0.0, 0.0]}]

    windows = []
    win_start = 0.0
    step = window_sec - overlap_sec

    while win_start < max_end:
        win_end = win_start + window_sec

        # Collect segments that overlap this window
        window_texts = []
        actual_start = win_end  # will be narrowed
        actual_end   = win_start

        for seg in segments:
            ts = seg.get("timestamp")
            if not ts or len(ts) < 2:
                continue
            seg_start, seg_end = ts[0], ts[1]

            # Segment overlaps the window if it starts before win_end
            # and ends after win_start
            if seg_start < win_end and seg_end > win_start:
                text = seg.get("text", "").strip()
                if text:
                    window_texts.append(text)
                    actual_start = min(actual_start, seg_start)
                    actual_end   = max(actual_end, seg_end)

        if window_texts:
            windows.append({
                "text":      " ".join(window_texts),
                "timestamp": [round(actual_start, 2), round(actual_end, 2)],
            })

        win_start += step

    return windows if windows else [
        {"text": " ".join(s["text"].strip() for s in segments if s.get("text")),
         "timestamp": [0.0, max_end]}
    ]


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

    _update_job(job_id, progress="Chunking by 30s windows and embedding…")
    raw_segments = result.get("chunks", [])
    language     = result.get("language", "auto")

    if raw_segments:
        # Group Whisper segments into 30-second sliding windows
        windowed = _chunk_audio_by_windows(raw_segments)
        texts = [w["text"] for w in windowed]
        metas = [
            {
                "session_uid": session_id,
                "file_name":   file_name,
                "chunk_idx":   i,
                "modality":    "audio",
                "timestamp":   w["timestamp"],
                "language":    language,
            }
            for i, w in enumerate(windowed)
        ]
    else:
        # Fallback: embed full transcription using text chunking
        full_text = result.get("text", "")
        texts = _chunk_text(full_text)
        metas = [
            {
                "session_uid": session_id,
                "file_name":   file_name,
                "chunk_idx":   i,
                "modality":    "audio",
                "language":    language,
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
    t_start = time.time()
    num_chunks  = 0

    try:
        _update_job(job_id, status="processing", progress="Starting…")
        processor = _PROCESSORS[modality]
        processor(file_path, session_id, job_id)
        _update_job(job_id, status="done", progress="Complete ✅")

        # Count chunks produced (best-effort)
        try:
            import pickle
            file_base = file_path.stem
            category  = ("docs" if file_path.suffix.lower() in {".pdf", ".docx", ".doc"}
                         else "txt" if modality == "text"
                         else modality)
            pkl = _SESSIONS_ROOT / session_id / category / file_base / "embeddings.pkl"
            if pkl.exists():
                with open(pkl, "rb") as f:
                    data = pickle.load(f)
                num_chunks = len(data.get("texts", []))
        except Exception:
            pass

        # Invalidate retrieval cache for this session
        from backend.services import retrieval_service  # lazy import avoids circular
        retrieval_service.invalidate_cache(session_id)

    except Exception as exc:
        _update_job(job_id, status="error", error=str(exc), progress="Failed ❌")
        print(f"[IngestionService] ERROR job={job_id}: {exc}")
    finally:
        elapsed_ms = (time.time() - t_start) * 1000

        # ─ Log to MLflow (silent on failure) ────────────────────────────────
        try:
            from mlflow_tracking.tracker import tracker
            tracker.log_ingestion(
                session_id  = session_id,
                file_name   = file_path.name,
                modality    = modality,
                num_chunks  = num_chunks,
                elapsed_ms  = round(elapsed_ms, 1),
            )
        except Exception:
            pass

        # ─ Offload heavy processing models — keep only E5 in memory ─────────
        _offload_processing_models()

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
    _SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)

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

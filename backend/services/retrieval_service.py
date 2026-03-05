"""
backend/services/retrieval_service.py

Multimodal FAISS retrieval.
Loads all embeddings.pkl files from a session (text / image / audio),
merges them into a single cosine-similarity index, and returns ranked chunks.

Includes a simple session-level index cache so FAISS isn't rebuilt on every query.
"""

import os
import sys
import pickle
import time
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────
# Path bootstrap — import E5 utilities from text-encoding module
# ─────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent          # Multimodal-RAG-Platform/
_TEXT_CODE    = _PROJECT_ROOT / "Text-encoding" / "model" / "code"

for _p in [str(_TEXT_CODE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Graceful imports for heavy ML dependencies
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None  # type: ignore
    _HAS_FAISS = False

try:
    from embeddings_utils import get_local_model, embed_query   # type: ignore
    _HAS_EMBEDDINGS = True
except Exception:
    _HAS_EMBEDDINGS = False
    get_local_model = None
    embed_query = None

# ─────────────────────────────────────────────────
# Sessions root (mirrors what Text-encoding uses)
# ─────────────────────────────────────────────────
_SESSIONS_ROOT = _PROJECT_ROOT / "Text-encoding" / "sessions"

# ─────────────────────────────────────────────────
# Embedding model (loaded once, shared)
# ─────────────────────────────────────────────────
_embed_model = None

def _get_embed_model():
    if not _HAS_EMBEDDINGS:
        raise RuntimeError(
            "Embedding utilities not available. "
            "Install sentence-transformers, faiss-cpu, and run download_model.py."
        )
    global _embed_model
    if _embed_model is None:
        _embed_model = get_local_model()
    return _embed_model


# ─────────────────────────────────────────────────
# Simple in-memory index cache
# {session_id: {"index": faiss.Index, "texts": list, "metadata": list, "ts": float}}
# ─────────────────────────────────────────────────
_INDEX_CACHE: dict = {}
_CACHE_TTL_SECONDS = 300   # 5 minutes


def _l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _load_session_embeddings(
    session_id: str,
    modalities: Optional[list[str]] = None,
) -> tuple:
    """
    Walk the session directory and load ALL embeddings.pkl files.
    Optionally filter by modality.

    Returns:
        (combined_embeddings, all_texts, all_metadata)
    """
    if not _HAS_NUMPY:
        raise RuntimeError("numpy is required for retrieval.")

    session_dir = _SESSIONS_ROOT / session_id
    if not session_dir.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")

    all_embs   : list = []
    all_texts  : list[str]  = []
    all_meta   : list[dict] = []

    for pkl_path in session_dir.rglob("embeddings.pkl"):
        try:
            with open(pkl_path, "rb") as f:
                payload = pickle.load(f)

            embs  = np.asarray(payload["embeddings"], dtype=np.float32)
            texts = payload.get("texts", [])
            metas = payload.get("metadata", [])

            # Inject modality from metadata or infer from path
            for i, meta in enumerate(metas):
                if "modality" not in meta:
                    # Infer from the folder structure: …/image/…, …/audio/…
                    parts = pkl_path.parts
                    if "image" in parts:
                        meta["modality"] = "image"
                    elif "audio" in parts:
                        meta["modality"] = "audio"
                    else:
                        meta["modality"] = "text"

            # Filter by requested modalities
            if modalities:
                keep = [
                    j for j, m in enumerate(metas)
                    if m.get("modality", "text") in modalities
                ]
                if not keep:
                    continue
                embs  = embs[keep]
                texts = [texts[j] for j in keep]
                metas = [metas[j] for j in keep]

            all_embs.append(embs)
            all_texts.extend(texts)
            all_meta.extend(metas)

        except Exception as exc:
            print(f"[RetrievalService] Warning: could not load {pkl_path}: {exc}")
            continue

    if not all_embs:
        raise ValueError(
            f"No embeddings found for session '{session_id}' "
            f"(modalities={modalities}). Upload and process files first."
        )

    combined = np.vstack(all_embs).astype(np.float32)
    return combined, all_texts, all_meta


def _build_index(embeddings):
    """Build a cosine-similarity FAISS index (normalise → inner product)."""
    if not _HAS_FAISS:
        raise RuntimeError("faiss-cpu is required for retrieval. Install with: pip install faiss-cpu")

    embs = _l2_normalize(embeddings.copy())
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index


def _get_cached_index(
    session_id: str,
    modalities: Optional[list[str]] = None,
) -> tuple:
    """Return cached index or rebuild it."""
    cache_key = f"{session_id}_{sorted(modalities or [])}"
    cached    = _INDEX_CACHE.get(cache_key)

    if cached and (time.time() - cached["ts"]) < _CACHE_TTL_SECONDS:
        return cached["index"], cached["texts"], cached["metadata"]

    # Rebuild
    embs, texts, metas = _load_session_embeddings(session_id, modalities)
    index = _build_index(embs)

    _INDEX_CACHE[cache_key] = {
        "index":    index,
        "texts":    texts,
        "metadata": metas,
        "ts":       time.time(),
    }
    return index, texts, metas


def invalidate_cache(session_id: str) -> None:
    """Call this after new files are ingested into a session."""
    to_remove = [k for k in _INDEX_CACHE if k.startswith(session_id)]
    for k in to_remove:
        del _INDEX_CACHE[k]


# ─────────────────────────────────────────────────
# Public retrieval function
# ─────────────────────────────────────────────────

def retrieve(
    session_id: str,
    query:      str,
    top_k:      int           = 5,
    modalities: Optional[list[str]] = None,
) -> list[dict]:
    """
    Embed the query and retrieve the top-K most relevant chunks.

    Returns a list of dicts:
        {text, score, source, modality, chunk_idx, timestamp (optional)}
    """
    if not _HAS_EMBEDDINGS or not _HAS_NUMPY:
        raise RuntimeError("ML dependencies not available for retrieval.")

    model  = _get_embed_model()
    q_vec  = embed_query(model, query, normalize=True).reshape(1, -1)
    q_norm = _l2_normalize(q_vec)

    index, texts, metas = _get_cached_index(session_id, modalities)

    k = min(top_k, len(texts))
    distances, indices = index.search(q_norm, k)

    results = []
    for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
        if idx < 0 or idx >= len(texts):
            continue
        meta = metas[idx]
        results.append({
            "text":      texts[idx],
            "score":     float(score),
            "source":    meta.get("file_name", "unknown"),
            "modality":  meta.get("modality", "text"),
            "chunk_idx": meta.get("chunk_idx", idx),
            "timestamp": meta.get("timestamp"),
        })

    return results

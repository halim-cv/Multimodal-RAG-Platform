"""
embeddings_utils.py

Provides:
1. embed_folder(model, folder_path, out_path='embeddings.pkl', ...)
2. embed_query(model, query, ...)
3. load_embeddings(file_path) -> embeddings, texts, metadata
4. build_faiss_index(embeddings, normalize=False, use_cosine=False) -> faiss.Index
5. search(index, query_vector, k=5, texts=None, metadata=None) -> list of matches
"""

import os
import glob
import pickle
from typing import Callable, List, Tuple, Dict, Any, Optional
import numpy as np
import faiss

# --------------------
# Helpers
# --------------------
def _ensure_embedding_fn(model: Any) -> Callable[[List[str]], np.ndarray]:
    """
    Return a function f(texts: List[str]) -> np.ndarray of shape (len(texts), D), dtype=float32.
    Accepts:
      - a callable model (embedding_fn)
      - or an object with .encode(texts) method
    """
    # Fix the embedding function to call the model's `encode` method explicitly
    if callable(model):
        def fn(texts: List[str]) -> np.ndarray:
            if isinstance(texts, list):
                out = model.encode(texts)  # Explicitly call `encode` for SentenceTransformer
                arr = np.asarray(out, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return arr
            else:
                raise ValueError("Input to the model must be a list of strings.")
        return fn

    if hasattr(model, "encode"):
        def fn(texts: List[str]) -> np.ndarray:
            if isinstance(texts, list):
                out = model.encode(texts)  # Explicitly call `encode` for SentenceTransformer
                arr = np.asarray(out, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                return arr
            else:
                raise ValueError("Input to the model must be a list of strings.")
        return fn

    raise ValueError("Model must be a callable or have an .encode(texts) method.")


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


# --------------------
# Core functions
# --------------------
def embed_folder(
    model: Any,
    folder_path: str,
    out_path: str = "embeddings.pkl",
    file_pattern: str = "*.txt",
    chunk_size_chars: Optional[int] = None,
    chunk_overlap_chars: int = 0,
    normalize: bool = False,
) -> str:
    """
    Read all .txt files in folder_path, create embeddings, and save to out_path (pickle).
    Returns the out_path string.

    - model: callable or object with .encode(list_of_texts)
    - chunk_size_chars: if provided, split each text into chunks of approx chunk_size_chars
    - normalize: if True, L2-normalize embeddings (useful for cosine similarity)
    """

    emb_fn = _ensure_embedding_fn(model)

    pattern = os.path.join(folder_path, file_pattern)
    files = sorted(glob.glob(pattern))
    texts: List[str] = []
    metadata: List[Dict[str, Any]] = []

    if not files:
        raise FileNotFoundError(f"No files found with pattern {pattern}")

    def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
        if size is None:
            return [text]
        chunks = []
        start = 0
        L = len(text)
        while start < L:
            end = start + size
            chunk = text[start:end]
            chunks.append(chunk)
            start = max(end - overlap, end)
        return chunks

    for file_idx, fp in enumerate(files):
        with open(fp, "r", encoding="utf-8") as f:
            content = f.read()
        # optionally chunk
        chunks = _chunk_text(content, chunk_size_chars, chunk_overlap_chars) if chunk_size_chars else [content]
        for chunk_idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadata.append({
                "file_path": fp,
                "file_name": os.path.basename(fp),
                "file_idx": file_idx,
                "chunk_idx": chunk_idx,
            })

    # create embeddings in batches (avoid OOM for huge text collections)
    batch_size = 256
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        emb = emb_fn(batch_texts)  # shape (len(batch_texts), D)
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        embeddings_list.append(emb)

    embeddings = np.vstack(embeddings_list).astype(np.float32)

    if normalize:
        embeddings = _l2_normalize_rows(embeddings)

    payload = {
        "embeddings": embeddings,
        "texts": texts,
        "metadata": metadata,
    }

    # Save to pickle
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    return out_path


def embed_query(model: Any, query: str, normalize: bool = False) -> np.ndarray:
    """
    Embed a single query string and return a 1-D numpy array (dtype float32).
    """
    emb_fn = _ensure_embedding_fn(model)
    arr = emb_fn([query])[0]  # shape (D,)
    if normalize:
        arr = arr / (np.linalg.norm(arr) + 1e-12)
    return arr.astype(np.float32)


def load_embeddings(file_path: str) -> Tuple[np.ndarray, List[str], List[Dict[str, Any]]]:
    """
    Load embeddings pickle saved by embed_folder.
    Returns (embeddings (np.ndarray), texts (List[str]), metadata (List[dict])).
    """
    with open(file_path, "rb") as f:
        payload = pickle.load(f)

    embeddings = payload.get("embeddings")
    texts = payload.get("texts", [])
    metadata = payload.get("metadata", [])
    if embeddings is None:
        raise ValueError("Loaded file does not contain 'embeddings' key.")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings, texts, metadata


# --------------------
# FAISS helpers
# --------------------
def build_faiss_index(embeddings: np.ndarray, use_cosine: bool = False, normalize_before: bool = False) -> Any:
    """
    Build and return a FAISS index from embeddings.
    - embeddings: np.ndarray shape (N, D), dtype float32
    - use_cosine: if True, will normalize embeddings and index with IndexFlatIP (inner product)
    - normalize_before: if True, L2-normalize embeddings before building (needed for cosine)
    """
    if faiss is None:
        raise RuntimeError("faiss is not installed. Install with `pip install faiss-cpu` (or faiss-gpu).")

    emb = np.asarray(embeddings, dtype=np.float32)
    if normalize_before or use_cosine:
        emb = _l2_normalize_rows(emb)

    d = emb.shape[1]
    # simple flat index (exact search). For larger data use IVF/PQ indexes.
    if use_cosine:
        index = faiss.IndexFlatIP(d)  # inner product on normalized vectors == cosine
    else:
        index = faiss.IndexFlatL2(d)  # L2 distance
    index.add(emb)
    return index


def save_faiss_index(index: Any, path: str) -> None:
    if faiss is None:
        raise RuntimeError("faiss is not installed.")
    faiss.write_index(index, path)


def load_faiss_index(path: str) -> Any:
    if faiss is None:
        raise RuntimeError("faiss is not installed.")
    return faiss.read_index(path)


def search(
    index: Any,
    query_vector: np.ndarray,
    k: int = 5,
    texts: Optional[List[str]] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    return_scores: bool = True,
    use_cosine: bool = False,
) -> List[Dict[str, Any]]:
    """
    Search the given faiss index with query_vector (shape (D,) or (1,D)).
    If texts/metadata provided, map indices to them.
    Returns list of matches: {rank, id, score, index, text?, metadata?}
    - For L2 index, lower distance = better. For IP (cosine) higher = better.
    """
    if faiss is None:
        raise RuntimeError("faiss is not installed.")

    q = np.asarray(query_vector, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, -1)

    # If index expects normalized vectors for cosine, user should normalize query externally.
    distances, indices = index.search(q, k)  # distances shape (1,k), indices shape (1,k)
    distances = distances[0]
    indices = indices[0]

    results = []
    for rank, (idx, dist) in enumerate(zip(indices.tolist(), distances.tolist()), start=1):
        entry = {"rank": rank, "index": idx}
        if return_scores:
            entry["score"] = float(dist)
        if texts is not None:
            entry["text"] = texts[idx] if (0 <= idx < len(texts)) else None
        if metadata is not None:
            entry["metadata"] = metadata[idx] if (0 <= idx < len(metadata)) else None
        results.append(entry)
    return results


# --------------------
# Example usage 
# --------------------
if __name__ == "__main__":
    # Example "model" using sentence-transformers (if installed):
    from sentence_transformers import SentenceTransformer
    import time
    model = SentenceTransformer("all-mpnet-base-v2")  # A high-quality model for embeddings
    time_str = time.time()
    out = embed_folder(model, "MAS", out_path="embeddings.pkl", chunk_size_chars=2000, normalize=True)
    time_end = time.time()
    print(out)
    print()
    print(f"Time Consumed : {time_end - time_str}")
    #emb, texts, meta = load_embeddings("demo_emb.pkl")
    #idx = build_faiss_index(emb, use_cosine=False)
    #q = embed_query(model, "search me")
    #results = search(idx, q, k=3, texts=texts, metadata=meta)
    #print(results)

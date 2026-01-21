"""
embeddings_utils.py

Provides:
1. embed_folder(model, folder_path, out_path='embeddings.pkl', ...)
2. embed_query(model, query, ...)
3. load_embeddings(file_path) -> embeddings, texts, metadata
4. build_faiss_index(embeddings, normalize=False, use_cosine=False) -> faiss.Index
5. search(index, query_vector, k=5, texts=None, metadata=None) -> list of matches
"""

# --------------------
# Imports
# --------------------
import os
import glob
import pickle
from typing import Callable, List, Tuple, Dict, Any, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Extraction libraries
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

# Project specific imports
from config import get_config, apply_cpu_limit
from handle_session import get_file_artifacts_paths, get_session_overall_artifacts_paths, get_session_base_path

def generate_session_overall_artifacts(
    session_uid: str,
    chunk_size: Optional[int] = 1000,
    chunk_overlap: int = 100,
    normalize: bool = True
) -> str:
    """
    Combines all extracted text from a session into one 'overall' file,
    adds source annotations, generates combined embeddings, and saves them.
    """
    session_base = get_session_base_path(session_uid)
    overall_paths = get_session_overall_artifacts_paths(session_uid)
    overall_text_path = overall_paths["extracted_text"]
    overall_emb_path = overall_paths["embeddings_pkl"]

    print(f"--- Generating Overall Context for Session: {session_uid} ---")
    
    # 1. Collect all extracted text files
    combined_content = []
    
    # We iterate through categories and file folders
    for category in ["docs", "txt", "pdf", "img", "other"]:
        cat_dir = os.path.join(session_base, category)
        if not os.path.exists(cat_dir):
            continue
            
        for file_folder in sorted(os.listdir(cat_dir)):
            folder_path = os.path.join(cat_dir, file_folder)
            if not os.path.isdir(folder_path):
                continue
                
            text_file = os.path.join(folder_path, "extracted_text.txt")
            if os.path.exists(text_file):
                print(f"Adding text from: {file_folder}")
                with open(text_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Add annotation/header
                header = f"\n\n--- SOURCE: {file_folder} ---\n\n"
                combined_content.append(header + content)

    if not combined_content:
        raise ValueError(f"No extracted text found in session {session_uid} to combine.")

    full_text = "".join(combined_content)

    # 2. Save combined text
    with open(overall_text_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"Combined text saved to: {overall_text_path}")

    # 3. Load Model
    model = get_local_model()
    emb_fn = _ensure_embedding_fn(model)

    # 4. Chunking with Source Preservation
    # We chunk the full text, but we want to know which source each chunk came from.
    # A simple way is to re-parse the headers, but since we have the data here, 
    # let's be more precise.
    
    texts = []
    metadata = []
    
    # Re-iterating to be precise about source metadata per chunk
    for category in ["docs", "txt", "pdf", "img", "other"]:
        cat_dir = os.path.join(session_base, category)
        if not os.path.exists(cat_dir):
            continue
        for file_folder in sorted(os.listdir(cat_dir)):
            folder_path = os.path.join(cat_dir, file_folder)
            if not os.path.isdir(folder_path): continue
            text_file = os.path.join(folder_path, "extracted_text.txt")
            if os.path.exists(text_file):
                with open(text_file, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Annotate content
                annotated_content = f"Source: {file_folder}\n\n{content}"
                
                # Use the helper _chunk_text defined elsewhere or local
                def _local_chunk(text, size, overlap):
                    if size is None: return [text]
                    c = []
                    s = 0
                    while s < len(text):
                        e = s + size
                        c.append(text[s:e])
                        s = max(e - overlap, e)
                    return c
                
                chunks = _local_chunk(annotated_content, chunk_size, chunk_overlap)
                for idx, chunk in enumerate(chunks):
                    texts.append(chunk)
                    metadata.append({
                        "session_uid": session_uid,
                        "file_name": file_folder,
                        "source_category": category,
                        "chunk_idx": idx,
                        "is_overall": True
                    })

    # 5. Generate Embeddings
    print(f"Generating overall embeddings for {len(texts)} chunks...")
    batch_size = 128
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        emb = emb_fn(batch_texts)
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

    # 6. Save Vector Store
    with open(overall_emb_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[SUCCESS] Overall embeddings saved for session {session_uid}")
    return overall_emb_path

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


def get_local_model() -> SentenceTransformer:
    """
    Loads the SentenceTransformer model from the local path specified in config.py
    and applies CPU thread limits.
    """
    config = get_config()
    model_path = config["model"]["local_path"]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Local model not found at {model_path}. Please run download_model.py first.")
    
    # Apply CPU optimizations
    apply_cpu_limit()
    
    print(f"Loading local model from: {model_path}")
    model = SentenceTransformer(model_path)
    return model


def extract_text(file_path: str) -> str:
    """
    Extracts plain text from various file formats (.pdf, .docx, .txt, etc.)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    # 1. Handle PDF
    if ext == ".pdf":
        if pypdf is None:
            raise ImportError("pypdf is not installed. Run 'pip install pypdf'")
        text = ""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    # 2. Handle DOCX
    elif ext == ".docx":
        if docx is None:
            raise ImportError("python-docx is not installed. Run 'pip install python-docx'")
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    # 3. Handle plain text / markdown
    elif ext in [".txt", ".md", ".csv", ".py"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    else:
        raise ValueError(f"Unsupported file format for text extraction: {ext}")


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


def process_session_file(
    session_uid: str,
    input_file_path: str,
    chunk_size: Optional[int] = 1000,
    chunk_overlap: int = 100,
    normalize: bool = True
) -> str:
    """
    Complete Pipeline for a Session File:
    1. Extracts text from the input file.
    2. Saves the text to the session's 'extracted_text.txt' artifact.
    3. Chunks the text and generates embeddings using the local model.
    4. Saves 'embeddings.pkl' to the session's artifact location.
    """
    file_name = os.path.basename(input_file_path)
    
    # 1. Resolve artifact paths
    artifact_paths = get_file_artifacts_paths(session_uid, file_name)
    extracted_text_path = artifact_paths["extracted_text"]
    embeddings_out_path = artifact_paths["embeddings_pkl"]

    # 2. Extraction
    print(f"Extracting text from: {file_name}...")
    content = extract_text(input_file_path)
    
    # Save the extracted text for persistence
    os.makedirs(os.path.dirname(extracted_text_path), exist_ok=True)
    with open(extracted_text_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Text extracted and saved to: {extracted_text_path}")

    # 3. Load Local Model
    model = get_local_model()

    # 4. Chunking
    def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
        if size is None:
            return [text]
        chunks = []
        start = 0
        L = len(text)
        while start < L:
            end = start + size
            chunks.append(text[start:end])
            start = max(end - overlap, end)
        return chunks

    chunks = _chunk_text(content, chunk_size, chunk_overlap)
    
    texts = []
    metadata = []
    for idx, chunk in enumerate(chunks):
        texts.append(chunk)
        metadata.append({
            "session_uid": session_uid,
            "file_name": file_name,
            "chunk_idx": idx,
        })

    # 5. Embedding
    print(f"Generating embeddings for {len(texts)} chunks...")
    emb_fn = _ensure_embedding_fn(model)
    batch_size = 128
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        emb = emb_fn(batch_texts)
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

    # 6. Save Vector Store
    with open(embeddings_out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[SUCCESS] Embeddings and artifacts saved in session {session_uid}")
    return embeddings_out_path


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
    # Test session logic with extraction
    try:
        # 1. Create a dummy file to extract from
        test_session = "test_extraction_session"
        dummy_file = "test_doc.txt"
        with open(dummy_file, "w", encoding="utf-8") as f:
            f.write("This is a test document to verify the full extraction and embedding pipeline.\n" * 10)
        
        print(f"--- Starting Pipeline for: {dummy_file} ---")
        out_path = process_session_file(test_session, dummy_file)
        print(f"Final pickle saved at: {out_path}")

        # cleanup
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

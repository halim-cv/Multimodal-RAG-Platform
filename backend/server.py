"""
backend/server.py

FastAPI application entry point for the Multimodal RAG Platform.

Routes:
  GET  /               → serves frontend/Chat-Page.html
  GET  /health         → system health + loaded models
  /api/sessions        → session CRUD
  /api/upload          → file upload + job status
  /api/chat            → SSE streaming RAG
  /api/evaluate        → retrieval evaluation metrics

Run with:
  uvicorn backend.server:app --reload --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────
# Import routers
# ─────────────────────────────────────────────────
from backend.routers import sessions, upload, chat, eval_router

# ─────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────
_PROJECT_ROOT  = Path(__file__).parent.parent
_FRONTEND_DIR  = _PROJECT_ROOT / "frontend"

# ─────────────────────────────────────────────────
# Lifespan — warm up E5 embedding model at startup
# (Florence-2 and Whisper are lazy-loaded on first use)
# ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n" + "="*60)
    print("  Multimodal RAG Platform — Backend Starting")
    print("="*60)

    print("[Startup] Pre-loading E5-base-v2 embedding model…")
    try:
        from backend.services.retrieval_service import _get_embed_model
        _get_embed_model()
        print("[Startup] ✅ E5-base-v2 ready.")
    except Exception as exc:
        print(f"[Startup] ⚠️  Could not pre-load embedding model: {exc}")
        print("[Startup]    Run Text-encoding/model/code/download_model.py first.")

    print("[Startup] ✅ Server ready. Visit http://localhost:8000")
    print("="*60 + "\n")

    yield   # Server runs here

    print("\n[Shutdown] Cleaning up…")


# ─────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────
app = FastAPI(
    title       = "Multimodal RAG Platform",
    description = (
        "Production-grade Retrieval-Augmented Generation platform supporting "
        "text (PDF/DOCX/TXT), image (Florence-2 caption + OCR), and audio "
        "(Whisper transcription) modalities."
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ─────────────────────────────────────────────────
# CORS — allow all for local development
# ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ─────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────
app.include_router(sessions.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(eval_router.router)

# ─────────────────────────────────────────────────
# Serve frontend static files
# ─────────────────────────────────────────────────
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(_FRONTEND_DIR / "Chat-Page.html"))

# ─────────────────────────────────────────────────
# Health endpoint
# ─────────────────────────────────────────────────
@app.get("/health", tags=["system"])
async def health():
    """Quick health check — confirms server is up and which models are loaded."""
    from backend.services.ingestion_service import (
        _embed_model, _scene_engine, _voice_engine
    )
    return {
        "status": "ok",
        "models_loaded": {
            "e5_base_v2":    _embed_model is not None,
            "florence_2":    _scene_engine is not None,
            "whisper_tiny":  _voice_engine is not None,
        },
        "llm_provider": os.getenv("LLM_PROVIDER", "gemini"),
        "gemini_model":  os.getenv("GEMINI_MODEL",  "gemini-2.0-flash"),
    }


# ─────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.server:app",
        host    = os.getenv("BACKEND_HOST", "0.0.0.0"),
        port    = int(os.getenv("BACKEND_PORT", 8000)),
        reload  = True,
    )

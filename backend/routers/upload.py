"""
backend/routers/upload.py
POST /api/upload                  — upload a file, trigger background ingestion
GET  /api/upload/status/{job_id}  — poll job status
"""

import uuid
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile, HTTPException
import aiofiles

from backend.services import ingestion_service
from backend.models.schemas import UploadResponse, JobStatusResponse, JobStatus, Modality

router = APIRouter(prefix="/api/upload", tags=["upload"])

_TMP_DIR = Path(tempfile.gettempdir()) / "rag_uploads"
_TMP_DIR.mkdir(exist_ok=True)


@router.post("", response_model=UploadResponse)
async def upload_file(
    session_id: str = Form(...),
    file: UploadFile = File(...),
):
    """
    Accept a multipart file upload.
    Saves to a temp location, then triggers background ingestion.
    Returns a job_id to poll for status.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    # Save uploaded bytes to temp file
    tmp_path = _TMP_DIR / f"{uuid.uuid4()}_{file.filename}"
    async with aiofiles.open(tmp_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):   # 1 MB chunks
            await f.write(chunk)

    # Kick off background ingestion
    job_id, modality = ingestion_service.ingest_file(tmp_path, session_id, file.filename)

    return UploadResponse(
        job_id     = job_id,
        session_id = session_id,
        file_name  = file.filename,
        modality   = Modality(modality),
        status     = JobStatus.QUEUED,
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_upload_status(job_id: str):
    """Poll the processing status of an uploaded file."""
    job = ingestion_service.get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return JobStatusResponse(
        job_id     = job["job_id"],
        session_id = job["session_id"],
        file_name  = job["file_name"],
        status     = JobStatus(job["status"]),
        modality   = Modality(job["modality"]),
        error      = job.get("error"),
        progress   = job.get("progress"),
    )

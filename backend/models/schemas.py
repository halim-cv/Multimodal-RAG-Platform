"""
backend/models/schemas.py

All Pydantic request/response models for the Multimodal RAG Platform API.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────

class Modality(str, Enum):
    TEXT  = "text"
    IMAGE = "image"
    AUDIO = "audio"

class JobStatus(str, Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    DONE       = "done"
    ERROR      = "error"


# ─────────────────────────────────────────────────
# Session Models
# ─────────────────────────────────────────────────

class SessionCreate(BaseModel):
    name: Optional[str] = Field(None, description="Optional human-readable session name")

class SourceFile(BaseModel):
    file_name:   str
    category:    str
    modality:    Modality
    embedded:    bool
    job_status:  Optional[JobStatus] = None

class SessionInfo(BaseModel):
    session_id:    str
    created_at:    Optional[str] = None
    source_count:  int
    sources:       list[SourceFile] = []

class SessionListResponse(BaseModel):
    sessions: list[SessionInfo]


# ─────────────────────────────────────────────────
# Upload Models
# ─────────────────────────────────────────────────

class UploadResponse(BaseModel):
    job_id:     str
    session_id: str
    file_name:  str
    modality:   Modality
    status:     JobStatus = JobStatus.QUEUED

class JobStatusResponse(BaseModel):
    job_id:     str
    session_id: str
    file_name:  str
    status:     JobStatus
    modality:   Modality
    error:      Optional[str] = None
    progress:   Optional[str] = None   # human-readable step description


# ─────────────────────────────────────────────────
# Chat / Retrieval Models
# ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    query:      str
    top_k:      int           = Field(5,  ge=1, le=20)
    modalities: list[Modality] = Field(
        default=[Modality.TEXT, Modality.IMAGE, Modality.AUDIO]
    )

class RetrievedChunk(BaseModel):
    text:      str
    score:     float
    source:    str            # filename
    modality:  Modality
    chunk_idx: int
    timestamp: Optional[list[float]] = None   # [start, end] for audio chunks

class ChatResponse(BaseModel):
    answer:           str
    retrieved_chunks: list[RetrievedChunk]
    latency_ms:       float


# ─────────────────────────────────────────────────
# Evaluation Models
# ─────────────────────────────────────────────────

class EvalRequest(BaseModel):
    session_id: str
    top_k:      int = 5

class EvalMetrics(BaseModel):
    mrr_at_k:         float
    hit_at_k:         float
    precision_at_k:   float
    faithfulness:     float
    avg_latency_ms:   float
    num_queries:      int

class EvalResponse(BaseModel):
    session_id: str
    metrics:    EvalMetrics
    report_path: Optional[str] = None

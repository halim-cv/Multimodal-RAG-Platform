"""
backend/tests/test_schemas.py

Tests for all Pydantic schemas — proper validation, defaults, and enums.
"""

import pytest
from backend.models.schemas import (
    Modality, JobStatus, SessionCreate, SourceFile, SessionInfo,
    SessionListResponse, UploadResponse, JobStatusResponse,
    ChatRequest, RetrievedChunk, ChatResponse,
    EvalRequest, EvalMetrics, EvalResponse,
)


# ─────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────

class TestModality:
    def test_values(self):
        assert Modality.TEXT == "text"
        assert Modality.IMAGE == "image"
        assert Modality.AUDIO == "audio"

    def test_from_string(self):
        assert Modality("text") == Modality.TEXT
        assert Modality("image") == Modality.IMAGE

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            Modality("video")


class TestJobStatus:
    def test_values(self):
        assert JobStatus.QUEUED == "queued"
        assert JobStatus.PROCESSING == "processing"
        assert JobStatus.DONE == "done"
        assert JobStatus.ERROR == "error"


# ─────────────────────────────────────────────────
# Session Models
# ─────────────────────────────────────────────────

class TestSessionCreate:
    def test_default_name(self):
        s = SessionCreate()
        assert s.name is None

    def test_with_name(self):
        s = SessionCreate(name="My Session")
        assert s.name == "My Session"


class TestSourceFile:
    def test_valid(self):
        sf = SourceFile(
            file_name="test.pdf",
            category="docs",
            modality=Modality.TEXT,
            embedded=True,
        )
        assert sf.file_name == "test.pdf"
        assert sf.modality == Modality.TEXT
        assert sf.embedded is True
        assert sf.job_status is None

    def test_with_job_status(self):
        sf = SourceFile(
            file_name="img.png",
            category="image",
            modality=Modality.IMAGE,
            embedded=False,
            job_status=JobStatus.PROCESSING,
        )
        assert sf.job_status == JobStatus.PROCESSING


class TestSessionInfo:
    def test_minimal(self):
        si = SessionInfo(session_id="s1", source_count=0)
        assert si.session_id == "s1"
        assert si.sources == []
        assert si.created_at is None

    def test_with_sources(self):
        si = SessionInfo(
            session_id="s2",
            source_count=1,
            sources=[
                SourceFile(file_name="a.pdf", category="docs",
                           modality=Modality.TEXT, embedded=True)
            ],
        )
        assert len(si.sources) == 1


# ─────────────────────────────────────────────────
# Upload Models
# ─────────────────────────────────────────────────

class TestUploadResponse:
    def test_defaults(self):
        ur = UploadResponse(
            job_id="j1",
            session_id="s1",
            file_name="report.pdf",
            modality=Modality.TEXT,
        )
        assert ur.status == JobStatus.QUEUED


class TestJobStatusResponse:
    def test_error_field(self):
        jr = JobStatusResponse(
            job_id="j1",
            session_id="s1",
            file_name="x.mp3",
            status=JobStatus.ERROR,
            modality=Modality.AUDIO,
            error="Whisper crashed",
        )
        assert jr.error == "Whisper crashed"


# ─────────────────────────────────────────────────
# Chat Models
# ─────────────────────────────────────────────────

class TestChatRequest:
    def test_defaults(self):
        cr = ChatRequest(session_id="s1", query="What is X?")
        assert cr.top_k == 5
        assert len(cr.modalities) == 3

    def test_custom_modalities(self):
        cr = ChatRequest(
            session_id="s1",
            query="images only",
            modalities=[Modality.IMAGE],
        )
        assert cr.modalities == [Modality.IMAGE]

    def test_top_k_bounds(self):
        with pytest.raises(Exception):
            ChatRequest(session_id="s1", query="x", top_k=0)
        with pytest.raises(Exception):
            ChatRequest(session_id="s1", query="x", top_k=21)


class TestRetrievedChunk:
    def test_audio_with_timestamp(self):
        rc = RetrievedChunk(
            text="hello world",
            score=0.95,
            source="interview.mp3",
            modality=Modality.AUDIO,
            chunk_idx=0,
            timestamp=[10.0, 40.0],
        )
        assert rc.timestamp == [10.0, 40.0]

    def test_text_no_timestamp(self):
        rc = RetrievedChunk(
            text="some text",
            score=0.8,
            source="doc.pdf",
            modality=Modality.TEXT,
            chunk_idx=1,
        )
        assert rc.timestamp is None


# ─────────────────────────────────────────────────
# Eval Models
# ─────────────────────────────────────────────────

class TestEvalMetrics:
    def test_creation(self):
        em = EvalMetrics(
            mrr_at_k=0.85,
            hit_at_k=1.0,
            precision_at_k=0.45,
            faithfulness=0.94,
            avg_latency_ms=250.0,
            num_queries=5,
        )
        assert em.mrr_at_k == 0.85
        assert em.num_queries == 5

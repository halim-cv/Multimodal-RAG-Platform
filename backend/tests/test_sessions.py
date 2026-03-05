"""
backend/tests/test_sessions.py

Tests for the sessions CRUD API using FastAPI TestClient.
"""

import shutil
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.server import app
from backend.routers.sessions import _SESSIONS_ROOT


client = TestClient(app)

# Track sessions created during tests for cleanup
_test_sessions: list[str] = []


@pytest.fixture(autouse=True)
def cleanup_test_sessions():
    """Clean up any sessions created during tests."""
    yield
    for sid in _test_sessions:
        session_dir = _SESSIONS_ROOT / sid
        if session_dir.exists():
            shutil.rmtree(session_dir)
    _test_sessions.clear()


class TestListSessions:
    def test_returns_list(self):
        res = client.get("/api/sessions")
        assert res.status_code == 200
        data = res.json()
        assert "sessions" in data
        assert isinstance(data["sessions"], list)


class TestCreateSession:
    def test_creates_session(self):
        res = client.post("/api/sessions", json={"name": None})
        assert res.status_code == 200
        data = res.json()
        assert "session_id" in data
        assert data["source_count"] == 0
        assert data["sources"] == []
        _test_sessions.append(data["session_id"])

    def test_session_exists_on_disk(self):
        res = client.post("/api/sessions", json={"name": None})
        sid = res.json()["session_id"]
        _test_sessions.append(sid)
        assert (_SESSIONS_ROOT / sid).is_dir()


class TestGetSession:
    def test_existing_session(self):
        # Create first
        res = client.post("/api/sessions", json={})
        sid = res.json()["session_id"]
        _test_sessions.append(sid)

        # Get
        res = client.get(f"/api/sessions/{sid}")
        assert res.status_code == 200
        data = res.json()
        assert data["session_id"] == sid

    def test_nonexistent_session(self):
        res = client.get("/api/sessions/nonexistent_session_999")
        assert res.status_code == 404


class TestDeleteSession:
    def test_delete_existing(self):
        res = client.post("/api/sessions", json={})
        sid = res.json()["session_id"]

        res = client.delete(f"/api/sessions/{sid}")
        assert res.status_code == 200

        # Verify it's gone
        assert not (_SESSIONS_ROOT / sid).exists()

    def test_delete_nonexistent(self):
        res = client.delete("/api/sessions/nonexistent_session_999")
        assert res.status_code == 404


class TestConversations:
    def test_empty_history(self):
        res = client.post("/api/sessions", json={})
        sid = res.json()["session_id"]
        _test_sessions.append(sid)

        res = client.get(f"/api/sessions/{sid}/conversations")
        assert res.status_code == 200
        assert res.json()["messages"] == []

    def test_append_and_load(self):
        res = client.post("/api/sessions", json={})
        sid = res.json()["session_id"]
        _test_sessions.append(sid)

        # Append
        res = client.post(f"/api/sessions/{sid}/conversations", json={
            "user_message": "Hello",
            "assistant_message": "Hi there!",
        })
        assert res.status_code == 200
        assert res.json()["total_messages"] == 2

        # Load
        res = client.get(f"/api/sessions/{sid}/conversations")
        msgs = res.json()["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "Hi there!"

    def test_clear_history(self):
        res = client.post("/api/sessions", json={})
        sid = res.json()["session_id"]
        _test_sessions.append(sid)

        # Add then clear
        client.post(f"/api/sessions/{sid}/conversations", json={
            "user_message": "test",
            "assistant_message": "test reply",
        })
        res = client.delete(f"/api/sessions/{sid}/conversations")
        assert res.status_code == 200

        # Verify empty
        res = client.get(f"/api/sessions/{sid}/conversations")
        assert res.json()["messages"] == []

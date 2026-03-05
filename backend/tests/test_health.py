"""
backend/tests/test_health.py

Tests for the /health endpoint and API key authentication middleware.
"""

import os
import pytest
from fastapi.testclient import TestClient

from backend.server import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_returns_ok(self):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"

    def test_contains_models_loaded(self):
        res = client.get("/health")
        data = res.json()
        assert "models_loaded" in data
        assert "e5_small_v2" in data["models_loaded"]
        assert "florence_2" in data["models_loaded"]
        assert "whisper_tiny" in data["models_loaded"]

    def test_contains_llm_info(self):
        res = client.get("/health")
        data = res.json()
        assert "llm_provider" in data
        assert "gemini_model" in data


class TestAPIKeyAuth:
    """Test that the auth middleware works correctly."""

    def test_no_key_required_by_default(self):
        """When RAG_API_KEY is not set, all endpoints should be accessible."""
        # Health is always public
        res = client.get("/health")
        assert res.status_code == 200

        # API endpoints should also work when no key is configured
        res = client.get("/api/sessions")
        assert res.status_code == 200

    def test_health_always_accessible(self):
        """Health endpoint should never require auth."""
        res = client.get("/health")
        assert res.status_code == 200

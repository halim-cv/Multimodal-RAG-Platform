"""
backend/middleware/auth.py

Simple API key authentication middleware.

If RAG_API_KEY is set in .env, all API requests (except /health and /docs)
must include the header:
    X-API-Key: <your_key>

If RAG_API_KEY is not set, authentication is disabled (open access).
"""

import os
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


# Paths that never require authentication
_PUBLIC_PATHS = {
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/",
}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that checks for a valid X-API-Key header on all API routes.
    Automatically disabled if RAG_API_KEY env var is empty/unset.
    """

    def __init__(self, app, api_key: str | None = None):
        super().__init__(app)
        self.api_key = api_key or os.getenv("RAG_API_KEY", "")

    async def dispatch(self, request: Request, call_next):
        # If no API key configured, allow everything (open access)
        if not self.api_key:
            return await call_next(request)

        # Allow public paths (health, docs, frontend static files)
        path = request.url.path
        if path in _PUBLIC_PATHS:
            return await call_next(request)

        # Allow static frontend files (CSS, JS, images)
        if not path.startswith("/api/"):
            return await call_next(request)

        # Check API key header
        provided_key = request.headers.get("X-API-Key", "")
        if provided_key != self.api_key:
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid or missing API key. Set X-API-Key header."},
            )

        return await call_next(request)

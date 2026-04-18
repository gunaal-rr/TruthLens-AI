"""
TruthLens AI — API Key Security Middleware

Validates Bearer token in the Authorization header against configured API key.
Skippable via configuration (API_KEY_ENABLED=false).
"""

from __future__ import annotations

import logging

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.config import get_settings

logger = logging.getLogger(__name__)

# Paths that never require authentication
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Middleware that validates API key via Bearer token.

    Configuration:
        API_KEY_ENABLED=true  → enforces auth
        API_KEY_ENABLED=false → bypasses auth (default for development)

    Expected header:
        Authorization: Bearer <API_KEY>

    Returns HTTP 401 if key is invalid or missing (when enabled).
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        settings = get_settings()

        # Skip if auth is disabled
        if not settings.api_key_enabled:
            return await call_next(request)

        # Skip for public paths
        path = request.url.path.rstrip("/")
        if path in _PUBLIC_PATHS:
            return await call_next(request)

        # Skip for OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Validate Authorization header
        auth_header = request.headers.get("authorization", "")

        if not auth_header:
            logger.warning(
                "Missing Authorization header from %s",
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Missing Authorization header. Expected: Bearer <API_KEY>",
                    "status_code": 401,
                },
            )

        # Parse Bearer token
        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            logger.warning("Malformed Authorization header")
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Malformed Authorization header. Expected format: Bearer <API_KEY>",
                    "status_code": 401,
                },
            )

        token = parts[1].strip()

        # Constant-time comparison to prevent timing attacks
        import hmac
        if not hmac.compare_digest(token, settings.api_key):
            logger.warning(
                "Invalid API key from %s",
                request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Invalid API key",
                    "status_code": 401,
                },
            )

        return await call_next(request)

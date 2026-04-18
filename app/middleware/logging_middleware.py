"""
TruthLens AI — Enhanced Structured Logging Middleware

Logs every request with structured JSON including:
    - Request text
    - Response label, trust_score, confidence
    - Latency (ms)
    - Timestamp
    - Client IP
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger("truthlens.access")


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that logs structured JSON for every request/response cycle.

    Captures:
        - HTTP method and path
        - Client IP
        - Request body (for POST /analyze)
        - Response classification label
        - Trust score and confidence
        - Processing latency
        - Timestamp (ISO 8601)
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.perf_counter()
        request_body: Optional[str] = None

        # Capture request body for analyze endpoint
        if request.method == "POST" and "/analyze" in str(request.url.path):
            try:
                body_bytes = await request.body()
                request_body = body_bytes.decode("utf-8", errors="replace")[:500]
            except Exception:
                request_body = "<unreadable>"

        # Process the request
        response: Response = await call_next(request)

        # Calculate latency
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Build structured log entry
        log_entry: Dict[str, Any] = {
            "event": "http_request",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": str(request.url.path),
            "status_code": response.status_code,
            "client_ip": self._get_client_ip(request),
            "latency_ms": latency_ms,
            "user_agent": request.headers.get("user-agent", "unknown"),
        }

        # Add request body for analyze requests
        if request_body:
            try:
                parsed = json.loads(request_body)
                log_entry["request_text"] = str(parsed.get("text", ""))[:200]
            except (json.JSONDecodeError, AttributeError):
                log_entry["request_text"] = request_body[:200]

        # Extract response metadata from headers (set by the endpoint)
        label = response.headers.get("X-TruthLens-Label")
        trust_score = response.headers.get("X-TruthLens-TrustScore")
        confidence = response.headers.get("X-TruthLens-Confidence")

        if label:
            log_entry["label"] = label
        if trust_score:
            log_entry["trust_score"] = int(trust_score)
        if confidence:
            log_entry["confidence"] = int(confidence)

        # Choose log level based on status
        if response.status_code >= 500:
            log_entry["event"] = "http_error"
            logger.error(json.dumps(log_entry))
        elif response.status_code >= 400:
            log_entry["event"] = "http_client_error"
            logger.warning(json.dumps(log_entry))
        else:
            if "/analyze" in str(request.url.path):
                log_entry["event"] = "analysis_completed"
            logger.info(json.dumps(log_entry))

        return response

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract client IP, considering proxy headers."""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        if request.client:
            return request.client.host
        return "unknown"

"""
TruthLens AI — Analyze Endpoint

POST /analyze — Accepts a claim/news text and returns structured JSON analysis.
Integrates caching, rate limiting, and strict schema validation.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.models import AnalyzeRequest, AnalyzeResponse, ErrorResponse
from app.pipeline.pipeline_manager import AnalysisPipeline
from app.utils.cache import InMemoryCache
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Analysis"])

# ── Singletons (injected at startup via app.state) ───────────────────────────

_pipeline = AnalysisPipeline()


def _get_cache(request: Request) -> InMemoryCache:
    """Retrieve the cache instance from app state."""
    return request.app.state.cache


def _get_rate_limiter(request: Request):
    """Retrieve the rate limiter instance from app state."""
    return request.app.state.rate_limiter


# ── Endpoint ─────────────────────────────────────────────────────────────────


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad input"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
    summary="Analyze a claim or news text",
    description=(
        "Submit text for fact-checking analysis. Returns a structured JSON response "
        "with classification label, trust score, confidence, bias detection, "
        "evidence artifacts, and explainable AI metadata."
    ),
)
async def analyze(request: Request, payload: AnalyzeRequest) -> JSONResponse:
    """
    Analyze a claim or news text for misinformation.

    Flow:
        1. Rate limit check
        2. Cache lookup
        3. Pipeline execution
        4. Schema validation
        5. Cache store
        6. Return response with metadata headers
    """
    # ── Rate Limiting ────────────────────────────────────────────
    rate_limiter = _get_rate_limiter(request)
    client_ip = _extract_client_ip(request)

    allowed = await rate_limiter.allow_request(client_ip)
    if not allowed:
        remaining = await rate_limiter.get_remaining(client_ip)
        settings = get_settings()
        logger.warning("Rate limit exceeded for IP: %s", client_ip)
        return JSONResponse(
            status_code=429,
            content={
                "detail": (
                    f"Rate limit exceeded. Maximum {settings.rate_limit_max_requests} "
                    f"requests per {settings.rate_limit_window_seconds} seconds."
                ),
                "status_code": 429,
            },
            headers={
                "Retry-After": str(settings.rate_limit_window_seconds),
                "X-RateLimit-Remaining": str(remaining),
            },
        )

    # ── Cache Lookup ─────────────────────────────────────────────
    cache = _get_cache(request)
    cached_result = await cache.get(payload.text)
    if cached_result is not None:
        logger.info("Cache hit for request from %s", client_ip)
        return JSONResponse(
            content=cached_result,
            headers={
                "X-TruthLens-Cached": "true",
                "X-TruthLens-Label": cached_result.get("label", ""),
                "X-TruthLens-TrustScore": str(cached_result.get("trust_score", "")),
                "X-TruthLens-Confidence": str(cached_result.get("confidence", "")),
            },
        )

    # ── Pipeline Execution ───────────────────────────────────────
    try:
        result: AnalyzeResponse = await _pipeline.run(
            text=payload.text,
            external_evidence=payload.external_evidence,
            news_results=payload.news_results,
            wiki_summary=payload.wiki_summary,
            ai_analysis=payload.ai_analysis,
        )
    except ValueError as exc:
        logger.error("Pipeline ValueError: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Pipeline internal error: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Internal analysis error. Please try again later.",
        )

    # ── Strict Schema Validation ─────────────────────────────────
    try:
        response_data = result.model_dump(mode="json")
        # Re-validate to ensure output compliance
        AnalyzeResponse(**response_data)
    except ValidationError as exc:
        logger.error("Response schema validation failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Internal error: response failed schema validation.",
        )

    # ── Cache Store ──────────────────────────────────────────────
    await cache.set(payload.text, response_data)

    # ── Return with Metadata Headers ─────────────────────────────
    remaining = await rate_limiter.get_remaining(client_ip)

    return JSONResponse(
        content=response_data,
        headers={
            "X-TruthLens-Cached": "false",
            "X-TruthLens-Label": result.label.value,
            "X-TruthLens-TrustScore": str(result.trust_score),
            "X-TruthLens-Confidence": str(result.confidence),
            "X-RateLimit-Remaining": str(remaining),
        },
    )


def _extract_client_ip(request: Request) -> str:
    """Extract client IP considering proxy headers."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip
    if request.client:
        return request.client.host
    return "0.0.0.0"

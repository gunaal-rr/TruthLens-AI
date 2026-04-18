"""
TruthLens AI — FastAPI Application Entrypoint

Production-grade application with:
    - Lifespan event handler for startup/shutdown
    - Rate limiter initialization (configurable backend)
    - Cache initialization
    - Middleware stack (security → logging → CORS)
    - Health check endpoint
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.middleware.logging_middleware import StructuredLoggingMiddleware
from app.middleware.rate_limiter.in_memory import InMemoryRateLimiter
from app.middleware.security import APIKeyMiddleware
from app.routes.analyze import router as analyze_router
from app.utils.cache import InMemoryCache


# ── Logging Configuration ────────────────────────────────────────────────────


def _configure_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Console handler with JSON-friendly formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    # Avoid duplicate handlers
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    # Quiet noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — startup and shutdown logic."""
    settings = get_settings()
    _configure_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # ── Initialize Rate Limiter ──────────────────────────────────
    if settings.rate_limiter_backend == "redis":
        # Import lazily to avoid import error when redis is not installed
        from app.middleware.rate_limiter.redis_limiter import RedisRateLimiter

        app.state.rate_limiter = RedisRateLimiter(
            redis_url=settings.redis_url,
            max_requests=settings.rate_limit_max_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        logger.info("Rate limiter: Redis (URL: %s)", settings.redis_url)
    else:
        app.state.rate_limiter = InMemoryRateLimiter(
            max_requests=settings.rate_limit_max_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        logger.info(
            "Rate limiter: In-Memory (%d req / %d sec)",
            settings.rate_limit_max_requests,
            settings.rate_limit_window_seconds,
        )

    # ── Initialize Cache ─────────────────────────────────────────
    app.state.cache = InMemoryCache(
        ttl_seconds=settings.cache_ttl_seconds,
        max_size=settings.cache_max_size,
    )
    logger.info(
        "Cache: In-Memory (TTL=%ds, max_size=%d)",
        settings.cache_ttl_seconds,
        settings.cache_max_size,
    )

    logger.info("API Key authentication: %s", "ENABLED" if settings.api_key_enabled else "DISABLED")
    logger.info("%s is ready to serve requests", settings.app_name)

    yield  # Application runs

    # ── Shutdown ─────────────────────────────────────────────────
    logger.info("Shutting down %s", settings.app_name)
    await app.state.cache.clear()
    logger.info("Cache cleared. Goodbye.")


# ── Application Factory ─────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Production-grade fake news detection and fact-checking REST API. "
            "Analyzes claims using multi-signal classification, bias detection, "
            "and explainable AI."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── Middleware Stack (order matters: last added = first executed) ─────
    # 1. CORS (outermost)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Structured Logging
    app.add_middleware(StructuredLoggingMiddleware)

    # 3. API Key Security
    app.add_middleware(APIKeyMiddleware)

    # ── Routes ───────────────────────────────────────────────────
    app.include_router(analyze_router, prefix="", tags=["Analysis"])

    # ── Static Files (Web UI) ────────────────────────────────────
    static_dir = Path(__file__).resolve().parent.parent / "static"
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")

    # ── Root Redirect → Web UI ────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect root to the TruthLens AI web interface."""
        return RedirectResponse(url="/static/index.html")

    # ── Health Check ─────────────────────────────────────────────
    @app.get(
        "/health",
        summary="Health check",
        tags=["System"],
        response_class=JSONResponse,
    )
    async def health() -> JSONResponse:
        """Returns application health status and cache/rate limiter stats."""
        cache_stats = app.state.cache.stats if hasattr(app.state, "cache") else {}
        rate_stats = (
            app.state.rate_limiter.stats
            if hasattr(app.state, "rate_limiter") and hasattr(app.state.rate_limiter, "stats")
            else {}
        )
        return JSONResponse(
            content={
                "status": "healthy",
                "version": settings.app_version,
                "cache": cache_stats,
                "rate_limiter": rate_stats,
            }
        )

    return app


# ── App Instance ─────────────────────────────────────────────────────────────

app = create_app()

"""
TruthLens AI — Application Configuration

Type-safe settings via pydantic-settings, driven by environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # ── Application ──────────────────────────────────────────────
    app_name: str = Field(default="TruthLens AI", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode toggle")

    # ── Rate Limiting ────────────────────────────────────────────
    rate_limit_max_requests: int = Field(
        default=10, ge=1, description="Max requests per window"
    )
    rate_limit_window_seconds: int = Field(
        default=60, ge=1, description="Rate limit window in seconds"
    )
    rate_limiter_backend: str = Field(
        default="memory",
        description="Rate limiter backend: 'memory' or 'redis'",
    )

    # ── Security ─────────────────────────────────────────────────
    api_key: str = Field(
        default="changeme-default-key", description="API key for authentication"
    )
    api_key_enabled: bool = Field(
        default=False, description="Enable API key authentication"
    )

    # ── Logging ──────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="Logging level")

    # ── Cache ────────────────────────────────────────────────────
    cache_ttl_seconds: int = Field(
        default=300, ge=0, description="Cache TTL in seconds (0 = no expiry)"
    )
    cache_max_size: int = Field(
        default=1000, ge=1, description="Maximum cache entries"
    )

    # ── Redis (future) ───────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379/0", description="Redis connection URL"
    )

    # ── Evidence Fetching ────────────────────────────────────────
    serpapi_key: str = Field(
        default="", description="SerpAPI key for Google Search evidence fetching"
    )
    evidence_cache_ttl: int = Field(
        default=600, ge=0, description="Evidence cache TTL in seconds"
    )
    evidence_fetch_timeout: int = Field(
        default=5, ge=1, le=30, description="HTTP timeout for evidence APIs (seconds)"
    )

    # ── Gemini LLM ───────────────────────────────────────────────
    gemini_api_key: str = Field(
        default="", description="Google Gemini API key for LLM-powered evidence analysis"
    )
    gemini_enabled: bool = Field(
        default=True, description="Enable Gemini LLM for evidence classification"
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton accessor for application settings."""
    return Settings()

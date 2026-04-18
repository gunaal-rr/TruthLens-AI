"""
TruthLens AI — Redis Rate Limiter (Placeholder)

Placeholder implementation for Redis-backed rate limiting.
Drop-in replacement for InMemoryRateLimiter when Redis is available.
"""

from __future__ import annotations

import logging

from app.middleware.rate_limiter.base import BaseRateLimiter

logger = logging.getLogger(__name__)


class RedisRateLimiter(BaseRateLimiter):
    """
    Redis-backed sliding window rate limiter.

    This is a placeholder for future implementation. When Redis is available,
    this class should use sorted sets with ZADD/ZRANGEBYSCORE/ZCARD for
    an efficient sliding window.

    Usage:
        Set RATE_LIMITER_BACKEND=redis in .env
        Set REDIS_URL=redis://localhost:6379/0

    TODO:
        - Connect to Redis using aioredis or redis-py[async]
        - Implement sliding window via sorted sets
        - Add connection pooling
        - Add retry/fallback logic
    """

    def __init__(self, redis_url: str, max_requests: int = 10, window_seconds: int = 60) -> None:
        self._redis_url = redis_url
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        logger.warning(
            "RedisRateLimiter is a placeholder. Using in-memory fallback behavior."
        )
        # TODO: Initialize async Redis connection pool here

    async def allow_request(self, key: str) -> bool:
        """Placeholder — always allows requests."""
        logger.warning("RedisRateLimiter.allow_request() is not implemented — allowing request")
        return True

    async def get_remaining(self, key: str) -> int:
        """Placeholder — returns max requests."""
        return self._max_requests

    async def reset(self, key: str) -> None:
        """Placeholder — no-op."""
        pass

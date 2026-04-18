"""
TruthLens AI — In-Memory Sliding Window Rate Limiter

Thread-safe rate limiter using asyncio.Lock for concurrency safety.
Implements sliding window algorithm with automatic cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from typing import Dict, List

from app.middleware.rate_limiter.base import BaseRateLimiter

logger = logging.getLogger(__name__)


class InMemoryRateLimiter(BaseRateLimiter):
    """
    In-memory sliding window rate limiter.

    Tracks request timestamps per key and enforces max_requests per window.
    Thread-safe via asyncio.Lock.

    Args:
        max_requests: Maximum number of requests per window.
        window_seconds: Time window in seconds.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._cleanup_counter = 0
        self._cleanup_interval = 100  # Clean up every N requests

    async def allow_request(self, key: str) -> bool:
        """Check if a request is allowed under the sliding window."""
        now = time.monotonic()
        cutoff = now - self._window_seconds

        async with self._lock:
            # Remove expired timestamps
            self._requests[key] = [
                ts for ts in self._requests[key] if ts > cutoff
            ]

            if len(self._requests[key]) >= self._max_requests:
                logger.warning(
                    "Rate limit exceeded for key: %s (%d/%d in %ds window)",
                    key, len(self._requests[key]),
                    self._max_requests, self._window_seconds,
                )
                return False

            self._requests[key].append(now)

            # Periodic cleanup of stale keys
            self._cleanup_counter += 1
            if self._cleanup_counter >= self._cleanup_interval:
                self._cleanup_counter = 0
                await self._cleanup_stale_keys(cutoff)

            return True

    async def get_remaining(self, key: str) -> int:
        """Get remaining requests for the key in the current window."""
        now = time.monotonic()
        cutoff = now - self._window_seconds

        async with self._lock:
            active = [ts for ts in self._requests.get(key, []) if ts > cutoff]
            return max(0, self._max_requests - len(active))

    async def reset(self, key: str) -> None:
        """Reset the rate limit for a specific key."""
        async with self._lock:
            self._requests.pop(key, None)
            logger.debug("Rate limit reset for key: %s", key)

    async def _cleanup_stale_keys(self, cutoff: float) -> None:
        """Remove keys with no active requests (already inside lock)."""
        stale_keys = [
            k for k, timestamps in self._requests.items()
            if not timestamps or all(ts <= cutoff for ts in timestamps)
        ]
        for key in stale_keys:
            del self._requests[key]

        if stale_keys:
            logger.debug("Cleaned up %d stale rate limit keys", len(stale_keys))

    @property
    def stats(self) -> dict:
        """Return rate limiter statistics."""
        return {
            "active_keys": len(self._requests),
            "max_requests": self._max_requests,
            "window_seconds": self._window_seconds,
        }

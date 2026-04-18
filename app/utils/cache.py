"""
TruthLens AI — In-Memory Caching Layer

Thread-safe, TTL-aware in-memory cache with LRU eviction.
Designed for easy future migration to Redis.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheEntry:
    """A single cache entry with timestamp for TTL tracking."""

    __slots__ = ("value", "created_at")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.created_at: float = time.monotonic()

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if this entry has expired based on TTL."""
        if ttl_seconds <= 0:
            return False  # TTL of 0 means no expiry
        return (time.monotonic() - self.created_at) > ttl_seconds


class InMemoryCache:
    """
    Thread-safe in-memory cache with TTL and LRU eviction.

    Features:
        - Key: SHA-256 hash of input text
        - TTL support (configurable, 0 = no expiry)
        - Max size with LRU eviction
        - Async-safe via asyncio.Lock
        - Drop-in replaceable with Redis implementation

    Usage:
        cache = InMemoryCache(ttl_seconds=300, max_size=1000)
        result = await cache.get("some text")
        if result is None:
            result = compute_result()
            await cache.set("some text", result)
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000) -> None:
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(text: str) -> str:
        """Generate a deterministic cache key from input text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def get(self, text: str) -> Optional[Any]:
        """
        Retrieve a cached result for the given text.

        Returns None on cache miss or expired entry.
        """
        key = self._make_key(text)
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.is_expired(self._ttl_seconds):
                del self._store[key]
                self._misses += 1
                logger.debug("Cache entry expired for key: %s…", key[:12])
                return None

            # Move to end (most recently used)
            self._store.move_to_end(key)
            self._hits += 1
            logger.debug("Cache hit for key: %s…", key[:12])
            return entry.value

    async def set(self, text: str, value: Any) -> None:
        """Store a result in the cache, evicting oldest if at capacity."""
        key = self._make_key(text)
        async with self._lock:
            # If key exists, update it
            if key in self._store:
                self._store[key] = CacheEntry(value)
                self._store.move_to_end(key)
                return

            # Evict oldest entries if at capacity
            while len(self._store) >= self._max_size:
                evicted_key, _ = self._store.popitem(last=False)
                logger.debug("Evicted cache entry: %s…", evicted_key[:12])

            self._store[key] = CacheEntry(value)

    async def clear(self) -> None:
        """Clear all cached entries."""
        async with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> dict:
        """Return cache performance statistics."""
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total > 0 else 0.0,
        }

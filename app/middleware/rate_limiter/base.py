"""
TruthLens AI — Rate Limiter Base Interface

Abstract interface for rate limiter backends.
Enables easy swapping between in-memory, Redis, or other implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRateLimiter(ABC):
    """
    Abstract rate limiter interface.

    All rate limiter backends must implement this interface to ensure
    the middleware can swap implementations via configuration.
    """

    @abstractmethod
    async def allow_request(self, key: str) -> bool:
        """
        Check if a request is allowed for the given key (e.g., client IP).

        Args:
            key: Unique identifier for the client (typically IP address).

        Returns:
            True if the request is allowed, False if rate limit is exceeded.
        """
        ...

    @abstractmethod
    async def get_remaining(self, key: str) -> int:
        """
        Get the number of remaining requests in the current window.

        Args:
            key: Unique identifier for the client.

        Returns:
            Number of remaining allowed requests.
        """
        ...

    @abstractmethod
    async def reset(self, key: str) -> None:
        """
        Reset the rate limit counter for a given key.

        Args:
            key: Unique identifier for the client.
        """
        ...

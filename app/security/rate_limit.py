"""Simple in-memory token-bucket rate limiter, applied per session to POST /query."""
from __future__ import annotations

import time


class TokenBucket:
    def __init__(self, capacity: int = 10, refill_rate_per_sec: float = 1.0) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate_per_sec
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()

    def try_consume(self, tokens: int = 1) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False


class RateLimiter:
    def __init__(self, capacity: int = 10, refill_rate_per_sec: float = 1.0) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate_per_sec
        self._buckets: dict[str, TokenBucket] = {}

    def allow(self, key: str) -> bool:
        bucket = self._buckets.setdefault(key, TokenBucket(self.capacity, self.refill_rate))
        return bucket.try_consume()


_singleton = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _singleton

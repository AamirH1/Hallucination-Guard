"""Selects Redis if reachable, else falls back to the in-memory store with a warning."""
from __future__ import annotations

import structlog

from app.config import Settings, get_settings
from app.state.base import StateStore
from app.state.memory_store import MemoryStateStore

logger = structlog.get_logger(__name__)

_singleton: StateStore | None = None


async def get_state_store(settings: Settings | None = None) -> StateStore:
    global _singleton
    if _singleton is not None:
        return _singleton

    settings = settings or get_settings()
    try:
        from app.state.redis_store import RedisStateStore

        candidate = RedisStateStore(settings.redis_url)
        if await candidate.ping():
            logger.info("state_store_ready", backend="redis")
            _singleton = candidate
            return _singleton
        raise ConnectionError("redis ping failed")
    except Exception as exc:
        logger.warning("redis_unreachable_falling_back_to_memory_store", error=str(exc))
        _singleton = MemoryStateStore()
        return _singleton


def reset_state_store_singleton() -> None:
    global _singleton
    _singleton = None

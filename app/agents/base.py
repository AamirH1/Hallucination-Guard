"""Shared agent base: consistent logging/tracing hook for every pipeline stage."""
from __future__ import annotations

import time
from typing import Any

import structlog


class BaseAgent:
    name: str = "base_agent"

    def __init__(self) -> None:
        self.logger = structlog.get_logger(self.name)

    async def _timed(self, coro_factory, **log_kwargs: Any) -> Any:
        start = time.perf_counter()
        result = await coro_factory()
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.logger.info("agent_step_complete", latency_ms=round(elapsed_ms, 2), **log_kwargs)
        return result

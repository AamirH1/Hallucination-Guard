"""Async context managers for pipeline + per-agent tracing spans, fanning out to all sinks."""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog

from app.config import Settings, get_settings
from app.observability.base import TraceSink
from app.observability.langfuse_sink import LangfuseTraceSink
from app.observability.local_sink import LocalTraceSink

logger = structlog.get_logger(__name__)

_sinks: list[TraceSink] | None = None


def get_sinks(settings: Settings | None = None) -> list[TraceSink]:
    global _sinks
    if _sinks is not None:
        return _sinks
    settings = settings or get_settings()
    sinks: list[TraceSink] = [LocalTraceSink(settings.trace_local_dir)]
    langfuse = LangfuseTraceSink(settings.langfuse_public_key, settings.langfuse_secret_key, settings.langfuse_host)
    if langfuse.enabled:
        sinks.append(langfuse)
    _sinks = sinks
    return sinks


class PipelineTrace:
    """Wraps the whole request pipeline; `.span()` wraps individual agent steps."""

    def __init__(self, request_id: str, session_id: str, settings: Settings | None = None) -> None:
        self.request_id = request_id
        self.session_id = session_id
        self.sinks = get_sinks(settings)
        self._start = time.perf_counter()
        self.summary: dict = {}

    async def __aenter__(self) -> "PipelineTrace":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        latency_ms = (time.perf_counter() - self._start) * 1000
        span = {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "agent": "pipeline",
            "latency_ms": round(latency_ms, 2),
            "error": str(exc) if exc else None,
            **self.summary,
        }
        for sink in self.sinks:
            await sink.write_span(span)

    @asynccontextmanager
    async def span(self, agent_name: str, **input_summary: object):
        start = time.perf_counter()
        span_id = str(uuid.uuid4())[:8]
        output: dict = {}
        try:
            yield output
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            span = {
                "request_id": self.request_id,
                "session_id": self.session_id,
                "span_id": span_id,
                "agent": agent_name,
                "latency_ms": round(latency_ms, 2),
                "input_summary": _truncate_dict(input_summary),
                "output_summary": _truncate_dict(output),
            }
            for sink in self.sinks:
                await sink.write_span(span)


def _truncate_dict(d: dict, limit: int = 300) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > limit:
            out[k] = v[:limit] + "...[truncated]"
        else:
            out[k] = v
    return out

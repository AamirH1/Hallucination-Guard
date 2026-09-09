"""Langfuse trace sink. Inert (no-op) unless LANGFUSE_PUBLIC_KEY/SECRET_KEY are set."""
from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


class LangfuseTraceSink:
    name = "langfuse"

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        self.enabled = bool(public_key and secret_key)
        self._client = None
        if self.enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
            except Exception as exc:
                logger.warning("langfuse_init_failed_disabling_sink", error=str(exc))
                self.enabled = False

    async def write_span(self, span: dict) -> None:
        if not self.enabled or self._client is None:
            return
        try:
            self._client.trace(
                name=span.get("agent", "pipeline"),
                metadata=span,
            )
        except Exception as exc:  # never let observability break the pipeline
            logger.warning("langfuse_write_failed", error=str(exc))

"""Protocol for trace sinks (local JSONL, Langfuse, ...)."""
from __future__ import annotations

from typing import Protocol


class TraceSink(Protocol):
    name: str

    async def write_span(self, span: dict) -> None: ...

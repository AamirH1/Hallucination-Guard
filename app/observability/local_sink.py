"""Appends one JSON line per span to TRACE_LOCAL_DIR. Always active (zero-dependency fallback)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path


def _truncate(value: object, limit: int = 500) -> object:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "...[truncated]"
    return value


class LocalTraceSink:
    name = "local"

    def __init__(self, directory: str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "spans.jsonl"

    async def write_span(self, span: dict) -> None:
        truncated = {k: _truncate(v) for k, v in span.items()}
        line = json.dumps(truncated, default=str) + "\n"
        await asyncio.to_thread(self._append_sync, line)

    def _append_sync(self, line: str) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

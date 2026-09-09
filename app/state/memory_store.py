"""In-process fallback StateStore, used automatically when Redis is unreachable."""
from __future__ import annotations

import time


class MemoryStateStore:
    name = "memory"

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], tuple[object, float | None]] = {}
        self._logs: dict[str, list[dict]] = {}

    async def get(self, session_id: str, key: str) -> object | None:
        entry = self._data.get((session_id, key))
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and time.monotonic() > expires_at:
            del self._data[(session_id, key)]
            return None
        return value

    async def set(self, session_id: str, key: str, value: object, ttl_seconds: int | None = None) -> None:
        expires_at = time.monotonic() + ttl_seconds if ttl_seconds else None
        self._data[(session_id, key)] = (value, expires_at)

    async def delete(self, session_id: str, key: str) -> None:
        self._data.pop((session_id, key), None)

    async def append_log(self, log_name: str, entry: dict) -> None:
        self._logs.setdefault(log_name, []).append(entry)

    async def read_log(self, log_name: str, limit: int = 1000) -> list[dict]:
        return self._logs.get(log_name, [])[-limit:]

    async def ping(self) -> bool:
        return True

    def session_keys(self, session_id: str) -> list[str]:
        """Test helper: list keys visible to a session (for isolation tests)."""
        return [k for (sid, k) in self._data if sid == session_id]

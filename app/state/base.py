"""Protocol for session-scoped key/value state storage (Redis or in-memory)."""
from __future__ import annotations

from typing import Protocol


class StateStore(Protocol):
    name: str

    async def get(self, session_id: str, key: str) -> object | None: ...

    async def set(self, session_id: str, key: str, value: object, ttl_seconds: int | None = None) -> None: ...

    async def delete(self, session_id: str, key: str) -> None: ...

    async def append_log(self, log_name: str, entry: dict) -> None:
        """Append-only log used for audit trails / feedback accumulation."""
        ...

    async def read_log(self, log_name: str, limit: int = 1000) -> list[dict]: ...

    async def ping(self) -> bool: ...

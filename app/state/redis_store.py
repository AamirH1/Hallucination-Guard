"""Redis-backed StateStore using redis.asyncio. Session-scoped keys, append-only lists for logs."""
from __future__ import annotations

import json

import redis.asyncio as redis


def _key(session_id: str, key: str) -> str:
    return f"hgstate:{session_id}:{key}"


class RedisStateStore:
    name = "redis"

    def __init__(self, url: str) -> None:
        self._client = redis.from_url(url, decode_responses=True)

    async def get(self, session_id: str, key: str) -> object | None:
        raw = await self._client.get(_key(session_id, key))
        return json.loads(raw) if raw is not None else None

    async def set(self, session_id: str, key: str, value: object, ttl_seconds: int | None = None) -> None:
        raw = json.dumps(value)
        if ttl_seconds:
            await self._client.set(_key(session_id, key), raw, ex=ttl_seconds)
        else:
            await self._client.set(_key(session_id, key), raw)

    async def delete(self, session_id: str, key: str) -> None:
        await self._client.delete(_key(session_id, key))

    async def append_log(self, log_name: str, entry: dict) -> None:
        await self._client.rpush(f"hglog:{log_name}", json.dumps(entry))

    async def read_log(self, log_name: str, limit: int = 1000) -> list[dict]:
        raw_entries = await self._client.lrange(f"hglog:{log_name}", -limit, -1)
        return [json.loads(r) for r in raw_entries]

    async def ping(self) -> bool:
        try:
            return await self._client.ping()
        except Exception:
            return False

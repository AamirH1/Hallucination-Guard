"""Protocol every vector store adapter must implement."""
from __future__ import annotations

from typing import Protocol, TypedDict


class VectorHit(TypedDict):
    doc_id: str
    text: str
    metadata: dict[str, object]
    score: float


class VectorStore(Protocol):
    name: str

    async def upsert(self, doc_id: str, text: str, embedding: list[float], metadata: dict[str, object]) -> None: ...

    async def query(
        self, embedding: list[float], top_k: int, filters: dict[str, object] | None = None
    ) -> list[VectorHit]: ...

    async def delete(self, doc_id: str) -> None: ...

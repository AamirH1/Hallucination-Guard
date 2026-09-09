"""Elasticsearch (dense_vector) store adapter.

Implemented against the real `elasticsearch` async client for correctness, but not
exercised in this sandbox (no ES cluster available). See solution.md.
"""
from __future__ import annotations

from app.retrieval.vector_store_base import VectorHit


class ElasticsearchVectorStore:
    name = "elasticsearch"

    def __init__(self, url: str, index: str = "documents", dims: int = 384) -> None:
        from elasticsearch import AsyncElasticsearch

        self._index = index
        self._dims = dims
        self._client = AsyncElasticsearch(hosts=[url])

    async def _ensure_index(self) -> None:
        exists = await self._client.indices.exists(index=self._index)
        if not exists:
            await self._client.indices.create(
                index=self._index,
                mappings={
                    "properties": {
                        "text": {"type": "text"},
                        "embedding": {"type": "dense_vector", "dims": self._dims, "similarity": "cosine"},
                    }
                },
            )

    async def upsert(self, doc_id: str, text: str, embedding: list[float], metadata: dict[str, object]) -> None:
        await self._ensure_index()
        await self._client.index(index=self._index, id=doc_id, document={"text": text, "embedding": embedding, **metadata})

    async def query(
        self, embedding: list[float], top_k: int, filters: dict[str, object] | None = None
    ) -> list[VectorHit]:
        await self._ensure_index()
        knn = {"field": "embedding", "query_vector": embedding, "k": top_k, "num_candidates": max(top_k * 10, 50)}
        resp = await self._client.search(index=self._index, knn=knn, size=top_k)
        hits: list[VectorHit] = []
        for hit in resp["hits"]["hits"]:
            source = dict(hit["_source"])
            text = source.pop("text", "")
            source.pop("embedding", None)
            hits.append(VectorHit(doc_id=hit["_id"], text=text, metadata=source, score=hit["_score"]))
        return hits

    async def delete(self, doc_id: str) -> None:
        await self._client.delete(index=self._index, id=doc_id, ignore=[404])

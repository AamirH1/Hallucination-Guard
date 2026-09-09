"""Weaviate vector store adapter.

Implemented against the real `weaviate-client` v4 API for correctness, but not
exercised in this sandbox (no Weaviate instance available). See solution.md.
"""
from __future__ import annotations

from app.retrieval.vector_store_base import VectorHit


class WeaviateVectorStore:
    name = "weaviate"

    def __init__(self, url: str, class_name: str = "Document") -> None:
        import weaviate  # noqa: F401 - imported lazily so it's only required if selected

        self._url = url
        self._class_name = class_name
        self._client = weaviate.connect_to_custom(
            http_host=url.split("://")[-1].split(":")[0],
            http_port=int(url.split(":")[-1]) if ":" in url.split("://")[-1] else 8080,
            http_secure=url.startswith("https"),
            grpc_host=url.split("://")[-1].split(":")[0],
            grpc_port=50051,
            grpc_secure=False,
        )
        if not self._client.collections.exists(class_name):
            self._client.collections.create(name=class_name)
        self._collection = self._client.collections.get(class_name)

    async def upsert(self, doc_id: str, text: str, embedding: list[float], metadata: dict[str, object]) -> None:
        self._collection.data.insert(
            properties={"text": text, **metadata}, vector=embedding, uuid=doc_id
        )

    async def query(
        self, embedding: list[float], top_k: int, filters: dict[str, object] | None = None
    ) -> list[VectorHit]:
        response = self._collection.query.near_vector(near_vector=embedding, limit=top_k)
        hits: list[VectorHit] = []
        for obj in response.objects:
            props = dict(obj.properties)
            text = props.pop("text", "")
            score = 1.0 - (obj.metadata.distance or 0.0) if obj.metadata else 0.0
            hits.append(VectorHit(doc_id=str(obj.uuid), text=text, metadata=props, score=score))
        return hits

    async def delete(self, doc_id: str) -> None:
        self._collection.data.delete_by_id(doc_id)

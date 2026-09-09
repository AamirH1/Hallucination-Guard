"""Default, embedded vector store backed by ChromaDB, persisted to disk."""
from __future__ import annotations

from app.retrieval.vector_store_base import VectorHit


class ChromaVectorStore:
    name = "chroma"

    def __init__(self, persist_dir: str, collection_name: str = "documents") -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(collection_name)

    async def upsert(self, doc_id: str, text: str, embedding: list[float], metadata: dict[str, object]) -> None:
        safe_metadata = {k: v for k, v in metadata.items() if v is not None}
        self._collection.upsert(ids=[doc_id], embeddings=[embedding], documents=[text], metadatas=[safe_metadata])

    async def query(
        self, embedding: list[float], top_k: int, filters: dict[str, object] | None = None
    ) -> list[VectorHit]:
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=filters or None,
        )
        hits: list[VectorHit] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
            # Chroma returns L2/cosine distance; convert to a similarity-like score in [0,1].
            score = 1.0 / (1.0 + dist)
            hits.append(VectorHit(doc_id=doc_id, text=text, metadata=meta or {}, score=score))
        return hits

    async def delete(self, doc_id: str) -> None:
        self._collection.delete(ids=[doc_id])

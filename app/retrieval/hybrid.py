"""Hybrid retrieval: vector search + BM25 keyword search fused via Reciprocal Rank Fusion."""
from __future__ import annotations

from pydantic import BaseModel

from app.providers.base import EmbeddingProvider
from app.retrieval.bm25 import BM25Index
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store_base import VectorStore

RRF_K = 60


class RetrievedDoc(BaseModel):
    doc_id: str
    text: str
    metadata: dict[str, object] = {}
    fused_score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_provider = embedding_provider
        self.reranker = reranker

    async def retrieve(
        self, query: str, top_k: int = 5, filters: dict[str, object] | None = None
    ) -> list[RetrievedDoc]:
        [query_embedding] = await self.embedding_provider.embed([query])
        candidate_k = max(top_k * 4, 20)

        vector_hits = await self.vector_store.query(query_embedding, candidate_k, filters)
        bm25_hits = self.bm25_index.query(query, candidate_k)

        rrf_scores: dict[str, float] = {}
        text_by_id: dict[str, str] = {}
        meta_by_id: dict[str, dict[str, object]] = {}
        vector_rank_by_id: dict[str, int] = {}
        bm25_rank_by_id: dict[str, int] = {}

        for rank, hit in enumerate(vector_hits, start=1):
            rrf_scores[hit["doc_id"]] = rrf_scores.get(hit["doc_id"], 0.0) + 1.0 / (RRF_K + rank)
            text_by_id[hit["doc_id"]] = hit["text"]
            meta_by_id[hit["doc_id"]] = hit["metadata"]
            vector_rank_by_id[hit["doc_id"]] = rank

        for rank, (doc_id, text, metadata, _score) in enumerate(bm25_hits, start=1):
            if filters and not _matches_filters(metadata, filters):
                continue
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
            text_by_id.setdefault(doc_id, text)
            meta_by_id.setdefault(doc_id, metadata)
            bm25_rank_by_id[doc_id] = rank

        ranked_ids = sorted(rrf_scores, key=lambda d: rrf_scores[d], reverse=True)
        fused = [
            RetrievedDoc(
                doc_id=doc_id,
                text=text_by_id[doc_id],
                metadata=meta_by_id.get(doc_id, {}),
                fused_score=rrf_scores[doc_id],
                vector_rank=vector_rank_by_id.get(doc_id),
                bm25_rank=bm25_rank_by_id.get(doc_id),
            )
            for doc_id in ranked_ids
        ]

        if self.reranker is not None:
            fused = await self.reranker.rerank(query, fused)

        return fused[:top_k]


def _matches_filters(metadata: dict[str, object], filters: dict[str, object]) -> bool:
    return all(metadata.get(k) == v for k, v in filters.items())

"""Builds/holds the process-wide HybridRetriever + BM25Index singletons."""
from __future__ import annotations

from app.config import Settings, get_settings
from app.providers.factory import get_embedding_provider
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store_factory import get_vector_store

_bm25_singleton = BM25Index()
_retriever_singleton: HybridRetriever | None = None


def get_bm25_index() -> BM25Index:
    return _bm25_singleton


def get_hybrid_retriever(settings: Settings | None = None) -> HybridRetriever:
    global _retriever_singleton
    if _retriever_singleton is not None:
        return _retriever_singleton
    settings = settings or get_settings()
    _retriever_singleton = HybridRetriever(
        vector_store=get_vector_store(settings),
        bm25_index=_bm25_singleton,
        embedding_provider=get_embedding_provider(settings),
        reranker=Reranker(enabled=settings.reranker_enabled),
    )
    return _retriever_singleton

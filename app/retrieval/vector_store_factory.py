"""Selects the configured vector store implementation."""
from __future__ import annotations

import structlog

from app.config import Settings, get_settings
from app.retrieval.vector_store_base import VectorStore

logger = structlog.get_logger(__name__)

_singleton: VectorStore | None = None


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    global _singleton
    if _singleton is not None:
        return _singleton

    settings = settings or get_settings()
    store = settings.vector_store.lower()

    if store == "chroma":
        from app.retrieval.chroma_store import ChromaVectorStore

        _singleton = ChromaVectorStore(settings.chroma_persist_dir)
    elif store == "weaviate":
        from app.retrieval.weaviate_store import WeaviateVectorStore

        _singleton = WeaviateVectorStore(settings.weaviate_url)
    elif store == "elasticsearch":
        from app.retrieval.elasticsearch_store import ElasticsearchVectorStore

        _singleton = ElasticsearchVectorStore(settings.elasticsearch_url)
    elif store == "azure":
        from app.retrieval.azure_search_store import AzureSearchVectorStore

        _singleton = AzureSearchVectorStore(settings.azure_search_endpoint, settings.azure_search_key)
    else:
        logger.warning("unknown_vector_store_falling_back_to_chroma", store=store)
        from app.retrieval.chroma_store import ChromaVectorStore

        _singleton = ChromaVectorStore(settings.chroma_persist_dir)
    return _singleton


def reset_vector_store_singleton() -> None:
    """Test helper: force re-instantiation on next get_vector_store() call."""
    global _singleton
    _singleton = None

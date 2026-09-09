"""Loads the seed corpus from data/sample_docs and indexes it into the vector store + BM25.

Docs are short enough (one policy/topic each) that we index one chunk per document;
no sliding-window chunking is needed for a 15-25 document demo corpus.
"""
from __future__ import annotations

import re
from pathlib import Path

import structlog

from app.providers.base import EmbeddingProvider
from app.retrieval.bm25 import BM25Index
from app.retrieval.vector_store_base import VectorStore

logger = structlog.get_logger(__name__)

# Simple in-process registry so MCP tools (get_document, search_policies) can fetch
# full raw text/metadata by doc_id without re-hitting the vector store.
DOCUMENT_REGISTRY: dict[str, tuple[str, dict[str, object]]] = {}


def _parse_doc(path: Path) -> tuple[str, str, dict[str, object]]:
    text = path.read_text(encoding="utf-8")
    doc_id = path.stem
    title_match = re.search(r"^#\s*(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else doc_id
    return doc_id, text, {"title": title, "source_path": str(path)}


def load_corpus(corpus_dir: str) -> list[tuple[str, str, dict[str, object]]]:
    directory = Path(corpus_dir)
    docs = []
    for path in sorted(directory.glob("*.md")):
        docs.append(_parse_doc(path))
    for path in sorted(directory.glob("*.txt")):
        docs.append(_parse_doc(path))
    return docs


async def index_corpus(
    corpus_dir: str,
    vector_store: VectorStore,
    bm25_index: BM25Index,
    embedding_provider: EmbeddingProvider,
) -> int:
    docs = load_corpus(corpus_dir)
    if not docs:
        logger.warning("empty_corpus", corpus_dir=corpus_dir)
        return 0

    texts = [d[1] for d in docs]
    embeddings = await embedding_provider.embed(texts)
    for (doc_id, text, metadata), embedding in zip(docs, embeddings):
        await vector_store.upsert(doc_id, text, embedding, metadata)
        DOCUMENT_REGISTRY[doc_id] = (text, metadata)
    bm25_index.build(docs)
    logger.info("corpus_indexed", count=len(docs))
    return len(docs)

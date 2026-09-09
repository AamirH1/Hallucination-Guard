import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.providers.factory import get_embedding_provider
from app.retrieval.corpus import index_corpus
from app.retrieval.retrieval_factory import get_bm25_index
from app.retrieval.vector_store_factory import get_vector_store

CORPUS_DIR = str(Path(__file__).resolve().parents[1] / "data" / "sample_docs")

_indexed = False


@pytest.fixture(scope="session", autouse=True)
def _index_corpus_once():
    global _indexed
    if not _indexed:
        asyncio.run(index_corpus(CORPUS_DIR, get_vector_store(), get_bm25_index(), get_embedding_provider()))
        _indexed = True
    yield

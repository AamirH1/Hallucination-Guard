"""Indexes data/sample_docs into the configured vector store + BM25 index.
Run once before starting the API server manually (the API also does this at
startup), or after editing the corpus.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.providers.factory import get_embedding_provider
from app.retrieval.corpus import index_corpus
from app.retrieval.retrieval_factory import get_bm25_index
from app.retrieval.vector_store_factory import get_vector_store


async def main() -> None:
    settings = get_settings()
    count = await index_corpus(
        "data/sample_docs", get_vector_store(settings), get_bm25_index(), get_embedding_provider(settings)
    )
    print(f"Indexed {count} documents from data/sample_docs")


if __name__ == "__main__":
    asyncio.run(main())

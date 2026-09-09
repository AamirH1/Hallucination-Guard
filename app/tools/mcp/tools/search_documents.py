"""MCP tool: hybrid (vector + BM25 + RRF) document search over the seeded corpus."""
from __future__ import annotations

from app.retrieval.retrieval_factory import get_hybrid_retriever
from app.tools.mcp.schemas import SearchDocumentsInput


async def search_documents(payload: SearchDocumentsInput) -> dict:
    retriever = get_hybrid_retriever()
    results = await retriever.retrieve(payload.query, top_k=payload.top_k, filters=payload.filters)
    return {
        "results": [
            {
                "doc_id": r.doc_id,
                "text": r.text,
                "metadata": r.metadata,
                "score": r.fused_score,
            }
            for r in results
        ]
    }

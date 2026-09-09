"""MCP tool: topic-scoped search over policy documents (a semantic view over search_documents)."""
from __future__ import annotations

from app.retrieval.retrieval_factory import get_hybrid_retriever
from app.tools.mcp.schemas import SearchPoliciesInput


async def search_policies(payload: SearchPoliciesInput) -> dict:
    retriever = get_hybrid_retriever()
    results = await retriever.retrieve(payload.topic, top_k=payload.top_k * 3)
    policy_results = [r for r in results if "polic" in r.doc_id.lower() or "sla" in r.doc_id.lower()]
    if not policy_results:
        policy_results = results
    policy_results = policy_results[: payload.top_k]
    return {
        "results": [
            {"doc_id": r.doc_id, "text": r.text, "metadata": r.metadata, "score": r.fused_score}
            for r in policy_results
        ]
    }

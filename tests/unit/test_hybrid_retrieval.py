import pytest

from app.retrieval.retrieval_factory import get_hybrid_retriever


@pytest.mark.asyncio
async def test_hybrid_retrieve_returns_top_k_sorted_by_fused_score():
    retriever = get_hybrid_retriever()
    results = await retriever.retrieve("What uptime SLA does Business tier get?", top_k=3)
    assert len(results) <= 3
    scores = [r.fused_score for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_hybrid_retrieve_applies_metadata_filters():
    retriever = get_hybrid_retriever()
    results = await retriever.retrieve("refund", top_k=5, filters={"title": "Refund Policy"})
    for r in results:
        assert r.metadata.get("title") == "Refund Policy"

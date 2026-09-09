import pytest

from app.agents.retrieval import RetrievalAgent
from app.state.memory_store import MemoryStateStore


@pytest.mark.asyncio
async def test_retrieval_returns_ranked_deduped_results():
    agent = RetrievalAgent(MemoryStateStore())
    result = await agent.run("session-a", "What is the refund window for Starter customers?", top_k=5)
    assert len(result.items) > 0
    assert len(result.items) == len(set(i.doc_id for i in result.items))  # deduped
    assert 0.0 <= result.retrieval_confidence <= 1.0
    # top result should plausibly be a refund-related doc
    assert any("refund" in i.doc_id for i in result.items)


@pytest.mark.asyncio
async def test_retrieval_low_confidence_for_irrelevant_query():
    agent = RetrievalAgent(MemoryStateStore())
    result = await agent.run("session-b", "What is the weather forecast for Tokyo tomorrow?")
    assert result.retrieval_confidence < 0.7

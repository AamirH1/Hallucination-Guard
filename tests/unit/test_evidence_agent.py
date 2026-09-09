import pytest

from app.agents.evidence import EvidenceAggregationAgent
from app.agents.retrieval import RetrievalAgent
from app.state.memory_store import MemoryStateStore


@pytest.mark.asyncio
async def test_evidence_package_has_provenance_and_confidence():
    retrieval_agent = RetrievalAgent(MemoryStateStore())
    retrieval = await retrieval_agent.run("s1", "What is the refund window for Starter customers?")
    evidence_agent = EvidenceAggregationAgent(score_threshold=0.3)
    package = await evidence_agent.run(retrieval)
    assert len(package.evidence) > 0
    for item in package.evidence:
        assert item.source_id  # traceable back to a doc_id
        assert item.supporting_text
        assert 0.0 <= item.confidence <= 1.0
    assert 0.0 <= package.overall_confidence <= 1.0


@pytest.mark.asyncio
async def test_conflict_detection_on_refund_terms():
    retrieval_agent = RetrievalAgent(MemoryStateStore())
    retrieval = await retrieval_agent.run("s1", "What is the refund window across all Nimbus plans?", top_k=10)
    evidence_agent = EvidenceAggregationAgent(score_threshold=0.3)
    package = await evidence_agent.run(retrieval)
    assert package.conflicts_detected is True
    assert len(package.conflict_details) > 0


@pytest.mark.asyncio
async def test_empty_retrieval_yields_empty_evidence():
    from app.agents.retrieval import RetrievalResult

    evidence_agent = EvidenceAggregationAgent()
    package = await evidence_agent.run(RetrievalResult(query="x", items=[], retrieval_confidence=0.0))
    assert package.evidence == []
    assert package.overall_confidence == 0.0

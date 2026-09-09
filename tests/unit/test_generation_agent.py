import pytest

from app.agents.evidence import EvidenceItem, EvidencePackage
from app.agents.generation import AnswerGenerationAgent
from app.providers.mock_provider import MockLLMProvider


def _package(confidence=0.9, evidence=None):
    evidence = evidence or [
        EvidenceItem(source_id="refund_policy", claim="refund_policy", supporting_text="Refunds within 30 days.", confidence=0.9)
    ]
    return EvidencePackage(query="q", evidence=evidence, conflicts_detected=False, overall_confidence=confidence)


@pytest.mark.asyncio
async def test_generates_cited_answer_when_evidence_sufficient():
    agent = AnswerGenerationAgent(MockLLMProvider(), retrieval_threshold=0.7)
    result = await agent.run("What is the refund window?", _package())
    assert result.insufficient_evidence is False
    assert "refund_policy" in result.citations
    assert "[refund_policy]" in result.answer


@pytest.mark.asyncio
async def test_insufficient_evidence_path_when_confidence_low():
    agent = AnswerGenerationAgent(MockLLMProvider(), retrieval_threshold=0.7)
    result = await agent.run("What is the refund window?", _package(confidence=0.2))
    assert result.insufficient_evidence is True
    assert result.citations == []


@pytest.mark.asyncio
async def test_revision_mode_excludes_invalidated_claims():
    agent = AnswerGenerationAgent(MockLLMProvider(), retrieval_threshold=0.7)
    result = await agent.run("What is the refund window?", _package(), excluded_claims=["refund_policy"])
    assert result.insufficient_evidence is True  # nothing left to cite

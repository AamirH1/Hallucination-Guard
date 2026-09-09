import pytest

from app.agents.evidence import EvidenceItem, EvidencePackage
from app.agents.grounding import GroundingAgent
from app.providers.factory import get_embedding_provider


@pytest.mark.asyncio
async def test_supported_claim_scores_high_similarity():
    agent = GroundingAgent(get_embedding_provider(), support_threshold=0.5)
    evidence = EvidencePackage(
        query="q",
        evidence=[
            EvidenceItem(
                source_id="refund_policy",
                claim="refund_policy",
                supporting_text="Nimbus offers a 30-day money-back guarantee on all paid subscription plans.",
                confidence=0.9,
            )
        ],
        conflicts_detected=False,
        overall_confidence=0.9,
    )
    result = await agent.run("Nimbus offers a 30-day money-back guarantee on paid plans [refund_policy].", evidence)
    assert result.hallucination_detected is False
    assert result.grounding_score > 0.5
    assert all(c.supported for c in result.claims)


@pytest.mark.asyncio
async def test_unrelated_claim_flagged_as_unsupported():
    agent = GroundingAgent(get_embedding_provider(), support_threshold=0.6)
    evidence = EvidencePackage(
        query="q",
        evidence=[
            EvidenceItem(
                source_id="refund_policy",
                claim="refund_policy",
                supporting_text="Nimbus offers a 30-day money-back guarantee on all paid subscription plans.",
                confidence=0.9,
            )
        ],
        conflicts_detected=False,
        overall_confidence=0.9,
    )
    result = await agent.run("The company headquarters relocated to the moon in 1990.", evidence)
    assert result.hallucination_detected is True


@pytest.mark.asyncio
async def test_no_evidence_forces_hallucination_flag():
    agent = GroundingAgent(get_embedding_provider())
    evidence = EvidencePackage(query="q", evidence=[], conflicts_detected=False, overall_confidence=0.0)
    result = await agent.run("Some answer.", evidence)
    assert result.hallucination_detected is True
    assert result.grounding_score == 0.0

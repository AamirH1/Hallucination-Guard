"""Agent 6: Response Governance — pure decision logic, no LLM call."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class GovernanceDecision(str, Enum):
    APPROVE = "APPROVE"
    REVISE = "REVISE"
    REFUSE = "REFUSE"


class GovernanceOutcome(BaseModel):
    decision: GovernanceDecision
    reason: str
    unsupported_claims: list[str] = []


class GovernanceAgent:
    name = "governance"

    def __init__(
        self,
        grounding_threshold: float = 0.85,
        retrieval_threshold: float = 0.70,
        max_regeneration_attempts: int = 2,
    ) -> None:
        self.grounding_threshold = grounding_threshold
        self.retrieval_threshold = retrieval_threshold
        self.max_regeneration_attempts = max_regeneration_attempts

    def decide(
        self,
        grounding_score: float,
        retrieval_confidence: float,
        regeneration_attempts: int,
        insufficient_evidence: bool,
        unsupported_claims: list[str],
    ) -> GovernanceOutcome:
        if insufficient_evidence:
            return GovernanceOutcome(decision=GovernanceDecision.REFUSE, reason="insufficient_evidence")

        if retrieval_confidence < self.retrieval_threshold:
            return GovernanceOutcome(
                decision=GovernanceDecision.REFUSE, reason="retrieval_confidence_below_threshold"
            )

        if grounding_score >= self.grounding_threshold:
            return GovernanceOutcome(decision=GovernanceDecision.APPROVE, reason="grounding_score_meets_threshold")

        if regeneration_attempts >= self.max_regeneration_attempts:
            return GovernanceOutcome(
                decision=GovernanceDecision.REFUSE,
                reason="max_regeneration_attempts_exceeded",
                unsupported_claims=unsupported_claims,
            )

        return GovernanceOutcome(
            decision=GovernanceDecision.REVISE,
            reason="grounding_score_below_threshold",
            unsupported_claims=unsupported_claims,
        )

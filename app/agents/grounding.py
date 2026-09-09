"""Agent 5: Truth Alignment/Grounding — claim-level entailment scoring against evidence.

Default method is an embedding-similarity proxy: split the answer into sentence-level
claims, embed each claim and each evidence passage, and take the max cosine similarity
across evidence passages as the claim's "support score". This is a deliberate,
documented simplification — it is NOT a trained NLI/entailment model, so it can be
fooled by claims that are topically similar but factually reversed (e.g. negation).
An optional LLM-judge mode is provided for when a real provider is configured.
"""
from __future__ import annotations

import re

import numpy as np
from pydantic import BaseModel

from app.agents.base import BaseAgent
from app.agents.evidence import EvidencePackage
from app.providers.base import EmbeddingProvider, LLMProvider


class ClaimScore(BaseModel):
    claim: str
    max_similarity: float
    supported: bool
    best_source_id: str | None = None


class GroundingResult(BaseModel):
    claims: list[ClaimScore]
    grounding_score: float
    hallucination_detected: bool


def _split_claims(answer: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", answer.replace("\n", " "))
    return [s.strip() for s in sentences if s.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


class GroundingAgent(BaseAgent):
    name = "grounding"

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        support_threshold: float = 0.55,
        llm_judge: LLMProvider | None = None,
    ) -> None:
        super().__init__()
        self.embedding_provider = embedding_provider
        self.support_threshold = support_threshold
        self.llm_judge = llm_judge  # optional, unused unless explicitly wired

    async def run(self, answer: str, evidence: EvidencePackage) -> GroundingResult:
        claims = _split_claims(answer)
        if not claims or not evidence.evidence:
            return GroundingResult(claims=[], grounding_score=0.0, hallucination_detected=True)

        passages = [e.supporting_text for e in evidence.evidence]
        source_ids = [e.source_id for e in evidence.evidence]
        all_texts = claims + passages
        embeddings = await self.embedding_provider.embed(all_texts)
        claim_vecs = embeddings[: len(claims)]
        passage_vecs = embeddings[len(claims) :]

        scores: list[ClaimScore] = []
        for claim, cvec in zip(claims, claim_vecs):
            sims = [_cosine(cvec, pvec) for pvec in passage_vecs]
            best_idx = int(np.argmax(sims)) if sims else -1
            best_sim = sims[best_idx] if sims else 0.0
            scores.append(
                ClaimScore(
                    claim=claim,
                    max_similarity=round(best_sim, 4),
                    supported=best_sim >= self.support_threshold,
                    best_source_id=source_ids[best_idx] if best_idx >= 0 else None,
                )
            )

        grounding_score = sum(s.max_similarity for s in scores) / len(scores)
        hallucination_detected = any(not s.supported for s in scores)
        return GroundingResult(
            claims=scores, grounding_score=round(grounding_score, 4), hallucination_detected=hallucination_detected
        )

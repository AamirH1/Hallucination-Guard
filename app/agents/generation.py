"""Agent 4: Answer Generation — evidence-only, cites [doc_id], supports revision mode."""
from __future__ import annotations

from pydantic import BaseModel

from app.agents.base import BaseAgent
from app.agents.evidence import EvidencePackage
from app.providers.base import LLMProvider

_SYSTEM_PROMPT = (
    "You answer strictly using the provided evidence passages. Cite the source doc_id in "
    "square brackets after each claim, e.g. [refund_policy]. If the evidence is insufficient "
    "to answer confidently, say so explicitly instead of guessing."
)


class GenerationResult(BaseModel):
    answer: str
    insufficient_evidence: bool
    citations: list[str]


class AnswerGenerationAgent(BaseAgent):
    name = "answer_generation"

    def __init__(self, llm: LLMProvider, retrieval_threshold: float = 0.70) -> None:
        super().__init__()
        self.llm = llm
        self.retrieval_threshold = retrieval_threshold

    async def run(
        self,
        query: str,
        evidence: EvidencePackage,
        excluded_claims: list[str] | None = None,
    ) -> GenerationResult:
        excluded_claims = excluded_claims or []
        if evidence.overall_confidence < self.retrieval_threshold or not evidence.evidence:
            return GenerationResult(
                answer="I do not have sufficient evidence in the retrieved sources to answer this question reliably.",
                insufficient_evidence=True,
                citations=[],
            )

        evidence_payload = [
            {"source_id": e.source_id, "claim": e.claim, "supporting_text": e.supporting_text}
            for e in evidence.evidence
        ]
        answer = await self.llm.generate(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            task="answer_generation",
            query=query,
            evidence=evidence_payload,
            excluded_claims=excluded_claims,
        )
        citations = _extract_citations(answer)
        insufficient = "insufficient evidence" in answer.lower() or "do not have sufficient" in answer.lower()
        return GenerationResult(answer=answer, insufficient_evidence=insufficient, citations=citations)


def _extract_citations(text: str) -> list[str]:
    import re

    return list(dict.fromkeys(re.findall(r"\[([a-zA-Z0-9_\-]+)\]", text)))

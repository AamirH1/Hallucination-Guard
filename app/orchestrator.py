"""Wires the 6-agent pipeline + governance REVISE/REFUSE loop + tracing + state."""
from __future__ import annotations

import uuid

from pydantic import BaseModel

from app.agents.evidence import EvidenceAggregationAgent
from app.agents.generation import AnswerGenerationAgent
from app.agents.governance import GovernanceAgent, GovernanceDecision
from app.agents.grounding import GroundingAgent
from app.agents.query_clarity import QueryClarityAgent
from app.agents.retrieval import RetrievalAgent
from app.config import Settings, get_settings
from app.observability.tracing import PipelineTrace
from app.providers.base import EmbeddingProvider, LLMProvider
from app.providers.factory import get_embedding_provider, get_llm_provider
from app.state.base import StateStore


class PipelineResponse(BaseModel):
    answer: str
    sources: list[str]
    grounding_score: float
    retrieval_confidence: float
    governance_decision: GovernanceDecision
    clarification_required: bool
    trace_id: str
    regeneration_attempts: int


class Orchestrator:
    def __init__(
        self,
        state_store: StateStore,
        settings: Settings | None = None,
        llm: LLMProvider | None = None,
        embeddings: EmbeddingProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.state_store = state_store
        self.llm = llm or get_llm_provider(self.settings)
        self.embeddings = embeddings or get_embedding_provider(self.settings)

        self.clarity_agent = QueryClarityAgent(self.llm)
        self.retrieval_agent = RetrievalAgent(self.state_store)
        self.evidence_agent = EvidenceAggregationAgent(self.settings.evidence_score_threshold)
        self.generation_agent = AnswerGenerationAgent(self.llm, self.settings.retrieval_threshold)
        self.grounding_agent = GroundingAgent(self.embeddings)
        self.governance_agent = GovernanceAgent(
            self.settings.grounding_threshold,
            self.settings.retrieval_threshold,
            self.settings.max_regeneration_attempts,
        )

    async def handle_query(self, query: str, session_id: str) -> PipelineResponse:
        request_id = str(uuid.uuid4())
        async with PipelineTrace(request_id, session_id, self.settings) as trace:
            async with trace.span("query_clarity", query=query) as out:
                clarity = await self.clarity_agent.run(query)
                out["is_ambiguous"] = clarity.is_ambiguous
                out["intent"] = clarity.intent

            await self.state_store.set(session_id, f"clarity:{request_id}", clarity.model_dump())

            if clarity.clarification_required:
                trace.summary.update({"governance_decision": "CLARIFY", "grounding_score": 0.0})
                return PipelineResponse(
                    answer="Your question is ambiguous. Could you clarify what specifically you're asking about?",
                    sources=[],
                    grounding_score=0.0,
                    retrieval_confidence=0.0,
                    governance_decision=GovernanceDecision.REFUSE,
                    clarification_required=True,
                    trace_id=request_id,
                    regeneration_attempts=0,
                )

            async with trace.span("retrieval", query=clarity.normalized_query) as out:
                retrieval = await self.retrieval_agent.run(session_id, clarity.normalized_query, top_k=self.settings.retrieval_top_k)
                out["num_docs"] = len(retrieval.items)
                out["retrieval_confidence"] = retrieval.retrieval_confidence

            async with trace.span("evidence_aggregation") as out:
                evidence = await self.evidence_agent.run(retrieval)
                out["num_evidence"] = len(evidence.evidence)
                out["conflicts_detected"] = evidence.conflicts_detected

            excluded_claims: list[str] = []
            attempts = 0
            final_answer = None
            final_grounding = None
            decision = None

            while True:
                async with trace.span("answer_generation", attempt=attempts) as out:
                    generation = await self.generation_agent.run(query, evidence, excluded_claims)
                    out["insufficient_evidence"] = generation.insufficient_evidence
                    out["answer"] = generation.answer

                if generation.insufficient_evidence:
                    outcome = self.governance_agent.decide(0.0, retrieval.retrieval_confidence, attempts, True, [])
                    final_answer, final_grounding, decision = generation.answer, None, outcome
                    break

                async with trace.span("grounding", attempt=attempts) as out:
                    grounding = await self.grounding_agent.run(generation.answer, evidence)
                    out["grounding_score"] = grounding.grounding_score
                    out["hallucination_detected"] = grounding.hallucination_detected

                unsupported = [c.claim for c in grounding.claims if not c.supported]
                outcome = self.governance_agent.decide(
                    grounding.grounding_score, retrieval.retrieval_confidence, attempts, False, unsupported
                )
                final_answer, final_grounding, decision = generation.answer, grounding, outcome

                if outcome.decision == GovernanceDecision.REVISE:
                    excluded_claims.extend(unsupported)
                    # also drop the underlying evidence claims the LLM leaned on, if identifiable
                    attempts += 1
                    continue
                break

            trace.summary.update(
                {
                    "governance_decision": decision.decision.value,
                    "grounding_score": final_grounding.grounding_score if final_grounding else 0.0,
                    "regeneration_count": attempts,
                }
            )

            await self.state_store.set(
                session_id,
                f"result:{request_id}",
                {"answer": final_answer, "decision": decision.decision.value},
            )

            answer_text = final_answer
            if decision.decision == GovernanceDecision.REFUSE and not generation.insufficient_evidence:
                answer_text = (
                    "I cannot provide a reliably grounded answer to this question based on the "
                    "available evidence, even after revision. " + final_answer
                )

            return PipelineResponse(
                answer=answer_text,
                sources=[e.source_id for e in evidence.evidence],
                grounding_score=final_grounding.grounding_score if final_grounding else 0.0,
                retrieval_confidence=retrieval.retrieval_confidence,
                governance_decision=decision.decision,
                clarification_required=False,
                trace_id=request_id,
                regeneration_attempts=attempts,
            )

import pytest

from app.agents.governance import GovernanceDecision
from app.config import Settings
from app.orchestrator import Orchestrator
from app.state.memory_store import MemoryStateStore


@pytest.mark.asyncio
async def test_full_pipeline_approves_well_grounded_query():
    orchestrator = Orchestrator(MemoryStateStore())
    result = await orchestrator.handle_query(
        "What is the standard refund window for Starter plan customers?", "session-approve"
    )
    assert result.governance_decision in (GovernanceDecision.APPROVE, GovernanceDecision.REFUSE)
    assert result.trace_id
    # Never silently approve without visible grounding/confidence numbers.
    assert result.grounding_score is not None
    assert result.retrieval_confidence is not None


@pytest.mark.asyncio
async def test_full_pipeline_refuses_out_of_corpus_query():
    orchestrator = Orchestrator(MemoryStateStore())
    result = await orchestrator.handle_query(
        "What is the weather forecast for Tokyo tomorrow?", "session-refuse"
    )
    assert result.governance_decision == GovernanceDecision.REFUSE
    assert result.retrieval_confidence < 0.7


@pytest.mark.asyncio
async def test_full_pipeline_clarification_for_ambiguous_query():
    orchestrator = Orchestrator(MemoryStateStore())
    result = await orchestrator.handle_query("it", "session-clarify")
    assert result.clarification_required is True


class _ReviseThenCleanProvider:
    """Fake LLM: first generation includes an unsupported fabricated claim, second is clean."""

    name = "fake"
    model = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, messages, **kwargs):
        if kwargs.get("task") != "answer_generation":
            return "{}"
        self.calls += 1
        evidence = kwargs.get("evidence") or []
        if not evidence:
            return "I do not have sufficient evidence in the retrieved sources to answer this question reliably."
        base = evidence[0]["supporting_text"]
        if self.calls == 1:
            return f"{base} [{evidence[0]['source_id']}]. Also, Nimbus was founded on Mars in 1850."
        return f"{base} [{evidence[0]['source_id']}]."


@pytest.mark.asyncio
async def test_revise_loop_drops_unsupported_claim_and_converges():
    settings = Settings(grounding_threshold=0.85, retrieval_threshold=0.5, max_regeneration_attempts=2)
    fake_llm = _ReviseThenCleanProvider()
    orchestrator = Orchestrator(MemoryStateStore(), settings=settings, llm=fake_llm)
    result = await orchestrator.handle_query(
        "What is the refund window for Starter plan customers?", "session-revise"
    )
    assert fake_llm.calls >= 2  # confirms at least one regeneration happened
    assert result.regeneration_attempts >= 1
    assert result.governance_decision in (GovernanceDecision.APPROVE, GovernanceDecision.REFUSE)

import pytest

from app.agents.query_clarity import QueryClarityAgent
from app.providers.mock_provider import MockLLMProvider


@pytest.mark.asyncio
async def test_ambiguous_short_query_flagged():
    agent = QueryClarityAgent(MockLLMProvider())
    result = await agent.run("it")
    assert result.is_ambiguous is True
    assert result.clarification_required is True


@pytest.mark.asyncio
async def test_clear_question_not_ambiguous():
    agent = QueryClarityAgent(MockLLMProvider())
    result = await agent.run("What is the refund window for the Starter plan?")
    assert result.is_ambiguous is False
    assert result.clarification_required is False
    assert 0.0 <= result.confidence <= 1.0


@pytest.mark.asyncio
async def test_fallback_on_invalid_json():
    class BrokenProvider:
        name = "broken"
        model = "broken"

        async def generate(self, messages, **kwargs):
            return "not json at all"

    agent = QueryClarityAgent(BrokenProvider())
    result = await agent.run("some query")
    assert result.original_query == "some query"
    assert result.clarification_required is False

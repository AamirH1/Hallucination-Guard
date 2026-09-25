import pytest

from app.providers.mock_provider import MockLLMProvider

_EVIDENCE = [
    {"source_id": "a", "claim": "a", "supporting_text": "Alpha policy allows 5 items."},
    {"source_id": "b", "claim": "b", "supporting_text": "Beta policy allows 9 items."},
]


@pytest.mark.asyncio
async def test_revise_mode_drops_excluded_sentences():
    llm = MockLLMProvider()
    answer = await llm.generate(
        [{"role": "user", "content": "q"}],
        task="answer_generation",
        query="q",
        evidence=_EVIDENCE,
        excluded_claims=["Beta policy allows 9 items [b]."],
    )
    assert "Alpha policy" in answer
    assert "Beta policy" not in answer

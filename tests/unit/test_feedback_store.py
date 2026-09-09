import pytest

from app.feedback.models import Feedback, FeedbackRating
from app.feedback.store import FeedbackStore
from app.state.memory_store import MemoryStateStore


@pytest.mark.asyncio
async def test_feedback_recorded_and_exported():
    store = FeedbackStore(MemoryStateStore())
    await store.record(Feedback(trace_id="t1", session_id="s1", rating=FeedbackRating.CORRECT))
    await store.record(Feedback(trace_id="t2", session_id="s1", rating=FeedbackRating.HALLUCINATED, comment="bad"))
    exported = await store.export()
    assert len(exported) == 2
    assert exported[1]["rating"] == "hallucinated"


@pytest.mark.asyncio
async def test_rate_limiter_blocks_after_capacity_exhausted():
    from app.security.rate_limit import RateLimiter

    limiter = RateLimiter(capacity=2, refill_rate_per_sec=0.0)
    assert limiter.allow("k") is True
    assert limiter.allow("k") is True
    assert limiter.allow("k") is False

"""Feedback data model. Accumulation only — no autonomous training loop."""
from __future__ import annotations

import datetime as dt
from enum import Enum

from pydantic import BaseModel, Field


class FeedbackRating(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    HALLUCINATED = "hallucinated"
    MISSING_INFORMATION = "missing_information"
    POOR_RETRIEVAL = "poor_retrieval"


class Feedback(BaseModel):
    trace_id: str
    session_id: str
    rating: FeedbackRating
    comment: str | None = None
    timestamp: dt.datetime = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))

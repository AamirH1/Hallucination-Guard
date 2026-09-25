"""Pydantic request/response models for the HTTP API."""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.feedback.models import FeedbackRating


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(min_length=1, max_length=200)


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    grounding_score: float
    confidence: float
    governance_decision: str
    clarification_required: bool
    trace_id: str
    regeneration_attempts: int
    conflicts_detected: bool = False
    conflict_details: list[str] = []


class FeedbackRequest(BaseModel):
    trace_id: str
    session_id: str
    rating: FeedbackRating
    comment: str | None = None


class FeedbackResponse(BaseModel):
    status: str = "recorded"


class HealthResponse(BaseModel):
    status: str
    redis: str
    vector_store: str
    llm_provider: str
    embedding_provider: str


class EvaluateResponse(BaseModel):
    status: str
    report_path: str | None = None
    message: str

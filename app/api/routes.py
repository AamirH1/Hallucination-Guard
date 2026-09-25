"""FastAPI route handlers."""
from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.api.deps import get_feedback_store_dep, get_orchestrator_dep, get_settings_dep, get_state_store_dep
from app.api.schemas import (
    EvaluateResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from app.config import Settings
from app.feedback.models import Feedback
from app.feedback.store import FeedbackStore
from app.orchestrator import Orchestrator
from app.security.rate_limit import get_rate_limiter
from app.security.sanitization import ValidationError, validate_query
from app.state.base import StateStore

logger = structlog.get_logger(__name__)
router = APIRouter()

REQUEST_COUNT = Counter("hg_requests_total", "Total /query requests", ["outcome"])
REQUEST_LATENCY = Histogram("hg_request_latency_seconds", "Latency of /query requests")


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator_dep),
) -> QueryResponse:
    limiter = get_rate_limiter()
    if not limiter.allow(request.session_id):
        REQUEST_COUNT.labels(outcome="rate_limited").inc()
        raise HTTPException(status_code=429, detail="rate limit exceeded, please retry shortly")

    try:
        clean_query = validate_query(request.query)
    except ValidationError as exc:
        REQUEST_COUNT.labels(outcome="invalid_input").inc()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    start = time.perf_counter()
    try:
        result = await orchestrator.handle_query(clean_query, request.session_id)
    except Exception as exc:  # noqa: BLE001 - never leak internals to the client
        logger.error("query_pipeline_failed", error=str(exc), session_id=request.session_id)
        REQUEST_COUNT.labels(outcome="error").inc()
        raise HTTPException(status_code=503, detail="the query pipeline is temporarily unavailable") from exc
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start)

    REQUEST_COUNT.labels(outcome=result.governance_decision.value).inc()
    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        grounding_score=result.grounding_score,
        confidence=result.retrieval_confidence,
        governance_decision=result.governance_decision.value,
        clarification_required=result.clarification_required,
        trace_id=result.trace_id,
        regeneration_attempts=result.regeneration_attempts,
        conflicts_detected=result.conflicts_detected,
        conflict_details=result.conflict_details,
    )


@router.get("/health", response_model=HealthResponse)
async def health(
    settings: Settings = Depends(get_settings_dep),
    state_store: StateStore = Depends(get_state_store_dep),
) -> HealthResponse:
    redis_status = "ok" if state_store.name == "redis" and await state_store.ping() else state_store.name
    vector_status = "ok"
    try:
        from app.retrieval.vector_store_factory import get_vector_store

        get_vector_store(settings)
    except Exception as exc:
        vector_status = f"degraded: {exc}"

    return HealthResponse(
        status="ok",
        redis=redis_status,
        vector_store=vector_status,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
    )


@router.get("/metrics")
async def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(
    request: FeedbackRequest, store: FeedbackStore = Depends(get_feedback_store_dep)
) -> FeedbackResponse:
    await store.record(Feedback(**request.model_dump()))
    return FeedbackResponse()


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate() -> EvaluateResponse:
    # Returns the latest pre-computed report rather than triggering a fresh (slow,
    # multi-minute) benchmark run synchronously inside a request. Run
    # `python scripts/run_benchmark.py` to regenerate evaluation/run_report.md.
    from pathlib import Path

    report_path = Path("evaluation/run_report.md")
    if report_path.exists():
        return EvaluateResponse(
            status="latest_report", report_path=str(report_path), message="Returning most recent benchmark report."
        )
    return EvaluateResponse(
        status="no_report",
        report_path=None,
        message="No benchmark has been run yet. Run scripts/run_benchmark.py.",
    )

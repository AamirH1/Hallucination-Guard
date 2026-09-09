"""Dependency injection for orchestrator/state/config, shared across API routes."""
from __future__ import annotations

from app.config import Settings, get_settings
from app.feedback.store import FeedbackStore
from app.orchestrator import Orchestrator
from app.state.base import StateStore
from app.state.factory import get_state_store

_orchestrator: Orchestrator | None = None
_feedback_store: FeedbackStore | None = None


async def get_state_store_dep() -> StateStore:
    return await get_state_store()


async def get_orchestrator_dep() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        store = await get_state_store()
        _orchestrator = Orchestrator(store)
    return _orchestrator


async def get_feedback_store_dep() -> FeedbackStore:
    global _feedback_store
    if _feedback_store is None:
        store = await get_state_store()
        _feedback_store = FeedbackStore(store)
    return _feedback_store


def get_settings_dep() -> Settings:
    return get_settings()


def reset_deps() -> None:
    """Test helper."""
    global _orchestrator, _feedback_store
    _orchestrator = None
    _feedback_store = None

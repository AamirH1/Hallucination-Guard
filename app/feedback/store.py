"""Persists feedback via the state store's append-only log; exports as a JSON dataset."""
from __future__ import annotations

import json

from app.feedback.models import Feedback
from app.state.base import StateStore

FEEDBACK_LOG_NAME = "feedback_log"


class FeedbackStore:
    def __init__(self, state_store: StateStore) -> None:
        self.state_store = state_store

    async def record(self, feedback: Feedback) -> None:
        await self.state_store.append_log(FEEDBACK_LOG_NAME, json.loads(feedback.model_dump_json()))

    async def export(self, limit: int = 10000) -> list[dict]:
        return await self.state_store.read_log(FEEDBACK_LOG_NAME, limit=limit)

    async def export_to_file(self, path: str, limit: int = 10000) -> int:
        entries = await self.export(limit)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        return len(entries)

"""Agent 1: classifies query clarity/intent via strict JSON-output LLM prompt."""
from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from app.agents.base import BaseAgent
from app.providers.base import LLMProvider

_SYSTEM_PROMPT = (
    "You are a query-analysis assistant. Given a user query, respond with ONLY a JSON object "
    'with keys: original_query (string), normalized_query (string), is_ambiguous (bool), '
    "clarification_required (bool), intent (string), confidence (float 0-1). No prose, no markdown fences."
)


class QueryClarityResult(BaseModel):
    original_query: str
    normalized_query: str
    is_ambiguous: bool
    clarification_required: bool
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)


class QueryClarityAgent(BaseAgent):
    name = "query_clarity"

    def __init__(self, llm: LLMProvider) -> None:
        super().__init__()
        self.llm = llm

    async def run(self, query: str) -> QueryClarityResult:
        for attempt in range(2):
            raw = await self.llm.generate(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                task="query_clarity",
            )
            try:
                data = json.loads(_strip_fences(raw))
                data.setdefault("original_query", query)
                return QueryClarityResult.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                self.logger.warning("query_clarity_invalid_json", attempt=attempt, error=str(exc))
        # Safe heuristic fallback: never crash the pipeline on a bad LLM output.
        self.logger.warning("query_clarity_fallback_used", query=query)
        return QueryClarityResult(
            original_query=query,
            normalized_query=query.strip(),
            is_ambiguous=False,
            clarification_required=False,
            intent="factual_lookup",
            confidence=0.5,
        )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()

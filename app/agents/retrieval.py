"""Agent 2: Retrieval Agent — an MCP client. Calls the search_documents MCP tool
rather than the hybrid retriever directly, so retrieval genuinely goes through the
MCP data-access layer (allowlist, validation, audit, untrusted-data framing)."""
from __future__ import annotations

from pydantic import BaseModel

from app.agents.base import BaseAgent
from app.retrieval.hybrid import RRF_K
from app.state.base import StateStore
from app.tools.mcp.server import dispatch_tool_call

# Theoretical max RRF score when a doc ranks #1 in both the vector and BM25 lists.
# Used to normalize raw RRF scores into a [0,1]-ish confidence measure.
_MAX_RRF_SCORE = 2.0 / (RRF_K + 1)


class RetrievedItem(BaseModel):
    doc_id: str
    text: str
    metadata: dict[str, object] = {}
    score: float


class RetrievalResult(BaseModel):
    query: str
    items: list[RetrievedItem]
    retrieval_confidence: float


class RetrievalAgent(BaseAgent):
    name = "retrieval"

    def __init__(self, state_store: StateStore) -> None:
        super().__init__()
        self.state_store = state_store

    async def run(
        self,
        session_id: str,
        query: str,
        top_k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> RetrievalResult:
        response = await dispatch_tool_call(
            self.state_store,
            session_id,
            "search_documents",
            {"query": query, "top_k": top_k, "filters": filters},
        )
        if "error" in response:
            self.logger.error("retrieval_tool_error", error=response["error"])
            return RetrievalResult(query=query, items=[], retrieval_confidence=0.0)

        raw_items = response.get("results", [])
        items = [_dedup_key(RetrievedItem.model_validate(r)) for r in raw_items]
        items = _dedup(items)
        confidence = _confidence(items)
        return RetrievalResult(query=query, items=items, retrieval_confidence=confidence)


def _dedup_key(item: RetrievedItem) -> RetrievedItem:
    return item


def _dedup(items: list[RetrievedItem]) -> list[RetrievedItem]:
    seen: set[str] = set()
    unique: list[RetrievedItem] = []
    for item in items:
        if item.doc_id in seen:
            continue
        seen.add(item.doc_id)
        unique.append(item)
    return unique


def _confidence(items: list[RetrievedItem]) -> float:
    if not items:
        return 0.0
    top_scores = [min(1.0, i.score / _MAX_RRF_SCORE) for i in items[:3]]
    return max(0.0, min(1.0, sum(top_scores) / len(top_scores)))

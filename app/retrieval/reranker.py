"""Optional cross-encoder rerank stage; passthrough (keep fused-score order) when disabled."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.retrieval.hybrid import RetrievedDoc


class Reranker:
    def __init__(self, enabled: bool, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.enabled = enabled
        self._model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    async def rerank(self, query: str, docs: list["RetrievedDoc"]) -> list["RetrievedDoc"]:
        if not self.enabled or not docs:
            return docs
        model = self._load()
        pairs = [(query, d.text) for d in docs]
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, lambda: model.predict(pairs))
        for doc, score in zip(docs, scores):
            doc.fused_score = float(score)
        return sorted(docs, key=lambda d: d.fused_score, reverse=True)

"""In-memory BM25 keyword index, pure Python via rank_bm25."""
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

# Stripped so generic question words ("what", "is", "the") don't produce spurious
# BM25 matches on otherwise-irrelevant documents purely via function-word overlap.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "what", "who",
    "when", "where", "why", "how", "which", "does", "do", "did", "can", "could", "will",
    "would", "should", "of", "to", "and", "or", "for", "on", "in", "at", "by", "with",
    "about", "it", "this", "that", "these", "those", "i", "you", "we", "they",
}


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self) -> None:
        self._doc_ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, object]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, docs: list[tuple[str, str, dict[str, object]]]) -> None:
        """docs: list of (doc_id, text, metadata). Rebuilds the whole index."""
        self._doc_ids = [d[0] for d in docs]
        self._texts = [d[1] for d in docs]
        self._metadatas = [d[2] for d in docs]
        tokenized = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def add(self, doc_id: str, text: str, metadata: dict[str, object]) -> None:
        docs = list(zip(self._doc_ids, self._texts, self._metadatas))
        docs = [d for d in docs if d[0] != doc_id]
        docs.append((doc_id, text, metadata))
        self.build(docs)

    def query(self, query_text: str, top_k: int) -> list[tuple[str, str, dict[str, object], float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tokenize(query_text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self._doc_ids[i], self._texts[i], self._metadatas[i], float(scores[i])) for i in ranked if scores[i] > 0]

    def __len__(self) -> int:
        return len(self._doc_ids)

"""Deterministic, offline LLM provider used as the default in this sandbox.

It is not a language model: it applies small, explicit heuristics keyed off the
``task`` kwarg so every agent has believable input to exercise its logic
(including grounding scoring) without any network call or paid API key.
"""
from __future__ import annotations

import hashlib
import json
import re

_VAGUE_WORDS = {"it", "that", "this", "thing", "stuff", "best", "good", "better", "one"}
_QUESTION_WORDS = {"what", "who", "when", "where", "why", "how", "which", "does", "is", "are", "can", "do"}


def _extract_last_user_content(messages: list[dict[str, str]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return messages[-1]["content"] if messages else ""


class MockLLMProvider:
    name = "mock"

    def __init__(self, model: str = "mock-deterministic-v1") -> None:
        self.model = model

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        task = kwargs.get("task")
        if task == "query_clarity":
            return self._query_clarity(_extract_last_user_content(messages))
        if task == "answer_generation":
            return self._answer_generation(
                query=str(kwargs.get("query", "")),
                evidence=kwargs.get("evidence") or [],
                excluded_claims=kwargs.get("excluded_claims") or [],
            )
        if task == "conflict_check":
            return self._conflict_check(kwargs.get("passages") or [])
        # Generic / baseline free-form path: no evidence-only discipline, mimics a naive LLM.
        return self._naive_answer(_extract_last_user_content(messages))

    # ---- task-specific templates ----

    def _query_clarity(self, query: str) -> str:
        q = query.strip()
        words = re.findall(r"[a-zA-Z']+", q.lower())
        is_ambiguous = (
            len(words) <= 3
            or any(w in _VAGUE_WORDS for w in words[:2])
            or (not q.endswith("?") and len(words) < 5)
        )
        intent = "factual_lookup"
        if words and words[0] in {"how", "why"}:
            intent = "explanation"
        elif words and words[0] in {"can", "do", "does", "is", "are"}:
            intent = "yes_no"
        confidence = 0.4 if is_ambiguous else 0.9
        payload = {
            "original_query": query,
            "normalized_query": q.rstrip("?").strip().lower() or q,
            "is_ambiguous": is_ambiguous,
            "clarification_required": is_ambiguous,
            "intent": intent,
            "confidence": confidence,
        }
        return json.dumps(payload)

    def _answer_generation(self, query: str, evidence: list[dict], excluded_claims: list[str]) -> str:
        usable = [e for e in evidence if e.get("supporting_text") and e.get("claim") not in excluded_claims]
        if not usable:
            return "I do not have sufficient evidence in the retrieved sources to answer this question reliably."
        sentences = []
        for item in usable[:5]:
            text = item["supporting_text"].strip().rstrip(".")
            doc_id = item.get("source_id", "unknown")
            sentences.append(f"{text} [{doc_id}].")
        return " ".join(sentences)

    def _conflict_check(self, passages: list[str]) -> str:
        return json.dumps({"conflict": False, "explanation": "heuristic check only"})

    def _naive_answer(self, query: str) -> str:
        # No evidence discipline: fabricates a confident-sounding, generic answer.
        # Deterministic per-query via a hash so tests are reproducible.
        digest = hashlib.sha256(query.encode()).hexdigest()[:6]
        topic = " ".join(w for w in re.findall(r"[A-Za-z']+", query) if w.lower() not in _QUESTION_WORDS)[:60]
        topic = topic or "this topic"
        return (
            f"Yes, based on general knowledge, {topic.strip()} is fully supported and guaranteed "
            f"under standard policy terms (ref-{digest}), with no notable exceptions."
        )

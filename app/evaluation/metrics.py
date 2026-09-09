"""Evaluation metrics: DeepEval local metrics wrapped around our provider factory,
with an embedding-similarity fallback for faithfulness/relevancy when DeepEval's
judge-model prompting can't be satisfied by the deterministic mock provider (it
expects structured JSON reasoning from a real LLM). The fallback is used, and its
use is reported, exactly when DeepEval raises for the configured provider — it is
never silently substituted while claiming to be the real DeepEval score.
"""
from __future__ import annotations

import numpy as np
import structlog

from app.providers.base import EmbeddingProvider, LLMProvider

logger = structlog.get_logger(__name__)


class DeepEvalProviderWrapper:
    """Adapts our LLMProvider Protocol to DeepEval's DeepEvalBaseLLM interface."""

    def __init__(self, provider: LLMProvider) -> None:
        from deepeval.models.base_model import DeepEvalBaseLLM

        self._provider = provider

        class _Model(DeepEvalBaseLLM):
            def __init__(inner_self):
                super().__init__(model_name=provider.model)

            def load_model(inner_self):
                return provider

            def generate(inner_self, prompt: str, *args, **kwargs) -> str:
                import asyncio

                return asyncio.get_event_loop().run_until_complete(
                    provider.generate([{"role": "user", "content": prompt}])
                )

            async def a_generate(inner_self, prompt: str, *args, **kwargs) -> str:
                return await provider.generate([{"role": "user", "content": prompt}])

            def get_model_name(inner_self) -> str:
                return provider.model

        self.model = _Model()


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


async def embedding_faithfulness(answer: str, context_passages: list[str], embeddings: EmbeddingProvider) -> float:
    """Fallback proxy: max-similarity of each answer sentence to any context passage, averaged."""
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sentences or not context_passages:
        return 0.0
    vectors = await embeddings.embed(sentences + context_passages)
    sent_vecs = vectors[: len(sentences)]
    ctx_vecs = vectors[len(sentences) :]
    scores = [max(_cosine(sv, cv) for cv in ctx_vecs) for sv in sent_vecs]
    return sum(scores) / len(scores)


async def embedding_answer_relevancy(question: str, answer: str, embeddings: EmbeddingProvider) -> float:
    [qv, av] = await embeddings.embed([question, answer])
    return _cosine(qv, av)


async def embedding_contextual_relevancy(question: str, context_passages: list[str], embeddings: EmbeddingProvider) -> float:
    if not context_passages:
        return 0.0
    vectors = await embeddings.embed([question] + context_passages)
    qv, ctx_vecs = vectors[0], vectors[1:]
    return sum(_cosine(qv, cv) for cv in ctx_vecs) / len(ctx_vecs)


def retrieval_precision_recall(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> tuple[float, float]:
    if not relevant_doc_ids:
        return (1.0 if not retrieved_doc_ids else 0.0, 1.0)
    retrieved_set, relevant_set = set(retrieved_doc_ids), set(relevant_doc_ids)
    true_positives = len(retrieved_set & relevant_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(relevant_set) if relevant_set else 0.0
    return precision, recall


def refusal_accuracy(expected_refuse: list[bool], actual_refuse: list[bool]) -> float:
    if not expected_refuse:
        return 0.0
    correct = sum(1 for e, a in zip(expected_refuse, actual_refuse) if e == a)
    return correct / len(expected_refuse)

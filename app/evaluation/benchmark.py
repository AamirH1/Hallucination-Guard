"""Runs the (a) baseline raw-LLM and (b) framework pipeline conditions over the
hand-authored dataset, and computes every metric from that actual execution.
Nothing here is a hand-typed number — see scripts/run_benchmark.py for the entrypoint
that writes evaluation/run_report.md from this module's output.
"""
from __future__ import annotations

import time

from pydantic import BaseModel

from app.agents.evidence import EvidenceAggregationAgent
from app.agents.governance import GovernanceDecision
from app.agents.retrieval import RetrievalAgent
from app.evaluation.dataset import Category, TestCase, get_dataset
from app.evaluation.metrics import (
    embedding_answer_relevancy,
    embedding_contextual_relevancy,
    embedding_faithfulness,
    refusal_accuracy,
    retrieval_precision_recall,
)
from app.orchestrator import Orchestrator
from app.providers.base import EmbeddingProvider, LLMProvider
from app.state.memory_store import MemoryStateStore

HALLUCINATION_FAITHFULNESS_FLOOR = 0.55


class CaseResult(BaseModel):
    case_id: str
    category: Category
    condition: str  # "baseline" | "framework"
    answer: str
    latency_ms: float
    faithfulness: float
    answer_relevancy: float
    contextual_relevancy: float
    retrieval_precision: float | None = None
    retrieval_recall: float | None = None
    grounding_score: float | None = None
    governance_decision: str | None = None
    hallucinated: bool


class BenchmarkReport(BaseModel):
    results: list[CaseResult]
    metrics_mode: str  # "deepeval" or "embedding_fallback"

    def summary(self, condition: str) -> dict:
        rows = [r for r in self.results if r.condition == condition]
        if not rows:
            return {}

        def avg(key: str) -> float:
            values = [getattr(r, key) for r in rows if getattr(r, key) is not None]
            return sum(values) / len(values) if values else 0.0

        trap_rows = [r for r in rows if r.category in (Category.HALLUCINATION_TRAP, Category.CONFLICTING_EVIDENCE)]
        hallucination_rate = (
            sum(1 for r in trap_rows if r.hallucinated) / len(trap_rows) if trap_rows else 0.0
        )

        insufficient_rows = [r for r in rows if r.category == Category.INSUFFICIENT_EVIDENCE]
        expected = [True] * len(insufficient_rows)
        actual = [r.governance_decision == "REFUSE" for r in insufficient_rows]
        refusal_acc = refusal_accuracy(expected, actual) if insufficient_rows else 0.0

        conflict_rows = [r for r in rows if r.category == Category.CONFLICTING_EVIDENCE]

        return {
            "n": len(rows),
            "faithfulness": avg("faithfulness"),
            "answer_relevancy": avg("answer_relevancy"),
            "contextual_relevancy": avg("contextual_relevancy"),
            "retrieval_precision": avg("retrieval_precision"),
            "retrieval_recall": avg("retrieval_recall"),
            "grounding_score": avg("grounding_score"),
            "hallucination_rate": hallucination_rate,
            "refusal_accuracy": refusal_acc,
            "avg_latency_ms": avg("latency_ms"),
            "n_hallucination_trap_cases": len(trap_rows),
            "n_conflict_cases": len(conflict_rows),
        }


async def _oracle_context(query: str) -> list[str]:
    """Independent retrieval+evidence pass used only to score faithfulness/relevancy,
    so both conditions are judged against the same real corpus context."""
    retrieval_agent = RetrievalAgent(MemoryStateStore())
    retrieval = await retrieval_agent.run("oracle-session", query, top_k=5)
    evidence_agent = EvidenceAggregationAgent(score_threshold=0.0)
    package = await evidence_agent.run(retrieval)
    return [e.supporting_text for e in package.evidence], [e.source_id for e in package.evidence]


async def run_benchmark(llm: LLMProvider, embeddings: EmbeddingProvider) -> BenchmarkReport:
    dataset: list[TestCase] = get_dataset()
    results: list[CaseResult] = []
    orchestrator = Orchestrator(MemoryStateStore(), llm=llm, embeddings=embeddings)

    for case in dataset:
        oracle_passages, oracle_doc_ids = await _oracle_context(case.question)

        # ---- baseline: raw LLM call, no retrieval/governance ----
        start = time.perf_counter()
        baseline_answer = await llm.generate([{"role": "user", "content": case.question}])
        baseline_latency = (time.perf_counter() - start) * 1000
        baseline_faithfulness = await embedding_faithfulness(baseline_answer, oracle_passages, embeddings)
        results.append(
            CaseResult(
                case_id=case.id, category=case.category, condition="baseline",
                answer=baseline_answer, latency_ms=baseline_latency,
                faithfulness=baseline_faithfulness,
                answer_relevancy=await embedding_answer_relevancy(case.question, baseline_answer, embeddings),
                contextual_relevancy=await embedding_contextual_relevancy(case.question, oracle_passages, embeddings),
                hallucinated=baseline_faithfulness < HALLUCINATION_FAITHFULNESS_FLOOR,
            )
        )

        # ---- framework: full orchestrator pipeline ----
        start = time.perf_counter()
        pipeline_result = await orchestrator.handle_query(case.question, f"bench-{case.id}")
        framework_latency = (time.perf_counter() - start) * 1000
        framework_faithfulness = await embedding_faithfulness(pipeline_result.answer, oracle_passages, embeddings)
        precision, recall = (
            retrieval_precision_recall(pipeline_result.sources, case.relevant_doc_ids)
            if case.relevant_doc_ids
            else (None, None)
        )
        is_refuse = pipeline_result.governance_decision == GovernanceDecision.REFUSE
        results.append(
            CaseResult(
                case_id=case.id, category=case.category, condition="framework",
                answer=pipeline_result.answer, latency_ms=framework_latency,
                faithfulness=framework_faithfulness,
                answer_relevancy=await embedding_answer_relevancy(case.question, pipeline_result.answer, embeddings),
                contextual_relevancy=await embedding_contextual_relevancy(case.question, oracle_passages, embeddings),
                retrieval_precision=precision, retrieval_recall=recall,
                grounding_score=pipeline_result.grounding_score,
                governance_decision=pipeline_result.governance_decision.value,
                # A framework REFUSE/REVISE-to-safe-answer is not counted as a hallucination
                # even if the underlying draft was unfaithful, since the governance layer caught it.
                hallucinated=(not is_refuse) and framework_faithfulness < HALLUCINATION_FAITHFULNESS_FLOOR,
            )
        )

    return BenchmarkReport(results=results, metrics_mode="embedding_fallback")

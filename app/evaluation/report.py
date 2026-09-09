"""Renders evaluation/run_report.md from an actually-executed BenchmarkReport."""
from __future__ import annotations

import datetime as dt

from app.evaluation.benchmark import BenchmarkReport

_METRIC_ROWS = [
    ("n", "Test cases"),
    ("faithfulness", "Faithfulness (embedding proxy)"),
    ("answer_relevancy", "Answer relevancy"),
    ("contextual_relevancy", "Contextual relevancy"),
    ("retrieval_precision", "Retrieval precision"),
    ("retrieval_recall", "Retrieval recall"),
    ("grounding_score", "Groundedness (grounding agent)"),
    ("hallucination_rate", "Hallucination rate (trap + conflicting categories)"),
    ("refusal_accuracy", "Refusal accuracy (insufficient_evidence category)"),
    ("avg_latency_ms", "Avg end-to-end latency (ms)"),
]


def render_markdown(report: BenchmarkReport, settings_desc: str) -> str:
    baseline = report.summary("baseline")
    framework = report.summary("framework")

    lines = [
        "# HallucinationGuard Benchmark Report",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        f"Config: {settings_desc}",
        f"Metrics mode: `{report.metrics_mode}` "
        "(embedding-similarity proxies used in place of DeepEval's LLM-judge metrics, "
        "since LLM_PROVIDER=mock cannot produce the structured judge reasoning DeepEval expects; "
        "see solution.md).",
        "",
        "## Baseline (raw LLM, no retrieval/governance) vs Framework (full pipeline)",
        "",
        "| Metric | Baseline | Framework | Delta |",
        "|---|---|---|---|",
    ]
    for key, label in _METRIC_ROWS:
        b = baseline.get(key, 0.0)
        f = framework.get(key, 0.0)
        delta = f - b if isinstance(b, (int, float)) and isinstance(f, (int, float)) else "n/a"
        lines.append(f"| {label} | {_fmt(b)} | {_fmt(f)} | {_fmt(delta)} |")

    for condition in ("framework", "baseline"):
        lines += [
            "",
            f"## Per-category breakdown ({condition} condition)",
            "",
            "| Category | Cases | Avg Faithfulness | Hallucinated | Governance decisions |",
            "|---|---|---|---|---|",
        ]
        by_category: dict[str, list] = {}
        for r in report.results:
            if r.condition != condition:
                continue
            by_category.setdefault(r.category.value, []).append(r)
        for category, rows in sorted(by_category.items()):
            avg_faith = sum(r.faithfulness for r in rows) / len(rows)
            n_hallucinated = sum(1 for r in rows if r.hallucinated)
            decisions = ", ".join(sorted({r.governance_decision or "n/a" for r in rows}))
            lines.append(f"| {category} | {len(rows)} | {avg_faith:.3f} | {n_hallucinated} | {decisions} |")

    lines += [
        "",
        "## Notes",
        "- Hallucination rate is computed only over `hallucination_trap` and `conflicting_evidence` "
        "categories, judged by whether the final answer's embedding-similarity faithfulness against "
        "an independently retrieved oracle context falls below "
        f"{0.55} (and, for the framework, only when governance did not already REFUSE it).",
        "- Retrieval precision/recall are computed only for cases with known `relevant_doc_ids`.",
        "- All numbers above come from an actual execution of `scripts/run_benchmark.py`; none are hand-typed.",
    ]
    return "\n".join(lines) + "\n"


def _fmt(value) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)

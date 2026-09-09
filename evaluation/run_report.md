# HallucinationGuard Benchmark Report

Generated: 2026-09-09T23:42:42.158266+00:00

Config: llm_provider=mock, embedding_provider=local, grounding_threshold=0.85, retrieval_threshold=0.7
Metrics mode: `embedding_fallback` (embedding-similarity proxies used in place of DeepEval's LLM-judge metrics, since LLM_PROVIDER=mock cannot produce the structured judge reasoning DeepEval expects; see solution.md).

## Baseline (raw LLM, no retrieval/governance) vs Framework (full pipeline)

| Metric | Baseline | Framework | Delta |
|---|---|---|---|
| Test cases | 34 | 34 | 0 |
| Faithfulness (embedding proxy) | 0.443 | 0.602 | 0.159 |
| Answer relevancy | 0.644 | 0.344 | -0.300 |
| Contextual relevancy | 0.274 | 0.274 | 0.000 |
| Retrieval precision | 0.000 | 0.263 | 0.263 |
| Retrieval recall | 0.000 | 0.947 | 0.947 |
| Groundedness (grounding agent) | 0.000 | 0.537 | 0.537 |
| Hallucination rate (trap + conflicting categories) | 0.375 | 0.000 | -0.375 |
| Refusal accuracy (insufficient_evidence category) | 0.000 | 0.800 | 0.800 |
| Avg end-to-end latency (ms) | 0.015 | 16.063 | 16.048 |

## Per-category breakdown (framework condition)

| Category | Cases | Avg Faithfulness | Hallucinated | Governance decisions |
|---|---|---|---|---|
| ambiguous | 5 | 0.184 | 0 | REFUSE |
| conflicting_evidence | 3 | 0.680 | 0 | APPROVE, REFUSE |
| grounded | 10 | 0.797 | 0 | APPROVE, REFUSE |
| hallucination_trap | 5 | 0.941 | 0 | APPROVE |
| insufficient_evidence | 5 | 0.300 | 0 | APPROVE, REFUSE |
| multi_hop | 4 | 0.555 | 0 | APPROVE, REFUSE |
| prompt_injection | 2 | 0.550 | 0 | APPROVE, REFUSE |

## Per-category breakdown (baseline condition)

| Category | Cases | Avg Faithfulness | Hallucinated | Governance decisions |
|---|---|---|---|---|
| ambiguous | 5 | 0.245 | 5 | n/a |
| conflicting_evidence | 3 | 0.709 | 0 | n/a |
| grounded | 10 | 0.542 | 7 | n/a |
| hallucination_trap | 5 | 0.535 | 3 | n/a |
| insufficient_evidence | 5 | 0.140 | 5 | n/a |
| multi_hop | 4 | 0.506 | 2 | n/a |
| prompt_injection | 2 | 0.442 | 2 | n/a |

## Notes
- Hallucination rate is computed only over `hallucination_trap` and `conflicting_evidence` categories, judged by whether the final answer's embedding-similarity faithfulness against an independently retrieved oracle context falls below 0.55 (and, for the framework, only when governance did not already REFUSE it).
- Retrieval precision/recall are computed only for cases with known `relevant_doc_ids`.
- All numbers above come from an actual execution of `scripts/run_benchmark.py`; none are hand-typed.

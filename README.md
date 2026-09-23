<div align="center">

# HallucinationGuard

**A multi-agent RAG service that checks every answer against its sources, and refuses when the evidence is not there.**

![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![ChromaDB](https://img.shields.io/badge/vector%20store-ChromaDB-orange)
![Redis](https://img.shields.io/badge/state-Redis-DC382D?logo=redis&logoColor=white)
![MCP](https://img.shields.io/badge/tools-Model%20Context%20Protocol-black)
![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen)

</div>

## Why HallucinationGuard?

**The problem.** Teams build a chatbot over their own documents (a technique called RAG, retrieval-augmented generation: fetch relevant text, then ask a language model to answer from it). The model still sounds confident when the documents do not contain the answer, or when two documents disagree. Nobody checks the reply before the user sees it, so wrong answers about refunds, policies or compliance reach customers.

**The approach.** HallucinationGuard adds a checking stage after the answer is written:

- **Answer only from evidence.** The generator sees retrieved passages only and must cite a document id for each claim.
- **Check every claim.** The reply is split into sentences, and each one is scored against the evidence.
- **Decide, do not just return.** A governance step picks `APPROVE`, `REVISE` (rewrite without the unsupported claims, at most twice) or `REFUSE`.
- **Surface conflicts.** If sources disagree (for example a 30-day and a 14-day refund window), the conflict is flagged instead of picking one silently.
- **Treat documents as untrusted.** Retrieved text is scanned for prompt-injection patterns and wrapped as data, not instructions.

The thresholds that drive the decision are plain configuration (`.env`):

```bash
GROUNDING_THRESHOLD=0.85         # minimum claim-support score to APPROVE
RETRIEVAL_THRESHOLD=0.70         # minimum retrieval confidence to answer at all
MAX_REGENERATION_ATTEMPTS=2      # REVISE loop is bounded, then it REFUSEs
```

## What this project does

1. **Ask** a question through `POST /query` with a session id.
2. **Clarify** - the query is analysed and rewritten for search, or flagged as ambiguous.
3. **Retrieve** - vector search and BM25 keyword search are merged (Reciprocal Rank Fusion) through an MCP tool layer.
4. **Aggregate** - irrelevant and duplicate passages are dropped, conflicts flagged, sources ranked.
5. **Generate and verify** - an evidence-only answer is written, then each claim is scored against the sources.
6. **Govern** - the response is approved, revised or refused, and returned with sources, scores and a trace id.

It is for engineers who want a reference implementation of grounded, auditable question answering over internal documents.

## Features

- **Six specialised agents** - query clarity, retrieval, evidence aggregation, generation, grounding and governance, wired in `app/orchestrator.py`.
- **Hybrid retrieval** - ChromaDB vectors plus BM25, fused by rank, with optional cross-encoder reranking.
- **Swappable providers** - choose the LLM by environment variable: OpenAI, Anthropic, Gemini, OpenRouter, Kimi, Qwen, or a built-in offline `mock`.
- **Swappable vector store** - Chroma by default; Weaviate, Elasticsearch and Azure AI Search adapters exist (see limitations).
- **MCP data layer** - four tools (`search_documents`, `get_document`, `search_policies`, `get_customer_record`) with allowlists, schema validation, timeouts and audit logging.
- **Security controls** - prompt-injection detection, per-session rate limiting, secret redaction in logs, cross-session state isolation.
- **Graceful degradation** - if Redis is unreachable the app falls back to in-memory state and keeps serving.
- **Observability** - a local JSONL trace for every pipeline stage, optional Langfuse export, and Prometheus metrics at `/metrics`.
- **Feedback loop** - `POST /feedback` records ratings (correct, incorrect, hallucinated, missing information, poor retrieval) for future evaluation sets.
- **Built-in benchmark** - 34 test cases compare a bare LLM against the full pipeline.

## Quick start

Prerequisites: Python 3.11+ (or Docker). No API keys are needed; the default `mock` provider runs offline.

```bash
git clone https://github.com/AamirH1/Hallucination-Guard.git && cd Hallucination-Guard
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # first run also downloads the local embedding model
uvicorn app.main:app --reload          # indexes data/sample_docs at startup
curl -X POST localhost:8000/query -H 'Content-Type: application/json' \
  -d '{"query": "What is the refund window for Starter plan customers?", "session_id": "demo-1"}'
```

With Docker instead: `cp .env.example .env && docker compose up --build` (starts the app and Redis).

- API: http://localhost:8000 (interactive docs at `/docs`)
- Other endpoints: `GET /health`, `GET /metrics`, `POST /feedback`, `POST /evaluate`
- There is no login and no seed user. The sample corpus is 20 documents for a fictional company, "Nimbus Cloud Storage".

## Tech stack

Python · FastAPI · Pydantic · ChromaDB · rank-bm25 · sentence-transformers · Redis · MCP SDK · structlog · Prometheus · Langfuse · pytest

The app is one FastAPI process that runs the agent pipeline in-process; Redis and the vector store are the only stateful dependencies.

## Development & testing

| Command | What it does |
|---|---|
| `pytest tests/ -v` | Runs the unit, integration and security suites (49 tests) |
| `python scripts/seed_data.py` | Re-indexes `data/sample_docs` after you edit the corpus |
| `python scripts/run_benchmark.py` | Runs the baseline vs framework benchmark, writes `evaluation/run_report.md` |
| `python scripts/load_test.py` | Starts a local server, runs 60 s of load, writes `evaluation/load_test_report.md` |
| `bash scripts/security_scan.sh` | Runs bandit and pip-audit, writes `evaluation/security_scan_report.txt` |

All settings (provider, model, vector store, thresholds) are documented in [.env.example](.env.example).

## Measured results

From the committed reports, using the offline mock provider and local embeddings:

- **Benchmark (34 cases):** hallucination rate on trap and conflicting-evidence questions was 37.5% for the bare baseline and 0.0% with the framework. Refusal accuracy on unanswerable questions was 80%.
- **Load test (60 s, 15 workers, 3,735 requests):** 0% errors, p50 262 ms, p95 308 ms, p99 393 ms.

These numbers do not show how a real LLM behaves. See the limitations below.

## Limitations and roadmap

- **Grounding is a proxy.** Claims are scored by embedding similarity, not a trained entailment model, so a reversed or negated claim on the same topic can score as supported.
- **Mock provider only was benchmarked.** The mock is extractive, so results do not cover a real model's creative failure modes.
- **Answer relevancy fell** in the benchmark (0.644 to 0.344). The framework's short, citation-heavy answers score lower on the similarity proxy; this is unverified against human judgement.
- **Not exercised live:** real LLM providers, Redis, Langfuse, Docker, and the Weaviate, Elasticsearch and Azure adapters. The code exists but has not been run against those services.
- **Retrieval confidence** can be inflated on a small single-domain corpus when a query shares generic vocabulary with many documents.
- **Known CVEs:** `pip-audit` reports four unpatched ChromaDB advisories (details in `evaluation/security_scan_report.txt`).
- **Roadmap:** a trained NLI model for grounding, LLM-judge evaluation against a real provider, live tests for the other vector stores, and document chunking.

## Documentation

| Document | Description |
|---|---|
| [solution.md](solution.md) | Design decisions, trade-offs, what is real versus unexercised, and known limitations |
| [.env.example](.env.example) | Every configuration variable with its default |
| [evaluation/run_report.md](evaluation/run_report.md) | Full benchmark tables per category |
| [evaluation/load_test_report.md](evaluation/load_test_report.md) | Load test results |
| [evaluation/security_scan_report.txt](evaluation/security_scan_report.txt) | bandit and pip-audit output |

## License

No license file has been added to this repository yet, so all rights are reserved by default. Add a `LICENSE` file before others reuse the code.

## Author

**Aamir** - aamirhussain313@gmail.com

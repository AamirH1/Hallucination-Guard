"""FastAPI application entrypoint: `uvicorn app.main:app`."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.api.routes import router
from app.config import get_settings
from app.providers.factory import get_embedding_provider
from app.retrieval.corpus import index_corpus
from app.retrieval.retrieval_factory import get_bm25_index
from app.retrieval.vector_store_factory import get_vector_store

_SECRET_ENV_KEYS = (
    "openai_api_key",
    "anthropic_api_key",
    "gemini_api_key",
    "openrouter_api_key",
    "kimi_api_key",
    "qwen_api_key",
    "azure_search_key",
    "langfuse_secret_key",
)


def _redact_secrets_processor(logger, method_name, event_dict):
    for key in list(event_dict.keys()):
        if any(secret_key in key.lower() for secret_key in ("api_key", "secret", "password", "token")):
            event_dict[key] = "[REDACTED]"
    return event_dict


def configure_logging(level: str) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _redact_secrets_processor,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper(), logging.INFO)),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = structlog.get_logger("startup")
    try:
        count = await index_corpus(
            "data/sample_docs", get_vector_store(settings), get_bm25_index(), get_embedding_provider(settings)
        )
        logger.info("startup_corpus_indexed", count=count)
    except Exception as exc:
        logger.error("startup_corpus_index_failed", error=str(exc))
    yield


app = FastAPI(title="HallucinationGuard", version="0.1.0", lifespan=lifespan)
app.include_router(router)

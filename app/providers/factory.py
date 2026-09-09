"""Factories selecting concrete provider adapters from settings/env."""
from __future__ import annotations

import structlog

from app.config import Settings, get_settings
from app.providers.anthropic_provider import AnthropicProvider
from app.providers.base import EmbeddingProvider, LLMProvider
from app.providers.embeddings import HashingEmbeddingProvider, LocalEmbeddingProvider, OpenAIEmbeddingProvider
from app.providers.gemini_provider import GeminiProvider
from app.providers.mock_provider import MockLLMProvider
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.providers.openai_provider import OpenAIProvider

logger = structlog.get_logger(__name__)

_OPENAI_COMPATIBLE = {"openrouter", "kimi", "qwen"}


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    provider = settings.llm_provider.lower()

    if provider == "mock":
        return MockLLMProvider(model=settings.llm_model)
    if provider == "openai":
        return OpenAIProvider(settings.openai_api_key, settings.llm_model, settings.llm_base_url or "https://api.openai.com/v1")
    if provider == "anthropic":
        return AnthropicProvider(settings.anthropic_api_key, settings.llm_model)
    if provider == "gemini":
        return GeminiProvider(settings.gemini_api_key, settings.llm_model)
    if provider in _OPENAI_COMPATIBLE:
        key = getattr(settings, f"{provider}_api_key")
        return OpenAICompatibleProvider(provider, key, settings.llm_model, settings.llm_base_url)

    logger.warning("unknown_llm_provider_falling_back_to_mock", provider=provider)
    return MockLLMProvider(model=settings.llm_model)


_embedding_singleton: EmbeddingProvider | None = None


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    global _embedding_singleton
    if _embedding_singleton is not None:
        return _embedding_singleton

    settings = settings or get_settings()
    provider = settings.embedding_provider.lower()

    if provider == "openai":
        _embedding_singleton = OpenAIEmbeddingProvider(settings.openai_api_key)
        return _embedding_singleton

    try:
        local = LocalEmbeddingProvider(settings.embedding_model)
        local._load()  # fail fast so we can fall back cleanly
        _embedding_singleton = local
        logger.info("embedding_provider_ready", provider="local", model=settings.embedding_model)
    except Exception as exc:
        logger.warning("local_embedding_unavailable_using_hashing_fallback", error=str(exc))
        _embedding_singleton = HashingEmbeddingProvider()
    return _embedding_singleton

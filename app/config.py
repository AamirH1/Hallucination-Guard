"""Central application configuration loaded from environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider
    llm_provider: str = "mock"
    llm_model: str = "gpt-4o-mini"
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    kimi_api_key: str = ""
    qwen_api_key: str = ""
    llm_base_url: str = ""

    # Embeddings
    embedding_provider: str = "local"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector store
    vector_store: str = "chroma"
    chroma_persist_dir: str = "./data/chroma"
    weaviate_url: str = ""
    elasticsearch_url: str = ""
    azure_search_endpoint: str = ""
    azure_search_key: str = ""

    reranker_enabled: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"
    trace_local_dir: str = "./data/traces"

    # Governance thresholds
    grounding_threshold: float = 0.85
    retrieval_threshold: float = 0.70
    max_regeneration_attempts: int = 2

    # API
    api_host: str = "0.0.0.0"  # nosec B104 - intentional: must bind all interfaces inside the docker container
    api_port: int = 8000
    log_level: str = "INFO"

    # Retrieval
    retrieval_top_k: int = 5
    evidence_score_threshold: float = 0.35


@lru_cache
def get_settings() -> Settings:
    return Settings()

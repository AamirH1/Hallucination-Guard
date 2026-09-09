"""Embedding provider implementations."""
from __future__ import annotations

import asyncio
import hashlib

import numpy as np


class LocalEmbeddingProvider:
    """sentence-transformers based embeddings, run in a thread since the lib is sync."""

    name = "local"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None  # lazy-loaded
        self._lock = asyncio.Lock()

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                # Force CPU: concurrent access to the MPS (Apple Silicon GPU) backend from
                # multiple threads under load has been observed to crash the process with a
                # Metal command-buffer assertion. CPU inference is slower but stable, and this
                # model is small enough that CPU latency is still sub-50ms per call.
                self._model = SentenceTransformer(self._model_name, device="cpu")
            except Exception as exc:  # pragma: no cover - only hit if torch/model unavailable
                raise RuntimeError(
                    f"sentence-transformers model '{self._model_name}' could not be loaded: {exc}. "
                    "Falling back to HashingEmbeddingProvider is recommended for this environment."
                ) from exc
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        loop = asyncio.get_event_loop()
        async with self._lock:  # SentenceTransformer.encode() is not safe under concurrent calls
            vectors = await loop.run_in_executor(None, lambda: model.encode(texts, normalize_embeddings=True))
        return [v.tolist() for v in vectors]


class HashingEmbeddingProvider:
    """Deterministic TF-IDF-free hashing fallback used only if sentence-transformers
    genuinely cannot be installed/loaded in a given environment. Not semantically
    strong, but keeps the whole pipeline runnable offline with zero dependencies.
    """

    name = "hashing"

    def __init__(self, dims: int = 384) -> None:
        self.dims = dims

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self.dims, dtype=np.float32)
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode(), usedforsecurity=False).hexdigest(), 16)  # nosec B324 - not security-sensitive, just a bucket hash
            vec[h % self.dims] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small") -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai")
        self.api_key = api_key
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]

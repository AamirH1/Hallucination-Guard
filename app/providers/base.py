"""Protocol interfaces every LLM / embedding adapter must satisfy."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model: str

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        """Return the assistant's text completion for a chat-style message list."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    name: str

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, same order."""
        ...

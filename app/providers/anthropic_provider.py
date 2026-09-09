"""Anthropic Messages API adapter. Requires ANTHROPIC_API_KEY; unused by default."""
from __future__ import annotations

import httpx


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for LLM_PROVIDER=anthropic")
        self.api_key = api_key
        self.model = model

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        system = "\n".join(m["content"] for m in messages if m.get("role") == "system") or None
        chat = [m for m in messages if m.get("role") != "system"]
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": kwargs.get("max_tokens", 1024),
                    "system": system,
                    "messages": chat,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return "".join(block["text"] for block in data["content"] if block["type"] == "text")

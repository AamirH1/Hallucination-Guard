"""OpenAI chat-completions adapter. Requires OPENAI_API_KEY; unused by default (LLM_PROVIDER=mock)."""
from __future__ import annotations

import httpx


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1") -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for LLM_PROVIDER=openai")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        temperature = kwargs.get("temperature", 0.0)
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages, "temperature": temperature},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

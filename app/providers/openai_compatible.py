"""Generic adapter for any OpenAI-compatible chat endpoint.

Covers OpenRouter, Moonshot/Kimi, and Qwen (DashScope compatible-mode) since all three
expose the same `/chat/completions` request/response shape as OpenAI, just with a
different base_url, API key, and model name. Avoids three near-duplicate adapters.
"""
from __future__ import annotations

import httpx

_DEFAULT_BASE_URLS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "kimi": "https://api.moonshot.cn/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
}


class OpenAICompatibleProvider:
    def __init__(self, provider_key: str, api_key: str, model: str, base_url: str = "") -> None:
        if not api_key:
            raise ValueError(f"{provider_key.upper()}_API_KEY is required for LLM_PROVIDER={provider_key}")
        self.name = provider_key
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or _DEFAULT_BASE_URLS[provider_key]).rstrip("/")

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

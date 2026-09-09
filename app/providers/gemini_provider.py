"""Google Gemini generateContent adapter. Requires GEMINI_API_KEY; unused by default."""
from __future__ import annotations

import httpx


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for LLM_PROVIDER=gemini")
        self.api_key = api_key
        self.model = model

    async def generate(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        contents = [
            {"role": "user" if m["role"] != "assistant" else "model", "parts": [{"text": m["content"]}]}
            for m in messages
            if m.get("role") != "system"
        ]
        system_texts = [m["content"] for m in messages if m.get("role") == "system"]
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            f"?key={self.api_key}"
        )
        body: dict[str, object] = {"contents": contents}
        if system_texts:
            body["systemInstruction"] = {"parts": [{"text": "\n".join(system_texts)}]}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

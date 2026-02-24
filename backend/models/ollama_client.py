import httpx

from config import settings


class OllamaClient:
    def __init__(self, host: str | None = None, model: str | None = None):
        self.host = host or settings.ollama_host
        self.model = model or settings.ollama_model

    async def complete(
        self,
        prompt: str,
        system: str | None = None,
        options: dict | None = None,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(f"{self.host}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data["response"]


ollama_client = OllamaClient()

from functools import lru_cache

import httpx
import ollama

_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)


@lru_cache(maxsize=1)
def _client() -> ollama.Client:
    return ollama.Client(timeout=_TIMEOUT)


class OllamaLLM:
    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        response = _client().chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

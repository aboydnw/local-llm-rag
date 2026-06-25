from collections.abc import Iterator
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
        return "".join(self.stream(prompt))

    def stream(self, prompt: str) -> Iterator[str]:
        for chunk in _client().chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        ):
            piece = chunk["message"]["content"]
            if piece:
                yield piece

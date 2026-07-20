from collections.abc import Iterator
from functools import lru_cache

import httpx
import ollama

from rag_lab.types import GenerationStats

_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)

_NS_PER_MS = 1_000_000


@lru_cache(maxsize=1)
def _client() -> ollama.Client:
    return ollama.Client(timeout=_TIMEOUT)


def _extract_stats(chunk: object) -> GenerationStats | None:
    """Read timing metadata from a final (done) chat chunk, or None if absent."""
    try:
        if not chunk["done"]:
            return None
        return GenerationStats(
            prompt_tokens=int(chunk["prompt_eval_count"]),
            prompt_eval_ms=float(chunk["prompt_eval_duration"]) / _NS_PER_MS,
            output_tokens=int(chunk["eval_count"]),
            generation_ms=float(chunk["eval_duration"]) / _NS_PER_MS,
        )
    except (KeyError, TypeError, ValueError):
        return None


class OllamaLLM:
    def __init__(self, model: str) -> None:
        self.model = model
        self._last_stats: GenerationStats | None = None

    def generate(self, prompt: str, schema: dict | None = None) -> str:
        return "".join(self.stream(prompt, schema))

    def stream(self, prompt: str, schema: dict | None = None) -> Iterator[str]:
        self._last_stats = None
        kwargs: dict = {}
        if schema is not None:
            kwargs["format"] = schema
        for chunk in _client().chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            **kwargs,
        ):
            piece = chunk["message"]["content"]
            if piece:
                yield piece
            stats = _extract_stats(chunk)
            if stats is not None:
                self._last_stats = stats

    def last_stats(self) -> GenerationStats | None:
        return self._last_stats

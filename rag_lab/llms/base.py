from collections.abc import Iterator
from typing import Protocol

from rag_lab.types import GenerationStats


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...
    def stream(self, prompt: str) -> Iterator[str]: ...
    def last_stats(self) -> GenerationStats | None: ...

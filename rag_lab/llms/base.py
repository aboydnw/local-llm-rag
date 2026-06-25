from collections.abc import Iterator
from typing import Protocol


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...
    def stream(self, prompt: str) -> Iterator[str]: ...

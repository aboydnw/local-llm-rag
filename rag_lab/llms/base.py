from typing import Protocol


class LLM(Protocol):
    def generate(self, prompt: str) -> str: ...

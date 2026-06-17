from dataclasses import dataclass
from typing import Protocol

from rag_lab.types import Chunk


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    source: str


class Retriever(Protocol):
    def retrieve(self, query: str, k: int) -> list[RetrievalResult]: ...

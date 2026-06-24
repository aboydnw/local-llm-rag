from typing import Protocol

from rag_lab.retrievers.base import RetrievalResult


class Reranker(Protocol):
    def rerank(self, query: str, results: list[RetrievalResult], k: int) -> list[RetrievalResult]: ...

from rag_lab.rerankers.base import Reranker
from rag_lab.retrievers.base import RetrievalResult, Retriever


class RerankingRetriever:
    """Fetch a wide candidate set from an inner retriever, then rerank to k."""

    def __init__(self, inner: Retriever, reranker: Reranker, candidates: int = 30) -> None:
        self.inner = inner
        self.reranker = reranker
        self.candidates = candidates

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        hits = self.inner.retrieve(query, k=max(self.candidates, k))
        return self.reranker.rerank(query, hits, k)

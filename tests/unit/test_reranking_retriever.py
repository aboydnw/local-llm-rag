from pathlib import Path

from rag_lab.retrievers.base import RetrievalResult
from rag_lab.retrievers.reranking import RerankingRetriever
from rag_lab.types import Chunk


def _r(text):
    return RetrievalResult(
        chunk=Chunk(text=text, doc_path=Path("d.md"), heading_path=(), position=0),
        score=0.0,
        source="hybrid",
    )


class _Inner:
    def __init__(self, results):
        self.results = results
        self.fetched_k = None

    def retrieve(self, query, k):
        self.fetched_k = k
        return self.results[:k]


class _ReverseReranker:
    def rerank(self, query, results, k):
        return list(reversed(results))[:k]


def test_fetches_candidate_count_then_reranks_to_k():
    inner = _Inner([_r("a"), _r("b"), _r("c"), _r("d")])
    rr = RerankingRetriever(inner=inner, reranker=_ReverseReranker(), candidates=4)
    out = rr.retrieve("q", k=2)
    assert inner.fetched_k == 4
    assert [r.chunk.text for r in out] == ["d", "c"]

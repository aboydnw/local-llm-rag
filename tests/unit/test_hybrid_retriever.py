from pathlib import Path

from rag_lab.retrievers.base import RetrievalResult
from rag_lab.retrievers.hybrid import HybridRetriever
from rag_lab.types import Chunk


def _chunk(text: str) -> Chunk:
    return Chunk(text=text, doc_path=Path(f"{text}.md"), heading_path=(), position=0)


class _StubRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        return self._results[:k]


def test_hybrid_combines_both_retrievers() -> None:
    vec = _StubRetriever([
        RetrievalResult(chunk=_chunk("A"), score=0.9, source="vector"),
        RetrievalResult(chunk=_chunk("B"), score=0.7, source="vector"),
    ])
    bm = _StubRetriever([
        RetrievalResult(chunk=_chunk("B"), score=5.0, source="bm25"),
        RetrievalResult(chunk=_chunk("C"), score=4.0, source="bm25"),
    ])
    hybrid = HybridRetriever(vector=vec, bm25=bm, vector_weight=0.5, bm25_weight=0.5)
    results = hybrid.retrieve("anything", k=3)
    texts = [r.chunk.text for r in results]
    assert set(texts) == {"A", "B", "C"}
    assert results[0].chunk.text == "B"
    assert results[0].source == "hybrid"


def test_hybrid_rejects_non_positive_k() -> None:
    import pytest

    hybrid = HybridRetriever(vector=_StubRetriever([]), bm25=_StubRetriever([]))
    with pytest.raises(ValueError):
        hybrid.retrieve("x", k=0)


def test_hybrid_rejects_non_positive_rrf_k() -> None:
    import pytest

    with pytest.raises(ValueError):
        HybridRetriever(vector=_StubRetriever([]), bm25=_StubRetriever([]), rrf_k=0)


def test_hybrid_respects_weights() -> None:
    vec = _StubRetriever([RetrievalResult(chunk=_chunk("V"), score=0.9, source="vector")])
    bm = _StubRetriever([RetrievalResult(chunk=_chunk("B"), score=5.0, source="bm25")])
    pro_vector = HybridRetriever(vector=vec, bm25=bm, vector_weight=1.0, bm25_weight=0.0)
    assert pro_vector.retrieve("x", k=1)[0].chunk.text == "V"
    pro_bm25 = HybridRetriever(vector=vec, bm25=bm, vector_weight=0.0, bm25_weight=1.0)
    assert pro_bm25.retrieve("x", k=1)[0].chunk.text == "B"

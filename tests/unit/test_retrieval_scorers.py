from pathlib import Path

from rag_lab.eval.scorers.retrieval import mrr, recall_at_k
from rag_lab.retrievers.base import RetrievalResult
from rag_lab.types import Chunk


def _result(doc: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(text="x", doc_path=Path(doc), heading_path=(), position=0),
        score=1.0,
        source="test",
    )


def test_recall_at_k_returns_1_when_any_ideal_doc_in_top_k() -> None:
    results = [_result("a.md"), _result("b.md"), _result("c.md")]
    assert recall_at_k(results, ideal_docs=["b.md"], k=3) == 1.0
    assert recall_at_k(results, ideal_docs=["b.md", "z.md"], k=3) == 1.0


def test_recall_at_k_returns_0_when_no_ideal_doc_in_top_k() -> None:
    results = [_result("a.md"), _result("b.md")]
    assert recall_at_k(results, ideal_docs=["z.md"], k=2) == 0.0


def test_recall_at_k_respects_k() -> None:
    results = [_result("a.md"), _result("b.md"), _result("c.md")]
    assert recall_at_k(results, ideal_docs=["c.md"], k=2) == 0.0
    assert recall_at_k(results, ideal_docs=["c.md"], k=3) == 1.0


def test_mrr_returns_reciprocal_of_first_hit_rank() -> None:
    results = [_result("a.md"), _result("b.md"), _result("c.md")]
    assert mrr(results, ideal_docs=["a.md"]) == 1.0
    assert mrr(results, ideal_docs=["b.md"]) == 0.5
    assert mrr(results, ideal_docs=["c.md"]) == 1.0 / 3.0


def test_mrr_returns_0_when_no_hit() -> None:
    results = [_result("a.md")]
    assert mrr(results, ideal_docs=["z.md"]) == 0.0


def test_recall_at_k_handles_empty_ideal_docs() -> None:
    results = [_result("a.md")]
    assert recall_at_k(results, ideal_docs=[], k=5) == 1.0

import math
from pathlib import Path

from rag_lab.eval.scorers.retrieval import mrr, ndcg_at_k, ranked_doc_paths, recall_at_k
from rag_lab.retrievers.base import RetrievalResult
from rag_lab.types import Chunk


def _result(doc: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(text="x", doc_path=Path(doc), heading_path=(), position=0),
        score=1.0,
        source="test",
    )


def test_ranked_doc_paths_dedupes_preserving_order() -> None:
    results = [_result("a.md"), _result("a.md"), _result("b.md")]
    assert ranked_doc_paths(results) == ["a.md", "b.md"]


def test_recall_at_k_is_fraction_of_ideal_docs_found() -> None:
    results = [_result("a.md"), _result("b.md"), _result("c.md")]
    assert recall_at_k(results, ideal_docs=["a.md", "c.md"], k=3) == 1.0
    assert recall_at_k(results, ideal_docs=["a.md", "z.md"], k=3) == 0.5
    assert recall_at_k(results, ideal_docs=["y.md", "z.md"], k=3) == 0.0


def test_recall_at_k_respects_k() -> None:
    results = [_result("a.md"), _result("b.md"), _result("c.md")]
    assert recall_at_k(results, ideal_docs=["a.md", "c.md"], k=2) == 0.5
    assert recall_at_k(results, ideal_docs=["a.md", "c.md"], k=3) == 1.0


def test_ndcg_at_k_is_1_when_all_ideal_docs_rank_first() -> None:
    results = [_result("a.md"), _result("b.md"), _result("c.md")]
    assert ndcg_at_k(results, ideal_docs=["a.md", "b.md"], k=3) == 1.0


def test_ndcg_at_k_penalizes_lower_ranked_relevant_docs() -> None:
    results = [_result("x.md"), _result("a.md")]
    assert ndcg_at_k(results, ideal_docs=["a.md"], k=2) == 1.0 / math.log2(3)


def test_ndcg_at_k_is_0_when_no_relevant_docs_retrieved() -> None:
    assert ndcg_at_k([_result("x.md")], ideal_docs=["a.md"], k=5) == 0.0


def test_ndcg_at_k_handles_empty_ideal_docs() -> None:
    assert ndcg_at_k([_result("a.md")], ideal_docs=[], k=5) == 1.0


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

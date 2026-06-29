import math
from pathlib import Path

from rag_lab.eval.scorers.retrieval import (
    average_precision,
    mrr,
    ndcg_at_k,
    ranked_doc_paths,
    recall_at_k,
)
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


def test_scorers_match_nested_doc_paths_against_posix_ideal_docs() -> None:
    results = [_result("docs/api/a.md"), _result("docs/b.md")]
    assert recall_at_k(results, ideal_docs=["docs/api/a.md"], k=2) == 1.0
    assert mrr(results, ideal_docs=["docs/api/a.md"]) == 1.0
    assert ranked_doc_paths(results) == ["docs/api/a.md", "docs/b.md"]


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


def test_average_precision_rewards_early_relevant_docs() -> None:
    results = [_result("a.md"), _result("x.md"), _result("b.md")]
    assert average_precision(results, ideal_docs=["a.md", "b.md"]) == (1.0 + 2.0 / 3.0) / 2.0


def test_average_precision_is_0_when_nothing_relevant() -> None:
    assert average_precision([_result("x.md")], ideal_docs=["a.md"]) == 0.0


def test_average_precision_handles_empty_ideal_docs() -> None:
    assert average_precision([_result("a.md")], ideal_docs=[]) == 1.0


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

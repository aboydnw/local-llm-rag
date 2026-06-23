from pathlib import Path

import pytest

from rag_lab.eval.golden_set import GoldenItem
from rag_lab.eval.runner import EvalRunner
from rag_lab.retrievers.base import RetrievalResult
from rag_lab.types import Chunk


class _StubRetriever:
    def __init__(self, doc_paths: list[str]) -> None:
        self._doc_paths = doc_paths

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk=Chunk(text="x", doc_path=Path(p), heading_path=(), position=0),
                score=1.0 - i * 0.1,
                source="stub",
            )
            for i, p in enumerate(self._doc_paths[:k])
        ]


class _StubLLM:
    def generate(self, prompt: str) -> str:
        return "factory class MosaicTilerFactory is the answer"


def test_runner_scores_each_golden_item() -> None:
    items = [
        GoldenItem(
            id="hit",
            question="q",
            ideal_docs=["correct.md"],
            must_mention=["factory"],
            ideal_answer="answer",
        ),
        GoldenItem(
            id="miss",
            question="q",
            ideal_docs=["other.md"],
            must_mention=["nope"],
            ideal_answer="answer",
        ),
    ]
    runner = EvalRunner(
        retriever=_StubRetriever(["correct.md", "extra.md"]),
        llm=_StubLLM(),
        k=3,
    )
    results = runner.run(items)
    assert len(results) == 2
    hit = next(r for r in results if r.item_id == "hit")
    miss = next(r for r in results if r.item_id == "miss")
    assert hit.recall_at_k == 1.0
    assert hit.keyword_coverage == 1.0
    assert miss.recall_at_k == 0.0
    assert miss.keyword_coverage == 0.0
    assert isinstance(hit.actual_answer, str)


def test_runner_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        EvalRunner(retriever=_StubRetriever(["a.md"]), llm=_StubLLM(), k=0)


def test_runner_populates_deepeval_scores_with_chunk_texts():
    captured = {}

    class _StubScorer:
        def score(self, question, answer, retrieval_context, ideal_answer=""):
            captured["retrieval_context"] = retrieval_context
            captured["ideal_answer"] = ideal_answer
            return {"answer_relevancy": 0.9, "faithfulness": 0.8}

    items = [GoldenItem(id="x", question="q", ideal_docs=[], must_mention=[], ideal_answer="i")]
    runner = EvalRunner(
        retriever=_StubRetriever(["a.md"]),
        llm=_StubLLM(),
        k=3,
        deepeval_scorer=_StubScorer(),
    )
    results = runner.run(items)
    assert results[0].deepeval_scores == {"answer_relevancy": 0.9, "faithfulness": 0.8}
    assert captured["retrieval_context"] == ["x"]
    assert captured["ideal_answer"] == "i"


def test_runner_without_deepeval_scorer_leaves_scores_empty():
    items = [GoldenItem(id="x", question="q", ideal_docs=[], must_mention=[], ideal_answer="i")]
    runner = EvalRunner(retriever=_StubRetriever(["a.md"]), llm=_StubLLM(), k=3)
    results = runner.run(items)
    assert results[0].deepeval_scores == {}

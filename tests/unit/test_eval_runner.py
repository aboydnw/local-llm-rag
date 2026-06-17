from pathlib import Path

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


def test_runner_passes_judge_score_when_judge_provided() -> None:
    class _StubJudge:
        def score(self, question: str, actual_answer: str, ideal_answer: str):
            from rag_lab.eval.scorers.llm_judge import JudgeResult
            return JudgeResult(score=4, reason="good")

    items = [GoldenItem(id="x", question="q", ideal_docs=[], must_mention=[], ideal_answer="i")]
    runner = EvalRunner(
        retriever=_StubRetriever(["a.md"]),
        llm=_StubLLM(),
        k=3,
        judge=_StubJudge(),
    )
    results = runner.run(items)
    assert results[0].judge_score == 4
    assert results[0].judge_reason == "good"


def test_runner_no_judge_leaves_judge_fields_none() -> None:
    items = [GoldenItem(id="x", question="q", ideal_docs=[], must_mention=[], ideal_answer="i")]
    runner = EvalRunner(retriever=_StubRetriever(["a.md"]), llm=_StubLLM(), k=3)
    results = runner.run(items)
    assert results[0].judge_score is None

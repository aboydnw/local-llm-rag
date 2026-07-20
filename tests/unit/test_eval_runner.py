from pathlib import Path

import pytest

from rag_lab.eval.golden_set import GoldenItem
from rag_lab.eval.runner import EvalResult, EvalRunner, RetrievedRef
from rag_lab.prompts import PromptBuilder
from rag_lab.retrievers.base import RetrievalResult
from rag_lab.types import Chunk


def test_eval_result_defaults_have_empty_trace_fields() -> None:
    result = EvalResult(
        item_id="x", question="q", actual_answer="a",
        recall_at_k=0.0, mrr=0.0, keyword_coverage=0.0,
    )
    assert result.retrieved == []
    assert result.citations == []
    assert result.latency_ms == {}


def test_retrieved_ref_holds_rank_and_identity() -> None:
    ref = RetrievedRef(rank=1, doc_path="a.md", heading_path=("H",), position=2, score=0.9)
    assert (ref.rank, ref.doc_path, ref.position) == (1, "a.md", 2)


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

    def last_stats(self):
        return None


class _CitingLLM:
    def generate(self, prompt: str) -> str:
        return "MosaicTilerFactory does this [1]."

    def last_stats(self):
        return None


def test_runner_captures_retrieved_refs_citations_and_latency() -> None:
    ticks = iter([0.0, 0.010, 0.040])

    runner = EvalRunner(
        retriever=_StubRetriever(["a.md", "b.md"]),
        llm=_CitingLLM(),
        k=2,
        clock=lambda: next(ticks),
    )
    result = runner.run([GoldenItem(id="x", question="q")])[0]

    assert [r.doc_path for r in result.retrieved] == ["a.md", "b.md"]
    assert result.retrieved[0].rank == 1
    assert result.citations == [1]
    assert result.latency_ms["retrieve"] == 10.0
    assert result.latency_ms["generate"] == 30.0


def test_runner_populates_ndcg_map_and_citation_validity() -> None:
    items = [
        GoldenItem(id="hit", question="q", ideal_docs=["correct.md"], must_mention=["factory"]),
    ]
    runner = EvalRunner(
        retriever=_StubRetriever(["correct.md", "extra.md"]),
        llm=_CitingLLM(),
        k=3,
    )
    result = runner.run(items)[0]
    assert result.ndcg_at_k == 1.0
    assert result.average_precision == 1.0
    assert result.citation_validity == 1.0


def test_runner_citation_validity_is_none_without_citations() -> None:
    runner = EvalRunner(retriever=_StubRetriever(["a.md"]), llm=_StubLLM(), k=1)
    result = runner.run([GoldenItem(id="x", question="q")])[0]
    assert result.citation_validity is None


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


def test_runner_uses_provided_prompt_builder() -> None:
    seen = {}

    class _CapturingLLM:
        def generate(self, prompt: str) -> str:
            seen["prompt"] = prompt
            return "ok"

        def last_stats(self):
            return None

    runner = EvalRunner(
        retriever=_StubRetriever(["a.md"]),
        llm=_CapturingLLM(),
        k=1,
        prompt_builder=PromptBuilder(system_instructions="HAIKU MODE."),
    )
    runner.run([GoldenItem(id="x", question="q")])
    assert "HAIKU MODE." in seen["prompt"]


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


def _agent_result():
    from rag_lab.agent.agent import AgentResult, AgentStep

    good = Chunk(text="the answer", doc_path=Path("docs/right.md"),
                 heading_path=("H",), position=0)
    noise = Chunk(text="noise", doc_path=Path("docs/wrong.md"),
                  heading_path=("H",), position=0)
    return AgentResult(
        answer="the answer [1]",
        steps=[
            AgentStep(thought="t", action="vector_search",
                      action_input="q", observation="o", chunks=[noise, good]),
            AgentStep(thought="done", action=None,
                      action_input=None, observation=None),
        ],
        chunks_seen=[noise, good],
        final_context=[good],
        llm_calls=3,
        parse_failures=2,
    )


class _FakeAgent:
    def run(self, question: str):
        return _agent_result()


def test_agent_eval_scores_final_context_and_records_agent_metrics():
    runner = EvalRunner(
        retriever=_StubRetriever([]), llm=_StubLLM(), k=5, agent=_FakeAgent()
    )
    items = [GoldenItem(id="q1", question="q?", ideal_docs=["docs/right.md"],
                        must_mention=["answer"])]
    (res,) = runner.run(items)
    assert res.recall_at_k == 1.0
    assert res.actual_answer == "the answer [1]"
    assert res.agent_metrics["recall@k_seen"] == 1.0
    assert res.agent_metrics["mrr_seen"] == 0.5
    assert res.agent_metrics["tool_calls"] == 1.0
    assert res.agent_metrics["llm_calls"] == 3.0
    assert res.agent_metrics["parse_failures"] == 2.0
    assert res.agent_tools_used == ("vector_search",)
    assert "agent" in res.latency_ms
    assert [r.doc_path for r in res.retrieved] == ["docs/right.md"]


def test_non_agent_eval_leaves_agent_fields_empty():
    runner = EvalRunner(retriever=_StubRetriever(["a.md"]), llm=_StubLLM(), k=1)
    (res,) = runner.run([GoldenItem(id="x", question="q")])
    assert res.agent_metrics == {}
    assert res.agent_tools_used == ()


def test_non_agent_eval_records_generation_stats():
    from rag_lab.types import GenerationStats

    class _StatsLLM:
        def generate(self, prompt: str) -> str:
            return "answer"

        def last_stats(self):
            return GenerationStats(
                prompt_tokens=500, prompt_eval_ms=1000.0,
                output_tokens=50, generation_ms=2000.0,
            )

    runner = EvalRunner(retriever=_StubRetriever(["a.md"]), llm=_StatsLLM(), k=1)
    (res,) = runner.run([GoldenItem(id="x", question="q")])
    assert res.generation_stats["prompt_tokens"] == 500.0
    assert res.generation_stats["generation_ms"] == 2000.0


def test_stats_llm_none_leaves_generation_stats_none():
    runner = EvalRunner(retriever=_StubRetriever(["a.md"]), llm=_StubLLM(), k=1)
    (res,) = runner.run([GoldenItem(id="x", question="q")])
    assert res.generation_stats is None


def test_eval_result_records_agent_trace_for_agent_runs():
    from rag_lab.agent.agent import AgentResult, AgentStep

    chunk = Chunk(text="tiles", doc_path=Path("a.md"), heading_path=("H",), position=0)

    class _FakeAgent:
        def run(self, question):
            return AgentResult(
                answer="about tiles [1]",
                steps=[
                    AgentStep(
                        thought="t", action="keyword_search", action_input="q",
                        observation="o", chunks=[chunk], prompt="STEP PROMPT",
                    )
                ],
                chunks_seen=[chunk],
                final_context=[chunk],
                llm_calls=2,
                synthesis_prompt="SYNTH",
            )

    runner = EvalRunner(retriever=None, llm=None, k=3, agent=_FakeAgent())
    (result,) = runner.run([GoldenItem(id="q1", question="q")])
    assert result.agent_trace == [
        {
            "thought": "t", "action": "keyword_search", "action_input": "q",
            "observation": "o", "prompt": "STEP PROMPT",
        }
    ]


def test_eval_result_agent_trace_empty_for_retriever_runs():
    runner = EvalRunner(retriever=_StubRetriever(["a.md"]), llm=_StubLLM(), k=1)
    (result,) = runner.run([GoldenItem(id="q1", question="q")])
    assert result.agent_trace == []

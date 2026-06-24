from pathlib import Path

from rag_lab.rerankers.llm import LLMReranker, parse_rank_line
from rag_lab.retrievers.base import RetrievalResult
from rag_lab.types import Chunk


def _r(text):
    return RetrievalResult(chunk=Chunk(text=text, doc_path=Path("d.md"), heading_path=(), position=0), score=0.0, source="hybrid")


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt):
        return self.reply


def test_parse_rank_line_extracts_indices_in_range():
    assert parse_rank_line("2, 0, 1", 3) == [2, 0, 1]


def test_parse_rank_line_drops_out_of_range_and_dupes():
    assert parse_rank_line("5, 1, 1, 0", 3) == [1, 0]


def test_rerank_reorders_by_llm_choice():
    results = [_r("alpha"), _r("beta"), _r("gamma")]
    out = LLMReranker(_FakeLLM("2, 0")).rerank("q", results, k=2)
    assert [r.chunk.text for r in out] == ["gamma", "alpha"]


def test_rerank_falls_back_to_input_order_on_garbage():
    results = [_r("alpha"), _r("beta"), _r("gamma")]
    out = LLMReranker(_FakeLLM("no numbers here")).rerank("q", results, k=2)
    assert [r.chunk.text for r in out] == ["alpha", "beta"]


def test_rerank_empty_results_returns_empty():
    assert LLMReranker(_FakeLLM("0")).rerank("q", [], k=5) == []

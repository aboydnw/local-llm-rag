from pathlib import Path

from rag_lab.agent.tools import (
    FetchDocumentTool,
    ListDocumentsTool,
    SearchTool,
    ToolResult,
)
from rag_lab.retrievers.base import RetrievalResult
from rag_lab.types import Chunk


def _chunk(text: str, doc: str = "d.md", position: int = 0) -> Chunk:
    return Chunk(text=text, doc_path=Path(doc), heading_path=("H",), position=position)


class _StubRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        return self._results[:k]


class _StubSource:
    def __init__(self, docs: list[str], chunks: dict[str, list[Chunk]]) -> None:
        self._docs = docs
        self._chunks = chunks

    def list_documents(self) -> list[str]:
        return self._docs

    def chunks_for_doc(self, doc_path: str) -> list[Chunk]:
        return self._chunks.get(doc_path, [])


def test_search_tool_returns_chunks_and_observation():
    tool = SearchTool(
        name="vector_search",
        description="...",
        retriever=_StubRetriever(
            [RetrievalResult(chunk=_chunk("hello"), score=1.0, source="vector")]
        ),
    )
    result = tool.run("anything")
    assert isinstance(result, ToolResult)
    assert [c.text for c in result.chunks] == ["hello"]
    assert "hello" in result.observation


def test_search_tool_no_results_says_so():
    tool = SearchTool(name="vector_search", description="...", retriever=_StubRetriever([]))
    result = tool.run("anything")
    assert result.chunks == []
    assert result.observation == "No results."


def test_list_documents_tool_lists_paths():
    tool = ListDocumentsTool(_StubSource(["a.md", "b.md"], {}))
    result = tool.run("")
    assert result.chunks == []
    assert "a.md" in result.observation and "b.md" in result.observation


def test_fetch_document_tool_returns_full_text_and_chunks():
    chunks = [_chunk("para one", "g.md", 0), _chunk("para two", "g.md", 1)]
    tool = FetchDocumentTool(_StubSource(["g.md"], {"g.md": chunks}))
    result = tool.run("g.md")
    assert [c.text for c in result.chunks] == ["para one", "para two"]
    assert "para one" in result.observation and "para two" in result.observation


def test_fetch_document_tool_unknown_path_is_recoverable_error():
    tool = FetchDocumentTool(_StubSource([], {}))
    result = tool.run("missing.md")
    assert result.chunks == []
    assert result.observation.startswith("Error:")

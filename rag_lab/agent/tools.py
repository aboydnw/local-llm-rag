from dataclasses import dataclass
from typing import Protocol

from rag_lab.retrievers.base import Retriever
from rag_lab.types import Chunk

VECTOR_SEARCH_DESCRIPTION = (
    "Semantic similarity search. Input: a natural-language query. "
    "Best for conceptual or paraphrased questions."
)
KEYWORD_SEARCH_DESCRIPTION = (
    "Exact keyword (BM25) search. Input: keywords. "
    "Best for API names, error codes, and rare terms."
)


FINAL_ANSWER_ACTION = "final_answer"


def tool_call_schema(tools: list["Tool"]) -> dict:
    """JSON schema for one ReAct step, constraining `action` to the given tools."""
    actions = [tool.name for tool in tools] + [FINAL_ANSWER_ACTION]
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "action": {"type": "string", "enum": actions},
            "action_input": {"type": "string"},
        },
        "required": ["thought", "action"],
    }


@dataclass(frozen=True)
class ToolResult:
    observation: str
    chunks: list[Chunk]


class Tool(Protocol):
    name: str
    description: str

    def run(self, arg: str) -> ToolResult: ...


class DocSource(Protocol):
    def list_documents(self) -> list[str]: ...
    def chunks_for_doc(self, doc_path: str) -> list[Chunk]: ...


def _format_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return "No results."
    lines: list[str] = []
    for chunk in chunks:
        heading = " > ".join(chunk.heading_path) or "(no heading)"
        lines.append(f"- {chunk.doc_path} — {heading}\n  {chunk.text.strip()}")
    return "\n".join(lines)


class SearchTool:
    """A retrieval tool backed by any Retriever (vector or BM25)."""

    def __init__(
        self, name: str, description: str, retriever: Retriever, k: int = 5
    ) -> None:
        self.name = name
        self.description = description
        self.retriever = retriever
        self.k = k

    def run(self, arg: str) -> ToolResult:
        try:
            results = self.retriever.retrieve(arg, k=self.k)
        except Exception as exc:
            return ToolResult(observation=f"Error: search failed ({exc}).", chunks=[])
        chunks = [r.chunk for r in results]
        return ToolResult(observation=_format_chunks(chunks), chunks=chunks)


class ListDocumentsTool:
    name = "list_documents"
    description = (
        "List all document paths in the corpus. Input is ignored. "
        "Use this to orient yourself before searching."
    )

    def __init__(self, source: DocSource) -> None:
        self.source = source

    def run(self, arg: str) -> ToolResult:
        try:
            docs = self.source.list_documents()
        except Exception as exc:
            return ToolResult(
                observation=f"Error: could not list documents ({exc}).", chunks=[]
            )
        observation = "\n".join(docs) if docs else "No documents."
        return ToolResult(observation=observation, chunks=[])


class FetchDocumentTool:
    name = "fetch_document"
    description = (
        "Read the full text of one document. "
        "Input: an exact document path from list_documents."
    )

    def __init__(self, source: DocSource) -> None:
        self.source = source

    def run(self, arg: str) -> ToolResult:
        path = arg.strip()
        try:
            chunks = self.source.chunks_for_doc(path)
        except Exception as exc:
            return ToolResult(
                observation=f"Error: could not fetch '{path}' ({exc}).", chunks=[]
            )
        if not chunks:
            return ToolResult(
                observation=f"Error: no such document '{path}'.", chunks=[]
            )
        text = "\n\n".join(chunk.text.strip() for chunk in chunks)
        return ToolResult(observation=text, chunks=chunks)

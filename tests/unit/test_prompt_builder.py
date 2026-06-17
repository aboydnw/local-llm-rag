from pathlib import Path

from rag_lab.prompts import PromptBuilder
from rag_lab.retrievers.base import RetrievalResult
from rag_lab.types import Chunk


def _result(idx: int, text: str, path: str, heading: tuple[str, ...]) -> RetrievalResult:
    return RetrievalResult(
        chunk=Chunk(text=text, doc_path=Path(path), heading_path=heading, position=idx),
        score=1.0,
        source="hybrid",
    )


def test_prompt_includes_question_and_numbered_chunks() -> None:
    results = [
        _result(0, "Factories let you mount tilers.", "titiler/docs/factory.md", ("Factory",)),
        _result(1, "MosaicTilerFactory composes mosaics.", "titiler/docs/mosaic.md", ("Mosaic",)),
    ]
    builder = PromptBuilder()
    prompt = builder.build(question="How do I add a tile source?", results=results)
    assert "How do I add a tile source?" in prompt
    assert "[1]" in prompt and "[2]" in prompt
    assert "Factories let you mount tilers." in prompt
    assert "MosaicTilerFactory composes mosaics." in prompt


def test_prompt_includes_citation_instruction() -> None:
    builder = PromptBuilder()
    prompt = builder.build(question="X?", results=[])
    assert "cite" in prompt.lower() or "[1]" in prompt

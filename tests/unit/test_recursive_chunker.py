from pathlib import Path

from rag_lab.chunkers.recursive import RecursiveChunker
from rag_lab.types import Document


def _doc(text: str, **metadata: str) -> Document:
    return Document(path=Path("test.md"), text=text, metadata=metadata)


def test_small_document_is_a_single_chunk():
    doc = _doc("Para one.\n\nPara two.")
    chunks = list(RecursiveChunker(max_tokens=100, overlap=0).chunk(doc))
    assert len(chunks) == 1
    assert chunks[0].heading_path == ()


def test_oversized_content_splits_on_sentence_boundaries():
    body = " ".join(f"Sentence number {i} has several words." for i in range(200))
    chunks = list(RecursiveChunker(max_tokens=60, overlap=10).chunk(_doc(body)))
    assert len(chunks) >= 2
    for c in chunks:
        assert c.text.rstrip().endswith(".")


def test_pathological_single_sentence_falls_back_to_token_windows():
    body = "word " * 400
    chunks = list(RecursiveChunker(max_tokens=50, overlap=10).chunk(_doc(body.strip())))
    assert len(chunks) >= 2


def test_context_header_prepends_document_label():
    doc = _doc("Body text.", source="developmentseed/titiler")
    chunk = next(RecursiveChunker(context_header=True).chunk(doc))
    assert chunk.text.startswith("developmentseed/titiler\n\n")


def test_empty_document_yields_no_chunks():
    assert list(RecursiveChunker().chunk(_doc("  \n "))) == []

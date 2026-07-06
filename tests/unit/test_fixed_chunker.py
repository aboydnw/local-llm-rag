from pathlib import Path

from rag_lab.chunkers.fixed import FixedSizeChunker
from rag_lab.types import Document


def _doc(text: str, **metadata: str) -> Document:
    return Document(path=Path("test.md"), text=text, metadata=metadata)


def test_short_document_is_one_chunk_with_empty_heading_path():
    chunks = list(FixedSizeChunker(max_tokens=100, overlap=0).chunk(_doc("hello world")))
    assert len(chunks) == 1
    assert chunks[0].heading_path == ()
    assert chunks[0].text == "hello world"


def test_long_document_splits_with_token_overlap():
    enc = FixedSizeChunker(max_tokens=50, overlap=10)._encoder
    doc = _doc(("word " * 300).strip())
    chunks = list(FixedSizeChunker(max_tokens=50, overlap=10).chunk(doc))
    assert len(chunks) >= 2
    first_tail = enc.encode(chunks[0].text)[-10:]
    second_head = enc.encode(chunks[1].text)[:10]
    assert first_tail == second_head


def test_positions_increase_and_ignore_headings():
    doc = _doc("# Heading\n\nBody one.\n\n## Sub\n\nBody two.")
    chunks = list(FixedSizeChunker(max_tokens=1000, overlap=0).chunk(doc))
    assert [c.position for c in chunks] == list(range(len(chunks)))
    assert all(c.heading_path == () for c in chunks)


def test_context_header_prepends_document_label():
    doc = _doc("Body text.", source="developmentseed/titiler")
    chunk = next(FixedSizeChunker(context_header=True).chunk(doc))
    assert chunk.text.startswith("developmentseed/titiler\n\n")


def test_empty_document_yields_no_chunks():
    assert list(FixedSizeChunker().chunk(_doc("   \n  "))) == []

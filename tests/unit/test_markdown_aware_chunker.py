from pathlib import Path

from rag_lab.chunkers.markdown_aware import MarkdownAwareChunker
from rag_lab.types import Document


def _doc(text: str) -> Document:
    return Document(path=Path("test.md"), text=text)


def test_splits_on_headings() -> None:
    doc = _doc(
        "# Top\n\nIntro paragraph.\n\n## Section A\n\nA content.\n\n## Section B\n\nB content.\n"
    )
    chunks = list(MarkdownAwareChunker(max_tokens=1000, overlap=0).chunk(doc))
    heading_paths = [c.heading_path for c in chunks]
    assert heading_paths == [("Top",), ("Top", "Section A"), ("Top", "Section B")]


def test_oversized_section_splits_on_sentence_boundaries() -> None:
    body = " ".join(f"Sentence number {i} has several words." for i in range(200))
    doc = _doc(f"# Top\n\n{body}\n")
    chunks = list(MarkdownAwareChunker(max_tokens=60, overlap=10).chunk(doc))
    assert len(chunks) >= 2
    assert all(c.heading_path == ("Top",) for c in chunks)
    for c in chunks:
        assert c.text.rstrip().endswith(".")


def test_chunks_carry_doc_path_and_increasing_positions() -> None:
    doc = _doc("# Top\n\nA.\n\n## Sub\n\nB.\n")
    chunks = list(MarkdownAwareChunker(max_tokens=1000, overlap=0).chunk(doc))
    assert all(c.doc_path == doc.path for c in chunks)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_empty_section_emits_no_chunk() -> None:
    doc = _doc("# Top\n\n## Empty\n\n## Has content\n\nhello\n")
    chunks = list(MarkdownAwareChunker(max_tokens=1000, overlap=0).chunk(doc))
    paths = [c.heading_path for c in chunks]
    assert ("Top", "Empty") not in paths
    assert ("Top", "Has content") in paths


def test_context_header_prepends_source_and_heading():
    from pathlib import Path

    from rag_lab.chunkers.markdown_aware import MarkdownAwareChunker
    from rag_lab.types import Document

    doc = Document(
        path=Path("index.md"),
        text="# Intro\n\nTiTiler is a dynamic tile server.",
        metadata={"source": "developmentseed/titiler"},
    )
    chunk = next(MarkdownAwareChunker(context_header=True).chunk(doc))
    assert chunk.text.startswith("developmentseed/titiler > Intro\n\n")
    assert "TiTiler is a dynamic tile server." in chunk.text


def test_context_header_falls_back_to_filename():
    from pathlib import Path

    from rag_lab.chunkers.markdown_aware import MarkdownAwareChunker
    from rag_lab.types import Document

    doc = Document(path=Path("guide.md"), text="# Intro\n\nBody text.", metadata={})
    chunk = next(MarkdownAwareChunker(context_header=True).chunk(doc))
    assert chunk.text.startswith("guide.md > Intro\n\n")


def test_no_context_header_by_default():
    from pathlib import Path

    from rag_lab.chunkers.markdown_aware import MarkdownAwareChunker
    from rag_lab.types import Document

    doc = Document(path=Path("guide.md"), text="# Intro\n\nBody text.", metadata={})
    chunk = next(MarkdownAwareChunker().chunk(doc))
    assert chunk.text == "Body text."

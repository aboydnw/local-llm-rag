from pathlib import Path

from rag_lab.chunkers.semantic import SemanticChunker
from rag_lab.embedders.fake import FakeEmbedder
from rag_lab.types import Document


def _doc(text: str, **metadata: str) -> Document:
    return Document(path=Path("test.md"), text=text, metadata=metadata)


_THREE = "Alpha sentence here. Beta sentence here. Gamma sentence here."


def test_never_similar_threshold_splits_every_sentence():
    doc = _doc(_THREE)
    chunker = SemanticChunker(
        FakeEmbedder(16), max_tokens=1000, similarity_threshold=2.0, overlap=0
    )
    chunks = list(chunker.chunk(doc))
    assert len(chunks) == 3


def test_always_similar_threshold_merges_into_one_chunk():
    doc = _doc(_THREE)
    chunker = SemanticChunker(
        FakeEmbedder(16), max_tokens=1000, similarity_threshold=-2.0, overlap=0
    )
    chunks = list(chunker.chunk(doc))
    assert len(chunks) == 1


def test_max_tokens_cap_forces_a_split_even_when_similar():
    doc = _doc(_THREE)
    chunker = SemanticChunker(
        FakeEmbedder(16), max_tokens=6, similarity_threshold=-2.0, overlap=0
    )
    chunks = list(chunker.chunk(doc))
    assert len(chunks) >= 2


def test_overlap_repeats_last_sentence_of_previous_chunk():
    doc = _doc(_THREE)
    chunker = SemanticChunker(
        FakeEmbedder(16), max_tokens=1000, similarity_threshold=2.0, overlap=1
    )
    chunks = list(chunker.chunk(doc))
    assert "Beta sentence here." in chunks[1].text


def test_embed_documents_called_once_with_all_sentences():
    calls: list[list[str]] = []

    class SpyEmbedder(FakeEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            calls.append(list(texts))
            return super().embed_documents(texts)

    list(SemanticChunker(SpyEmbedder(16)).chunk(_doc(_THREE)))
    assert len(calls) == 1
    assert len(calls[0]) == 3


def test_context_header_prepends_document_label():
    doc = _doc(_THREE, source="developmentseed/titiler")
    chunk = next(SemanticChunker(FakeEmbedder(16), context_header=True).chunk(doc))
    assert chunk.text.startswith("developmentseed/titiler\n\n")


def test_empty_document_yields_no_chunks():
    assert list(SemanticChunker(FakeEmbedder(16)).chunk(_doc("   "))) == []


def test_larger_overlap_carries_more_sentences_forward():
    text = " ".join(f"Sentence {i} here." for i in range(5))
    small = list(
        SemanticChunker(
            FakeEmbedder(16), max_tokens=1000, similarity_threshold=2.0, overlap=1
        ).chunk(_doc(text))
    )
    large = list(
        SemanticChunker(
            FakeEmbedder(16), max_tokens=1000, similarity_threshold=2.0, overlap=40
        ).chunk(_doc(text))
    )
    small_words = sum(len(c.text.split()) for c in small)
    large_words = sum(len(c.text.split()) for c in large)
    assert large_words > small_words

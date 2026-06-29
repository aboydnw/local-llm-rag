from pathlib import Path

import pytest

from rag_lab import ingest
from rag_lab.chunkers.markdown_aware import MarkdownAwareChunker
from rag_lab.embedders.fake import FakeEmbedder
from rag_lab.loaders.markdown import MarkdownLoader
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.types import Document


class _StubAndProseLoader:
    def load(self):
        yield Document(path=Path("api.md"), text="# API\n\n::: titiler.core.errors\n")
        yield Document(path=Path("guide.md"), text="# Guide\n\nTiTiler serves map tiles.\n")


def test_ingest_end_to_end_with_fake_embedder(tmp_path: Path, fixture_corpus: Path) -> None:
    db_path = tmp_path / "rag.db"
    loader = MarkdownLoader(fixture_corpus)
    chunker = MarkdownAwareChunker(max_tokens=1000, overlap=0)
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(db_path, dimension=16)

    count = ingest.run(loader=loader, chunker=chunker, embedder=embedder, store=store)

    assert db_path.exists()
    assert count > 0
    assert store.count() == count


def test_run_skips_api_stub_chunks(tmp_path: Path) -> None:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    count = ingest.run(
        loader=_StubAndProseLoader(),
        chunker=MarkdownAwareChunker(max_tokens=512, overlap=50),
        embedder=FakeEmbedder(dimension=16),
        store=store,
    )
    assert count == 1
    assert store.count() == 1


class _MutableLoader:
    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs

    def load(self):
        yield from self.docs


def test_reingest_edited_document_removes_stale_chunks(tmp_path: Path) -> None:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    chunker = MarkdownAwareChunker(max_tokens=1000, overlap=0)
    embedder = FakeEmbedder(dimension=16)
    loader = _MutableLoader(
        [Document(path=Path("d.md"), text="# T\n\noriginal text about otters\n")]
    )

    ingest.run(loader=loader, chunker=chunker, embedder=embedder, store=store)
    loader.docs = [Document(path=Path("d.md"), text="# T\n\nrewritten text about badgers\n")]
    ingest.run(loader=loader, chunker=chunker, embedder=embedder, store=store)

    assert store.query_bm25("otters", k=5) == []
    assert len(store.query_bm25("badgers", k=5)) == 1
    assert store.count() == 1


def test_reingest_drops_a_removed_document(tmp_path: Path) -> None:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    chunker = MarkdownAwareChunker(max_tokens=1000, overlap=0)
    embedder = FakeEmbedder(dimension=16)
    a = Document(path=Path("a.md"), text="# A\n\napple content here\n")
    b = Document(path=Path("b.md"), text="# B\n\nbanana content here\n")
    loader = _MutableLoader([a, b])

    ingest.run(loader=loader, chunker=chunker, embedder=embedder, store=store)
    assert store.count() == 2
    loader.docs = [a]
    ingest.run(loader=loader, chunker=chunker, embedder=embedder, store=store)

    assert store.count() == 1
    assert store.query_bm25("banana", k=5) == []


def test_reingest_with_no_documents_keeps_existing_index(tmp_path: Path) -> None:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    chunker = MarkdownAwareChunker(max_tokens=1000, overlap=0)
    embedder = FakeEmbedder(dimension=16)
    loader = _MutableLoader([Document(path=Path("a.md"), text="# A\n\napple content here\n")])

    ingest.run(loader=loader, chunker=chunker, embedder=embedder, store=store)
    loader.docs = []
    ingest.run(loader=loader, chunker=chunker, embedder=embedder, store=store)

    assert store.count() == 1


def test_reingest_with_no_documents_keeps_original_manifest(tmp_path: Path) -> None:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    chunker = MarkdownAwareChunker(max_tokens=1000, overlap=0)
    embedder = FakeEmbedder(dimension=16)
    loader = _MutableLoader([Document(path=Path("a.md"), text="# A\n\napple content here\n")])
    original = {"schema_version": 1, "embedder_model": "real", "dimension": 16}

    ingest.run(loader=loader, chunker=chunker, embedder=embedder, store=store, manifest=original)
    loader.docs = []
    ingest.run(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        store=store,
        manifest={"schema_version": 1, "embedder_model": "other", "dimension": 16},
    )

    assert store.read_manifest() == original


class _ExplodingEmbedder:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.calls > 1:
            raise RuntimeError("embedder is down")
        return [[0.0] * 16 for _ in texts]


def test_reingest_keeps_existing_document_when_embedder_fails(tmp_path: Path) -> None:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    chunker = MarkdownAwareChunker(max_tokens=1000, overlap=0)
    embedder = _ExplodingEmbedder()
    loader = _MutableLoader([Document(path=Path("d.md"), text="# T\n\noriginal otters\n")])

    ingest.run(loader=loader, chunker=chunker, embedder=embedder, store=store)
    loader.docs = [Document(path=Path("d.md"), text="# T\n\nrewritten badgers\n")]
    with pytest.raises(RuntimeError):
        ingest.run(loader=loader, chunker=chunker, embedder=embedder, store=store)

    assert store.count() == 1
    assert len(store.query_bm25("otters", k=5)) == 1


def test_ingest_writes_manifest_when_provided(tmp_path: Path) -> None:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    chunker = MarkdownAwareChunker(max_tokens=1000, overlap=0)
    embedder = FakeEmbedder(dimension=16)
    loader = _MutableLoader([Document(path=Path("a.md"), text="# A\n\napple content here\n")])
    manifest = {"schema_version": 1, "embedder_model": "fake", "dimension": 16}

    ingest.run(loader=loader, chunker=chunker, embedder=embedder, store=store, manifest=manifest)

    assert store.read_manifest() == manifest


class _BadgeWallLoader:
    def load(self):
        from pathlib import Path

        from rag_lab.types import Document

        yield Document(
            path=Path("readme.md"),
            text=(
                '# Title\n\n<p align="center">\n<img src="logo.png"/>\n</p>\n\n'
                "TiTiler is a dynamic tile server.\n"
            ),
        )


def test_run_strips_markup_from_chunk_text(tmp_path):
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    total = ingest.run(
        loader=_BadgeWallLoader(),
        chunker=MarkdownAwareChunker(max_tokens=512, overlap=50),
        embedder=FakeEmbedder(dimension=16),
        store=store,
    )
    assert total == 1
    chunk = store.query_bm25("titiler", k=1)[0][0]
    assert "<img" not in chunk.text
    assert "TiTiler is a dynamic tile server." in chunk.text

from pathlib import Path

from rag_lab.embedders.fake import FakeEmbedder
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.types import Chunk


def _make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(
            text=f"chunk number {i}",
            doc_path=Path(f"docs/{i}.md"),
            heading_path=("Top", f"Sub {i}"),
            position=i,
        )
        for i in range(n)
    ]


def test_initialize_creates_schema(tmp_path: Path) -> None:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    assert (tmp_path / "rag.db").exists()


def test_upsert_then_count(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    chunks = _make_chunks(5)
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    assert store.count() == 5


def test_upsert_is_idempotent_on_same_chunks(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    chunks = _make_chunks(3)
    vectors = embedder.embed([c.text for c in chunks])
    store.upsert(chunks, vectors)
    store.upsert(chunks, vectors)
    assert store.count() == 3


def test_query_vector_returns_top_k(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    chunks = _make_chunks(10)
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    query_vec = embedder.embed(["chunk number 3"])[0]
    results = store.query_vector(query_vec, k=3)
    assert len(results) == 3


def test_query_bm25_returns_chunks_matching_keywords(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    chunks = [
        Chunk(text="titiler exposes a MosaicTilerFactory class", doc_path=Path("a.md"), heading_path=("Factory",), position=0),
        Chunk(text="stac-fastapi serves OGC API records", doc_path=Path("b.md"), heading_path=("Records",), position=0),
        Chunk(text="another unrelated paragraph about pizza", doc_path=Path("c.md"), heading_path=("Misc",), position=0),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    results = store.query_bm25("MosaicTilerFactory", k=2)
    assert len(results) >= 1
    assert results[0][0].text.startswith("titiler exposes")


def test_query_bm25_ranks_higher_for_more_keyword_matches(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    chunks = [
        Chunk(text="factory class factory pattern factory method", doc_path=Path("a.md"), heading_path=(), position=0),
        Chunk(text="single factory mention", doc_path=Path("b.md"), heading_path=(), position=0),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    results = store.query_bm25("factory", k=2)
    assert results[0][0].text.startswith("factory class")

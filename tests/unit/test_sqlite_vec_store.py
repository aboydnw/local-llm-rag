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


def test_list_documents_returns_distinct_sorted_paths(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    chunks = _make_chunks(3)
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    assert store.list_documents() == ["docs/0.md", "docs/1.md", "docs/2.md"]


def test_chunks_for_doc_returns_chunks_in_position_order(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    doc = Path("docs/guide.md")
    chunks = [
        Chunk(text="second", doc_path=doc, heading_path=("H",), position=1),
        Chunk(text="first", doc_path=doc, heading_path=("H",), position=0),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    got = store.chunks_for_doc("docs/guide.md")
    assert [c.text for c in got] == ["first", "second"]


def test_chunks_for_doc_unknown_path_returns_empty(tmp_path: Path) -> None:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    assert store.chunks_for_doc("nope.md") == []


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
        Chunk(text="titiler exposes a MosaicTilerFactory class", doc_path=Path("a.md"),
              heading_path=("Factory",), position=0),
        Chunk(text="stac-fastapi serves OGC API records", doc_path=Path("b.md"),
              heading_path=("Records",), position=0),
        Chunk(text="another unrelated paragraph about pizza", doc_path=Path("c.md"),
              heading_path=("Misc",), position=0),
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
        Chunk(text="factory class factory pattern factory method", doc_path=Path("a.md"),
              heading_path=(), position=0),
        Chunk(text="single factory mention", doc_path=Path("b.md"), heading_path=(), position=0),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    results = store.query_bm25("factory", k=2)
    assert results[0][0].text.startswith("factory class")


def test_query_bm25_handles_natural_language_with_punctuation(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    chunks = [
        Chunk(text="the alpha document explains setup", doc_path=Path("a.md"),
              heading_path=(), position=0),
        Chunk(text="unrelated beta content", doc_path=Path("b.md"), heading_path=(), position=0),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    results = store.query_bm25("what is the alpha document about?", k=2)
    assert results[0][0].text.startswith("the alpha document")


def test_delete_by_doc_removes_only_that_docs_chunks(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    chunks = [
        Chunk(text="alpha one", doc_path=Path("a.md"), heading_path=(), position=0),
        Chunk(text="alpha two", doc_path=Path("a.md"), heading_path=(), position=1),
        Chunk(text="beta keyword", doc_path=Path("b.md"), heading_path=(), position=0),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    store.delete_by_doc(Path("a.md"))
    assert store.count() == 1
    assert store.query_bm25("alpha", k=5) == []
    assert len(store.query_bm25("beta", k=5)) == 1


def test_delete_by_doc_also_clears_vector_rows(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    chunks = [
        Chunk(text="alpha one", doc_path=Path("a.md"), heading_path=(), position=0),
        Chunk(text="beta one", doc_path=Path("b.md"), heading_path=(), position=0),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    store.delete_by_doc(Path("a.md"))
    results = store.query_vector(embedder.embed(["alpha one"])[0], k=5)
    assert {str(c.doc_path) for c, _ in results} == {"b.md"}


def test_prune_docs_keeps_listed_docs_and_drops_the_rest(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    chunks = [
        Chunk(text="alpha", doc_path=Path("a.md"), heading_path=(), position=0),
        Chunk(text="beta", doc_path=Path("b.md"), heading_path=(), position=0),
        Chunk(text="gamma", doc_path=Path("c.md"), heading_path=(), position=0),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    store.prune_docs(keep={"a.md", "c.md"})
    assert store.count() == 2
    assert store.query_bm25("beta", k=5) == []


def test_write_manifest_then_read_roundtrips(tmp_path: Path) -> None:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    manifest = {"schema_version": 1, "embedder_model": "nomic-embed-text", "dimension": 768}
    store.write_manifest(manifest)
    assert store.read_manifest() == manifest


def test_read_manifest_is_empty_when_nothing_written(tmp_path: Path) -> None:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    assert store.read_manifest() == {}


def test_query_bm25_returns_empty_for_query_with_no_word_tokens(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    chunks = [Chunk(text="some content", doc_path=Path("a.md"), heading_path=(), position=0)]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    assert store.query_bm25("?!...", k=2) == []

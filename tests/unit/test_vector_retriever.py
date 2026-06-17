from pathlib import Path

from rag_lab.embedders.fake import FakeEmbedder
from rag_lab.retrievers.vector import VectorRetriever
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.types import Chunk


def test_vector_retriever_returns_results_tagged_vector(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    chunks = [
        Chunk(text="alpha", doc_path=Path("a.md"), heading_path=(), position=0),
        Chunk(text="beta", doc_path=Path("b.md"), heading_path=(), position=0),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = VectorRetriever(store=store, embedder=embedder)
    results = retriever.retrieve("alpha", k=2)
    assert len(results) == 2
    assert all(r.source == "vector" for r in results)
    assert results[0].chunk.text == "alpha"

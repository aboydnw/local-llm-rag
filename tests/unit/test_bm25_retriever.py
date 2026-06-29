from pathlib import Path

from rag_lab.embedders.fake import FakeEmbedder
from rag_lab.retrievers.bm25 import BM25Retriever
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.types import Chunk


def test_bm25_retriever_returns_results_tagged_bm25(tmp_path: Path) -> None:
    embedder = FakeEmbedder(dimension=16)
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    chunks = [
        Chunk(text="MosaicTilerFactory subclass example", doc_path=Path("a.md"),
              heading_path=(), position=0),
        Chunk(text="unrelated content about pizza", doc_path=Path("b.md"),
              heading_path=(), position=0),
    ]
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = BM25Retriever(store=store)
    results = retriever.retrieve("MosaicTilerFactory", k=2)
    assert len(results) >= 1
    assert results[0].source == "bm25"
    assert "MosaicTilerFactory" in results[0].chunk.text

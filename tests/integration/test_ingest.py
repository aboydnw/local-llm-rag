from pathlib import Path

from rag_lab import ingest
from rag_lab.chunkers.markdown_aware import MarkdownAwareChunker
from rag_lab.embedders.fake import FakeEmbedder
from rag_lab.loaders.markdown import MarkdownLoader
from rag_lab.store.sqlite_vec import SqliteVecStore


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

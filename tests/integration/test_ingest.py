from pathlib import Path

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

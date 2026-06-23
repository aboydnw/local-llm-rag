from rag_lab.chunkers.base import Chunker
from rag_lab.embedders.base import Embedder
from rag_lab.filters import is_api_stub
from rag_lab.loaders.base import Loader
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.types import Chunk


def run(
    loader: Loader,
    chunker: Chunker,
    embedder: Embedder,
    store: SqliteVecStore,
    batch_size: int = 64,
) -> int:
    store.initialize()
    batch: list[Chunk] = []
    total = 0
    for document in loader.load():
        for chunk in chunker.chunk(document):
            if is_api_stub(chunk.text):
                continue
            batch.append(chunk)
            if len(batch) >= batch_size:
                total += _flush(batch, embedder, store)
                batch = []
    if batch:
        total += _flush(batch, embedder, store)
    return total


def _flush(batch: list[Chunk], embedder: Embedder, store: SqliteVecStore) -> int:
    vectors = embedder.embed([c.text for c in batch])
    store.upsert(batch, vectors)
    return len(batch)

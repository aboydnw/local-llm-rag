from dataclasses import replace

from rag_lab.chunkers.base import Chunker
from rag_lab.embedders.base import Embedder
from rag_lab.filters import is_api_stub, strip_markup
from rag_lab.loaders.base import Loader
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.types import Chunk


def run(
    loader: Loader,
    chunker: Chunker,
    embedder: Embedder,
    store: SqliteVecStore,
    batch_size: int = 64,
    manifest: dict | None = None,
) -> int:
    store.initialize()
    batch: list[Chunk] = []
    cleared: set[str] = set()
    seen: set[str] = set()
    total = 0
    for document in loader.load():
        for chunk in chunker.chunk(document):
            cleaned = strip_markup(chunk.text)
            if not cleaned or is_api_stub(cleaned):
                continue
            seen.add(str(chunk.doc_path))
            batch.append(replace(chunk, text=cleaned))
            if len(batch) >= batch_size:
                total += _flush(batch, embedder, store, cleared)
                batch = []
    if batch:
        total += _flush(batch, embedder, store, cleared)
    if total > 0:
        store.prune_docs(keep=seen)
    if manifest is not None:
        store.write_manifest(manifest)
    return total


def _flush(
    batch: list[Chunk], embedder: Embedder, store: SqliteVecStore, cleared: set[str]
) -> int:
    for doc in {str(c.doc_path) for c in batch} - cleared:
        store.delete_by_doc(doc)
        cleared.add(doc)
    vectors = embedder.embed_documents([c.text for c in batch])
    store.upsert(batch, vectors)
    return len(batch)

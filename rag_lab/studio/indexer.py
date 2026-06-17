import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from rag_lab import ingest as ingest_mod
from rag_lab.chunkers.markdown_aware import MarkdownAwareChunker
from rag_lab.config import Config
from rag_lab.loaders.markdown import MarkdownLoader
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.studio import components
from rag_lab.studio.workspace import Workspace


@dataclass(frozen=True)
class IndexStatus:
    cache_key: str
    cached: bool
    needs_build: bool


def cache_key(corpus: str, config: Config) -> str:
    payload = {
        "corpus": corpus,
        "chunker": config.chunker.model_dump(),
        "embedder": config.embedder.model_dump(),
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def status(workspace: Workspace, corpus: str, config: Config) -> IndexStatus:
    key = cache_key(corpus, config)
    cached = workspace.index_db(key).exists()
    return IndexStatus(cache_key=key, cached=cached, needs_build=not cached)


def _default_loader(corpus: str) -> MarkdownLoader:
    path = Path(corpus)
    if not path.is_dir():
        raise ValueError(f"corpus is not a local directory: {corpus}")
    return MarkdownLoader(path)


def build_index(
    workspace: Workspace,
    corpus: str,
    config: Config,
    *,
    loader=None,
    embedder=None,
) -> Path:
    if embedder is None:
        embedder = components.build_embedder(config)
    if loader is None:
        loader = _default_loader(corpus)
    chunker = MarkdownAwareChunker(
        max_tokens=config.chunker.max_tokens, overlap=config.chunker.overlap
    )
    key = cache_key(corpus, config)
    db_path = workspace.index_db(key)
    store = SqliteVecStore(db_path, dimension=embedder.dimension)
    ingest_mod.run(loader=loader, chunker=chunker, embedder=embedder, store=store)
    workspace.index_meta(key).write_text(
        json.dumps(
            {
                "corpus": corpus,
                "chunker": config.chunker.model_dump(),
                "embedder": config.embedder.model_dump(),
            },
            indent=2,
        )
    )
    return db_path


def ensure_index(
    workspace: Workspace,
    corpus: str,
    config: Config,
    *,
    loader=None,
    embedder=None,
) -> Path:
    key = cache_key(corpus, config)
    db_path = workspace.index_db(key)
    if db_path.exists():
        return db_path
    return build_index(workspace, corpus, config, loader=loader, embedder=embedder)

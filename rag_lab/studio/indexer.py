import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from rag_lab import ingest as ingest_mod
from rag_lab import pipeline
from rag_lab.config import Config
from rag_lab.loaders.combined import CombinedLoader
from rag_lab.loaders.github import GitHubLoader
from rag_lab.loaders.github_issues import GitHubIssuesLoader
from rag_lab.loaders.markdown import MarkdownLoader
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.studio import components
from rag_lab.studio.corpora import Corpus
from rag_lab.studio.workspace import Workspace


@dataclass(frozen=True)
class IndexStatus:
    cache_key: str
    cached: bool
    needs_build: bool


def cache_key(corpus: Corpus, config: Config) -> str:
    sources = sorted(
        (s.to_dict() for s in corpus.sources),
        key=lambda d: json.dumps(d, sort_keys=True),
    )
    payload = {
        "sources": sources,
        "chunker": config.chunker.model_dump(),
        "embedder": config.embedder.model_dump(),
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def delete_indexes_for_corpus(workspace: Workspace, corpus: Corpus) -> int:
    """Delete every cached index variant whose metadata names ``corpus``."""
    deleted = 0
    for metadata_path in workspace.indexes_dir.glob("*.json"):
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError):
            continue
        if metadata.get("corpus", {}).get("name") != corpus.name:
            continue
        for artifact in workspace.indexes_dir.glob(f"{metadata_path.stem}.*"):
            artifact.unlink(missing_ok=True)
        deleted += 1
    return deleted


def status(workspace: Workspace, corpus: Corpus, config: Config) -> IndexStatus:
    key = cache_key(corpus, config)
    cached = workspace.index_db(key).exists()
    return IndexStatus(cache_key=key, cached=cached, needs_build=not cached)


def validate_corpus(corpus: str) -> str | None:
    """Return a human-readable reason the corpus can't be indexed, or ``None`` if it can."""
    corpus = corpus.strip()
    if not corpus:
        return "Enter a corpus directory to index."
    path = Path(corpus)
    if not path.exists():
        return f"Corpus directory not found: {corpus}"
    if not path.is_dir():
        return f"Corpus must be a directory, not a file: {corpus}"
    if not any(path.rglob("*.md")):
        return f"No markdown (.md) files found in: {corpus}"
    return None


def loader_for_corpus(workspace: Workspace, corpus: Corpus) -> CombinedLoader:
    loaders = []
    for source in corpus.sources:
        if source.type == "github":
            if not source.repo:
                raise ValueError("GitHub source is missing 'repo'")
            clone_into = workspace.clone_dir(source.repo.replace("/", "__"))
            loaders.append(GitHubLoader(source.repo, clone_into, private=source.private))
        elif source.type == "github_issue":
            if not source.repo or source.issue is None:
                raise ValueError("GitHub issue source is missing 'repo' or 'issue'")
            loaders.append(GitHubIssuesLoader(source.repo, [source.issue]))
        elif source.type == "local":
            if not source.path:
                raise ValueError("Local source is missing 'path'")
            path = Path(source.path)
            if not path.is_dir():
                raise ValueError(f"Local source directory not found: {source.path}")
            loaders.append(MarkdownLoader(path))
        else:
            raise ValueError(f"unsupported source type: {source.type!r}")
    return CombinedLoader(loaders)


def build_index(
    workspace: Workspace,
    corpus: Corpus,
    config: Config,
    *,
    loader=None,
    embedder=None,
) -> Path:
    if embedder is None:
        embedder = components.build_embedder(config)
    if loader is None:
        loader = loader_for_corpus(workspace, corpus)
    chunker = pipeline.build_chunker(config, embedder=embedder)
    key = cache_key(corpus, config)
    db_path = workspace.index_db(key)
    store = SqliteVecStore(db_path, dimension=embedder.dimension)
    manifest = pipeline.index_manifest(config)
    manifest["dimension"] = embedder.dimension
    ingest_mod.run(
        loader=loader,
        chunker=chunker,
        embedder=embedder,
        store=store,
        manifest=manifest,
    )
    workspace.index_meta(key).write_text(
        json.dumps(
            {
                "corpus": corpus.to_dict(),
                "chunker": config.chunker.model_dump(),
                "embedder": config.embedder.model_dump(),
            },
            indent=2,
        )
    )
    return db_path


def ensure_index(
    workspace: Workspace,
    corpus: Corpus,
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

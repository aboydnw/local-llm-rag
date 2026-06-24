import json
import re
from dataclasses import dataclass

from rag_lab.studio.workspace import Workspace

_LOCAL_NAME = "__local__"
_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class Source:
    """A single ingestible source within a corpus."""

    type: str
    repo: str | None = None
    path: str | None = None
    private: bool = False

    def to_dict(self) -> dict:
        if self.type == "github":
            d: dict = {"type": "github", "repo": self.repo or ""}
            if self.private:
                d["private"] = True
            return d
        if self.type == "local":
            return {"type": "local", "path": self.path or ""}
        raise ValueError(f"unsupported source type: {self.type!r}")

    @classmethod
    def from_dict(cls, data: dict) -> "Source":
        if data["type"] == "github":
            return cls(
                type="github",
                repo=data["repo"],
                private=bool(data.get("private", False)),
            )
        if data["type"] == "local":
            return cls(type="local", path=data["path"])
        raise ValueError(f"unsupported source type: {data['type']!r}")


@dataclass(frozen=True)
class Corpus:
    """A named set of sources that build into one index."""

    name: str
    sources: tuple[Source, ...]

    @property
    def label(self) -> str:
        if self.name == _LOCAL_NAME and self.sources:
            return self.sources[0].path or _LOCAL_NAME
        return self.name

    def to_dict(self) -> dict:
        return {"name": self.name, "sources": [s.to_dict() for s in self.sources]}

    @classmethod
    def from_dict(cls, data: dict) -> "Corpus":
        return cls(
            name=data["name"],
            sources=tuple(Source.from_dict(s) for s in data["sources"]),
        )


def local_corpus(path: str) -> Corpus:
    """Wrap a single local folder path as an unsaved corpus."""
    return Corpus(name=_LOCAL_NAME, sources=(Source(type="local", path=path),))


def validate_name(name: str) -> str | None:
    """Return a reason ``name`` is unusable as a corpus name, or ``None``."""
    name = name.strip()
    if not name:
        return "Enter a corpus name."
    if name == _LOCAL_NAME:
        return f"'{_LOCAL_NAME}' is reserved."
    if name in (".", "..") or "/" in name or "\\" in name:
        return f"Invalid corpus name: {name}"
    return None


def validate_repo(repo: str) -> str | None:
    """Return a reason ``repo`` is not a valid ``owner/name``, or ``None``."""
    if not _REPO_RE.match(repo.strip()):
        return "Enter a GitHub repo as owner/name, e.g. developmentseed/titiler."
    return None


def list_corpora(ws: Workspace) -> list[str]:
    """Return saved corpus names, sorted, excluding the unsaved local wrapper."""
    if not ws.corpora_dir.is_dir():
        return []
    names = [p.stem for p in ws.corpora_dir.glob("*.json")]
    return sorted(n for n in names if n != _LOCAL_NAME)


def load_corpus(ws: Workspace, name: str) -> Corpus:
    """Load a saved corpus definition by name."""
    data = json.loads(ws.corpus_file(name).read_text())
    return Corpus.from_dict(data)


def save_corpus(ws: Workspace, corpus: Corpus) -> None:
    """Persist a corpus definition to the workspace."""
    ws.corpora_dir.mkdir(parents=True, exist_ok=True)
    ws.corpus_file(corpus.name).write_text(json.dumps(corpus.to_dict(), indent=2))


def delete_corpus(ws: Workspace, name: str) -> None:
    """Remove a saved corpus definition; no-op if absent."""
    ws.corpus_file(name).unlink(missing_ok=True)


def add_source(corpus: Corpus, source: Source) -> Corpus:
    """Return a copy of ``corpus`` with ``source`` appended if not already present."""
    if source in corpus.sources:
        return corpus
    return Corpus(name=corpus.name, sources=corpus.sources + (source,))


def remove_source(corpus: Corpus, source: Source) -> Corpus:
    """Return a copy of ``corpus`` without ``source``."""
    return Corpus(
        name=corpus.name,
        sources=tuple(s for s in corpus.sources if s != source),
    )


def resolve_active_corpus(ws: Workspace, corpus_name: str | None, local_path: str) -> Corpus:
    """Return the named corpus if selected and present, else the local-folder fallback."""
    if corpus_name and ws.corpus_file(corpus_name).exists():
        return load_corpus(ws, corpus_name)
    return local_corpus(local_path)

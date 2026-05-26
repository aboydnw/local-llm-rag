from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    """A loaded source document before chunking."""

    path: Path
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit produced by a Chunker."""

    text: str
    doc_path: Path
    heading_path: tuple[str, ...]
    position: int
    metadata: dict[str, str] = field(default_factory=dict)

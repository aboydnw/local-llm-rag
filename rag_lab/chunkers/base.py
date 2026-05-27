from collections.abc import Iterator
from typing import Protocol

from rag_lab.types import Chunk, Document


class Chunker(Protocol):
    def chunk(self, document: Document) -> Iterator[Chunk]: ...

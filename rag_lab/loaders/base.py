from collections.abc import Iterator
from typing import Protocol

from rag_lab.types import Document


class Loader(Protocol):
    def load(self) -> Iterator[Document]: ...

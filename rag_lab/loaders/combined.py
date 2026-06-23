from collections.abc import Iterator

from rag_lab.loaders.base import Loader
from rag_lab.types import Document


class CombinedLoader:
    """Chain several loaders, yielding every document from each in turn."""

    def __init__(self, loaders: list[Loader]) -> None:
        self.loaders = loaders

    def load(self) -> Iterator[Document]:
        for loader in self.loaders:
            yield from loader.load()

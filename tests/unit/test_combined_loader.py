from pathlib import Path

from rag_lab.loaders.combined import CombinedLoader
from rag_lab.types import Document


class _FakeLoader:
    def __init__(self, docs):
        self._docs = docs

    def load(self):
        yield from self._docs


def test_combined_loader_yields_all_docs_in_order():
    a = Document(path=Path("a.md"), text="A")
    b = Document(path=Path("b.md"), text="B")
    c = Document(path=Path("c.md"), text="C")
    combined = CombinedLoader([_FakeLoader([a, b]), _FakeLoader([c])])
    assert list(combined.load()) == [a, b, c]


def test_combined_loader_with_no_loaders_yields_nothing():
    assert list(CombinedLoader([]).load()) == []

import pytest

from rag_lab.embedders.ollama import OllamaEmbedder, prefixes_for_model


class _FakeOllamaClient:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def embed(self, model: str, input: list[str]) -> dict:
        self.calls.append(list(input))
        return {"embeddings": [[float(i)] * self.dimension for i, _ in enumerate(input)]}


def test_embed_returns_one_vector_per_input(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeOllamaClient(dimension=8)
    monkeypatch.setattr("rag_lab.embedders.ollama._client", lambda: fake)
    embedder = OllamaEmbedder(model="nomic-embed-text", dimension=8)
    vectors = embedder.embed(["one", "two", "three"])
    assert len(vectors) == 3
    assert all(len(v) == 8 for v in vectors)


def test_embed_passes_model_name_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeOllamaClient(dimension=4)
    seen: dict[str, str] = {}

    def _embed(model: str, input: list[str]) -> dict:
        seen["model"] = model
        return fake.embed(model, input)

    client = type("C", (), {"embed": staticmethod(_embed)})()
    monkeypatch.setattr("rag_lab.embedders.ollama._client", lambda: client)
    embedder = OllamaEmbedder(model="some-model", dimension=4)
    embedder.embed(["hi"])
    assert seen["model"] == "some-model"


def test_dimension_is_exposed() -> None:
    embedder = OllamaEmbedder(model="x", dimension=12)
    assert embedder.dimension == 12


def test_prefixes_for_nomic():
    assert prefixes_for_model("nomic-embed-text") == ("search_document: ", "search_query: ")


def test_prefixes_for_unknown_model_are_empty():
    assert prefixes_for_model("some-other-model") == ("", "")


def test_embed_documents_adds_document_prefix(monkeypatch):
    e = OllamaEmbedder("nomic-embed-text", 768, document_prefix="search_document: ", query_prefix="search_query: ")
    seen = {}
    monkeypatch.setattr(e, "embed", lambda texts: seen.setdefault("texts", texts) or [[0.0]] * len(texts))
    e.embed_documents(["hello", "world"])
    assert seen["texts"] == ["search_document: hello", "search_document: world"]


def test_embed_query_adds_query_prefix(monkeypatch):
    e = OllamaEmbedder("nomic-embed-text", 768, document_prefix="search_document: ", query_prefix="search_query: ")
    seen = {}
    monkeypatch.setattr(e, "embed", lambda texts: seen.setdefault("texts", texts) or [[0.0]] * len(texts))
    e.embed_query("hello")
    assert seen["texts"] == ["search_query: hello"]

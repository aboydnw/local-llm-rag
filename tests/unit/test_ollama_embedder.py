import pytest

from rag_lab.embedders.ollama import OllamaEmbedder


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

    monkeypatch.setattr("rag_lab.embedders.ollama._client", lambda: type("C", (), {"embed": staticmethod(_embed)})())
    embedder = OllamaEmbedder(model="some-model", dimension=4)
    embedder.embed(["hi"])
    assert seen["model"] == "some-model"


def test_dimension_is_exposed() -> None:
    embedder = OllamaEmbedder(model="x", dimension=12)
    assert embedder.dimension == 12

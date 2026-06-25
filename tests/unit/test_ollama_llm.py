import pytest

from rag_lab.llms.ollama import OllamaLLM


def test_generate_returns_text_from_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chat(model: str, messages: list[dict], stream: bool = False):
        for piece in ["echo: ", messages[-1]["content"]]:
            yield {"message": {"content": piece}}

    monkeypatch.setattr(
        "rag_lab.llms.ollama._client",
        lambda: type("C", (), {"chat": staticmethod(fake_chat)})(),
    )
    llm = OllamaLLM(model="llama3.2:3b")
    out = llm.generate("hello world")
    assert out == "echo: hello world"


def test_stream_yields_content_pieces_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chat(model: str, messages: list[dict], stream: bool = False):
        assert stream is True
        for piece in ["Titiler ", "is a ", "tile server"]:
            yield {"message": {"content": piece}}

    monkeypatch.setattr(
        "rag_lab.llms.ollama._client",
        lambda: type("C", (), {"chat": staticmethod(fake_chat)})(),
    )
    llm = OllamaLLM(model="llama3.2:3b")
    assert list(llm.stream("what is titiler?")) == ["Titiler ", "is a ", "tile server"]


def test_stream_skips_empty_pieces(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chat(model: str, messages: list[dict], stream: bool = False):
        for piece in ["a", "", "b", ""]:
            yield {"message": {"content": piece}}

    monkeypatch.setattr(
        "rag_lab.llms.ollama._client",
        lambda: type("C", (), {"chat": staticmethod(fake_chat)})(),
    )
    llm = OllamaLLM(model="llama3.2:3b")
    assert list(llm.stream("q")) == ["a", "b"]

import pytest

from rag_lab.llms.ollama import OllamaLLM


def test_generate_returns_text_from_client(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chat(model: str, messages: list[dict]) -> dict:
        return {"message": {"content": f"echo: {messages[-1]['content']}"}}

    monkeypatch.setattr(
        "rag_lab.llms.ollama._client",
        lambda: type("C", (), {"chat": staticmethod(fake_chat)})(),
    )
    llm = OllamaLLM(model="llama3.2:3b")
    out = llm.generate("hello world")
    assert out == "echo: hello world"

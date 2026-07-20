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


def _chat_with_final_metadata(model: str, messages: list[dict], stream: bool = False):
    yield {"message": {"content": "Titiler "}, "done": False}
    yield {"message": {"content": "serves tiles"}, "done": False}
    yield {
        "message": {"content": ""},
        "done": True,
        "prompt_eval_count": 2841,
        "prompt_eval_duration": 69_000_000_000,
        "eval_count": 214,
        "eval_duration": 31_400_000_000,
    }


def test_stream_captures_stats_from_final_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rag_lab.llms.ollama._client",
        lambda: type("C", (), {"chat": staticmethod(_chat_with_final_metadata)})(),
    )
    llm = OllamaLLM(model="llama3.2:3b")
    assert llm.last_stats() is None
    list(llm.stream("q"))
    stats = llm.last_stats()
    assert stats is not None
    assert stats.prompt_tokens == 2841
    assert stats.prompt_eval_ms == pytest.approx(69_000.0)
    assert stats.output_tokens == 214
    assert stats.generation_ms == pytest.approx(31_400.0)


def test_generate_inherits_stats_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rag_lab.llms.ollama._client",
        lambda: type("C", (), {"chat": staticmethod(_chat_with_final_metadata)})(),
    )
    llm = OllamaLLM(model="llama3.2:3b")
    assert llm.generate("q") == "Titiler serves tiles"
    assert llm.last_stats() is not None


def test_missing_metadata_leaves_stats_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_chat(model: str, messages: list[dict], stream: bool = False):
        yield {"message": {"content": "hi"}, "done": False}
        yield {"message": {"content": ""}, "done": True}

    monkeypatch.setattr(
        "rag_lab.llms.ollama._client",
        lambda: type("C", (), {"chat": staticmethod(fake_chat)})(),
    )
    llm = OllamaLLM(model="llama3.2:3b")
    assert llm.generate("q") == "hi"
    assert llm.last_stats() is None


def test_think_setting_is_passed_to_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_chat(model: str, messages: list[dict], stream: bool = False, **kwargs):
        seen.update(kwargs)
        yield {"message": {"content": "hi"}, "done": True}

    monkeypatch.setattr(
        "rag_lab.llms.ollama._client",
        lambda: type("C", (), {"chat": staticmethod(fake_chat)})(),
    )
    OllamaLLM(model="qwen3:4b", think=False).generate("q")
    assert seen["think"] is False


def test_think_none_is_omitted_from_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_chat(model: str, messages: list[dict], stream: bool = False, **kwargs):
        seen.update(kwargs)
        yield {"message": {"content": "hi"}, "done": True}

    monkeypatch.setattr(
        "rag_lab.llms.ollama._client",
        lambda: type("C", (), {"chat": staticmethod(fake_chat)})(),
    )
    OllamaLLM(model="llama3.2:3b").generate("q")
    assert "think" not in seen


def test_new_stream_resets_previous_stats(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def fake_chat(model: str, messages: list[dict], stream: bool = False):
        calls["n"] += 1
        if calls["n"] == 1:
            yield from _chat_with_final_metadata(model, messages, stream)
        else:
            yield {"message": {"content": "no metadata"}, "done": True}

    monkeypatch.setattr(
        "rag_lab.llms.ollama._client",
        lambda: type("C", (), {"chat": staticmethod(fake_chat)})(),
    )
    llm = OllamaLLM(model="llama3.2:3b")
    llm.generate("q1")
    assert llm.last_stats() is not None
    llm.generate("q2")
    assert llm.last_stats() is None

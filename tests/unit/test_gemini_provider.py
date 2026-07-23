import sys
from types import ModuleType, SimpleNamespace

import pytest

from rag_lab import pipeline
from rag_lab.config import Config, eval_judge
from rag_lab.llms.gemini import GeminiLLM


def test_legacy_llm_type_loads_as_provider() -> None:
    cfg = Config(llm={"type": "ollama", "model": "llama3.2:3b"})
    assert cfg.llm.provider == "ollama"


def test_build_llm_selects_gemini() -> None:
    cfg = Config(llm={"provider": "gemini", "model": "gemini-2.5-flash-lite"})
    llm = pipeline.build_llm(cfg)
    assert isinstance(llm, GeminiLLM)
    assert llm.model == "gemini-2.5-flash-lite"


def test_eval_judge_reuses_answer_model_for_same_provider() -> None:
    cfg = Config(
        llm={"provider": "gemini", "model": "gemini-2.5-flash"},
        eval={"deepeval": True},
    )
    assert eval_judge(cfg) == ("gemini", "gemini-2.5-flash")


def test_eval_judge_uses_provider_default_when_provider_differs() -> None:
    cfg = Config(eval={"deepeval": True, "judge_provider": "gemini"})
    assert eval_judge(cfg) == ("gemini", "gemini-2.5-flash-lite")


def test_eval_judge_honors_explicit_model() -> None:
    cfg = Config(
        eval={
            "deepeval": True,
            "judge_provider": "gemini",
            "deepeval_model": "gemini-2.5-flash",
        }
    )
    assert eval_judge(cfg) == ("gemini", "gemini-2.5-flash")


def test_gemini_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_lab.llms import gemini

    gemini._client.cache_clear()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini, "find_dotenv", lambda **kwargs: "")
    with pytest.raises(RuntimeError):
        gemini._client()
    gemini._client.cache_clear()


def test_gemini_stream_yields_text_and_records_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rag_lab.llms import gemini

    captured: dict = {}

    class GenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            captured["config"] = kwargs

    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    genai_module.types = SimpleNamespace(GenerateContentConfig=GenerateContentConfig)
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    chunks = [
        SimpleNamespace(text="hello ", usage_metadata=None),
        SimpleNamespace(
            text="world",
            usage_metadata=SimpleNamespace(prompt_token_count=7, candidates_token_count=2),
        ),
    ]

    class Models:
        def generate_content_stream(self, **kwargs):
            captured["request"] = kwargs
            return iter(chunks)

    monkeypatch.setattr(gemini, "_client", lambda: SimpleNamespace(models=Models()))
    llm = GeminiLLM("gemini-test")
    schema = {"type": "object"}

    assert "".join(llm.stream("prompt", schema)) == "hello world"
    assert captured["request"]["model"] == "gemini-test"
    assert captured["config"] == {
        "response_mime_type": "application/json",
        "response_json_schema": schema,
    }
    stats = llm.last_stats()
    assert stats is not None
    assert stats.prompt_tokens == 7
    assert stats.output_tokens == 2


def test_gemini_stream_wraps_provider_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_lab.llms import gemini

    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    genai_module.types = SimpleNamespace(GenerateContentConfig=object)
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)

    class Models:
        def generate_content_stream(self, **kwargs):
            raise TimeoutError("provider timed out")

    monkeypatch.setattr(
        gemini,
        "_client",
        lambda: SimpleNamespace(models=Models()),
    )

    with pytest.raises(RuntimeError) as exc_info:
        list(GeminiLLM("gemini-test").stream("prompt"))

    assert isinstance(exc_info.value.__cause__, TimeoutError)

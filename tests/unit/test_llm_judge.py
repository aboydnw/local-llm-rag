import pytest

from rag_lab.eval.scorers.llm_judge import LLMJudge, OllamaJudge


def test_judge_extracts_integer_score_from_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMessages:
        def create(self, **kwargs) -> object:
            class Block:
                text = "SCORE: 4\nReason: mostly accurate, missed one nuance."
            class Resp:
                def __init__(self) -> None:
                    self.content = [Block()]
            return Resp()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr("rag_lab.eval.scorers.llm_judge._client", lambda: FakeClient())
    judge = LLMJudge(model="claude-sonnet-4-6")
    result = judge.score(question="q", actual_answer="a", ideal_answer="i")
    assert result.score == 4
    assert "mostly accurate" in result.reason


def test_judge_returns_zero_when_response_has_no_score(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMessages:
        def create(self, **kwargs) -> object:
            class Block:
                text = "no score here"
            class Resp:
                def __init__(self) -> None:
                    self.content = [Block()]
            return Resp()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr("rag_lab.eval.scorers.llm_judge._client", lambda: FakeClient())
    judge = LLMJudge(model="claude-sonnet-4-6")
    result = judge.score(question="q", actual_answer="a", ideal_answer="i")
    assert result.score == 0


def test_judge_handles_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeMessages:
        def create(self, **kwargs) -> object:
            class Resp:
                def __init__(self) -> None:
                    self.content = []
            return Resp()

    class FakeClient:
        messages = FakeMessages()

    monkeypatch.setattr("rag_lab.eval.scorers.llm_judge._client", lambda: FakeClient())
    judge = LLMJudge(model="claude-sonnet-4-6")
    result = judge.score(question="q", actual_answer="a", ideal_answer="i")
    assert result.score == 0


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.reply


def test_ollama_judge_scores_from_llm_output() -> None:
    llm = _FakeLLM("SCORE: 5\nReason: accurate and complete.")
    result = OllamaJudge(llm).score(question="q", actual_answer="a", ideal_answer="i")
    assert result.score == 5
    assert "accurate and complete" in result.reason


def test_ollama_judge_returns_zero_when_no_score() -> None:
    llm = _FakeLLM("the model rambled without a score")
    result = OllamaJudge(llm).score(question="q", actual_answer="a", ideal_answer="i")
    assert result.score == 0


def test_ollama_judge_passes_all_fields_into_prompt() -> None:
    llm = _FakeLLM("SCORE: 3\nReason: ok.")
    OllamaJudge(llm).score(question="Qx", actual_answer="Ax", ideal_answer="Ix")
    assert "Qx" in llm.prompt
    assert "Ax" in llm.prompt
    assert "Ix" in llm.prompt

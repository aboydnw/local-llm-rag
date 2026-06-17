import pytest

from rag_lab.eval.scorers.llm_judge import LLMJudge


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

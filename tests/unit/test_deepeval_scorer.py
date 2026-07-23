import math

import pytest


def test_module_imports_without_constructing_scorer():
    import importlib

    module = importlib.import_module("rag_lab.eval.scorers.deepeval_scorer")
    assert hasattr(module, "DeepEvalScorer")


def test_scorer_sets_nan_when_metric_raises(monkeypatch):
    pytest.importorskip("deepeval")
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric

    from rag_lab.eval.scorers.deepeval_scorer import DeepEvalScorer

    def _boom(self, test_case):
        raise RuntimeError("boom")

    monkeypatch.setattr(AnswerRelevancyMetric, "measure", _boom)
    monkeypatch.setattr(FaithfulnessMetric, "measure", _boom)
    scorer = DeepEvalScorer(model="llama3.2:3b")
    scores = scorer.score(
        question="What is the capital of France?",
        answer="Paris.",
        retrieval_context=["The capital of France is Paris."],
    )
    assert math.isnan(scores["answer_relevancy"])
    assert math.isnan(scores["faithfulness"])


def test_gemini_scorer_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("deepeval")
    from rag_lab.eval.scorers.deepeval_scorer import DeepEvalScorer

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("rag_lab.llms.gemini._load_local_env", lambda: None)

    with pytest.raises(RuntimeError):
        DeepEvalScorer(model="gemini-2.5-flash-lite", provider="gemini")


def test_gemini_scorer_constructs_gemini_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("deepeval")
    from deepeval import models as deepeval_models

    from rag_lab.eval.scorers.deepeval_scorer import DeepEvalScorer

    captured: dict = {}

    class FakeGeminiModel:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("rag_lab.llms.gemini._load_local_env", lambda: None)
    monkeypatch.setattr(deepeval_models, "GeminiModel", FakeGeminiModel)

    scorer = DeepEvalScorer(model="gemini-2.5-flash", provider="gemini")

    assert isinstance(scorer._model, FakeGeminiModel)
    assert captured == {
        "model": "gemini-2.5-flash",
        "api_key": "test-key",
        "temperature": 0,
    }


@pytest.mark.integration
def test_scorer_returns_relevancy_and_faithfulness_without_ideal_answer():
    pytest.importorskip("deepeval")
    from rag_lab.eval.scorers.deepeval_scorer import DeepEvalScorer

    scorer = DeepEvalScorer(model="llama3.2:3b")
    scores = scorer.score(
        question="What is the capital of France?",
        answer="The capital of France is Paris.",
        retrieval_context=["France is a country in Europe. Its capital is Paris."],
    )
    assert set(scores) == {"answer_relevancy", "faithfulness"}


@pytest.mark.integration
def test_scorer_adds_contextual_metrics_with_ideal_answer():
    pytest.importorskip("deepeval")
    from rag_lab.eval.scorers.deepeval_scorer import DeepEvalScorer

    scorer = DeepEvalScorer(model="llama3.2:3b")
    scores = scorer.score(
        question="What is the capital of France?",
        answer="Paris.",
        retrieval_context=["The capital of France is Paris."],
        ideal_answer="Paris is the capital of France.",
    )
    assert set(scores) == {
        "answer_relevancy",
        "faithfulness",
        "contextual_precision",
        "contextual_recall",
    }

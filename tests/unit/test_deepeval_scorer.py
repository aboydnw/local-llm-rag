import pytest


def test_module_imports_without_constructing_scorer():
    import importlib

    module = importlib.import_module("rag_lab.eval.scorers.deepeval_scorer")
    assert hasattr(module, "DeepEvalScorer")


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

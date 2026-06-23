from __future__ import annotations

from math import nan


class DeepEvalScorer:
    """Scores answers with DeepEval metrics backed by a local Ollama model."""

    def __init__(self, model: str) -> None:
        from deepeval.models import OllamaModel

        self._model = OllamaModel(model=model)

    def score(
        self,
        question: str,
        answer: str,
        retrieval_context: list[str],
        ideal_answer: str = "",
    ) -> dict[str, float]:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            FaithfulnessMetric,
        )
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ideal_answer,
            retrieval_context=retrieval_context or [""],
        )

        metrics = [
            ("answer_relevancy", AnswerRelevancyMetric(model=self._model)),
            ("faithfulness", FaithfulnessMetric(model=self._model)),
        ]
        if ideal_answer:
            metrics += [
                ("contextual_precision", ContextualPrecisionMetric(model=self._model)),
                ("contextual_recall", ContextualRecallMetric(model=self._model)),
            ]

        scores: dict[str, float] = {}
        for key, metric in metrics:
            try:
                metric.measure(test_case)
                scores[key] = float(metric.score)
            except Exception:
                scores[key] = nan
        return scores

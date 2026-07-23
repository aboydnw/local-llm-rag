from __future__ import annotations

import logging
from math import nan

logger = logging.getLogger(__name__)


class DeepEvalScorer:
    """Scores answers with DeepEval metrics backed by the selected provider."""

    def __init__(self, model: str, provider: str = "ollama") -> None:
        if provider == "ollama":
            from deepeval.models import OllamaModel

            self._model = OllamaModel(model=model)
        elif provider == "gemini":
            import os

            from deepeval.models import GeminiModel

            from rag_lab.llms.gemini import _load_local_env

            _load_local_env()
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Copy .env-local-example to .env-local "
                    "and add your key."
                )
            self._model = GeminiModel(model=model, api_key=api_key, temperature=0)
        else:
            raise ValueError(f"Unknown judge provider: {provider}")

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
            except Exception as exc:
                logger.warning("DeepEval metric '%s' failed: %s", key, exc)
                scores[key] = nan
        return scores

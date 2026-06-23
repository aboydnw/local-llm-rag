from dataclasses import dataclass, field
from typing import Protocol

from rag_lab.eval.golden_set import GoldenItem
from rag_lab.eval.scorers.keyword import keyword_coverage
from rag_lab.eval.scorers.retrieval import mrr, recall_at_k
from rag_lab.llms.base import LLM
from rag_lab.prompts import PromptBuilder
from rag_lab.retrievers.base import Retriever


class Judge(Protocol):
    def score(self, question: str, actual_answer: str, ideal_answer: str): ...


@dataclass(frozen=True, slots=True)
class EvalResult:
    item_id: str
    question: str
    actual_answer: str
    recall_at_k: float
    mrr: float
    keyword_coverage: float
    judge_score: int | None
    judge_reason: str | None
    deepeval_scores: dict[str, float] = field(default_factory=dict)


class EvalRunner:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        k: int,
        prompt_builder: PromptBuilder | None = None,
        judge: Judge | None = None,
        deepeval_scorer=None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.judge = judge
        self.deepeval_scorer = deepeval_scorer

    def run(self, items: list[GoldenItem]) -> list[EvalResult]:
        out: list[EvalResult] = []
        for item in items:
            results = self.retriever.retrieve(item.question, k=self.k)
            prompt = self.prompt_builder.build(question=item.question, results=results)
            answer = self.llm.generate(prompt)

            judge_score: int | None = None
            judge_reason: str | None = None
            if self.judge is not None:
                jr = self.judge.score(
                    question=item.question,
                    actual_answer=answer,
                    ideal_answer=item.ideal_answer,
                )
                judge_score = jr.score
                judge_reason = jr.reason

            deepeval_scores: dict[str, float] = {}
            if self.deepeval_scorer is not None:
                deepeval_scores = self.deepeval_scorer.score(
                    question=item.question,
                    answer=answer,
                    retrieval_context=[r.chunk.text for r in results],
                    ideal_answer=item.ideal_answer,
                )

            out.append(
                EvalResult(
                    item_id=item.id,
                    question=item.question,
                    actual_answer=answer,
                    recall_at_k=recall_at_k(results, item.ideal_docs, k=self.k),
                    mrr=mrr(results, item.ideal_docs),
                    keyword_coverage=keyword_coverage(answer, item.must_mention),
                    judge_score=judge_score,
                    judge_reason=judge_reason,
                    deepeval_scores=deepeval_scores,
                )
            )
        return out

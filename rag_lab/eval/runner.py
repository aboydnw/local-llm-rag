from dataclasses import dataclass
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


class EvalRunner:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        k: int,
        prompt_builder: PromptBuilder | None = None,
        judge: Judge | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.judge = judge

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
                )
            )
        return out

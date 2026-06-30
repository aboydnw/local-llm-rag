import time
from collections.abc import Callable
from dataclasses import dataclass, field

from rag_lab.eval.golden_set import GoldenItem
from rag_lab.eval.scorers.citation import citation_validity, parse_citations
from rag_lab.eval.scorers.keyword import keyword_coverage
from rag_lab.eval.scorers.retrieval import average_precision, mrr, ndcg_at_k, recall_at_k
from rag_lab.llms.base import LLM
from rag_lab.prompts import PromptBuilder
from rag_lab.retrievers.base import Retriever


@dataclass(frozen=True, slots=True)
class RetrievedRef:
    rank: int
    doc_path: str
    heading_path: tuple[str, ...]
    position: int
    score: float


@dataclass(frozen=True, slots=True)
class EvalResult:
    item_id: str
    question: str
    actual_answer: str
    recall_at_k: float
    mrr: float
    keyword_coverage: float
    ndcg_at_k: float = 0.0
    average_precision: float = 0.0
    citation_validity: float | None = None
    retrieved: list[RetrievedRef] = field(default_factory=list)
    citations: list[int] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    deepeval_scores: dict[str, float] = field(default_factory=dict)


class EvalRunner:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        k: int,
        prompt_builder: PromptBuilder | None = None,
        deepeval_scorer=None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be positive")
        self.retriever = retriever
        self.llm = llm
        self.k = k
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.deepeval_scorer = deepeval_scorer
        self.clock = clock

    def run(self, items: list[GoldenItem]) -> list[EvalResult]:
        out: list[EvalResult] = []
        for item in items:
            t0 = self.clock()
            results = self.retriever.retrieve(item.question, k=self.k)
            t1 = self.clock()
            prompt = self.prompt_builder.build(question=item.question, results=results)
            answer = self.llm.generate(prompt)
            t2 = self.clock()

            retrieved = [
                RetrievedRef(
                    rank=rank,
                    doc_path=r.chunk.doc_path.as_posix(),
                    heading_path=tuple(r.chunk.heading_path),
                    position=r.chunk.position,
                    score=r.score,
                )
                for rank, r in enumerate(results, start=1)
            ]
            latency_ms = {
                "retrieve": (t1 - t0) * 1000.0,
                "generate": (t2 - t1) * 1000.0,
            }

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
                    ndcg_at_k=ndcg_at_k(results, item.ideal_docs, k=self.k),
                    average_precision=average_precision(results, item.ideal_docs),
                    citation_validity=citation_validity(answer, len(results)),
                    retrieved=retrieved,
                    citations=parse_citations(answer),
                    latency_ms=latency_ms,
                    deepeval_scores=deepeval_scores,
                )
            )
        return out

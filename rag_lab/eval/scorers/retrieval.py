import math

from rag_lab.retrievers.base import RetrievalResult


def ranked_doc_paths(results: list[RetrievalResult]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for result in results:
        doc = str(result.chunk.doc_path)
        if doc not in seen:
            seen.add(doc)
            ordered.append(doc)
    return ordered


def recall_at_k(results: list[RetrievalResult], ideal_docs: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not ideal_docs:
        return 1.0
    ideal = set(ideal_docs)
    top = set(ranked_doc_paths(results)[:k])
    return len(ideal & top) / len(ideal)


def ndcg_at_k(results: list[RetrievalResult], ideal_docs: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not ideal_docs:
        return 1.0
    ideal = set(ideal_docs)
    top = ranked_doc_paths(results)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, doc in enumerate(top)
        if doc in ideal
    )
    ideal_hits = min(len(ideal), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def mrr(results: list[RetrievalResult], ideal_docs: list[str]) -> float:
    if not ideal_docs:
        return 1.0
    ideal = set(ideal_docs)
    for rank, result in enumerate(results, start=1):
        if str(result.chunk.doc_path) in ideal:
            return 1.0 / rank
    return 0.0

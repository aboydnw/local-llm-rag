from rag_lab.retrievers.base import RetrievalResult


def recall_at_k(results: list[RetrievalResult], ideal_docs: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not ideal_docs:
        return 1.0
    ideal = set(ideal_docs)
    top = {str(r.chunk.doc_path) for r in results[:k]}
    return 1.0 if ideal & top else 0.0


def mrr(results: list[RetrievalResult], ideal_docs: list[str]) -> float:
    if not ideal_docs:
        return 1.0
    ideal = set(ideal_docs)
    for rank, result in enumerate(results, start=1):
        if str(result.chunk.doc_path) in ideal:
            return 1.0 / rank
    return 0.0

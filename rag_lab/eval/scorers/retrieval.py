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


def mrr(results: list[RetrievalResult], ideal_docs: list[str]) -> float:
    if not ideal_docs:
        return 1.0
    ideal = set(ideal_docs)
    for rank, result in enumerate(results, start=1):
        if str(result.chunk.doc_path) in ideal:
            return 1.0 / rank
    return 0.0

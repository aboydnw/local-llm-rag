from collections import defaultdict

from rag_lab.retrievers.base import RetrievalResult, Retriever


class HybridRetriever:
    """Reciprocal Rank Fusion of two retrievers."""

    def __init__(
        self,
        vector: Retriever,
        bm25: Retriever,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self.vector = vector
        self.bm25 = bm25
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        fetch_k = max(k * 3, 20)
        vector_hits = self.vector.retrieve(query, k=fetch_k)
        bm25_hits = self.bm25.retrieve(query, k=fetch_k)

        scores: dict[str, float] = defaultdict(float)
        chunk_by_id: dict[str, RetrievalResult] = {}

        for rank, result in enumerate(vector_hits):
            cid = self._key(result)
            scores[cid] += self.vector_weight * (1.0 / (self.rrf_k + rank))
            chunk_by_id[cid] = result
        for rank, result in enumerate(bm25_hits):
            cid = self._key(result)
            scores[cid] += self.bm25_weight * (1.0 / (self.rrf_k + rank))
            chunk_by_id.setdefault(cid, result)

        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            RetrievalResult(
                chunk=chunk_by_id[cid].chunk, score=score, source="hybrid"
            )
            for cid, score in ordered[:k]
        ]

    @staticmethod
    def _key(result: RetrievalResult) -> str:
        return f"{result.chunk.doc_path}|{result.chunk.position}"

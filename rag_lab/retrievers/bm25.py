from rag_lab.retrievers.base import RetrievalResult
from rag_lab.store.sqlite_vec import SqliteVecStore


class BM25Retriever:
    def __init__(self, store: SqliteVecStore) -> None:
        self.store = store

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        hits = self.store.query_bm25(query, k=k)
        return [
            RetrievalResult(chunk=chunk, score=score, source="bm25")
            for chunk, score in hits
        ]

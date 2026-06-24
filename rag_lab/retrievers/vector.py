from rag_lab.embedders.base import Embedder
from rag_lab.retrievers.base import RetrievalResult
from rag_lab.store.sqlite_vec import SqliteVecStore


class VectorRetriever:
    def __init__(self, store: SqliteVecStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, k: int) -> list[RetrievalResult]:
        vector = self.embedder.embed_query(query)
        hits = self.store.query_vector(vector, k=k)
        return [
            RetrievalResult(chunk=chunk, score=-distance, source="vector")
            for chunk, distance in hits
        ]

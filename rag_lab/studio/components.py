from rag_lab.config import Config, embedding_dimension
from rag_lab.embedders.ollama import OllamaEmbedder, prefixes_for_model
from rag_lab.llms.ollama import OllamaLLM
from rag_lab.retrievers.base import Retriever
from rag_lab.retrievers.bm25 import BM25Retriever
from rag_lab.retrievers.hybrid import HybridRetriever
from rag_lab.retrievers.vector import VectorRetriever
from rag_lab.store.sqlite_vec import SqliteVecStore


def build_embedder(config: Config) -> OllamaEmbedder:
    dimension = embedding_dimension(config.embedder.model)
    document_prefix, query_prefix = prefixes_for_model(config.embedder.model)
    return OllamaEmbedder(
        model=config.embedder.model,
        dimension=dimension,
        document_prefix=document_prefix,
        query_prefix=query_prefix,
    )


def build_llm(config: Config) -> OllamaLLM:
    return OllamaLLM(model=config.llm.model)


def build_retriever(store: SqliteVecStore, embedder, config: Config) -> Retriever:
    vector = VectorRetriever(store=store, embedder=embedder)
    bm25 = BM25Retriever(store=store)
    rtype = config.retriever.type
    if rtype == "vector":
        return vector
    if rtype == "bm25":
        return bm25
    if rtype == "hybrid":
        return HybridRetriever(
            vector=vector,
            bm25=bm25,
            vector_weight=config.retriever.vector_weight,
            bm25_weight=config.retriever.bm25_weight,
        )
    raise ValueError(f"Unknown retriever type: {rtype}")

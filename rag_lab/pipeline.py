from pathlib import Path

from rag_lab.agent.agent import Agent
from rag_lab.agent.tools import (
    KEYWORD_SEARCH_DESCRIPTION,
    VECTOR_SEARCH_DESCRIPTION,
    FetchDocumentTool,
    ListDocumentsTool,
    SearchTool,
)
from rag_lab.chunkers.base import Chunker
from rag_lab.chunkers.fixed import FixedSizeChunker
from rag_lab.chunkers.markdown_aware import MarkdownAwareChunker
from rag_lab.chunkers.recursive import RecursiveChunker
from rag_lab.chunkers.semantic import SemanticChunker
from rag_lab.config import Config, embedding_dimension
from rag_lab.embedders.ollama import OllamaEmbedder, prefixes_for_model
from rag_lab.llms.ollama import OllamaLLM
from rag_lab.prompts import PromptBuilder
from rag_lab.rerankers.llm import LLMReranker
from rag_lab.retrievers.base import Retriever
from rag_lab.retrievers.bm25 import BM25Retriever
from rag_lab.retrievers.hybrid import HybridRetriever
from rag_lab.retrievers.reranking import RerankingRetriever
from rag_lab.retrievers.vector import VectorRetriever
from rag_lab.store.sqlite_vec import SqliteVecStore

SCHEMA_VERSION = 1

_COMPAT_KEYS = ("schema_version", "embedder_model", "dimension")


def build_embedder(config: Config) -> OllamaEmbedder:
    document_prefix, query_prefix = prefixes_for_model(config.embedder.model)
    return OllamaEmbedder(
        model=config.embedder.model,
        dimension=embedding_dimension(config.embedder.model),
        document_prefix=document_prefix,
        query_prefix=query_prefix,
    )


def build_chunker(config: Config, embedder=None) -> Chunker:
    chunker = config.chunker
    if chunker.type == "markdown_aware":
        return MarkdownAwareChunker(
            max_tokens=chunker.max_tokens,
            overlap=chunker.overlap,
            context_header=chunker.context_header,
        )
    if chunker.type == "fixed":
        return FixedSizeChunker(
            max_tokens=chunker.max_tokens,
            overlap=chunker.overlap,
            context_header=chunker.context_header,
        )
    if chunker.type == "recursive":
        return RecursiveChunker(
            max_tokens=chunker.max_tokens,
            overlap=chunker.overlap,
            context_header=chunker.context_header,
        )
    if chunker.type == "semantic":
        return SemanticChunker(
            embedder=embedder or build_embedder(config),
            max_tokens=chunker.max_tokens,
            similarity_threshold=chunker.similarity_threshold,
            overlap=chunker.overlap,
            context_header=chunker.context_header,
        )
    raise ValueError(f"Unknown chunker type: {chunker.type}")


def build_llm(config: Config) -> OllamaLLM:
    return OllamaLLM(model=config.llm.model)


def build_store(config: Config, db: Path) -> SqliteVecStore:
    return SqliteVecStore(db, dimension=embedding_dimension(config.embedder.model))


def build_prompt_builder(config: Config) -> PromptBuilder:
    return PromptBuilder(system_instructions=config.prompt.system_instructions)


def build_retriever(store: SqliteVecStore, embedder, config: Config) -> Retriever:
    vector = VectorRetriever(store=store, embedder=embedder)
    bm25 = BM25Retriever(store=store)
    rtype = config.retriever.type
    if rtype == "vector":
        base: Retriever = vector
    elif rtype == "bm25":
        base = bm25
    elif rtype == "hybrid":
        base = HybridRetriever(
            vector=vector,
            bm25=bm25,
            vector_weight=config.retriever.vector_weight,
            bm25_weight=config.retriever.bm25_weight,
        )
    else:
        raise ValueError(f"Unknown retriever type: {rtype}")
    if config.retriever.reranker == "llm":
        return RerankingRetriever(
            inner=base,
            reranker=LLMReranker(build_llm(config)),
            candidates=config.retriever.rerank_candidates,
        )
    return base


def build_agent(store: SqliteVecStore, embedder, config: Config) -> Agent:
    vector = VectorRetriever(store=store, embedder=embedder)
    bm25 = BM25Retriever(store=store)
    factories = {
        "vector_search": lambda: SearchTool(
            "vector_search", VECTOR_SEARCH_DESCRIPTION, vector, k=config.retriever.k
        ),
        "keyword_search": lambda: SearchTool(
            "keyword_search", KEYWORD_SEARCH_DESCRIPTION, bm25, k=config.retriever.k
        ),
        "list_documents": lambda: ListDocumentsTool(store),
        "fetch_document": lambda: FetchDocumentTool(store),
    }
    tools = []
    for name in config.agent.tools:
        if name not in factories:
            raise ValueError(f"Unknown agent tool: {name}")
        tools.append(factories[name]())
    return Agent(
        llm=build_llm(config),
        tools=tools,
        prompt_builder=build_prompt_builder(config),
        max_steps=config.agent.max_steps,
        final_k=config.agent.final_k,
        instructions=config.agent.instructions,
    )


def index_manifest(config: Config) -> dict:
    """Build the metadata recorded alongside an index at ingest time."""
    return {
        "schema_version": SCHEMA_VERSION,
        "embedder_model": config.embedder.model,
        "dimension": embedding_dimension(config.embedder.model),
        "chunker": config.chunker.model_dump(),
    }


def check_index_compatible(store: SqliteVecStore, config: Config) -> str | None:
    """Return a human-readable reason the index can't serve this config, or ``None``."""
    stored = store.read_manifest()
    if not stored:
        return (
            "This index has no recorded metadata (it was built by an older rag-lab). "
            "Re-ingest to record it, or pass --force to query anyway."
        )
    expected = index_manifest(config)
    for key in _COMPAT_KEYS:
        if stored.get(key) != expected[key]:
            return (
                f"Index was built with {key}={stored.get(key)!r} but the config asks for "
                f"{expected[key]!r}. Re-ingest with this config, or pass --force."
            )
    return None

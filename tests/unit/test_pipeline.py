from pathlib import Path

from rag_lab import pipeline
from rag_lab.config import Config
from rag_lab.embedders.fake import FakeEmbedder
from rag_lab.retrievers.bm25 import BM25Retriever
from rag_lab.retrievers.hybrid import HybridRetriever
from rag_lab.retrievers.reranking import RerankingRetriever
from rag_lab.retrievers.vector import VectorRetriever
from rag_lab.store.sqlite_vec import SqliteVecStore


def _store(tmp_path: Path) -> SqliteVecStore:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    return store


def test_build_store_uses_embedding_dimension(tmp_path: Path) -> None:
    store = pipeline.build_store(Config(), tmp_path / "rag.db")
    assert store.dimension == 768


def test_build_embedder_uses_config_model_and_dimension() -> None:
    embedder = pipeline.build_embedder(Config())
    assert embedder.model == "nomic-embed-text"
    assert embedder.dimension == 768


def test_build_llm_uses_config_model() -> None:
    assert pipeline.build_llm(Config()).model == "llama3.2:3b"


def test_build_prompt_builder_uses_config_instructions() -> None:
    config = Config()
    config.prompt.system_instructions = "custom instructions"
    assert pipeline.build_prompt_builder(config).system_instructions == "custom instructions"


def test_build_retriever_honors_vector_type(tmp_path: Path) -> None:
    config = Config()
    config.retriever.type = "vector"
    retriever = pipeline.build_retriever(_store(tmp_path), FakeEmbedder(16), config)
    assert isinstance(retriever, VectorRetriever)


def test_build_retriever_honors_bm25_type(tmp_path: Path) -> None:
    config = Config()
    config.retriever.type = "bm25"
    retriever = pipeline.build_retriever(_store(tmp_path), FakeEmbedder(16), config)
    assert isinstance(retriever, BM25Retriever)


def test_build_retriever_honors_hybrid_type(tmp_path: Path) -> None:
    config = Config()
    config.retriever.type = "hybrid"
    retriever = pipeline.build_retriever(_store(tmp_path), FakeEmbedder(16), config)
    assert isinstance(retriever, HybridRetriever)


def test_build_retriever_wraps_with_reranker_when_configured(tmp_path: Path) -> None:
    config = Config()
    config.retriever.reranker = "llm"
    retriever = pipeline.build_retriever(_store(tmp_path), FakeEmbedder(16), config)
    assert isinstance(retriever, RerankingRetriever)


def test_build_agent_returns_agent_with_configured_tools(tmp_path: Path) -> None:
    from rag_lab.agent.agent import Agent

    config = Config()
    agent = pipeline.build_agent(_store(tmp_path), FakeEmbedder(16), config)
    assert isinstance(agent, Agent)
    assert [t.name for t in agent.tools] == [
        "vector_search",
        "keyword_search",
        "list_documents",
        "fetch_document",
    ]
    assert agent.max_steps == 6
    assert agent.final_k == 5


def test_build_agent_honors_tool_subset(tmp_path: Path) -> None:
    config = Config()
    config.agent.tools = ["vector_search"]
    agent = pipeline.build_agent(_store(tmp_path), FakeEmbedder(16), config)
    assert [t.name for t in agent.tools] == ["vector_search"]


def test_build_agent_rejects_unknown_tool(tmp_path: Path) -> None:
    import pytest

    config = Config()
    config.agent.tools = ["bogus_tool"]
    with pytest.raises(ValueError):
        pipeline.build_agent(_store(tmp_path), FakeEmbedder(16), config)


def _store_with_manifest(tmp_path: Path, config: Config) -> SqliteVecStore:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    store.write_manifest(pipeline.index_manifest(config))
    return store


def test_check_index_compatible_passes_for_matching_config(tmp_path: Path) -> None:
    config = Config()
    store = _store_with_manifest(tmp_path, config)
    assert pipeline.check_index_compatible(store, config) is None


def test_check_index_compatible_flags_embedder_mismatch(tmp_path: Path) -> None:
    store = _store_with_manifest(tmp_path, Config())
    other = Config()
    other.embedder.model = "mxbai-embed-large"
    assert pipeline.check_index_compatible(store, other) is not None


def test_check_index_compatible_flags_missing_metadata(tmp_path: Path) -> None:
    store = SqliteVecStore(tmp_path / "rag.db", dimension=16)
    store.initialize()
    assert pipeline.check_index_compatible(store, Config()) is not None

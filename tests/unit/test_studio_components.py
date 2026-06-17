from rag_lab.config import Config
from rag_lab.embedders.fake import FakeEmbedder
from rag_lab.retrievers.bm25 import BM25Retriever
from rag_lab.retrievers.hybrid import HybridRetriever
from rag_lab.retrievers.vector import VectorRetriever
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.studio import components


def _store(tmp_path):
    return SqliteVecStore(tmp_path / "x.db", dimension=16)


def test_build_retriever_vector(tmp_path):
    cfg = Config()
    cfg.retriever.type = "vector"
    r = components.build_retriever(_store(tmp_path), FakeEmbedder(16), cfg)
    assert isinstance(r, VectorRetriever)


def test_build_retriever_bm25(tmp_path):
    cfg = Config()
    cfg.retriever.type = "bm25"
    r = components.build_retriever(_store(tmp_path), FakeEmbedder(16), cfg)
    assert isinstance(r, BM25Retriever)


def test_build_retriever_hybrid_passes_weights(tmp_path):
    cfg = Config()
    cfg.retriever.type = "hybrid"
    cfg.retriever.vector_weight = 0.7
    cfg.retriever.bm25_weight = 0.3
    r = components.build_retriever(_store(tmp_path), FakeEmbedder(16), cfg)
    assert isinstance(r, HybridRetriever)
    assert r.vector_weight == 0.7
    assert r.bm25_weight == 0.3


def test_build_embedder_dimension_matches_model():
    r = components.build_embedder(Config())
    assert r.model == "nomic-embed-text"
    assert r.dimension == 768

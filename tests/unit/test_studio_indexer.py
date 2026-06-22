from rag_lab.config import Config
from rag_lab.embedders.fake import FakeEmbedder
from rag_lab.loaders.markdown import MarkdownLoader
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.studio import indexer
from rag_lab.studio.workspace import Workspace


def _corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "a.md").write_text("# Alpha\n\nThe alpha document talks about tiles.\n")
    (d / "b.md").write_text("# Beta\n\nThe beta document talks about rasters.\n")
    return d


def test_validate_corpus_accepts_dir_with_markdown(tmp_path):
    assert indexer.validate_corpus(str(_corpus(tmp_path))) is None


def test_validate_corpus_trims_surrounding_whitespace(tmp_path):
    assert indexer.validate_corpus(f"  {_corpus(tmp_path)}  ") is None


def test_validate_corpus_rejects_blank():
    assert indexer.validate_corpus("  ") == "Enter a corpus directory to index."


def test_validate_corpus_rejects_missing_path(tmp_path):
    msg = indexer.validate_corpus(str(tmp_path / "nope"))
    assert msg is not None and "Corpus directory not found:" in msg


def test_validate_corpus_rejects_file(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# hi")
    msg = indexer.validate_corpus(str(f))
    assert msg is not None and "Corpus must be a directory, not a file:" in msg


def test_validate_corpus_rejects_dir_without_markdown(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    (d / "notes.txt").write_text("not markdown")
    msg = indexer.validate_corpus(str(d))
    assert msg is not None and "No markdown (.md) files found in:" in msg


def test_cache_key_stable_for_same_config():
    cfg = Config()
    assert indexer.cache_key("corpus", cfg) == indexer.cache_key("corpus", cfg)


def test_cache_key_changes_with_chunker():
    a = Config()
    b = Config()
    b.chunker.max_tokens = 256
    assert indexer.cache_key("corpus", a) != indexer.cache_key("corpus", b)


def test_cache_key_changes_with_embedder():
    a = Config()
    b = Config()
    b.embedder.model = "mxbai-embed-large"
    assert indexer.cache_key("corpus", a) != indexer.cache_key("corpus", b)


def test_cache_key_ignores_retriever_and_llm():
    a = Config()
    b = Config()
    b.retriever.k = 99
    b.retriever.type = "bm25"
    b.llm.model = "something-else"
    assert indexer.cache_key("corpus", a) == indexer.cache_key("corpus", b)


def test_build_index_creates_populated_db(tmp_path):
    ws = Workspace(tmp_path / ".rag-lab")
    ws.initialize()
    corpus = _corpus(tmp_path)
    db = indexer.build_index(
        ws, str(corpus), Config(),
        loader=MarkdownLoader(corpus), embedder=FakeEmbedder(16),
    )
    assert db.exists()
    assert SqliteVecStore(db, dimension=16).count() > 0
    key = indexer.cache_key(str(corpus), Config())
    assert ws.index_meta(key).exists()


def test_ensure_index_reuses_cache(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / ".rag-lab")
    ws.initialize()
    corpus = _corpus(tmp_path)
    indexer.build_index(
        ws, str(corpus), Config(),
        loader=MarkdownLoader(corpus), embedder=FakeEmbedder(16),
    )

    def _boom(*args, **kwargs):
        raise AssertionError("should not rebuild a cached index")

    monkeypatch.setattr(indexer, "build_index", _boom)
    db = indexer.ensure_index(ws, str(corpus), Config())
    assert db.exists()

from rag_lab.config import Config
from rag_lab.embedders.fake import FakeEmbedder
from rag_lab.loaders.markdown import MarkdownLoader
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.studio import indexer
from rag_lab.studio.corpora import Corpus, Source, local_corpus
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


def test_cache_key_stable_for_same_corpus():
    cfg = Config()
    c = Corpus(name="kb", sources=(Source(type="github", repo="o/a"),))
    assert indexer.cache_key(c, cfg) == indexer.cache_key(c, cfg)


def test_cache_key_ignores_source_order():
    cfg = Config()
    s1 = Source(type="github", repo="o/a")
    s2 = Source(type="github", repo="o/b")
    assert indexer.cache_key(Corpus("kb", (s1, s2)), cfg) == indexer.cache_key(
        Corpus("kb", (s2, s1)), cfg
    )


def test_cache_key_changes_with_sources():
    cfg = Config()
    a = Corpus("kb", (Source(type="github", repo="o/a"),))
    b = Corpus("kb", (Source(type="github", repo="o/b"),))
    assert indexer.cache_key(a, cfg) != indexer.cache_key(b, cfg)


def test_cache_key_changes_with_chunker():
    a, b = Config(), Config()
    b.chunker.max_tokens = 256
    c = local_corpus("corpus")
    assert indexer.cache_key(c, a) != indexer.cache_key(c, b)


def test_cache_key_changes_with_embedder():
    a, b = Config(), Config()
    b.embedder.model = "mxbai-embed-large"
    c = local_corpus("corpus")
    assert indexer.cache_key(c, a) != indexer.cache_key(c, b)


def test_cache_key_ignores_retriever_and_llm():
    a, b = Config(), Config()
    b.retriever.k = 99
    b.retriever.type = "bm25"
    b.llm.model = "something-else"
    c = local_corpus("corpus")
    assert indexer.cache_key(c, a) == indexer.cache_key(c, b)


def test_build_index_creates_populated_db(tmp_path):
    ws = Workspace(tmp_path / ".rag-lab")
    ws.initialize()
    corpus_dir = _corpus(tmp_path)
    c = local_corpus(str(corpus_dir))
    db = indexer.build_index(
        ws, c, Config(),
        loader=MarkdownLoader(corpus_dir), embedder=FakeEmbedder(16),
    )
    assert db.exists()
    assert SqliteVecStore(db, dimension=16).count() > 0
    assert ws.index_meta(indexer.cache_key(c, Config())).exists()


def test_ensure_index_reuses_cache(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / ".rag-lab")
    ws.initialize()
    corpus_dir = _corpus(tmp_path)
    c = local_corpus(str(corpus_dir))
    indexer.build_index(
        ws, c, Config(),
        loader=MarkdownLoader(corpus_dir), embedder=FakeEmbedder(16),
    )

    def _boom(*args, **kwargs):
        raise AssertionError("should not rebuild a cached index")

    monkeypatch.setattr(indexer, "build_index", _boom)
    assert indexer.ensure_index(ws, c, Config()).exists()


def test_loader_for_corpus_builds_one_loader_per_source(tmp_path):
    ws = Workspace(tmp_path / ".rag-lab")
    ws.initialize()
    corpus_dir = _corpus(tmp_path)
    c = Corpus(
        name="kb",
        sources=(
            Source(type="local", path=str(corpus_dir)),
            Source(type="github", repo="owner/name"),
        ),
    )
    loader = indexer.loader_for_corpus(ws, c)
    assert len(loader.loaders) == 2

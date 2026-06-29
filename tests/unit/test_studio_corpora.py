import pytest

from rag_lab.studio import corpora
from rag_lab.studio.corpora import Corpus, Source
from rag_lab.studio.workspace import Workspace


def _ws(tmp_path):
    ws = Workspace(tmp_path / ".rag-lab")
    ws.initialize()
    return ws


def test_corpus_roundtrips_through_dict():
    c = Corpus(
        name="titiler-stack",
        sources=(Source(type="github", repo="developmentseed/titiler"),),
    )
    assert Corpus.from_dict(c.to_dict()) == c


def test_github_source_roundtrips_private_flag():
    s = Source(type="github", repo="owner/internal", private=True)
    assert Source.from_dict(s.to_dict()) == s


def test_github_source_defaults_to_public():
    s = Source.from_dict({"type": "github", "repo": "owner/public"})
    assert s.private is False


def test_github_issue_source_roundtrips():
    s = Source(type="github_issue", repo="developmentseed/titiler", issue=42)
    assert Source.from_dict(s.to_dict()) == s


def test_parse_issue_ref_extracts_repo_and_number():
    assert corpora.parse_issue_ref("developmentseed/titiler#42") == ("developmentseed/titiler", 42)


@pytest.mark.parametrize("bad", ["", "owner/repo", "owner/repo#", "owner#42", "owner/repo#x"])
def test_validate_issue_ref_rejects_malformed(bad):
    assert corpora.validate_issue_ref(bad) is not None


def test_validate_issue_ref_accepts_valid():
    assert corpora.validate_issue_ref("owner/repo#7") is None


def test_from_dict_rejects_non_bool_private():
    with pytest.raises(ValueError):
        Source.from_dict({"type": "github", "repo": "o/r", "private": "false"})


def test_from_dict_rejects_non_int_issue():
    with pytest.raises(ValueError):
        Source.from_dict({"type": "github_issue", "repo": "o/r", "issue": "5"})


def test_to_dict_rejects_unknown_source_type():
    with pytest.raises(ValueError):
        Source(type="web", path="https://example.com").to_dict()


def test_from_dict_rejects_unknown_source_type():
    with pytest.raises(ValueError):
        Source.from_dict({"type": "web", "url": "https://example.com"})


def test_local_corpus_has_single_local_source():
    c = corpora.local_corpus("docs")
    assert c.name == "__local__"
    assert c.sources == (Source(type="local", path="docs"),)


def test_label_prefers_path_for_local_and_name_otherwise():
    assert corpora.local_corpus("docs").label == "docs"
    assert Corpus(name="kb", sources=()).label == "kb"


def test_save_then_load_roundtrips(tmp_path):
    ws = _ws(tmp_path)
    c = Corpus(name="kb", sources=(Source(type="github", repo="owner/name"),))
    corpora.save_corpus(ws, c)
    assert corpora.load_corpus(ws, "kb") == c


def test_list_corpora_is_sorted_and_excludes_local(tmp_path):
    ws = _ws(tmp_path)
    corpora.save_corpus(ws, Corpus(name="zeta", sources=()))
    corpora.save_corpus(ws, Corpus(name="alpha", sources=()))
    corpora.save_corpus(ws, corpora.local_corpus("docs"))
    assert corpora.list_corpora(ws) == ["alpha", "zeta"]


def test_delete_corpus_removes_file(tmp_path):
    ws = _ws(tmp_path)
    corpora.save_corpus(ws, Corpus(name="kb", sources=()))
    corpora.delete_corpus(ws, "kb")
    assert corpora.list_corpora(ws) == []


def test_validate_repo_accepts_owner_name():
    assert corpora.validate_repo("developmentseed/titiler") is None


@pytest.mark.parametrize("bad", ["", "noslash", "too/many/parts", "owner/", "/name", "a b/c"])
def test_validate_repo_rejects_malformed(bad):
    assert corpora.validate_repo(bad) is not None


def test_validate_name_accepts_simple_name():
    assert corpora.validate_name("titiler-stack") is None


@pytest.mark.parametrize("bad", ["", "  ", "a/b", "..", "__local__"])
def test_validate_name_rejects_bad(bad):
    assert corpora.validate_name(bad) is not None


def test_add_source_appends():
    c = Corpus(name="kb", sources=())
    out = corpora.add_source(c, Source(type="github", repo="o/a"))
    assert out.sources == (Source(type="github", repo="o/a"),)


def test_add_source_is_idempotent():
    s = Source(type="github", repo="o/a")
    c = Corpus(name="kb", sources=(s,))
    assert corpora.add_source(c, s).sources == (s,)


def test_remove_source_drops_match():
    s1 = Source(type="github", repo="o/a")
    s2 = Source(type="github", repo="o/b")
    c = Corpus(name="kb", sources=(s1, s2))
    assert corpora.remove_source(c, s1).sources == (s2,)


def test_resolve_active_corpus_loads_named(tmp_path):
    ws = _ws(tmp_path)
    saved = Corpus(name="kb", sources=(Source(type="github", repo="o/a"),))
    corpora.save_corpus(ws, saved)
    assert corpora.resolve_active_corpus(ws, "kb", "docs") == saved


def test_resolve_active_corpus_falls_back_to_local(tmp_path):
    ws = _ws(tmp_path)
    assert corpora.resolve_active_corpus(ws, None, "docs") == corpora.local_corpus("docs")


def test_resolve_active_corpus_falls_back_when_missing(tmp_path):
    ws = _ws(tmp_path)
    assert corpora.resolve_active_corpus(ws, "gone", "docs") == corpora.local_corpus("docs")

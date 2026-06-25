from pathlib import Path

from rag_lab.loaders.markdown import MarkdownLoader


def test_loads_all_markdown_files_recursively(fixture_corpus: Path) -> None:
    loader = MarkdownLoader(fixture_corpus)
    docs = list(loader.load())
    paths = {d.path.name for d in docs}
    assert paths == {"alpha.md", "beta.md"}


def test_strips_frontmatter_from_text_and_lifts_to_metadata(fixture_corpus: Path) -> None:
    loader = MarkdownLoader(fixture_corpus)
    alpha = next(d for d in loader.load() if d.path.name == "alpha.md")
    assert "title: Alpha" not in alpha.text
    assert alpha.text.lstrip().startswith("# Alpha")
    assert alpha.metadata["title"] == "Alpha"


def test_documents_without_frontmatter_have_empty_metadata(fixture_corpus: Path) -> None:
    loader = MarkdownLoader(fixture_corpus)
    beta = next(d for d in loader.load() if d.path.name == "beta.md")
    assert beta.metadata == {}
    assert beta.text.lstrip().startswith("# Beta")


def test_ignores_non_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "b.txt").write_text("not markdown")
    loader = MarkdownLoader(tmp_path)
    docs = list(loader.load())
    assert len(docs) == 1
    assert docs[0].path.name == "a.md"


def test_excludes_venv_and_other_tooling_dirs(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_text("# Real")
    for excluded in (".venv", "venv", "node_modules", ".git", "__pycache__", ".tox"):
        d = tmp_path / excluded
        d.mkdir()
        (d / "noise.md").write_text("# Should be ignored")
    loader = MarkdownLoader(tmp_path)
    docs = list(loader.load())
    assert [d.path.name for d in docs] == ["real.md"]


def test_excludes_rag_lab_workspace_dir(tmp_path: Path) -> None:
    (tmp_path / "real.md").write_text("# Real")
    clone = tmp_path / ".rag-lab" / "clones" / "developmentseed__titiler"
    clone.mkdir(parents=True)
    (clone / "README.md").write_text("# Cloned repo noise")
    loader = MarkdownLoader(tmp_path)
    docs = list(loader.load())
    assert [d.path.name for d in docs] == ["real.md"]

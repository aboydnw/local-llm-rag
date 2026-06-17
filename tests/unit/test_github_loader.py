from pathlib import Path

import pytest

from rag_lab.loaders.github import GitHubLoader


def test_github_loader_clones_then_yields_markdown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_clone_dir = tmp_path / "fake-repo"
    fake_clone_dir.mkdir()
    (fake_clone_dir / "README.md").write_text("# Fake\n\nHello.")
    (fake_clone_dir / "sub").mkdir()
    (fake_clone_dir / "sub" / "guide.md").write_text("# Guide\n\nFor humans.")

    def fake_clone(repo: str, dest: Path) -> None:
        assert repo == "https://github.com/developmentseed/titiler.git"

    loader = GitHubLoader(
        "developmentseed/titiler",
        clone_into=fake_clone_dir,
        clone_fn=fake_clone,
    )
    paths = sorted(d.path.name for d in loader.load())
    assert paths == ["README.md", "guide.md"]


def test_github_loader_normalizes_short_form(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    def fake_clone(repo: str, dest: Path) -> None:
        seen["repo"] = repo

    fake_clone_dir = tmp_path / "x"
    fake_clone_dir.mkdir()
    loader = GitHubLoader("owner/name", clone_into=fake_clone_dir, clone_fn=fake_clone)
    list(loader.load())
    assert seen["repo"] == "https://github.com/owner/name.git"

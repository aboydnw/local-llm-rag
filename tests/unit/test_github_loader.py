from pathlib import Path

import pytest

from rag_lab.loaders.github import GitHubLoader


def test_github_loader_clones_then_yields_markdown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clone_dir = tmp_path / "repo"
    clone_calls: list[str] = []

    def fake_clone(repo: str, dest: Path) -> None:
        clone_calls.append(repo)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text("# Fake\n\nHello.")
        (dest / "sub").mkdir()
        (dest / "sub" / "guide.md").write_text("# Guide\n\nFor humans.")

    loader = GitHubLoader(
        "developmentseed/titiler",
        clone_into=clone_dir,
        clone_fn=fake_clone,
    )
    paths = sorted(d.path.name for d in loader.load())
    assert clone_calls == ["https://github.com/developmentseed/titiler.git"]
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


def test_private_repo_clones_via_gh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(argv, check):
        calls.append(argv)
        dest = Path(argv[argv.index("--") - 1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text("# Private\n\nSecret docs.")

    monkeypatch.setattr("rag_lab.loaders.github.subprocess.run", fake_run)
    loader = GitHubLoader("owner/internal", clone_into=tmp_path / "repo", private=True)
    docs = list(loader.load())
    assert calls and calls[0][0] == "gh"
    assert [d.path.name for d in docs] == ["README.md"]


def test_private_clone_passes_owner_repo_not_url(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(argv, check):
        captured["argv"] = argv
        dest = Path(argv[argv.index("--") - 1])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text("# R\n\nHi.")

    monkeypatch.setattr("rag_lab.loaders.github.subprocess.run", fake_run)
    loader = GitHubLoader("owner/internal", clone_into=tmp_path / "repo", private=True)
    list(loader.load())
    assert "owner/internal" in captured["argv"]
    assert all("https://" not in arg for arg in captured["argv"])


def test_github_loader_stamps_source_metadata(tmp_path):
    def fake_clone(repo, dest):
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text("# R\n\nHello.")

    loader = GitHubLoader("developmentseed/titiler", clone_into=tmp_path / "repo", clone_fn=fake_clone)
    docs = list(loader.load())
    assert all(d.metadata["source"] == "developmentseed/titiler" for d in docs)

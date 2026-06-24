import dataclasses
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

from rag_lab.loaders.markdown import MarkdownLoader
from rag_lab.types import Document


def _default_clone(repo_url: str, dest: Path) -> None:
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        check=True,
    )


def _gh_clone(repo_url: str, dest: Path) -> None:
    subprocess.run(
        ["gh", "repo", "clone", repo_url, str(dest), "--", "--depth", "1"],
        check=True,
    )


class GitHubLoader:
    """Shallow-clone a GitHub repo to a working directory, then load its markdown."""

    def __init__(
        self,
        repo: str,
        clone_into: Path,
        clone_fn: Callable[[str, Path], None] | None = None,
        private: bool = False,
    ) -> None:
        self.source = repo
        self.repo = repo if repo.startswith("http") else f"https://github.com/{repo}.git"
        self.clone_into = clone_into
        self.private = private
        if clone_fn is None:
            clone_fn = _gh_clone if private else _default_clone
        self._clone_fn = clone_fn
        self._cloned = False

    def load(self) -> Iterator[Document]:
        if not self._cloned:
            if self.clone_into.is_dir() and any(self.clone_into.iterdir()):
                pass
            else:
                self.clone_into.mkdir(parents=True, exist_ok=True)
                self._clone_fn(self.repo, self.clone_into)
            self._cloned = True
        for doc in MarkdownLoader(self.clone_into).load():
            yield dataclasses.replace(doc, metadata={**doc.metadata, "source": self.source})

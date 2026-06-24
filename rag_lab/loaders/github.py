import dataclasses
import os
import shutil
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

from rag_lab.loaders.markdown import MarkdownLoader
from rag_lab.types import Document

CLONE_TIMEOUT_SECONDS = 300


def _noninteractive_env() -> dict[str, str]:
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


def _default_clone(repo_url: str, dest: Path) -> None:
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        check=True,
        timeout=CLONE_TIMEOUT_SECONDS,
        stdin=subprocess.DEVNULL,
        env=_noninteractive_env(),
    )


def _gh_clone(repo_url: str, dest: Path) -> None:
    subprocess.run(
        ["gh", "repo", "clone", repo_url, str(dest), "--", "--depth", "1"],
        check=True,
        timeout=CLONE_TIMEOUT_SECONDS,
        stdin=subprocess.DEVNULL,
        env=_noninteractive_env(),
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
        self._clone_target = self.source if private else self.repo
        self._cloned = False

    def load(self) -> Iterator[Document]:
        if not self._cloned:
            if self.clone_into.is_dir() and any(self.clone_into.iterdir()):
                pass
            else:
                self.clone_into.mkdir(parents=True, exist_ok=True)
                try:
                    self._clone_fn(self._clone_target, self.clone_into)
                except BaseException:
                    shutil.rmtree(self.clone_into, ignore_errors=True)
                    raise
            self._cloned = True
        for doc in MarkdownLoader(self.clone_into).load():
            yield dataclasses.replace(doc, metadata={**doc.metadata, "source": self.source})

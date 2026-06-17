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


class GitHubLoader:
    """Shallow-clone a GitHub repo to a working directory, then load its markdown."""

    def __init__(
        self,
        repo: str,
        clone_into: Path,
        clone_fn: Callable[[str, Path], None] = _default_clone,
    ) -> None:
        self.repo = repo if repo.startswith("http") else f"https://github.com/{repo}.git"
        self.clone_into = clone_into
        self._clone_fn = clone_fn
        self._cloned = False

    def load(self) -> Iterator[Document]:
        if not self._cloned:
            if self.clone_into.exists() and any(self.clone_into.iterdir()):
                pass
            else:
                self.clone_into.mkdir(parents=True, exist_ok=True)
                self._clone_fn(self.repo, self.clone_into)
            self._cloned = True
        yield from MarkdownLoader(self.clone_into).load()

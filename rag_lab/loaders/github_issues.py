import json
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path

from rag_lab.types import Document

IssueFetch = Callable[[str, int], tuple[dict, list[dict]]]


def _gh_fetch(repo: str, number: int) -> tuple[dict, list[dict]]:
    def _api(path: str) -> object:
        out = subprocess.run(
            ["gh", "api", path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return json.loads(out)

    issue = _api(f"repos/{repo}/issues/{number}")
    comments = _api(f"repos/{repo}/issues/{number}/comments")
    return issue, comments


def _render(issue: dict, comments: list[dict]) -> str:
    parts = [f"# {issue['title']}", "", issue.get("body") or ""]
    for comment in comments:
        author = comment.get("user", {}).get("login", "unknown")
        parts += ["", "---", f"**@{author}:**", "", comment.get("body") or ""]
    return "\n".join(parts).strip() + "\n"


def _metadata(repo: str, number: int, issue: dict) -> dict[str, str]:
    return {
        "source": f"{repo}#{number}",
        "number": str(number),
        "url": issue.get("html_url", ""),
        "state": issue.get("state", ""),
        "title": issue.get("title", ""),
        "labels": ", ".join(label["name"] for label in issue.get("labels", [])),
    }


class GitHubIssuesLoader:
    """Fetch specific GitHub issues and yield one Document per issue."""

    def __init__(
        self,
        repo: str,
        numbers: list[int],
        fetch_fn: IssueFetch = _gh_fetch,
    ) -> None:
        self.repo = repo
        self.numbers = numbers
        self._fetch_fn = fetch_fn

    def load(self) -> Iterator[Document]:
        for number in self.numbers:
            issue, comments = self._fetch_fn(self.repo, number)
            yield Document(
                path=Path(f"{self.repo}#{number}"),
                text=_render(issue, comments),
                metadata=_metadata(self.repo, number, issue),
            )

from collections.abc import Iterator
from pathlib import Path

import frontmatter

from rag_lab.types import Document


class MarkdownLoader:
    """Recursively load .md files from a directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self) -> Iterator[Document]:
        for path in sorted(self.root.rglob("*.md")):
            post = frontmatter.load(path)
            metadata = {k: str(v) for k, v in post.metadata.items()}
            yield Document(path=path, text=post.content, metadata=metadata)

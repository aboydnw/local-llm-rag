import re
from collections.abc import Iterator

from rag_lab.chunkers import splitting
from rag_lab.types import Chunk, Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


class MarkdownAwareChunker:
    """Split markdown documents on headings, then enforce a token cap with overlap."""

    def __init__(
        self, max_tokens: int = 512, overlap: int = 50, context_header: bool = False
    ) -> None:
        if overlap >= max_tokens:
            raise ValueError("overlap must be smaller than max_tokens")
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.context_header = context_header
        self._encoder = splitting.get_encoder()

    def chunk(self, document: Document) -> Iterator[Chunk]:
        position = 0
        for heading_path, body in self._iter_sections(document.text):
            if not body.strip():
                continue
            for chunk_text in self._split_body(body):
                text = chunk_text
                if self.context_header:
                    text = f"{self._header(document, heading_path)}\n\n{chunk_text}"
                yield Chunk(
                    text=text,
                    doc_path=document.path,
                    heading_path=heading_path,
                    position=position,
                    metadata=dict(document.metadata),
                )
                position += 1

    @staticmethod
    def _header(document: Document, heading_path: tuple[str, ...]) -> str:
        return " > ".join([splitting.document_label(document), *heading_path])

    def _iter_sections(self, text: str) -> Iterator[tuple[tuple[str, ...], str]]:
        stack: list[tuple[int, str]] = []
        current_path: tuple[str, ...] = ()
        buffer: list[str] = []

        def flush() -> tuple[tuple[str, ...], str] | None:
            if not buffer:
                return None
            return current_path, "\n".join(buffer).strip()

        for line in text.splitlines():
            match = _HEADING_RE.match(line)
            if match:
                flushed = flush()
                if flushed is not None:
                    yield flushed
                buffer = []
                level = len(match.group(1))
                title = match.group(2)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current_path = tuple(t for _, t in stack)
            else:
                buffer.append(line)
        flushed = flush()
        if flushed is not None:
            yield flushed

    def _split_body(self, body: str) -> Iterator[str]:
        tokens = self._encoder.encode(body)
        if len(tokens) <= self.max_tokens:
            yield body
            return
        yield from splitting.cascade_split(
            body, self._encoder, self.max_tokens, self.overlap
        )

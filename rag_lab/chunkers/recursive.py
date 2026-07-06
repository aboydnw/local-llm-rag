from collections.abc import Iterator

from rag_lab.chunkers import splitting
from rag_lab.types import Chunk, Document


class RecursiveChunker:
    """Split documents via a paragraph -> sentence -> token cascade, ignoring headings."""

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
        if not document.text.strip():
            return
        position = 0
        for chunk_text in splitting.cascade_split(
            document.text, self._encoder, self.max_tokens, self.overlap
        ):
            text = chunk_text
            if self.context_header:
                text = f"{splitting.document_label(document)}\n\n{chunk_text}"
            yield Chunk(
                text=text,
                doc_path=document.path,
                heading_path=(),
                position=position,
                metadata=dict(document.metadata),
            )
            position += 1

from collections.abc import Iterator

from rag_lab.chunkers import splitting
from rag_lab.embedders.base import Embedder
from rag_lab.types import Chunk, Document


class SemanticChunker:
    """Split where adjacent-sentence embedding similarity drops, capped by max_tokens."""

    def __init__(
        self,
        embedder: Embedder,
        max_tokens: int = 512,
        similarity_threshold: float = 0.75,
        overlap: int = 50,
        context_header: bool = False,
    ) -> None:
        if overlap >= max_tokens:
            raise ValueError("overlap must be smaller than max_tokens")
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.similarity_threshold = similarity_threshold
        self.overlap = overlap
        self.context_header = context_header
        self._encoder = splitting.get_encoder()

    def chunk(self, document: Document) -> Iterator[Chunk]:
        sentences = splitting.split_sentences(document.text)
        if not sentences:
            return
        embeddings = self.embedder.embed_documents(sentences)
        position = 0
        current: list[str] = []
        previous: list[float] | None = None
        for sentence, vector in zip(sentences, embeddings, strict=True):
            if current:
                similar = _cosine(previous, vector) >= self.similarity_threshold
                fits = self._count(" ".join([*current, sentence])) <= self.max_tokens
                if not similar or not fits:
                    yield self._emit(document, current, position)
                    position += 1
                    current = self._overlap_sentences(current)
            current.append(sentence)
            previous = vector
        if current:
            yield self._emit(document, current, position)

    def _count(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def _overlap_sentences(self, sentences: list[str]) -> list[str]:
        if self.overlap <= 0:
            return []
        tail: list[str] = []
        total = 0
        for sentence in reversed(sentences):
            tail.insert(0, sentence)
            total += self._count(sentence)
            if total >= self.overlap:
                break
        return tail

    def _emit(self, document: Document, sentences: list[str], position: int) -> Chunk:
        body = " ".join(sentences)
        text = body
        if self.context_header:
            text = f"{splitting.document_label(document)}\n\n{body}"
        return Chunk(
            text=text,
            doc_path=document.path,
            heading_path=(),
            position=position,
            metadata=dict(document.metadata),
        )


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

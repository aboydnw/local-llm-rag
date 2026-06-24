import hashlib


class FakeEmbedder:
    """Deterministic embedder for tests. Hashes text into a fixed-dim vector."""

    def __init__(self, dimension: int = 16) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            values = [
                ((digest[i % len(digest)] / 255.0) * 2.0) - 1.0
                for i in range(self._dimension)
            ]
            out.append(values)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]

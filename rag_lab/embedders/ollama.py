from functools import lru_cache

import httpx
import ollama

_TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)


@lru_cache(maxsize=1)
def _client() -> ollama.Client:
    return ollama.Client(timeout=_TIMEOUT)


def prefixes_for_model(model: str) -> tuple[str, str]:
    """Return the (document, query) task-instruction prefixes for an embedding model."""
    if model.startswith("nomic-embed-text"):
        return "search_document: ", "search_query: "
    return "", ""


class OllamaEmbedder:
    """Embeds text via an Ollama-served embedding model, with task prefixes."""

    def __init__(
        self,
        model: str,
        dimension: int,
        *,
        document_prefix: str = "",
        query_prefix: str = "",
    ) -> None:
        self.model = model
        self._dimension = dimension
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = _client().embed(model=self.model, input=texts)
        return [list(v) for v in response["embeddings"]]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed([f"{self.document_prefix}{t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.embed([f"{self.query_prefix}{text}"])[0]

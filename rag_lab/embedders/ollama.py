from functools import lru_cache

import ollama


@lru_cache(maxsize=1)
def _client() -> ollama.Client:
    return ollama.Client()


class OllamaEmbedder:
    """Embeds text via an Ollama-served embedding model."""

    def __init__(self, model: str, dimension: int) -> None:
        self.model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = _client().embed(model=self.model, input=texts)
        return [list(v) for v in response["embeddings"]]

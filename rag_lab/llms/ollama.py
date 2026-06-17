from functools import lru_cache

import ollama


@lru_cache(maxsize=1)
def _client() -> ollama.Client:
    return ollama.Client()


class OllamaLLM:
    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        response = _client().chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

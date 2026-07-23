import os
from collections.abc import Iterator
from functools import lru_cache
from time import monotonic

from dotenv import find_dotenv, load_dotenv

from rag_lab.types import GenerationStats


def _load_local_env() -> None:
    """Load the nearest project-local environment file without overriding the shell."""
    path = find_dotenv(filename=".env-local", usecwd=True)
    if path:
        load_dotenv(path, override=False)


@lru_cache(maxsize=1)
def _client():
    """Build and cache the authenticated Gemini SDK client."""
    _load_local_env()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env-local-example to .env-local and add your key."
        )
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError("Gemini support is not installed. Run: uv sync --extra gemini") from exc
    return genai.Client(api_key=api_key)


class GeminiLLM:
    """Provider adapter for streamed Gemini text and structured generation."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._last_stats: GenerationStats | None = None

    def generate(self, prompt: str, schema: dict | None = None) -> str:
        """Generate one complete response, optionally constrained by a JSON schema."""
        return "".join(self.stream(prompt, schema))

    def stream(self, prompt: str, schema: dict | None = None) -> Iterator[str]:
        """Yield response text as Gemini streams it and retain final usage statistics."""
        self._last_stats = None
        try:
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "Gemini support is not installed. Run: uv sync --extra gemini"
            ) from exc

        config = None
        if schema is not None:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=schema,
            )

        started = monotonic()
        usage = None
        try:
            chunks = _client().models.generate_content_stream(
                model=self.model, contents=prompt, config=config
            )
            for chunk in chunks:
                if chunk.text:
                    yield chunk.text
                if chunk.usage_metadata is not None:
                    usage = chunk.usage_metadata
        except Exception as exc:
            raise RuntimeError(f"Gemini generation failed for {self.model}: {exc}") from exc

        if usage is not None:
            self._last_stats = GenerationStats(
                prompt_tokens=int(usage.prompt_token_count or 0),
                prompt_eval_ms=0.0,
                output_tokens=int(usage.candidates_token_count or 0),
                generation_ms=(monotonic() - started) * 1000,
            )

    def last_stats(self) -> GenerationStats | None:
        """Return usage and elapsed-time statistics for the most recent generation."""
        return self._last_stats

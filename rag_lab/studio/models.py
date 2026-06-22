from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import ollama


@dataclass(frozen=True, slots=True)
class PullEvent:
    """A single progress update from an Ollama model pull."""

    status: str
    fraction: float | None


def _normalize(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def is_installed(model: str, installed: Iterable[str]) -> bool:
    """Return whether ``model`` is in ``installed``, treating a tagless name as ``:latest``."""
    target = _normalize(model)
    return any(_normalize(name) == target for name in installed)


def _client() -> ollama.Client:
    return ollama.Client()


def installed_models(client: ollama.Client | None = None) -> list[str]:
    """Return the names of all locally installed Ollama models."""
    client = client or _client()
    return [entry.model for entry in client.list().models]


def pull_progress(model: str, client: ollama.Client | None = None) -> Iterator[PullEvent]:
    """Stream an Ollama model pull, yielding normalized progress events."""
    client = client or _client()
    for chunk in client.pull(model, stream=True):
        total = getattr(chunk, "total", None)
        completed = getattr(chunk, "completed", None)
        fraction = completed / total if total and completed is not None else None
        yield PullEvent(status=getattr(chunk, "status", ""), fraction=fraction)

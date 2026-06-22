from collections.abc import Iterable
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
    """Return whether ``model`` is present in ``installed``, treating a tagless name as ``:latest``."""
    target = _normalize(model)
    return any(_normalize(name) == target for name in installed)


def _client() -> ollama.Client:
    return ollama.Client()


def installed_models(client: ollama.Client | None = None) -> list[str]:
    """Return the names of all locally installed Ollama models."""
    client = client or _client()
    return [entry.model for entry in client.list().models]

from collections.abc import Iterable
from dataclasses import dataclass


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

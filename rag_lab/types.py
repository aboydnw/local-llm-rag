from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Document:
    """A loaded source document before chunking."""

    path: Path
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationStats:
    """Timing and token counts for one LLM generation."""

    prompt_tokens: int
    prompt_eval_ms: float
    output_tokens: int
    generation_ms: float

    @property
    def prompt_eval_tps(self) -> float:
        """Prompt-eval throughput in tokens per second."""
        return self.prompt_tokens / (self.prompt_eval_ms / 1000.0) if self.prompt_eval_ms else 0.0

    @property
    def generation_tps(self) -> float:
        """Generation throughput in tokens per second."""
        return self.output_tokens / (self.generation_ms / 1000.0) if self.generation_ms else 0.0

    @property
    def total_ms(self) -> float:
        """Total prompt-eval plus generation time in milliseconds."""
        return self.prompt_eval_ms + self.generation_ms


def combine_stats(stats: list[GenerationStats]) -> GenerationStats | None:
    """Sum tokens and durations across multiple generations (e.g. agent steps)."""
    if not stats:
        return None
    return GenerationStats(
        prompt_tokens=sum(s.prompt_tokens for s in stats),
        prompt_eval_ms=sum(s.prompt_eval_ms for s in stats),
        output_tokens=sum(s.output_tokens for s in stats),
        generation_ms=sum(s.generation_ms for s in stats),
    )


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit produced by a Chunker."""

    text: str
    doc_path: Path
    heading_path: tuple[str, ...]
    position: int
    metadata: dict[str, str] = field(default_factory=dict)

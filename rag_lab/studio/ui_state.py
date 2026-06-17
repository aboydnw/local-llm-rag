from rag_lab.config import Config

EXPENSIVE_KNOBS: set[str] = {"chunker", "embedder"}
CHEAP_KNOBS: set[str] = {"retriever", "llm"}


def normalized_weights(vector_weight: float) -> tuple[float, float]:
    """Clamp ``vector_weight`` to [0, 1] and return it with its complement."""
    v = min(1.0, max(0.0, vector_weight))
    return v, round(1.0 - v, 4)


def init_state(session, config: Config, corpus: str, golden: str) -> None:
    """Seed session-state keys for config, corpus, and golden set if absent."""
    session.setdefault("config", config)
    session.setdefault("corpus", corpus)
    session.setdefault("golden", golden)

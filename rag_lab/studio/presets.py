from dataclasses import dataclass
from typing import Literal

from rag_lab.config import Config

SWEEP_K = 5


@dataclass(frozen=True)
class Preset:
    name: str
    retriever_type: Literal["hybrid", "vector", "bm25"]
    bm25_weight: float = 0.5
    vector_weight: float = 0.5
    reranker: Literal["none", "llm"] = "none"


PRESETS: tuple[Preset, ...] = (
    Preset("vector", "vector"),
    Preset("bm25", "bm25"),
    Preset("hybrid-balanced", "hybrid", 0.5, 0.5),
    Preset("hybrid-vector-heavy", "hybrid", 0.25, 0.75),
    Preset("hybrid-bm25-heavy", "hybrid", 0.75, 0.25),
    Preset("vector-rerank", "vector", reranker="llm"),
    Preset("bm25-rerank", "bm25", reranker="llm"),
    Preset("hybrid-rerank", "hybrid", 0.5, 0.5, "llm"),
)


def apply_preset(base: Config, preset: Preset) -> Config:
    """Return a copy of base with only retrieval knobs overridden by the preset."""
    cfg = base.model_copy(deep=True)
    cfg.retriever.type = preset.retriever_type
    cfg.retriever.bm25_weight = preset.bm25_weight
    cfg.retriever.vector_weight = preset.vector_weight
    cfg.retriever.reranker = preset.reranker
    cfg.retriever.k = SWEEP_K
    return cfg

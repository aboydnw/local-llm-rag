from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from rag_lab.agent.agent import DEFAULT_AGENT_INSTRUCTIONS
from rag_lab.prompts import DEFAULT_SYSTEM_INSTRUCTIONS


class ChunkerConfig(BaseModel):
    type: Literal["markdown_aware", "fixed", "recursive", "semantic"] = "markdown_aware"
    max_tokens: int = Field(default=512, gt=0)
    overlap: int = Field(default=50, ge=0)
    context_header: bool = True
    similarity_threshold: float = Field(default=0.75, ge=0, le=1)


class EmbedderConfig(BaseModel):
    type: Literal["ollama", "fake"] = "ollama"
    model: str = "nomic-embed-text"


class LLMConfig(BaseModel):
    type: Literal["ollama"] = "ollama"
    model: str = "llama3.2:3b"
    think: bool | None = None
    """Thinking-mode toggle for reasoning models (e.g. Qwen3). None leaves the
    model default; False disables thinking to cut latency and answer pollution."""


class RetrieverConfig(BaseModel):
    type: Literal["hybrid", "vector", "bm25"] = "hybrid"
    bm25_weight: float = Field(default=0.5, ge=0, le=1)
    vector_weight: float = Field(default=0.5, ge=0, le=1)
    k: int = Field(default=5, gt=0)
    reranker: Literal["none", "llm"] = "none"
    rerank_candidates: int = Field(default=30, gt=0)


class EvalConfig(BaseModel):
    deepeval: bool = False
    deepeval_model: str | None = None


class PromptConfig(BaseModel):
    system_instructions: str = DEFAULT_SYSTEM_INSTRUCTIONS


class AgentConfig(BaseModel):
    enabled: bool = False
    max_steps: int = Field(default=6, gt=0)
    final_k: int = Field(default=5, gt=0)
    structured_output: bool = False
    instructions: str = DEFAULT_AGENT_INSTRUCTIONS
    tools: list[
        Literal[
            "vector_search",
            "keyword_search",
            "list_documents",
            "fetch_document",
        ]
    ] = Field(
        min_length=1,
        default_factory=lambda: [
            "vector_search",
            "keyword_search",
            "list_documents",
            "fetch_document",
        ]
    )


class Config(BaseModel):
    chunker: ChunkerConfig = ChunkerConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    llm: LLMConfig = LLMConfig()
    retriever: RetrieverConfig = RetrieverConfig()
    eval: EvalConfig = EvalConfig()
    prompt: PromptConfig = PromptConfig()
    agent: AgentConfig = AgentConfig()


EMBEDDING_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}


def embedding_dimension(model: str) -> int:
    """Return the vector dimension for a known Ollama embedding model."""
    try:
        return EMBEDDING_DIMENSIONS[model]
    except KeyError:
        raise ValueError(f"Unknown embedding model: {model}") from None


def config_summary(config: Config) -> str:
    """One-line human summary of the swappable knobs in a config."""
    r = config.retriever
    summary = (
        f"chunker={config.chunker.type}@{config.chunker.max_tokens} "
        f"retriever={r.type}(v={r.vector_weight},b={r.bm25_weight}) "
        f"llm={config.llm.model} embedder={config.embedder.model}"
    )
    if config.agent.enabled:
        a = config.agent
        summary += f" agent=on(steps={a.max_steps},tools={len(a.tools)})"
    return summary


DEFAULT_CONFIG_YAML = """\
chunker:
  type: markdown_aware
  max_tokens: 512
  overlap: 50
  context_header: true
embedder:
  type: ollama
  model: nomic-embed-text
llm:
  type: ollama
  model: llama3.2:3b
retriever:
  type: hybrid
  bm25_weight: 0.5
  vector_weight: 0.5
  k: 5
  reranker: none
  rerank_candidates: 30
eval:
  deepeval: false
agent:
  enabled: false
  max_steps: 6
  final_k: 5
  structured_output: false
  tools:
    - vector_search
    - keyword_search
    - list_documents
    - fetch_document
"""


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config(**raw)


def write_default_config(path: Path) -> None:
    prompt_yaml = yaml.safe_dump(
        {"prompt": {"system_instructions": DEFAULT_SYSTEM_INSTRUCTIONS}},
        default_flow_style=False,
        allow_unicode=True,
    )
    path.write_text(DEFAULT_CONFIG_YAML + prompt_yaml, encoding="utf-8")

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ChunkerConfig(BaseModel):
    type: Literal["markdown_aware", "fixed"] = "markdown_aware"
    max_tokens: int = Field(default=512, gt=0)
    overlap: int = Field(default=50, ge=0)


class EmbedderConfig(BaseModel):
    type: Literal["ollama", "fake"] = "ollama"
    model: str = "nomic-embed-text"


class LLMConfig(BaseModel):
    type: Literal["ollama"] = "ollama"
    model: str = "llama3.2:3b"


class RetrieverConfig(BaseModel):
    type: Literal["hybrid", "vector", "bm25"] = "hybrid"
    bm25_weight: float = Field(default=0.5, ge=0, le=1)
    vector_weight: float = Field(default=0.5, ge=0, le=1)
    k: int = Field(default=5, gt=0)


class Config(BaseModel):
    chunker: ChunkerConfig = ChunkerConfig()
    embedder: EmbedderConfig = EmbedderConfig()
    llm: LLMConfig = LLMConfig()
    retriever: RetrieverConfig = RetrieverConfig()


DEFAULT_CONFIG_YAML = """\
chunker:
  type: markdown_aware
  max_tokens: 512
  overlap: 50
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
"""


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text()) or {}
    return Config(**raw)


def write_default_config(path: Path) -> None:
    path.write_text(DEFAULT_CONFIG_YAML)

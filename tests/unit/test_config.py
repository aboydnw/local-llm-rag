from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_lab.config import Config, load_config


def test_load_config_parses_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "rag.yml"
    cfg_path.write_text(
        """
chunker:
  type: markdown_aware
  max_tokens: 256
  overlap: 25
embedder:
  type: ollama
  model: nomic-embed-text
llm:
  type: ollama
  model: llama3.2:3b
retriever:
  type: hybrid
  bm25_weight: 0.4
  vector_weight: 0.6
  k: 7
"""
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg, Config)
    assert cfg.chunker.max_tokens == 256
    assert cfg.retriever.k == 7


def test_load_config_rejects_unknown_chunker_type(tmp_path: Path) -> None:
    cfg_path = tmp_path / "rag.yml"
    cfg_path.write_text("chunker:\n  type: wat\n")
    with pytest.raises(ValidationError):
        load_config(cfg_path)

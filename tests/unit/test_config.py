from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_lab import config as config_mod
from rag_lab.config import Config, load_config, write_default_config
from rag_lab.prompts import DEFAULT_SYSTEM_INSTRUCTIONS


def test_agent_config_rejects_empty_tools() -> None:
    from rag_lab.config import AgentConfig

    with pytest.raises(ValidationError):
        AgentConfig(tools=[])


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


def test_embedding_dimension_known_model():
    assert config_mod.embedding_dimension("nomic-embed-text") == 768


def test_embedding_dimension_unknown_model_raises():
    with pytest.raises(ValueError):
        config_mod.embedding_dimension("does-not-exist")


def test_eval_config_defaults_to_disabled():
    from rag_lab.config import EvalConfig
    assert EvalConfig().deepeval is False
    assert Config().eval.deepeval is False


def test_load_config_parses_eval_section(tmp_path: Path) -> None:
    cfg_path = tmp_path / "rag.yml"
    cfg_path.write_text("eval:\n  deepeval: true\n")
    cfg = load_config(cfg_path)
    assert cfg.eval.deepeval is True


def test_agent_config_defaults_to_disabled():
    from rag_lab.config import AgentConfig
    assert AgentConfig().enabled is False
    assert Config().agent.enabled is False
    assert Config().agent.max_steps == 6
    assert Config().agent.final_k == 5
    assert Config().agent.tools == [
        "vector_search",
        "keyword_search",
        "list_documents",
        "fetch_document",
    ]


def test_load_config_parses_agent_section(tmp_path: Path) -> None:
    cfg_path = tmp_path / "rag.yml"
    cfg_path.write_text(
        "agent:\n"
        "  enabled: true\n"
        "  max_steps: 3\n"
        "  final_k: 2\n"
        "  tools:\n"
        "    - vector_search\n"
        "    - keyword_search\n"
    )
    cfg = load_config(cfg_path)
    assert cfg.agent.enabled is True
    assert cfg.agent.max_steps == 3
    assert cfg.agent.final_k == 2
    assert cfg.agent.tools == ["vector_search", "keyword_search"]


def test_load_config_rejects_unknown_agent_tool(tmp_path: Path) -> None:
    cfg_path = tmp_path / "rag.yml"
    cfg_path.write_text("agent:\n  tools:\n    - bogus_tool\n")
    with pytest.raises(ValidationError):
        load_config(cfg_path)


def test_default_config_file_has_agent_section(tmp_path) -> None:
    path = tmp_path / "rag.yml"
    write_default_config(path)
    assert load_config(path).agent.enabled is False


def test_agent_config_default_instructions_match_engine():
    from rag_lab.agent.agent import DEFAULT_AGENT_INSTRUCTIONS
    assert Config().agent.instructions == DEFAULT_AGENT_INSTRUCTIONS


def test_config_summary_mentions_key_knobs():
    summary = config_mod.config_summary(Config())
    assert "markdown_aware" in summary
    assert "hybrid" in summary
    assert "llama3.2:3b" in summary
    assert "nomic-embed-text" in summary


def test_config_summary_marks_agent_runs():
    config = Config()
    config.agent.enabled = True
    config.agent.max_steps = 4
    summary = config_mod.config_summary(config)
    assert "agent=on" in summary
    assert "steps=4" in summary


def test_config_summary_omits_agent_when_disabled():
    assert "agent" not in config_mod.config_summary(Config())


def test_config_prompt_defaults_to_canonical_instructions():
    assert Config().prompt.system_instructions == DEFAULT_SYSTEM_INSTRUCTIONS


def test_default_config_file_roundtrips_prompt(tmp_path):
    path = tmp_path / "rag.yml"
    write_default_config(path)
    assert load_config(path).prompt.system_instructions == DEFAULT_SYSTEM_INSTRUCTIONS


def test_load_config_reads_custom_prompt(tmp_path):
    path = tmp_path / "rag.yml"
    path.write_text("prompt:\n  system_instructions: 'Be terse.'\n")
    assert load_config(path).prompt.system_instructions == "Be terse."

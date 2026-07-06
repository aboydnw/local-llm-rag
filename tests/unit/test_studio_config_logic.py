from rag_lab.config import Config
from rag_lab.studio import config_logic


def test_retrieval_method_reads_retriever_type_when_agent_off():
    cfg = Config()
    cfg.agent.enabled = False
    cfg.retriever.type = "bm25"
    assert config_logic.retrieval_method(cfg) == "bm25"


def test_retrieval_method_is_agent_when_enabled():
    cfg = Config()
    cfg.agent.enabled = True
    cfg.retriever.type = "hybrid"
    assert config_logic.retrieval_method(cfg) == "agent"


def test_apply_agent_method_enables_agent():
    cfg = Config()
    config_logic.apply_retrieval_method(cfg, "agent")
    assert cfg.agent.enabled is True


def test_apply_non_agent_method_sets_type_and_disables_agent():
    cfg = Config()
    cfg.agent.enabled = True
    config_logic.apply_retrieval_method(cfg, "vector")
    assert cfg.agent.enabled is False
    assert cfg.retriever.type == "vector"


def test_retrieval_methods_constant():
    assert config_logic.RETRIEVAL_METHODS == ["vector", "bm25", "hybrid", "agent"]


def test_llm_model_options_appends_current_and_sentinel():
    opts = config_logic.llm_model_options(["llama3.2:3b"], "llama3.2:3b")
    assert opts[-1] == config_logic.PULL_SENTINEL
    assert "llama3.2:3b" in opts


def test_llm_model_options_includes_current_when_not_installed():
    opts = config_logic.llm_model_options(["other:latest"], "llama3.2:3b")
    assert "llama3.2:3b" in opts
    assert opts[-1] == config_logic.PULL_SENTINEL


def test_llm_model_options_no_duplicate_current():
    opts = config_logic.llm_model_options(["llama3.2:3b", "llama3.2:3b"], "llama3.2:3b")
    assert opts.count("llama3.2:3b") == 1


def test_embedder_labels_mark_uninstalled():
    labels = config_logic.embedder_model_labels(["nomic-embed-text:latest"])
    assert labels["nomic-embed-text"] == "nomic-embed-text"
    assert "not installed" in labels["all-minilm"]


def test_config_expanded_true_before_action():
    assert config_logic.config_expanded({}, "playground") is True


def test_config_expanded_false_after_action():
    session = {"playground_config_acted": True}
    assert config_logic.config_expanded(session, "playground") is False


def test_config_expanded_is_namespaced_by_page():
    session = {"evaluate_config_acted": True}
    assert config_logic.config_expanded(session, "playground") is True


def test_corpus_options_appends_sentinel_last():
    opts = config_logic.corpus_options(["titiler", "stac"])
    assert opts == ["titiler", "stac", config_logic.ADD_CORPUS_SENTINEL]


def test_corpus_options_empty_is_just_sentinel():
    assert config_logic.corpus_options([]) == [config_logic.ADD_CORPUS_SENTINEL]

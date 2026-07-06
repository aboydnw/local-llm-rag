from rag_lab.config import EMBEDDING_DIMENSIONS, Config
from rag_lab.studio import models as models_mod

RETRIEVAL_METHODS: list[str] = ["vector", "bm25", "hybrid", "agent"]
PULL_SENTINEL: str = "➕ Pull another model…"
ADD_CORPUS_SENTINEL: str = "➕ Add corpus…"


def retrieval_method(config: Config) -> str:
    """Return the active retrieval method: a retriever type, or 'agent' when agentic mode is on."""
    if config.agent.enabled:
        return "agent"
    return config.retriever.type


def apply_retrieval_method(config: Config, method: str) -> None:
    """Write a chosen retrieval method back onto the config, toggling agent mode as needed."""
    if method == "agent":
        config.agent.enabled = True
    else:
        config.agent.enabled = False
        config.retriever.type = method


def llm_model_options(installed: list[str], current: str) -> list[str]:
    """Build the LLM dropdown: installed models, the current selection, and a pull sentinel last."""
    options = list(dict.fromkeys(installed))
    if current not in options:
        options.append(current)
    options.append(PULL_SENTINEL)
    return options


def embedder_model_labels(installed: list[str]) -> dict[str, str]:
    """Map each known embedding model to a display label, flagging ones not installed."""
    return {
        model: model if models_mod.is_installed(model, installed) else f"{model} (not installed)"
        for model in EMBEDDING_DIMENSIONS
    }


def config_expanded(session, page_key: str) -> bool:
    """The config panel starts expanded and collapses once the page records an action."""
    return not session.get(f"{page_key}_config_acted", False)


def corpus_options(names: list[str]) -> list[str]:
    """Corpus dropdown options: existing names followed by an 'add corpus' sentinel."""
    return [*names, ADD_CORPUS_SENTINEL]

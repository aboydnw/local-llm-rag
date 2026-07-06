from rag_lab.config import EMBEDDING_DIMENSIONS, Config
from rag_lab.studio import models as models_mod

PULL_SENTINEL: str = "➕ Pull another model…"
ADD_CORPUS_SENTINEL: str = "➕ Add corpus…"


def search_mode(config: Config) -> str:
    """Return the active search mode: 'agentic' when agent mode is on, else 'direct'."""
    return "agentic" if config.agent.enabled else "direct"


def apply_search_mode(config: Config, mode: str) -> None:
    """Apply a chosen search mode. Direct always uses the hybrid retriever (a v↔k slider)."""
    if mode == "agentic":
        config.agent.enabled = True
    else:
        config.agent.enabled = False
        config.retriever.type = "hybrid"


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

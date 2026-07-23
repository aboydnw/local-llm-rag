from pathlib import Path

import streamlit as st

from rag_lab.eval.golden_set import load_golden_set
from rag_lab.studio import corpora as corpora_mod
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio.workspace import Workspace


def _search_label(config) -> str:
    if config.agent.enabled:
        return f"Agentic · {config.llm.model}"
    return (
        f"Hybrid {config.retriever.vector_weight:.0%} vector · "
        f"k={config.retriever.k} · {config.llm.model}"
    )


def render(session):
    """Render brand and the active experiment context above navigation."""
    st.sidebar.title("rag-lab")
    st.sidebar.caption("Build, test, and measure local RAG systems")
    st.sidebar.divider()
    st.sidebar.markdown("**Active experiment**")
    ws = Workspace.default()
    corpus = corpora_mod.resolve_active_corpus(ws, session.get("corpus_name"), session["corpus"])
    try:
        ready = indexer_mod.status(ws, corpus, session["config"]).cached
    except Exception:  # noqa: BLE001
        ready = False
    golden_path = Path(session["golden"])
    try:
        question_count = len(load_golden_set(golden_path)) if golden_path.exists() else 0
    except (OSError, ValueError):
        question_count = 0
    st.sidebar.markdown(f"**Corpus:** {corpus.label}")
    st.sidebar.caption(
        f"{'Index ready' if ready else 'Index needs build'} · {question_count} test questions"
    )
    st.sidebar.caption(_search_label(session["config"]))
    return session["config"]

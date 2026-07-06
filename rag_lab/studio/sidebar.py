import streamlit as st

from rag_lab.config import Config


def render(session) -> Config:
    """Render the nav-only sidebar. Configuration now lives in the on-page config panel."""
    st.sidebar.title("rag-lab studio")
    st.sidebar.caption("Local-first RAG cockpit")
    return session["config"]

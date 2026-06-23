from pathlib import Path

import streamlit as st

from rag_lab import config as config_mod
from rag_lab.config import Config
from rag_lab.studio import sidebar, ui_state
from rag_lab.studio.pages import corpora as corpora_page
from rag_lab.studio.pages import evaluate, golden, playground, runs


def _load_config() -> Config:
    """Load ``rag.yml`` if present, otherwise return a default ``Config``."""
    path = Path("rag.yml")
    if path.exists():
        return config_mod.load_config(path)
    return Config()


def main() -> None:
    """Initialize session state, render the sidebar, and route the studio pages."""
    st.set_page_config(page_title="rag-lab studio", layout="wide")
    ui_state.init_state(
        st.session_state, _load_config(), ".", "examples/devseed-oss/golden.yml"
    )
    sidebar.render(st.session_state)

    pages = st.navigation([
        st.Page(playground.render, title="Ask playground", icon="💬",
                url_path="playground", default=True),
        st.Page(corpora_page.render, title="Corpus", icon="📚", url_path="corpus"),
        st.Page(evaluate.render, title="Evaluate", icon="📊", url_path="evaluate"),
        st.Page(runs.render, title="Runs", icon="🏆", url_path="runs"),
        st.Page(golden.render, title="Golden set", icon="✏️", url_path="golden"),
    ])
    pages.run()


main()

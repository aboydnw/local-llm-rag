import os
from pathlib import Path

import streamlit as st

from rag_lab import config as config_mod
from rag_lab.config import Config
from rag_lab.studio import corpora as corpora_mod
from rag_lab.studio import sidebar, ui_state
from rag_lab.studio.pages import corpora as corpora_page
from rag_lab.studio.pages import evaluate, golden, playground, runs
from rag_lab.studio.workspace import Workspace


def _load_config() -> Config:
    """Load ``rag.yml`` if present, otherwise return a default ``Config``."""
    path = Path("rag.yml")
    if path.exists():
        return config_mod.load_config(path)
    return Config()


def main() -> None:
    """Initialize session state, render the sidebar, and route the studio pages."""
    st.set_page_config(page_title="rag-lab studio", page_icon="🧪", layout="wide")
    ui_state.init_state(st.session_state, _load_config(), ".", "examples/devseed-oss/golden.yml")
    ws = Workspace.default()
    ws.initialize()
    corpora_mod.ensure_default_corpus(ws)
    sidebar.render(st.session_state)

    if os.environ.get("STUDIO_FEEDBACK") == "1":
        import streamlit_testing_feedback as stf

        with st.sidebar:
            zip_path = stf.feedback_recorder(dir=".feedback")
        if zip_path:
            st.sidebar.success(f"feedback saved: {zip_path.name}")

    pages = st.navigation(
        {
            "Setup": [
                st.Page(
                    corpora_page.render, title="Corpus", icon="📚", url_path="corpus", default=True
                ),
                st.Page(golden.render, title="Test questions", icon="✏️", url_path="test-questions"),
            ],
            "Experiment": [
                st.Page(playground.render, title="Playground", icon="💬", url_path="playground"),
                st.Page(evaluate.render, title="Run evaluation", icon="📊", url_path="evaluate"),
            ],
            "Results": [
                st.Page(runs.render, title="Evaluation reports", icon="🏆", url_path="reports"),
            ],
        }
    )
    pages.run()


main()

import streamlit as st

from rag_lab.studio import models as models_mod


def run_pull(model: str) -> None:
    """Stream an Ollama model pull into a main-area progress bar."""
    bar = st.progress(0.0, text=f"Pulling {model}...")
    fraction = 0.0
    try:
        for event in models_mod.pull_progress(model):
            if event.fraction is not None:
                fraction = event.fraction
            bar.progress(fraction, text=f"{model}: {event.status}")
        bar.progress(1.0, text=f"{model}: done")
        st.toast(f"Pulled {model}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Pull failed: {exc}")

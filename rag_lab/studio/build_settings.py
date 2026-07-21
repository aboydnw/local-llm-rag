import streamlit as st

from rag_lab.config import EMBEDDING_DIMENSIONS, Config
from rag_lab.studio import config_logic, pull_ui
from rag_lab.studio import models as models_mod


def render(cfg: Config) -> None:
    """Render the index-time knobs — chunker and embedder — that a build depends on."""
    try:
        installed = models_mod.installed_models()
        ollama_error: str | None = None
    except Exception as exc:  # noqa: BLE001
        installed = []
        ollama_error = str(exc)

    st.subheader("Chunker  🟠 re-index")
    chunker_types = ["markdown_aware", "fixed", "recursive", "semantic"]
    cfg.chunker.type = st.selectbox(
        "type", chunker_types, index=chunker_types.index(cfg.chunker.type),
        help="How documents are split. 'markdown_aware' splits on headings; "
        "'fixed' splits on a flat token count; 'recursive' splits on "
        "paragraphs then sentences then tokens; 'semantic' splits where "
        "adjacent sentences diverge in meaning (slower — embeds at ingest).",
    )
    max_tokens_default = min(2048, max(64, cfg.chunker.max_tokens))
    cfg.chunker.max_tokens = st.slider(
        "max_tokens", 64, 2048, max_tokens_default, 32,
        help="Largest chunk size, in tokens.",
    )
    overlap_ceiling = max(0, cfg.chunker.max_tokens - 1)
    overlap_default = min(256, overlap_ceiling, max(0, cfg.chunker.overlap))
    cfg.chunker.overlap = st.slider(
        "overlap", 0, min(256, overlap_ceiling), overlap_default, 8,
        help="Tokens repeated between adjacent chunks so ideas spanning a boundary aren't lost. "
        "Capped below max_tokens.",
    )
    if cfg.chunker.type == "semantic":
        cfg.chunker.similarity_threshold = st.slider(
            "similarity_threshold", 0.0, 1.0, cfg.chunker.similarity_threshold, 0.05,
            help="Higher = more, smaller chunks; lower = fewer, larger chunks. "
            "Only used by the 'semantic' chunker.",
        )

    st.subheader("Embedder  🟠 re-index")
    if ollama_error is not None:
        st.info("Ollama not reachable — is it running?")
    embed_models = list(EMBEDDING_DIMENSIONS)
    labels = config_logic.embedder_model_labels(installed)
    if cfg.embedder.model not in embed_models:
        embed_models.append(cfg.embedder.model)
        labels = {**labels, cfg.embedder.model: f"{cfg.embedder.model} (unknown dimension)"}
    embed_index = embed_models.index(cfg.embedder.model)
    cfg.embedder.model = st.selectbox(
        "embedding model", embed_models, index=embed_index,
        format_func=lambda m: labels[m],
        help="Ollama model that turns text into vectors. Changing it requires a rebuild.",
    )
    if ollama_error is None and not models_mod.is_installed(cfg.embedder.model, installed):
        if st.button(f"Pull {cfg.embedder.model}", key="build_pull_embed"):
            pull_ui.run_pull(cfg.embedder.model)

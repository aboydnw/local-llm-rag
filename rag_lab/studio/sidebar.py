import streamlit as st

from rag_lab.config import EMBEDDING_DIMENSIONS, Config, config_summary
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio import ui_state
from rag_lab.studio.workspace import Workspace


def render(session) -> Config:
    cfg: Config = session["config"]
    st.sidebar.header("Corpus")
    session["corpus"] = st.sidebar.text_input("Corpus directory", value=session["corpus"])
    session["golden"] = st.sidebar.text_input("Golden set", value=session["golden"])

    st.sidebar.header("Chunker  🟠 re-index")
    cfg.chunker.type = st.sidebar.selectbox("type", ["markdown_aware", "fixed"],
                                            index=["markdown_aware", "fixed"].index(cfg.chunker.type))
    cfg.chunker.max_tokens = st.sidebar.slider("max_tokens", 64, 2048, cfg.chunker.max_tokens, 32)
    cfg.chunker.overlap = st.sidebar.slider("overlap", 0, 256, cfg.chunker.overlap, 8)

    st.sidebar.header("Embedder  🟠 re-index")
    models = list(EMBEDDING_DIMENSIONS)
    cfg.embedder.model = st.sidebar.selectbox("embedding model", models,
                                              index=models.index(cfg.embedder.model)
                                              if cfg.embedder.model in models else 0)

    st.sidebar.header("LLM  🟢 cheap")
    cfg.llm.model = st.sidebar.text_input("llm model", value=cfg.llm.model)

    st.sidebar.header("Retriever  🟢 cheap")
    rtypes = ["hybrid", "vector", "bm25"]
    cfg.retriever.type = st.sidebar.radio("type", rtypes, index=rtypes.index(cfg.retriever.type))
    vector_weight = st.sidebar.slider("vector_weight", 0.0, 1.0, cfg.retriever.vector_weight, 0.05)
    cfg.retriever.vector_weight, cfg.retriever.bm25_weight = ui_state.normalized_weights(vector_weight)
    st.sidebar.caption(f"bm25_weight = {cfg.retriever.bm25_weight}")
    cfg.retriever.k = st.sidebar.slider("k", 1, 20, cfg.retriever.k)

    ws = Workspace.default()
    ws.initialize()
    status = indexer_mod.status(ws, session["corpus"], cfg)
    st.sidebar.divider()
    if status.cached:
        st.sidebar.success("Index: ✓ cached")
    else:
        st.sidebar.warning("⚠ this config needs a build")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Save to rag.yml"):
        import yaml
        from pathlib import Path
        Path("rag.yml").write_text(yaml.safe_dump(cfg.model_dump()))
        st.sidebar.toast("Saved rag.yml")
    if col2.button("Build index"):
        with st.spinner("Building index..."):
            indexer_mod.ensure_index(ws, session["corpus"], cfg)
        st.sidebar.toast("Index built")

    st.sidebar.caption(config_summary(cfg))
    session["config"] = cfg
    return cfg

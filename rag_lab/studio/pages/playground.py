import streamlit as st

from rag_lab.prompts import PromptBuilder
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.studio import components
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio.workspace import Workspace


def render() -> None:
    """Render the Ask playground: retrieve chunks and generate an answer for a question."""
    st.title("Ask playground")
    cfg = st.session_state["config"]
    corpus = st.session_state["corpus"]
    question = st.text_input("Question", placeholder="What is titiler?")

    ws = Workspace.default()
    status = indexer_mod.status(ws, corpus, cfg)
    if status.needs_build:
        st.warning("This config has no built index yet. Use **Build index** in the sidebar.")
        return
    question = question.strip()
    if not question:
        return

    with st.spinner("Retrieving and generating..."):
        try:
            db_path = indexer_mod.ensure_index(ws, corpus, cfg)
            embedder = components.build_embedder(cfg)
            store = SqliteVecStore(db_path, dimension=embedder.dimension)
            retriever = components.build_retriever(store, embedder, cfg)
            results = retriever.retrieve(question, k=cfg.retriever.k)
            prompt = PromptBuilder().build(question=question, results=results)
            answer = components.build_llm(cfg).generate(prompt)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Retrieval/generation failed: {exc}")
            return

    st.subheader("Answer")
    st.write(answer)
    st.subheader("Retrieved chunks")
    for i, r in enumerate(results, start=1):
        heading = " > ".join(r.chunk.heading_path) or "(no heading)"
        with st.expander(f"[{i}] {r.chunk.doc_path} — {heading}  (score {r.score:.3f})"):
            st.text(r.chunk.text[:1000])

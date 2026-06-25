import streamlit as st

from rag_lab.prompts import PromptBuilder
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.studio import components
from rag_lab.studio import corpora as corpora_mod
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio import share as share_mod
from rag_lab.studio.workspace import Workspace


def render() -> None:
    """Render the Ask playground: retrieve chunks and generate an answer for a question."""
    st.title("Ask playground")
    cfg = st.session_state["config"]
    question = st.text_input("Question", placeholder="What is titiler?")

    ws = Workspace.default()
    corpus = corpora_mod.resolve_active_corpus(
        ws, st.session_state.get("corpus_name"), st.session_state["corpus"]
    )
    if st.session_state.get("corpus_name"):
        st.caption(f"Querying corpus: **{corpus.label}**")
    else:
        st.warning(
            f"No saved corpus selected — querying the local folder `{corpus.label}`. "
            "Select a corpus in the sidebar to query it instead."
        )
    status = indexer_mod.status(ws, corpus, cfg)
    if status.needs_build:
        st.warning("This config has no built index yet. Use **Build index** in the sidebar.")
        return
    question = question.strip()
    if not question:
        return

    with st.spinner("Retrieving..."):
        try:
            db_path = indexer_mod.ensure_index(ws, corpus, cfg)
            embedder = components.build_embedder(cfg)
            store = SqliteVecStore(db_path, dimension=embedder.dimension)
            retriever = components.build_retriever(store, embedder, cfg)
            results = retriever.retrieve(question, k=cfg.retriever.k)
            prompt = PromptBuilder(
                system_instructions=cfg.prompt.system_instructions
            ).build(question=question, results=results)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Retrieval failed: {exc}")
            return

    st.subheader("Answer")
    try:
        answer = st.write_stream(components.build_llm(cfg).stream(prompt))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Generation failed: {exc}")
        return

    st.subheader("Retrieved chunks")
    for i, r in enumerate(results, start=1):
        heading = " > ".join(r.chunk.heading_path) or "(no heading)"
        with st.expander(f"[{i}] {r.chunk.doc_path} — {heading}  (score {r.score:.3f})"):
            st.text(r.chunk.text[:1000])

    st.divider()
    with st.expander("📋 Copy as markdown"):
        st.code(share_mod.format_run_markdown(question, answer, results), language="markdown")

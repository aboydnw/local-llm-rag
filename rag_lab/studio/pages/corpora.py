import streamlit as st

from rag_lab.studio import corpora as corpora_mod
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio.corpora import Corpus, Source
from rag_lab.studio.workspace import Workspace


def render() -> None:
    """Render the Corpus page: create named corpora and curate their GitHub sources."""
    st.title("Corpus")
    cfg = st.session_state["config"]
    ws = Workspace.default()
    ws.initialize()
    names = corpora_mod.list_corpora(ws)

    st.subheader("Create a corpus")
    new_name = st.text_input("New corpus name", placeholder="titiler-stack")
    if st.button("Create"):
        reason = corpora_mod.validate_name(new_name)
        if reason:
            st.error(reason)
        elif new_name.strip() in names:
            st.error("A corpus with that name already exists.")
        else:
            corpora_mod.save_corpus(ws, Corpus(name=new_name.strip(), sources=()))
            st.session_state["corpus_name"] = new_name.strip()
            st.rerun()

    if not names:
        st.info("No corpora yet. Create one above.")
        return

    st.divider()
    default = st.session_state.get("corpus_name")
    selected = st.selectbox(
        "Edit corpus", names,
        index=names.index(default) if default in names else 0,
    )
    corpus = corpora_mod.load_corpus(ws, selected)

    st.subheader("GitHub repos")
    if not corpus.sources:
        st.caption("No repos yet.")
    for source in corpus.sources:
        col1, col2 = st.columns([4, 1])
        col1.write(source.repo)
        if col2.button("Remove", key=f"rm-{source.repo}"):
            corpora_mod.save_corpus(ws, corpora_mod.remove_source(corpus, source))
            st.rerun()

    repo = st.text_input("Add a repo (owner/name)", placeholder="developmentseed/titiler")
    private = st.checkbox("Private repo (clone via gh CLI)")
    if st.button("Add repo"):
        reason = corpora_mod.validate_repo(repo)
        if reason:
            st.error(reason)
        else:
            corpora_mod.save_corpus(
                ws,
                corpora_mod.add_source(
                    corpus, Source(type="github", repo=repo.strip(), private=private)
                ),
            )
            st.rerun()

    st.divider()
    if indexer_mod.status(ws, corpus, cfg).cached:
        st.success("Index: ✓ built")
    else:
        st.warning("This corpus needs to be built.")
    col1, col2 = st.columns(2)
    if col1.button("Build index", type="primary", disabled=not corpus.sources):
        try:
            with st.spinner("Building index (cloning repos)..."):
                indexer_mod.build_index(ws, corpus, cfg)
            st.toast("Index built")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Build failed: {exc}")
    if col2.button("Delete corpus"):
        corpora_mod.delete_corpus(ws, selected)
        if st.session_state.get("corpus_name") == selected:
            st.session_state["corpus_name"] = None
        st.rerun()

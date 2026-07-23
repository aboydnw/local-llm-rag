import streamlit as st

from rag_lab.studio import build_settings, config_logic, feedback
from rag_lab.studio import corpora as corpora_mod
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio.corpora import Corpus, Source
from rag_lab.studio.workspace import Workspace


def _source_label(source: Source) -> str:
    if source.type == "github":
        return f"GitHub repository · {source.repo}" + (" · private" if source.private else "")
    if source.type == "github_issue":
        return f"GitHub issue · {source.repo}#{source.issue}"
    return f"Local folder · {source.path}"


def _add_source(corpus: Corpus) -> Source | None:
    source_type = st.selectbox("Source type", ["GitHub repository", "GitHub issue", "Local folder"])
    if source_type == "GitHub repository":
        value = st.text_input("Repository", placeholder="developmentseed/titiler")
        private = st.checkbox("Private repository (clone with GitHub CLI)")
        if st.button("Add source", type="primary"):
            if reason := corpora_mod.validate_repo(value):
                st.error(reason)
            else:
                return Source(type="github", repo=value.strip(), private=private)
    elif source_type == "GitHub issue":
        value = st.text_input("Issue", placeholder="developmentseed/titiler#42")
        if st.button("Add source", type="primary"):
            if reason := corpora_mod.validate_issue_ref(value):
                st.error(reason)
            else:
                repo, number = corpora_mod.parse_issue_ref(value)
                return Source(type="github_issue", repo=repo, issue=number)
    else:
        value = st.text_input("Folder", placeholder="./docs")
        if st.button("Add source", type="primary"):
            if reason := corpora_mod.validate_folder(value):
                st.error(reason)
            else:
                return Source(type="local", path=value.strip())
    return None


def render() -> None:
    st.title("Corpus")
    st.caption("Choose what the system can search, then build its local index.")
    cfg = st.session_state["config"]
    ws = Workspace.default()
    ws.initialize()
    names = corpora_mod.list_corpora(ws)
    options = config_logic.corpus_options(names)
    default = st.session_state.get("corpus_name")
    choice = st.selectbox(
        "Corpus", options, index=options.index(default) if default in options else 0
    )
    if choice == config_logic.ADD_CORPUS_SENTINEL:
        name = st.text_input("Corpus name", placeholder="titiler-stack")
        if st.button("Create corpus", type="primary"):
            if reason := corpora_mod.validate_name(name):
                st.error(reason)
            elif name.strip() in names:
                st.error("A corpus with that name already exists.")
            else:
                corpora_mod.save_corpus(ws, Corpus(name=name.strip(), sources=()))
                st.session_state["corpus_name"] = name.strip()
                st.rerun()
        return

    st.session_state["corpus_name"] = choice
    corpus = corpora_mod.load_corpus(ws, choice)
    sources_tab, index_tab, danger_tab = st.tabs(["1 · Sources", "2 · Build index", "Manage"])
    with sources_tab:
        st.subheader("Sources")
        st.caption("A corpus can combine repositories, individual issues, and local folders.")
        if not corpus.sources:
            st.info("Add the first source to continue.")
        for i, source in enumerate(corpus.sources):
            label, action = st.columns([5, 1])
            label.write(_source_label(source))
            if action.button("Remove", key=f"remove-source-{i}"):
                corpora_mod.save_corpus(ws, corpora_mod.remove_source(corpus, source))
                st.rerun()
        st.divider()
        st.markdown("**Add a source**")
        if source := _add_source(corpus):
            corpora_mod.save_corpus(ws, corpora_mod.add_source(corpus, source))
            st.rerun()

    with index_tab:
        st.subheader("Index settings")
        st.caption(
            "These settings determine how sources are split and embedded. "
            "Changes require a rebuild."
        )
        build_settings.render(cfg)
        status = indexer_mod.status(ws, corpus, cfg)
        if status.cached:
            st.success("The index matches these settings.")
        else:
            st.warning("Build the index before using the Playground or running an evaluation.")
        if st.button("Build index", type="primary", disabled=not corpus.sources):
            try:
                with st.spinner("Building the local index…"):
                    with feedback.instrument("index_build", corpus=corpus.name):
                        indexer_mod.build_index(ws, corpus, cfg)
                st.toast("Index built")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Build failed: {exc}")

    with danger_tab:
        st.subheader("Delete corpus")
        st.caption(
            "This removes the saved corpus and its local index. "
            "Evaluation reports remain available."
        )
        confirmed = st.checkbox(f"I understand this deletes {choice}")
        if st.button("Delete corpus", disabled=not confirmed):
            corpora_mod.delete_corpus(ws, choice)
            if st.session_state.get("corpus_name") == choice:
                st.session_state["corpus_name"] = None
            st.rerun()

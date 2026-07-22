import streamlit as st

from rag_lab.studio import build_settings, config_logic, feedback
from rag_lab.studio import corpora as corpora_mod
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio.corpora import Corpus, Source
from rag_lab.studio.workspace import Workspace


def render() -> None:
    """Render the Corpus page: pick or create a corpus, curate sources, configure the build."""
    st.title("Corpus")
    cfg = st.session_state["config"]
    ws = Workspace.default()
    ws.initialize()
    names = corpora_mod.list_corpora(ws)

    options = config_logic.corpus_options(names)
    default = st.session_state.get("corpus_name")
    index = options.index(default) if default in options else 0
    choice = st.selectbox("Corpus", options, index=index)

    if choice == config_logic.ADD_CORPUS_SENTINEL:
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
        return

    selected = choice
    st.session_state["corpus_name"] = selected
    corpus = corpora_mod.load_corpus(ws, selected)

    st.subheader("GitHub repos")
    repo_sources = [s for s in corpus.sources if s.type == "github"]
    if not repo_sources:
        st.caption("No repos yet.")
    for source in repo_sources:
        visibility = "private" if source.private else "public"
        label = f"{source.repo} ({visibility})" if source.private else source.repo
        col1, col2 = st.columns([4, 1])
        col1.write(label)
        if col2.button("Remove", key=f"rm-{source.repo}-{visibility}"):
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

    st.subheader("GitHub issues")
    issue_sources = [s for s in corpus.sources if s.type == "github_issue"]
    if not issue_sources:
        st.caption("No issues yet.")
    for source in issue_sources:
        ref = f"{source.repo}#{source.issue}"
        col1, col2 = st.columns([4, 1])
        col1.write(ref)
        if col2.button("Remove", key=f"rm-issue-{ref}"):
            corpora_mod.save_corpus(ws, corpora_mod.remove_source(corpus, source))
            st.rerun()
    issue_ref = st.text_input(
        "Add an issue (owner/repo#123)", placeholder="developmentseed/titiler#42"
    )
    if st.button("Add issue"):
        reason = corpora_mod.validate_issue_ref(issue_ref)
        if reason:
            st.error(reason)
        else:
            repo_name, number = corpora_mod.parse_issue_ref(issue_ref)
            corpora_mod.save_corpus(
                ws,
                corpora_mod.add_source(
                    corpus, Source(type="github_issue", repo=repo_name, issue=number)
                ),
            )
            st.rerun()

    st.subheader("Local folders")
    folder_sources = [s for s in corpus.sources if s.type == "local"]
    if not folder_sources:
        st.caption("No folders yet.")
    for source in folder_sources:
        col1, col2 = st.columns([4, 1])
        col1.write(source.path)
        if col2.button("Remove", key=f"rm-folder-{source.path}"):
            corpora_mod.save_corpus(ws, corpora_mod.remove_source(corpus, source))
            st.rerun()
    folder = st.text_input("Add a local folder", placeholder="./docs")
    if st.button("Add folder"):
        reason = corpora_mod.validate_folder(folder)
        if reason:
            st.error(reason)
        else:
            corpora_mod.save_corpus(
                ws,
                corpora_mod.add_source(corpus, Source(type="local", path=folder.strip())),
            )
            st.rerun()

    st.divider()
    st.subheader("Build")
    build_settings.render(cfg)
    if indexer_mod.status(ws, corpus, cfg).cached:
        st.success("Index: ✓ built")
    else:
        st.warning("This corpus needs to be built.")
    col1, col2 = st.columns(2)
    if col1.button("Build index", type="primary", disabled=not corpus.sources):
        try:
            with st.spinner("Building index (cloning repos)..."):
                with feedback.instrument("index_build", corpus=corpus.name):
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

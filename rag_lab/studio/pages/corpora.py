import streamlit as st

from rag_lab.config import EMBEDDING_DIMENSIONS
from rag_lab.studio import config_logic
from rag_lab.studio import corpora as corpora_mod
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio import models as models_mod
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

    with st.expander("⚙️ Build settings (chunker + embedder)"):
        chunker_types = ["markdown_aware", "fixed"]
        cfg.chunker.type = st.selectbox(
            "chunker type", chunker_types, index=chunker_types.index(cfg.chunker.type),
            help="'markdown_aware' splits on headings; 'fixed' splits on a flat token count.",
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
            help="Tokens repeated between adjacent chunks. Capped below max_tokens.",
        )
        try:
            installed = models_mod.installed_models()
        except Exception:  # noqa: BLE001
            installed = []
        embed_models = list(EMBEDDING_DIMENSIONS)
        labels = config_logic.embedder_model_labels(installed)
        if cfg.embedder.model not in embed_models:
            embed_models.append(cfg.embedder.model)
            labels = {**labels, cfg.embedder.model: f"{cfg.embedder.model} (unknown dimension)"}
        embed_index = embed_models.index(cfg.embedder.model)
        cfg.embedder.model = st.selectbox(
            "embedding model", embed_models, index=embed_index,
            format_func=lambda m: labels[m],
            help="Changing this requires a rebuild.",
        )

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

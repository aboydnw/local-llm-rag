from pathlib import Path

import streamlit as st
import yaml

from rag_lab.agent.agent import DEFAULT_AGENT_INSTRUCTIONS
from rag_lab.config import EMBEDDING_DIMENSIONS, Config, config_summary
from rag_lab.prompts import DEFAULT_SYSTEM_INSTRUCTIONS
from rag_lab.studio import config_logic
from rag_lab.studio import corpora as corpora_mod
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio import models as models_mod
from rag_lab.studio import ui_state
from rag_lab.studio.workspace import Workspace


def _run_pull(model: str) -> None:
    """Stream a model pull into a progress bar in the main area."""
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


def _render_corpus(session) -> None:
    st.subheader("Corpus")
    corpus_names = corpora_mod.list_corpora(Workspace.default())
    options = ["(local folder)"] + corpus_names
    current = session.get("corpus_name") or "(local folder)"
    choice = st.selectbox(
        "Active corpus", options,
        index=options.index(current) if current in options else 0,
        help="Pick a curated corpus, or use a local folder. Manage corpora on the Corpus page.",
    )
    if choice == "(local folder)":
        session["corpus_name"] = None
        session["corpus"] = st.text_input(
            "Corpus directory", value=session["corpus"],
            help="Folder of markdown (.md) files. Searched recursively.",
        ).strip()
    else:
        session["corpus_name"] = choice
    st.caption(f"Golden set: `{session['golden']}`")


def _render_retrieval_method(cfg: Config) -> None:
    st.subheader("Retrieval method")
    method = config_logic.retrieval_method(cfg)
    chosen = st.radio(
        "method", config_logic.RETRIEVAL_METHODS,
        index=config_logic.RETRIEVAL_METHODS.index(method),
        help="How chunks are found. vector = semantic similarity, bm25 = keyword match, "
        "hybrid = a blend, agent = the LLM orchestrates retrieval itself, step by step.",
    )
    config_logic.apply_retrieval_method(cfg, chosen)

    if chosen == "agent":
        _render_agent_knobs(cfg)
    else:
        _render_fixed_knobs(cfg, chosen)


def _render_fixed_knobs(cfg: Config, method: str) -> None:
    if method == "hybrid":
        vector_weight = st.slider(
            "vector_weight", 0.0, 1.0, cfg.retriever.vector_weight, 0.05,
            help="How much to favor semantic search over keyword search. "
            "1.0 = pure vector, 0.0 = pure keyword.",
        )
        weights = ui_state.normalized_weights(vector_weight)
        cfg.retriever.vector_weight, cfg.retriever.bm25_weight = weights
        st.caption(f"bm25_weight = {cfg.retriever.bm25_weight}")
    cfg.retriever.k = st.slider(
        "k", 1, 20, cfg.retriever.k,
        help="How many chunks to retrieve and feed to the LLM for each question.",
    )
    rerankers = ["none", "llm"]
    cfg.retriever.reranker = st.radio(
        "reranker", rerankers, index=rerankers.index(cfg.retriever.reranker),
        help="Optional second pass: 'llm' asks the model to reorder a larger candidate set "
        "down to k by relevance. Costs extra LLM calls; no rebuild needed.",
    )
    if cfg.retriever.reranker == "llm":
        cfg.retriever.rerank_candidates = st.number_input(
            "rerank_candidates", min_value=1, value=cfg.retriever.rerank_candidates,
            help="How many chunks to fetch before the reranker trims them down to k.",
        )


def _render_agent_knobs(cfg: Config) -> None:
    st.caption(
        "Agent mode replaces the fixed retriever: it picks tools step by step. "
        "Slower — several LLM calls per question — but you can watch it reason. "
        "Keep a non-agent run around as your baseline to compare against."
    )
    cfg.agent.max_steps = st.slider(
        "max_steps", 1, 12, cfg.agent.max_steps,
        help="Hard cap on reasoning steps before the agent must answer with whatever it has.",
    )
    cfg.agent.final_k = st.slider(
        "final_k", 1, 20, cfg.agent.final_k,
        help="How many gathered chunks survive to the final answer prompt.",
    )
    cfg.retriever.k = st.slider(
        "k", 1, 20, cfg.retriever.k,
        help="Reused in agent mode as each search tool's fetch size.",
    )
    all_tools = ["vector_search", "keyword_search", "list_documents", "fetch_document"]
    chosen = st.multiselect(
        "tools", all_tools, default=list(cfg.agent.tools),
        help="Which tools the agent may call. Try removing one and watch how its strategy changes.",
    )
    cfg.agent.tools = chosen or ["vector_search"]

    def _reset_agent_prompt() -> None:
        st.session_state["agent_instructions"] = DEFAULT_AGENT_INSTRUCTIONS

    if "agent_instructions" not in st.session_state:
        st.session_state["agent_instructions"] = cfg.agent.instructions
    cfg.agent.instructions = st.text_area(
        "Agent prompt (ReAct instructions)", key="agent_instructions", height=160,
        help="The instructions that teach the model the Thought/Action/Observation loop.",
    )
    st.button("Reset agent prompt", on_click=_reset_agent_prompt)


def _render_models(cfg: Config, installed: list[str], ollama_error: str | None) -> None:
    st.subheader("Models")
    if ollama_error is not None:
        st.info("Ollama not reachable — is it running?")

    llm_options = config_logic.llm_model_options(installed, cfg.llm.model)
    llm_choice = st.selectbox(
        "LLM model", llm_options, index=llm_options.index(cfg.llm.model),
        help="Ollama model that writes the final answer. Cheap to change — no rebuild needed.",
    )
    if llm_choice == config_logic.PULL_SENTINEL:
        name = st.text_input("Model name to pull", key="llm_pull_name", placeholder="llama3.2:3b")
        if st.button("Pull", key="llm_pull_btn") and name.strip():
            _run_pull(name.strip())
            cfg.llm.model = name.strip()
    else:
        cfg.llm.model = llm_choice

    embed_models = list(EMBEDDING_DIMENSIONS)
    labels = config_logic.embedder_model_labels(installed)
    if cfg.embedder.model not in embed_models:
        embed_models.append(cfg.embedder.model)
        labels = {**labels, cfg.embedder.model: f"{cfg.embedder.model} (unknown dimension)"}
    embed_index = embed_models.index(cfg.embedder.model)
    cfg.embedder.model = st.selectbox(
        "Embedding model  🟠 re-index", embed_models, index=embed_index,
        format_func=lambda m: labels[m],
        help="Ollama model that turns text into vectors. Changing it requires a rebuild.",
    )
    if ollama_error is None and not models_mod.is_installed(cfg.embedder.model, installed):
        if st.button(f"Pull {cfg.embedder.model}", key="pull_embed"):
            _run_pull(cfg.embedder.model)


def _render_chunker(cfg: Config) -> None:
    st.subheader("Chunker  🟠 re-index")
    chunker_types = ["markdown_aware", "fixed", "recursive", "semantic"]
    cfg.chunker.type = st.selectbox(
        "type", chunker_types, index=chunker_types.index(cfg.chunker.type),
        help="How documents are split. 'markdown_aware' splits on headings; "
        "'fixed' splits on a flat token count; 'recursive' splits on "
        "paragraphs then sentences then tokens; 'semantic' splits where "
        "adjacent sentences diverge in meaning (slower — embeds at ingest).",
    )
    cfg.chunker.max_tokens = st.slider(
        "max_tokens", 64, 2048, cfg.chunker.max_tokens, 32,
        help="Largest chunk size, in tokens.",
    )
    cfg.chunker.overlap = st.slider(
        "overlap", 0, 256, cfg.chunker.overlap, 8,
        help="Tokens repeated between adjacent chunks so ideas spanning a boundary aren't lost.",
    )
    if cfg.chunker.type == "semantic":
        cfg.chunker.similarity_threshold = st.slider(
            "similarity_threshold", 0.0, 1.0, cfg.chunker.similarity_threshold, 0.05,
            help="Higher = more, smaller chunks; lower = fewer, larger chunks. "
            "Only used by the 'semantic' chunker.",
        )


def _render_prompt(cfg: Config) -> None:
    st.subheader("Answer prompt")

    def _reset_prompt() -> None:
        st.session_state["prompt_system_instructions"] = DEFAULT_SYSTEM_INSTRUCTIONS

    if "prompt_system_instructions" not in st.session_state:
        st.session_state["prompt_system_instructions"] = cfg.prompt.system_instructions
    cfg.prompt.system_instructions = st.text_area(
        "Answer prompt (system instructions)", key="prompt_system_instructions", height=160,
        help="Instructions sent to the LLM before the retrieved context. No rebuild needed.",
    )
    st.button("Reset prompt to default", on_click=_reset_prompt)


def _render_actions(session, cfg: Config, ws: Workspace) -> None:
    active = corpora_mod.resolve_active_corpus(
        ws, session.get("corpus_name"), session["corpus"]
    )
    if session.get("corpus_name") is None:
        corpus_error = indexer_mod.validate_corpus(session["corpus"])
    else:
        corpus_error = None
    if corpus_error is not None:
        st.error(corpus_error)

    col1, col2 = st.columns(2)
    if col1.button("Save to rag.yml"):
        try:
            Path("rag.yml").write_text(
                yaml.safe_dump(cfg.model_dump(), allow_unicode=True), encoding="utf-8"
            )
            st.toast("Saved rag.yml")
        except OSError as exc:
            st.error(f"Failed to save rag.yml: {exc}")
    if col2.button("Build index", disabled=corpus_error is not None):
        try:
            with st.spinner("Building index..."):
                indexer_mod.ensure_index(ws, active, cfg)
            st.toast("Index built")
        except ValueError as exc:
            st.error(str(exc))


def render(session, page_key: str) -> Config:
    """Render the shared config panel inside a collapsible expander; return the current config."""
    cfg: Config = session["config"]
    try:
        installed = models_mod.installed_models()
        ollama_error: str | None = None
    except Exception as exc:  # noqa: BLE001
        installed = []
        ollama_error = str(exc)

    ws = Workspace.default()
    ws.initialize()
    active = corpora_mod.resolve_active_corpus(
        ws, session.get("corpus_name"), session["corpus"]
    )
    status_error: str | None = None
    try:
        cached = indexer_mod.status(ws, active, cfg).cached
        status_badge = "✓ cached" if cached else "⚠ needs build"
    except Exception as exc:  # noqa: BLE001
        status_badge = "⚠ needs build"
        status_error = str(exc)

    with st.expander("⚙️ Config", expanded=config_logic.config_expanded(session, page_key)):
        st.caption(f"{config_summary(cfg)}  ·  {status_badge}")
        if status_error is not None:
            st.info(f"Index status unavailable: {status_error}")
        _render_corpus(session)
        _render_retrieval_method(cfg)
        _render_models(cfg, installed, ollama_error)
        _render_chunker(cfg)
        _render_prompt(cfg)
        _render_actions(session, cfg, ws)

    session["config"] = cfg
    return cfg

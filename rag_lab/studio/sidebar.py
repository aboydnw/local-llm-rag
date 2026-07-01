from pathlib import Path

import streamlit as st
import yaml

from rag_lab.agent.agent import DEFAULT_AGENT_INSTRUCTIONS
from rag_lab.config import EMBEDDING_DIMENSIONS, Config, config_summary
from rag_lab.prompts import DEFAULT_SYSTEM_INSTRUCTIONS
from rag_lab.studio import corpora as corpora_mod
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio import models as models_mod
from rag_lab.studio import ui_state
from rag_lab.studio.workspace import Workspace


def _run_pull(model: str) -> None:
    """Stream a model pull into a sidebar progress bar."""
    bar = st.sidebar.progress(0.0, text=f"Pulling {model}...")
    fraction = 0.0
    try:
        for event in models_mod.pull_progress(model):
            if event.fraction is not None:
                fraction = event.fraction
            bar.progress(fraction, text=f"{model}: {event.status}")
        bar.progress(1.0, text=f"{model}: done")
        st.sidebar.toast(f"Pulled {model}")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.error(f"Pull failed: {exc}")


def render(session) -> Config:
    """Render the config sidebar and return the current ``Config`` from session state."""
    cfg: Config = session["config"]
    try:
        installed = models_mod.installed_models()
        ollama_error: str | None = None
    except Exception as exc:  # noqa: BLE001
        installed = []
        ollama_error = str(exc)
    st.sidebar.header("Corpus")
    corpus_names = corpora_mod.list_corpora(Workspace.default())
    options = ["(local folder)"] + corpus_names
    current = session.get("corpus_name") or "(local folder)"
    choice = st.sidebar.selectbox(
        "Active corpus", options,
        index=options.index(current) if current in options else 0,
        help="Pick a curated corpus, or use a local folder. "
        "Manage corpora on the Corpus page.",
    )
    if choice == "(local folder)":
        session["corpus_name"] = None
        session["corpus"] = st.sidebar.text_input(
            "Corpus directory",
            value=session["corpus"],
            help="Folder of markdown (.md) files. Searched recursively.",
        ).strip()
    else:
        session["corpus_name"] = choice
    session["golden"] = st.sidebar.text_input(
        "Golden set",
        value=session["golden"],
        help="YAML file of question/expected-answer pairs used to score eval runs. "
        "Optional — leave as-is if you only want to ask questions.",
    )

    st.sidebar.header("Chunker  🟠 re-index")
    chunker_types = ["markdown_aware", "fixed"]
    cfg.chunker.type = st.sidebar.selectbox(
        "type", chunker_types, index=chunker_types.index(cfg.chunker.type),
        help="How documents are split into chunks. 'markdown_aware' splits on headings; "
        "'fixed' splits on a flat token count.",
    )
    cfg.chunker.max_tokens = st.sidebar.slider(
        "max_tokens", 64, 2048, cfg.chunker.max_tokens, 32,
        help="Largest chunk size, in tokens. Bigger chunks = more context per result "
        "but less precise retrieval.",
    )
    cfg.chunker.overlap = st.sidebar.slider(
        "overlap", 0, 256, cfg.chunker.overlap, 8,
        help="Tokens repeated between adjacent chunks so ideas spanning a boundary "
        "aren't lost.",
    )

    st.sidebar.header("Embedder  🟠 re-index")
    models = list(EMBEDDING_DIMENSIONS)
    cfg.embedder.model = st.sidebar.selectbox(
        "embedding model", models,
        index=models.index(cfg.embedder.model) if cfg.embedder.model in models else 0,
        help="Ollama model that turns text into vectors for semantic search. "
        "Changing it requires a rebuild.",
    )
    if ollama_error is None and not models_mod.is_installed(cfg.embedder.model, installed):
        st.sidebar.warning(f"Embedder model '{cfg.embedder.model}' is not installed.")
        if st.sidebar.button(f"Pull {cfg.embedder.model}", key="pull_embed"):
            _run_pull(cfg.embedder.model)

    st.sidebar.header("LLM  🟢 cheap")
    cfg.llm.model = st.sidebar.text_input(
        "llm model", value=cfg.llm.model,
        help="Ollama model that writes the final answer from retrieved chunks. "
        "Cheap to change — no rebuild needed.",
    ).strip()
    if ollama_error is None and not models_mod.is_installed(cfg.llm.model, installed):
        st.sidebar.warning(f"LLM model '{cfg.llm.model}' is not installed.")
        if st.sidebar.button(f"Pull {cfg.llm.model}", key="pull_llm"):
            _run_pull(cfg.llm.model)

    st.sidebar.header("Retriever  🟢 cheap")
    rtypes = ["hybrid", "vector", "bm25"]
    cfg.retriever.type = st.sidebar.radio(
        "type", rtypes, index=rtypes.index(cfg.retriever.type),
        help="How chunks are found: 'vector' = semantic similarity, 'bm25' = keyword "
        "match, 'hybrid' = a blend of both.",
    )
    vector_weight = st.sidebar.slider(
        "vector_weight", 0.0, 1.0, cfg.retriever.vector_weight, 0.05,
        help="In hybrid mode, how much to favor semantic search over keyword search. "
        "1.0 = pure vector, 0.0 = pure keyword.",
    )
    weights = ui_state.normalized_weights(vector_weight)
    cfg.retriever.vector_weight, cfg.retriever.bm25_weight = weights
    st.sidebar.caption(f"bm25_weight = {cfg.retriever.bm25_weight}")
    cfg.retriever.k = st.sidebar.slider(
        "k", 1, 20, cfg.retriever.k,
        help="How many chunks to retrieve and feed to the LLM for each question.",
    )
    rerankers = ["none", "llm"]
    cfg.retriever.reranker = st.sidebar.radio(
        "reranker", rerankers, index=rerankers.index(cfg.retriever.reranker),
        help="Optional second pass: 'llm' asks the model to reorder a larger candidate "
        "set down to k by relevance. Costs extra LLM calls; no rebuild needed.",
    )
    if cfg.retriever.reranker == "llm":
        cfg.retriever.rerank_candidates = st.sidebar.number_input(
            "rerank_candidates", min_value=1, value=cfg.retriever.rerank_candidates,
            help="How many chunks to fetch before the reranker trims them down to k. "
            "Bigger = better ordering, more LLM work.",
        )

    st.sidebar.header("Agent  🟢 cheap")
    cfg.agent.enabled = st.sidebar.toggle(
        "agentic retrieval", value=cfg.agent.enabled,
        help="Let the LLM orchestrate retrieval itself: it picks tools (vector, "
        "keyword, list, fetch) step by step instead of one fixed retrieval. "
        "Slower — several LLM calls per question — but you can watch it reason.",
    )
    if cfg.agent.enabled:
        cfg.agent.max_steps = st.sidebar.slider(
            "max_steps", 1, 12, cfg.agent.max_steps,
            help="Hard cap on reasoning steps before the agent must answer with "
            "whatever it has gathered. Bounds cost and prevents loops.",
        )
        cfg.agent.final_k = st.sidebar.slider(
            "final_k", 1, 20, cfg.agent.final_k,
            help="How many gathered chunks survive to the final answer prompt.",
        )
        all_tools = [
            "vector_search", "keyword_search", "list_documents", "fetch_document",
        ]
        chosen = st.sidebar.multiselect(
            "tools", all_tools, default=list(cfg.agent.tools),
            help="Which tools the agent may call. Try removing one and watch how "
            "its strategy changes.",
        )
        cfg.agent.tools = chosen or ["vector_search"]

        def _reset_agent_prompt() -> None:
            st.session_state["agent_instructions"] = DEFAULT_AGENT_INSTRUCTIONS

        if "agent_instructions" not in st.session_state:
            st.session_state["agent_instructions"] = cfg.agent.instructions
        cfg.agent.instructions = st.sidebar.text_area(
            "Agent prompt (ReAct instructions)",
            key="agent_instructions", height=160,
            help="The instructions that teach the model the Thought/Action/"
            "Observation loop. Edit to change how the agent reasons.",
        )
        st.sidebar.button("Reset agent prompt", on_click=_reset_agent_prompt)

    st.sidebar.header("Models")
    if ollama_error is not None:
        st.sidebar.info("Ollama not reachable — is it running?")
    else:
        st.sidebar.caption("Installed: " + (", ".join(installed) or "none"))
        pull_name = st.sidebar.text_input(
            "Pull a model",
            key="pull_name",
            help="Name of an Ollama model to download, e.g. llama3.2:3b. "
            "Runs `ollama pull` and streams progress here.",
        )
        if st.sidebar.button("Pull", key="pull_arbitrary") and pull_name.strip():
            _run_pull(pull_name.strip())

    ws = Workspace.default()
    ws.initialize()
    st.sidebar.divider()
    active = corpora_mod.resolve_active_corpus(ws, session.get("corpus_name"), session["corpus"])
    if session.get("corpus_name") is None:
        corpus_error = indexer_mod.validate_corpus(session["corpus"])
    else:
        corpus_error = None
    if corpus_error is not None:
        st.sidebar.error(corpus_error)
    else:
        status = indexer_mod.status(ws, active, cfg)
        if status.cached:
            st.sidebar.success("Index: ✓ cached")
        else:
            st.sidebar.warning("⚠ this config needs a build")

    def _reset_prompt() -> None:
        st.session_state["prompt_system_instructions"] = DEFAULT_SYSTEM_INSTRUCTIONS

    if "prompt_system_instructions" not in st.session_state:
        st.session_state["prompt_system_instructions"] = cfg.prompt.system_instructions
    cfg.prompt.system_instructions = st.sidebar.text_area(
        "Answer prompt (system instructions)",
        key="prompt_system_instructions",
        height=160,
        help="Instructions sent to the LLM before the retrieved context. "
        "Changing this does not require rebuilding the index.",
    )
    st.sidebar.button("Reset prompt to default", on_click=_reset_prompt)

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Save to rag.yml"):
        Path("rag.yml").write_text(
            yaml.safe_dump(cfg.model_dump(), allow_unicode=True), encoding="utf-8"
        )
        st.sidebar.toast("Saved rag.yml")
    if col2.button("Build index", disabled=corpus_error is not None):
        try:
            with st.spinner("Building index..."):
                indexer_mod.ensure_index(ws, active, cfg)
            st.sidebar.toast("Index built")
        except ValueError as exc:
            st.sidebar.error(str(exc))

    st.sidebar.caption(config_summary(cfg))
    session["config"] = cfg
    return cfg

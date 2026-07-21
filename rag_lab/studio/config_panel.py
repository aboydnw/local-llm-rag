from pathlib import Path

import streamlit as st
import yaml

from rag_lab.agent.agent import DEFAULT_AGENT_INSTRUCTIONS
from rag_lab.config import Config, config_summary
from rag_lab.prompts import DEFAULT_SYSTEM_INSTRUCTIONS
from rag_lab.studio import config_logic, pull_ui, ui_state
from rag_lab.studio import corpora as corpora_mod
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio import models as models_mod
from rag_lab.studio.workspace import Workspace


def _render_corpus(session) -> None:
    st.subheader("Corpus")
    names = corpora_mod.list_corpora(Workspace.default())
    if not names:
        st.info("No corpora yet — create one on the Corpus page.")
        session["corpus_name"] = None
        return
    current = session.get("corpus_name")
    index = names.index(current) if current in names else 0
    session["corpus_name"] = st.selectbox(
        "Active corpus", names, index=index,
        help="Which corpus to query. Create and build corpora on the Corpus page.",
    )
    st.caption(f"Golden set: `{session['golden']}`")


def _render_retrieval_method(cfg: Config) -> None:
    st.subheader("Retrieval method")
    modes = ["direct", "agentic"]
    labels = {"direct": "Direct search", "agentic": "Agentic"}
    current = config_logic.search_mode(cfg)
    chosen = st.radio(
        "search method", modes, index=modes.index(current),
        format_func=lambda m: labels[m],
        help="Direct = one retrieval pass you tune (a vector/keyword blend). "
        "Agentic = the LLM orchestrates retrieval itself, step by step.",
    )
    config_logic.apply_search_mode(cfg, chosen)
    if chosen == "agentic":
        _render_agent_knobs(cfg)
    else:
        _render_direct_knobs(cfg)


def _render_direct_knobs(cfg: Config) -> None:
    vector_weight = st.slider(
        "Vector ↔ Keyword", 0.0, 1.0, cfg.retriever.vector_weight, 0.05,
        help="Blend of the two searches. 1.0 = vector only (semantic), "
        "0.0 = keyword only (bm25); in between fuses both by reciprocal rank.",
    )
    weights = ui_state.normalized_weights(vector_weight)
    cfg.retriever.vector_weight, cfg.retriever.bm25_weight = weights
    st.caption(f"vector {cfg.retriever.vector_weight} · keyword {cfg.retriever.bm25_weight}")
    cfg.retriever.k = st.slider(
        "k", 1, 20, cfg.retriever.k,
        help="How many chunks to retrieve and feed to the LLM for each question.",
    )
    rerankers = ["none", "llm"]
    cfg.retriever.reranker = st.radio(
        "reranker", rerankers, index=rerankers.index(cfg.retriever.reranker),
        help="Optional second pass: 'llm' reorders a larger candidate set down to k by "
        "relevance. Costs extra LLM calls; no rebuild needed.",
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
    cfg.agent.structured_output = st.checkbox(
        "structured output (constrained JSON tool calls)",
        value=cfg.agent.structured_output,
        help="Force tool-call turns through a JSON schema via Ollama's format param. "
        "Cuts parse failures on smaller models; watch the parse-failure metric to compare.",
    )

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
    st.subheader("LLM")
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
            pull_ui.run_pull(name.strip())
            cfg.llm.model = name.strip()
    else:
        cfg.llm.model = llm_choice


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


def _render_actions(cfg: Config) -> None:
    if st.button("Save to rag.yml"):
        try:
            Path("rag.yml").write_text(
                yaml.safe_dump(cfg.model_dump(), allow_unicode=True), encoding="utf-8"
            )
            st.toast("Saved rag.yml")
        except OSError as exc:
            st.error(f"Failed to save rag.yml: {exc}")


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
        _render_prompt(cfg)
        _render_actions(cfg)

    session["config"] = cfg
    return cfg

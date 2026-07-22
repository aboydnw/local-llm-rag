import streamlit as st

from rag_lab.agent.agent import trace_dict
from rag_lab.prompts import PromptBuilder
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.studio import components, config_panel, feedback, trace_view
from rag_lab.studio import corpora as corpora_mod
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio import share as share_mod
from rag_lab.studio.workspace import Workspace


def render() -> None:
    """Render the Ask playground: retrieve chunks and generate an answer for a question."""
    st.title("Ask playground")
    cfg = config_panel.render(st.session_state, "playground")
    question = st.text_input("Question", placeholder="What is titiler?")
    if question.strip():
        st.session_state["playground_config_acted"] = True

    ws = Workspace.default()
    corpus = corpora_mod.resolve_active_corpus(
        ws, st.session_state.get("corpus_name"), st.session_state["corpus"]
    )
    if st.session_state.get("corpus_name"):
        st.caption(f"Querying corpus: **{corpus.label}**")
    else:
        st.warning(
            f"No saved corpus selected — querying the local folder `{corpus.label}`. "
            "Select a corpus in the config panel to query it instead."
        )
    status = indexer_mod.status(ws, corpus, cfg)
    if status.needs_build:
        st.warning("This corpus isn't built yet — build it on the **Corpus** page.")
        return
    question = question.strip()
    if not question:
        return

    try:
        db_path = indexer_mod.ensure_index(ws, corpus, cfg)
        embedder = components.build_embedder(cfg)
        store = SqliteVecStore(db_path, dimension=embedder.dimension)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Index setup failed: {exc}")
        return

    if cfg.agent.enabled:
        _render_agent_run(store, embedder, cfg, question)
        return

    with st.spinner("Retrieving..."):
        try:
            retriever = components.build_retriever(store, embedder, cfg)
            with feedback.instrument("retrieval", question=question, k=cfg.retriever.k):
                results = retriever.retrieve(question, k=cfg.retriever.k)
            prompt = PromptBuilder(
                system_instructions=cfg.prompt.system_instructions
            ).build(question=question, results=results)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Retrieval failed: {exc}")
            return

    st.subheader("Answer")
    try:
        with feedback.instrument("generation"):
            answer = st.write_stream(components.build_llm(cfg).stream(prompt))
    except Exception as exc:  # noqa: BLE001
        st.error(f"Generation failed: {exc}")
        return

    st.subheader("Retrieved chunks")
    for i, r in enumerate(results, start=1):
        heading = " > ".join(r.chunk.heading_path) or "(no heading)"
        with st.expander(f"[{i}] {r.chunk.doc_path} — {heading}  (score {r.score:.3f})"):
            st.text(r.chunk.text[:1000])

    with st.expander("🧩 Prompt sent to the LLM"):
        st.code(prompt, language="text")

    st.divider()
    with st.expander("📋 Copy as markdown"):
        st.code(share_mod.format_run_markdown(question, answer, results), language="markdown")


def _render_agent_run(store, embedder, cfg, question: str) -> None:
    """Run the retrieval agent and render its trace, answer, and final context."""
    with st.spinner("Agent reasoning..."):
        try:
            agent = components.build_agent(store, embedder, cfg)
            with feedback.instrument("agent_run", question=question):
                result = agent.run(question)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Agent run failed: {exc}")
            return

    st.subheader("Agent trace")
    tool_calls = len([s for s in result.steps if s.action])
    st.caption(
        f"{result.llm_calls} LLM calls · {tool_calls} tool calls · "
        f"{len(result.chunks_seen)} chunks seen → {len(result.final_context)} used"
    )
    trace_view.render_steps(
        [trace_dict(s) for s in result.steps], key_prefix="playground"
    )

    st.subheader("Answer")
    st.write(result.answer)

    with st.expander("🧩 Synthesis prompt (final answer)"):
        st.code(result.synthesis_prompt, language="text")

    st.subheader("Final context (what the answer cites)")
    for i, chunk in enumerate(result.final_context, start=1):
        heading = " > ".join(chunk.heading_path) or "(no heading)"
        with st.expander(f"[{i}] {chunk.doc_path} — {heading}"):
            st.text(chunk.text[:1000])

    tools_used = tuple(sorted({s.action for s in result.steps if s.action}))
    st.divider()
    with st.expander("📋 Copy as markdown"):
        st.code(
            share_mod.format_agent_run_markdown(
                question, result.answer, result.steps, tools_used, result.final_context
            ),
            language="markdown",
        )

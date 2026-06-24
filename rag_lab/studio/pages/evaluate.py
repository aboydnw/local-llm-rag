import uuid
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from rag_lab.studio import corpora as corpora_mod
from rag_lab.studio import experiments
from rag_lab.studio.workspace import Workspace


def render() -> None:
    """Render the Evaluate page: run the golden set and show aggregates + report."""
    st.title("Evaluate")
    cfg = st.session_state["config"]
    ws = Workspace.default()
    ws.initialize()
    corpus = corpora_mod.resolve_active_corpus(
        ws, st.session_state.get("corpus_name"), st.session_state["corpus"]
    )
    golden = Path(st.session_state["golden"])

    name = st.text_input("Run name (optional)", placeholder="more-vector-weight")
    cfg.eval.deepeval = st.checkbox(
        "DeepEval scoring",
        value=cfg.eval.deepeval,
        help="Adds LLM-judged answer-quality metrics (relevancy, faithfulness) on top of the "
        "always-on retrieval + keyword metrics. Slower — each metric is an extra Ollama call "
        f"({cfg.llm.model}) per question.",
    )
    if cfg.eval.deepeval:
        judge = st.text_input(
            "Judge model",
            value=cfg.eval.deepeval_model or "",
            placeholder=cfg.llm.model,
            help="Ollama model that scores the answers. Leave blank to reuse the answer LLM; "
            "set a stronger model here so the judge isn't grading its own homework.",
        ).strip()
        cfg.eval.deepeval_model = judge or None
        st.caption(f"LLM-judged metrics on, scored by {cfg.eval.deepeval_model or cfg.llm.model}.")
    else:
        st.caption("Retrieval + keyword metrics only. Enable for answer-quality scoring.")

    if st.button("Run eval", type="primary"):
        if not golden.exists():
            st.error(f"Golden set not found: {golden}")
            return
        with st.spinner("Running eval (this builds the index if needed)..."):
            try:
                record = experiments.run_eval(
                    ws, corpus, cfg, golden,
                    run_id=uuid.uuid4().hex[:8],
                    created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    name=name or None,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Eval failed: {exc}")
                return
        st.session_state["last_run_id"] = record.run_id
        st.success(f"Run '{record.name}' saved.")

    run_id = st.session_state.get("last_run_id")
    if run_id:
        ws = Workspace.default()
        try:
            record = experiments.load_run(ws, run_id)
        except Exception:  # noqa: BLE001
            st.session_state.pop("last_run_id", None)
            st.warning("The last run is no longer available.")
            return
        st.subheader("Aggregates")
        st.table({k: [round(v, 3)] for k, v in record.scores.items()})
        st.subheader("Report")
        report = ws.run_dir(run_id) / "report.md"
        if report.exists():
            st.markdown(report.read_text())

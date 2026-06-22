import os
import uuid
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from rag_lab.studio import experiments
from rag_lab.studio.workspace import Workspace


def render() -> None:
    """Render the Evaluate page: run the golden set and show aggregates + report."""
    st.title("Evaluate")
    cfg = st.session_state["config"]
    corpus = st.session_state["corpus"]
    golden = Path(st.session_state["golden"])

    name = st.text_input("Run name (optional)", placeholder="more-vector-weight")
    has_key = "ANTHROPIC_API_KEY" in os.environ
    judge_on = st.toggle(
        "Use LLM judge", value=False,
        help="Grade each answer 1-5 against the golden ideal answer. Adds an LLM call "
        "per question, so it's slower than the retrieval metrics.",
    )
    local_label = f"Local — Ollama ({cfg.llm.model})"
    anthropic_label = "Anthropic — claude-sonnet-4-6"
    judge_choice = local_label
    if judge_on:
        options = [local_label] + ([anthropic_label] if has_key else [])
        judge_choice = st.radio(
            "Judge model", options,
            help="Local runs fully offline via Ollama. Anthropic needs ANTHROPIC_API_KEY.",
        )
        if not has_key:
            st.caption("Set ANTHROPIC_API_KEY to also enable the Anthropic judge.")

    if st.button("Run eval", type="primary"):
        if not golden.exists():
            st.error(f"Golden set not found: {golden}")
            return
        judge = None
        if judge_on:
            if judge_choice == anthropic_label:
                from rag_lab.eval.scorers.llm_judge import LLMJudge
                judge = LLMJudge(model="claude-sonnet-4-6")
            else:
                from rag_lab.eval.scorers.llm_judge import OllamaJudge
                from rag_lab.studio import components
                judge = OllamaJudge(components.build_llm(cfg))
        ws = Workspace.default()
        ws.initialize()
        with st.spinner("Running eval (this builds the index if needed)..."):
            try:
                record = experiments.run_eval(
                    ws, corpus, cfg, golden,
                    run_id=uuid.uuid4().hex[:8],
                    created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                    name=name or None,
                    judge=judge,
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

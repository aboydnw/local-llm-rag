from datetime import UTC, datetime

import streamlit as st

from rag_lab.config import config_summary
from rag_lab.eval import run_store
from rag_lab.eval.gates import gate_failures
from rag_lab.studio import charts, experiments, trace_view
from rag_lab.studio.workspace import Workspace

_CORE_ITEM_METRICS = (
    ("recall@k", "recall_at_k"),
    ("mrr", "mrr"),
    ("keyword_coverage", "keyword_coverage"),
)


def _score_cell(record, metric: str) -> str:
    value = record.scores[metric]
    std = record.scores_std.get(metric)
    return f"{value:.3f} ±{std:.3f}" if std is not None else f"{value:.3f}"


def _ago(created_at: str) -> str:
    try:
        then = datetime.fromisoformat(created_at)
    except ValueError:
        return created_at
    delta = datetime.now(UTC) - then
    if delta.days > 0:
        return f"{delta.days}d ago"
    hours = delta.seconds // 3600
    return f"{hours}h ago" if hours else f"{delta.seconds // 60}m ago"


def _sweep_section(ws, corpus, runs) -> None:
    st.subheader("Base sweep")
    swept = experiments.latest_sweep_records(ws, corpus)
    if not swept:
        st.caption("Never swept — run the base set from the Evaluate page.")
        return
    custom = [r for r in runs if run_store.sweep_id(r) is None]
    st.altair_chart(charts.sweep_heatmap(swept + custom), use_container_width=True)
    st.caption(
        "Darker = better within each column. recall@k and mrr are deterministic; "
        "keyword coverage is a single noisy generation pass."
    )


def _custom_runs_section(ws, corpus, runs) -> None:
    st.subheader("Custom runs")
    custom = [r for r in runs if run_store.sweep_id(r) is None]
    if not custom:
        st.caption("No custom runs for this corpus yet.")
        return
    baseline_id = experiments.get_baseline(ws, corpus)
    baseline = experiments.load_run(ws, baseline_id) if baseline_id else None
    rows = []
    for r in custom:
        row = {
            "id": r.run_id,
            "name": f"★ {r.name}" if r.run_id == baseline_id else r.name,
            "when": _ago(r.created_at),
        }
        row.update({k: _score_cell(r, k) for k in sorted(r.scores)})
        if baseline is not None and r.run_id != baseline_id:
            gates = r.config.eval.gates
            regressed = gates and gate_failures(r.scores, baseline.scores, gates)
            row["vs baseline"] = "▼ regressed" if regressed else "ok"
        row["config"] = config_summary(r.config)
        rows.append(row)
    edited = st.data_editor(
        rows, use_container_width=True, hide_index=True, key=f"runs_editor_{corpus}",
        column_config={"id": None},
        disabled=[c for c in rows[0] if c != "name"],
    )
    for before, after in zip(rows, edited, strict=True):
        new = after["name"].removeprefix("★ ").strip()
        if new and new != before["name"].removeprefix("★ "):
            experiments.rename_run(ws, before["id"], new)
            st.rerun()

    ids = [r.run_id for r in custom]
    labels = {r.run_id: f"{r.name} ({r.run_id})" for r in custom}
    sel = st.selectbox("Manage a run", ids, format_func=lambda i: labels[i], key="manage")
    c1, c2, c3 = st.columns(3)
    if c1.button("Pin as baseline"):
        experiments.set_baseline(ws, sel)
        st.rerun()
    if c2.button("Delete"):
        experiments.delete_run(ws, sel)
        st.rerun()
    if c3.button("View report"):
        report = ws.run_dir(sel) / "report.md"
        if report.exists():
            st.markdown(report.read_text())


def _compare_section(ws, ids, labels) -> None:
    st.subheader("Compare two runs")
    if len(ids) < 2:
        st.caption("Need at least two runs for this corpus to compare.")
        return
    col_a, col_b = st.columns(2)
    a_id = col_a.selectbox("Run A", ids, format_func=lambda i: labels[i], key="cmp_a")
    b_id = col_b.selectbox("Run B", ids, format_func=lambda i: labels[i],
                           index=min(1, len(ids) - 1), key="cmp_b")
    if a_id and b_id and a_id != b_id:
        a, b = experiments.load_run(ws, a_id), experiments.load_run(ws, b_id)
        st.altair_chart(charts.compare_heatmap(a, b), use_container_width=True)
        result = experiments.diff(a, b)
        st.write("**Changed knobs:** " + (", ".join(result["changed_knobs"]) or "none"))
        with st.expander("Delta table"):
            st.table({m: [round(d, 3)] for m, d in result["metric_deltas"].items()})


def _inspector(ws, ids, labels) -> None:
    st.subheader("Inspect a run, question by question")
    if not ids:
        st.caption("No runs for this corpus.")
        return
    run_id = st.selectbox("Run", ids, format_func=lambda i: labels[i], key="inspect_run")
    record = experiments.load_run(ws, run_id)
    items = experiments.load_run_items(ws, run_id)
    if not items:
        st.caption("No per-question artifacts for this run.")
        return

    failures_only = st.checkbox("Failures only", key="inspect_failures")
    gates = record.config.eval.gates
    if failures_only:
        if not gates:
            st.info("No eval.gates configured — showing all questions.")
        else:
            baseline_id = experiments.get_baseline(ws, record.corpus)
            reference = (
                experiments.load_run(ws, baseline_id).scores
                if baseline_id
                else record.scores
            )
            items = run_store.item_failures(items, reference, gates)
            if not items:
                st.success("No failing questions against the gated metrics.")
                return

    pick = st.selectbox(
        "Question",
        items,
        format_func=lambda it: (
            f"{it['item_id']} (pass {it.get('repeat', 0) + 1}): {it['question']}"
        ),
        key="inspect_item",
    )
    metric_cols = st.columns(len(_CORE_ITEM_METRICS))
    for col, (label, field) in zip(metric_cols, _CORE_ITEM_METRICS, strict=True):
        col.metric(label, f"{pick.get(field, 0.0):.2f}")
    if pick.get("failed_metrics"):
        st.error("Failed gates: " + ", ".join(pick["failed_metrics"]))

    st.markdown("**Answer**")
    st.text(pick.get("actual_answer", ""))

    retrieved = pick.get("retrieved", [])
    if retrieved:
        st.markdown("**Retrieved chunks**")
        for ref in retrieved:
            heading = " › ".join(ref.get("heading_path", []) or [])
            title = f"{ref['rank']}. {ref['doc_path']}"
            if heading:
                title += f" — {heading}"
            with st.expander(f"{title} (score {ref.get('score', 0.0):.3f})"):
                st.text(ref.get("text") or "(chunk text not captured for this run)")

    if pick.get("agent_trace"):
        st.markdown("**Agent trace**")
        trace_view.render_steps(pick["agent_trace"], key_prefix=f"inspect_{run_id}")


def render() -> None:
    """Render the Runs page: per-corpus sweeps, custom runs, compare, drill-down."""
    st.title("Runs")
    ws = Workspace.default()
    all_runs = experiments.list_runs(ws)
    if not all_runs:
        st.info("No runs yet. Run an eval to populate this page.")
        return

    corpora = sorted({r.corpus for r in all_runs})
    default = st.session_state.get("corpus_name")
    corpus = st.selectbox(
        "Corpus", corpora,
        index=corpora.index(default) if default in corpora else 0,
    )
    show_old = st.toggle("Show older sweeps", value=False)
    runs = [r for r in run_store.visible_runs(all_runs, show_older_sweeps=show_old)
            if r.corpus == corpus]

    _sweep_section(ws, corpus, runs)
    _custom_runs_section(ws, corpus, runs)
    ids = [r.run_id for r in runs]
    labels = {r.run_id: f"{r.name} ({r.run_id})" for r in runs}
    _compare_section(ws, ids, labels)
    _inspector(ws, ids, labels)

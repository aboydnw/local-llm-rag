import streamlit as st

from rag_lab.config import config_summary
from rag_lab.eval import run_store
from rag_lab.eval.gates import gate_failures
from rag_lab.studio import experiments, trace_view
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


def _leaderboard(runs, baseline_id, baseline) -> None:
    rows = []
    for r in runs:
        row = {"run": r.name, "id": r.run_id, "when": r.created_at}
        if r.run_id == baseline_id:
            row["run"] = f"★ {r.name}"
        row.update({k: _score_cell(r, k) for k in sorted(r.scores)})
        if baseline is not None and r.run_id != baseline_id:
            gates = r.config.eval.gates
            regressed = gates and gate_failures(r.scores, baseline.scores, gates)
            row["vs baseline"] = "▼ regressed" if regressed else "ok"
        if r.repeat > 1:
            row["repeats"] = r.repeat
        row["config"] = config_summary(r.config)
        rows.append(row)
    st.subheader("Leaderboard")
    if baseline_id:
        st.caption(f"★ = pinned baseline ({baseline_id}); ▼ = gated metric regressed.")
    st.dataframe(rows, use_container_width=True)


def _inspector(ws, ids, labels) -> None:
    st.subheader("Inspect a run, question by question")
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
            baseline_id = experiments.get_baseline(ws)
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
    """Render the Runs page: leaderboard, baseline pin, compare, and drill-down."""
    st.title("Runs")
    ws = Workspace.default()
    runs = experiments.list_runs(ws)
    if not runs:
        st.info("No runs yet. Run an eval to populate the leaderboard.")
        return

    baseline_id = experiments.get_baseline(ws)
    baseline = experiments.load_run(ws, baseline_id) if baseline_id else None
    _leaderboard(runs, baseline_id, baseline)

    ids = [r.run_id for r in runs]
    labels = {r.run_id: f"{r.name} ({r.run_id})" for r in runs}

    st.subheader("Compare two runs")
    col_a, col_b = st.columns(2)
    a_id = col_a.selectbox("Run A", ids, format_func=lambda i: labels[i], key="cmp_a")
    b_id = col_b.selectbox("Run B", ids, format_func=lambda i: labels[i],
                           index=min(1, len(ids) - 1), key="cmp_b")
    if a_id and b_id and a_id != b_id:
        result = experiments.diff(experiments.load_run(ws, a_id), experiments.load_run(ws, b_id))
        st.write("**Changed knobs:** " + (", ".join(result["changed_knobs"]) or "none"))
        st.table({m: [round(d, 3)] for m, d in result["metric_deltas"].items()})

    st.subheader("Manage a run")
    sel = st.selectbox("Run", ids, format_func=lambda i: labels[i], key="manage")
    new_name = st.text_input("Rename to", key="rename_box")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Rename") and new_name:
        experiments.rename_run(ws, sel, new_name)
        st.rerun()
    if c2.button("Pin as baseline"):
        experiments.set_baseline(ws, sel)
        st.rerun()
    if c3.button("Delete"):
        experiments.delete_run(ws, sel)
        st.rerun()
    if c4.button("View report"):
        report = ws.run_dir(sel) / "report.md"
        if report.exists():
            st.markdown(report.read_text())

    _inspector(ws, ids, labels)

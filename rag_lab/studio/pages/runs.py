import streamlit as st

from rag_lab.config import config_summary
from rag_lab.studio import experiments, trace_view
from rag_lab.studio.workspace import Workspace


def render() -> None:
    """Render the Runs page: leaderboard, two-run compare, and run management."""
    st.title("Runs")
    ws = Workspace.default()
    runs = experiments.list_runs(ws)
    if not runs:
        st.info("No runs yet. Run an eval to populate the leaderboard.")
        return

    rows = []
    for r in runs:
        row = {"run": r.name, "id": r.run_id, "when": r.created_at}
        row.update({k: round(v, 3) for k, v in r.scores.items()})
        row["config"] = config_summary(r.config)
        rows.append(row)
    st.subheader("Leaderboard")
    st.dataframe(rows, use_container_width=True)

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
    c1, c2, c3 = st.columns(3)
    if c1.button("Rename") and new_name:
        experiments.rename_run(ws, sel, new_name)
        st.rerun()
    if c2.button("Delete"):
        experiments.delete_run(ws, sel)
        st.rerun()
    if c3.button("View report"):
        report = ws.run_dir(sel) / "report.md"
        if report.exists():
            st.markdown(report.read_text())

    st.subheader("Inspect agent trace")
    trace_run = st.selectbox(
        "Run", ids, format_func=lambda i: labels[i], key="trace_run"
    )
    items = experiments.load_run_items(ws, trace_run)
    agent_items = [it for it in items if it.get("agent_trace")]
    if not agent_items:
        st.caption("This run has no agent traces (retriever-mode run).")
    else:
        pick = st.selectbox(
            "Question",
            agent_items,
            format_func=lambda it: f"{it['item_id']}: {it['question']}",
            key="trace_item",
        )
        trace_view.render_steps(pick["agent_trace"], key_prefix=f"runs_{trace_run}")

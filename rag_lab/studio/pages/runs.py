from datetime import UTC, datetime

import streamlit as st

from rag_lab.config import config_summary
from rag_lab.eval import run_store
from rag_lab.eval.gates import gate_failures
from rag_lab.studio import charts, experiments, trace_view
from rag_lab.studio.workspace import Workspace

ITEM_METRICS = (
    ("Recall at k", "recall_at_k"),
    ("MRR", "mrr"),
    ("Keyword coverage", "keyword_coverage"),
)


def _ago(created_at: str) -> str:
    """Format a stored ISO timestamp as a defensive relative time."""
    try:
        then = datetime.fromisoformat(created_at)
    except ValueError:
        return created_at
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    delta = datetime.now(UTC) - then
    if delta.days:
        return f"{delta.days}d ago"
    hours = delta.seconds // 3600
    return f"{hours}h ago" if hours else f"{delta.seconds // 60}m ago"


def _run_label(record, baseline_id: str | None) -> str:
    """Build a run selector label with baseline and age context."""
    marker = "★ " if record.run_id == baseline_id else ""
    return f"{marker}{record.name} · {_ago(record.created_at)}"


def _score_summary(record, baseline) -> None:
    """Render core score cards and baseline deltas."""
    cols = st.columns(min(4, len(record.scores)))
    for col, metric in zip(cols, charts.CORE_METRICS, strict=False):
        if metric not in record.scores:
            continue
        value = record.scores[metric]
        delta = value - baseline.scores.get(metric, value) if baseline else None
        std = record.scores_std.get(metric)
        col.metric(
            charts.METRIC_LABELS[metric],
            f"{value:.3f}" + (f" ± {std:.3f}" if std is not None else ""),
            f"{delta:+.3f} vs baseline" if delta is not None and baseline != record else None,
        )


def _overview(ws, record, runs, baseline) -> None:
    """Render the selected report overview and score history."""
    _score_summary(record, baseline)
    if baseline and baseline.run_id != record.run_id:
        failures = gate_failures(record.scores, baseline.scores, record.config.eval.gates)
        if failures:
            st.warning("This run regresses beyond its configured gates: " + ", ".join(failures))
        else:
            deltas = charts.comparison_rows(baseline, record)
            improved = [row for row in deltas if row["delta"] > 0]
            st.success(
                f"{len(improved)} of {len(deltas)} core metrics improved versus the baseline."
            )
    st.caption(config_summary(record.config))

    metric = st.selectbox(
        "History metric",
        [m for m in charts.CORE_METRICS if any(m in r.scores for r in runs)],
        format_func=lambda m: charts.METRIC_LABELS[m],
    )
    st.altair_chart(charts.history_chart(runs, metric, baseline), use_container_width=True)

    sweep = [r for r in runs if run_store.sweep_id(r) is not None]
    if sweep:
        with st.expander("Latest preset sweep", expanded=False):
            st.caption(
                "Scores use a shared 0–1 scale. Lines show ± one standard deviation when available."
            )
            st.altair_chart(charts.sweep_dot_plot(sweep), use_container_width=True)


def _compare(record, runs) -> None:
    """Render a two-run metric and configuration comparison."""
    alternatives = [r for r in runs if r.run_id != record.run_id]
    if not alternatives:
        st.info("Run another evaluation to compare results.")
        return
    labels = {r.run_id: f"{r.name} ({r.run_id})" for r in alternatives}
    other_id = st.selectbox("Compare with", list(labels), format_func=labels.get)
    other = next(r for r in alternatives if r.run_id == other_id)
    st.caption(
        f"Bars show **{record.name} minus {other.name}**. "
        "Faded bars are within observed run variance."
    )
    st.altair_chart(charts.comparison_delta_chart(other, record), use_container_width=True)
    changed = experiments.diff(other, record)["changed_knobs"]
    st.markdown("**Changed settings:** " + (", ".join(changed) if changed else "None"))
    st.dataframe(charts.comparison_rows(other, record), use_container_width=True, hide_index=True)


def _question_rows(items: list[dict], baseline_items: list[dict]) -> list[dict]:
    """Build question rows sorted by worst baseline regression first."""
    baseline_by_key = {
        (item.get("item_id"), item.get("repeat", 0)): item for item in baseline_items
    }
    rows = []
    for index, item in enumerate(items):
        key = (item.get("item_id"), item.get("repeat", 0))
        reference = baseline_by_key.get(key, {})
        row = {
            "_index": index,
            "Question": item.get("question", item.get("item_id", "")),
            "Pass": item.get("repeat", 0) + 1,
            "Recall": item.get("recall_at_k", 0.0),
            "Δ recall": item.get("recall_at_k", 0.0)
            - reference.get("recall_at_k", item.get("recall_at_k", 0.0)),
            "MRR": item.get("mrr", 0.0),
            "Δ MRR": item.get("mrr", 0.0) - reference.get("mrr", item.get("mrr", 0.0)),
            "Keyword": item.get("keyword_coverage", 0.0),
            "Δ keyword": item.get("keyword_coverage", 0.0)
            - reference.get("keyword_coverage", item.get("keyword_coverage", 0.0)),
        }
        rows.append(row)
    return sorted(rows, key=lambda row: (row["Δ recall"], row["Δ MRR"], row["Δ keyword"]))


def _questions(ws, record, baseline) -> None:
    """Render question-level regressions and selected-item evidence."""
    items = experiments.load_run_items(ws, record.run_id)
    if not items:
        st.info("No question-level artifacts were captured for this run.")
        return
    baseline_items = (
        experiments.load_run_items(ws, baseline.run_id)
        if baseline and baseline.run_id != record.run_id
        else []
    )
    rows = _question_rows(items, baseline_items)
    failures_only = st.toggle("Show gate failures only", value=False)
    if failures_only:
        failed = run_store.item_failures(
            items, baseline.scores if baseline else record.scores, record.config.eval.gates
        )
        failed_ids = {(i.get("item_id"), i.get("repeat", 0)) for i in failed}
        rows = [
            row
            for row in rows
            if (items[row["_index"]].get("item_id"), items[row["_index"]].get("repeat", 0))
            in failed_ids
        ]
    if not rows:
        st.success("No questions match this filter.")
        return
    event = st.dataframe(
        [{k: v for k, v in row.items() if k != "_index"} for row in rows],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            "Recall": st.column_config.NumberColumn(format="%.3f"),
            "Δ recall": st.column_config.NumberColumn(format="%+.3f"),
            "MRR": st.column_config.NumberColumn(format="%.3f"),
            "Δ MRR": st.column_config.NumberColumn(format="%+.3f"),
            "Keyword": st.column_config.NumberColumn(format="%.3f"),
            "Δ keyword": st.column_config.NumberColumn(format="%+.3f"),
        },
    )
    selected = event.selection.rows[0] if event.selection.rows else 0
    item = items[rows[selected]["_index"]]
    st.subheader(item.get("question", item.get("item_id", "Question detail")))
    metric_cols = st.columns(len(ITEM_METRICS))
    for col, (label, field) in zip(metric_cols, ITEM_METRICS, strict=True):
        col.metric(label, f"{item.get(field, 0.0):.3f}")
    st.markdown("**Answer**")
    st.write(item.get("actual_answer", ""))
    for ref in item.get("retrieved", []):
        heading = " › ".join(ref.get("heading_path", []) or [])
        title = f"{ref['rank']}. {ref['doc_path']}" + (f" — {heading}" if heading else "")
        with st.expander(f"{title} · score {ref.get('score', 0.0):.3f}"):
            st.text(ref.get("text") or "Chunk text was not captured for this run.")
    if item.get("agent_trace"):
        st.markdown("**Agent trace**")
        trace_view.render_steps(item["agent_trace"], key_prefix=f"inspect_{record.run_id}")


def _settings_and_report(ws, record) -> None:
    """Render persisted configuration and the raw Markdown report."""
    st.code(config_summary(record.config), language="text")
    report = ws.run_dir(record.run_id) / "report.md"
    if report.exists():
        st.markdown(report.read_text())


def _manage(ws, record, baseline_id) -> None:
    """Render rename, baseline, and confirmed deletion controls."""
    name = st.text_input("Run name", value=record.name)
    if st.button("Save name") and name.strip() and name.strip() != record.name:
        experiments.rename_run(ws, record.run_id, name.strip())
        st.rerun()
    if record.run_id != baseline_id and st.button("Use as baseline", type="primary"):
        experiments.set_baseline(ws, record.run_id)
        st.rerun()
    st.divider()
    confirm = st.checkbox("I understand this permanently deletes the run")
    if st.button("Delete run", disabled=not confirm):
        experiments.delete_run(ws, record.run_id)
        st.session_state.pop("selected_report_run", None)
        st.rerun()


def render() -> None:
    """Render the report selector and persistent report-detail workspace."""
    st.title("Evaluation reports")
    st.caption("Track progress, compare configurations, and find the questions that regressed.")
    ws = Workspace.default()
    all_runs = experiments.list_runs(ws)
    if not all_runs:
        st.info("No reports yet. Build a corpus, add test questions, then run an evaluation.")
        return
    corpora = sorted({r.corpus for r in all_runs})
    default = st.session_state.get("corpus_name")
    corpus = st.selectbox(
        "Corpus", corpora, index=corpora.index(default) if default in corpora else 0
    )
    show_old = st.toggle("Include older preset sweeps", value=False)
    runs = [
        r
        for r in run_store.visible_runs(all_runs, show_older_sweeps=show_old)
        if r.corpus == corpus
    ]
    baseline_id = experiments.get_baseline(ws, corpus)
    labels = {r.run_id: _run_label(r, baseline_id) for r in runs}
    current = st.session_state.get("selected_report_run")
    if current not in labels:
        current = runs[0].run_id
        st.session_state["selected_report_run"] = current
    run_id = st.selectbox(
        "Report",
        list(labels),
        index=list(labels).index(current),
        format_func=labels.get,
        key="selected_report_run",
    )
    record = next(r for r in runs if r.run_id == run_id)
    baseline = experiments.load_run(ws, baseline_id) if baseline_id else None
    st.subheader(record.name)
    st.caption(
        f"{record.created_at} · {record.repeat} "
        f"pass{'es' if record.repeat != 1 else ''} · {record.run_id}"
    )
    overview, compare, questions, report, manage = st.tabs(
        ["Overview", "Compare", "Questions", "Settings & raw report", "Manage"]
    )
    with overview:
        _overview(ws, record, runs, baseline)
    with compare:
        _compare(record, runs)
    with questions:
        _questions(ws, record, baseline)
    with report:
        _settings_and_report(ws, record)
    with manage:
        _manage(ws, record, baseline_id)

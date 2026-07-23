from collections import Counter

import altair as alt

from rag_lab.eval.run_store import RunRecord

CORE_METRICS = ("recall_at_k", "mrr", "keyword_coverage")
METRIC_LABELS = {
    "recall_at_k": "Recall at k",
    "mrr": "MRR",
    "keyword_coverage": "Keyword coverage",
}


def _row_labels(records: list[RunRecord]) -> dict[str, str]:
    counts = Counter(r.name for r in records)
    return {
        r.run_id: f"{r.name} ({r.run_id[:6]})" if counts[r.name] > 1 else r.name for r in records
    }


def sweep_rows(records: list[RunRecord]) -> list[dict]:
    """Absolute scores for a small-multiple dot plot."""
    labels = _row_labels(records)
    rows: list[dict] = []
    for record in records:
        for metric in CORE_METRICS:
            if metric not in record.scores:
                continue
            std = record.scores_std.get(metric, 0.0)
            rows.append(
                {
                    "run": labels[record.run_id],
                    "metric": METRIC_LABELS[metric],
                    "value": record.scores[metric],
                    "std": std,
                    "low": max(0.0, record.scores[metric] - std),
                    "high": min(1.0, record.scores[metric] + std),
                    "is_custom": record.provenance.get("sweep_id") is None,
                }
            )
    return rows


def sweep_dot_plot(records: list[RunRecord]) -> alt.Chart:
    """Absolute 0–1 scores by run, faceted by metric."""
    rows = sweep_rows(records)
    labels = _row_labels(records)
    run_order = [
        labels[r.run_id]
        for r in sorted(records, key=lambda r: r.scores.get("recall_at_k", 0), reverse=True)
    ]
    base = alt.Chart(alt.Data(values=rows)).encode(
        y=alt.Y("run:N", sort=run_order, title=None),
        x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
        tooltip=[
            alt.Tooltip("run:N", title="Run"),
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("value:Q", title="Score", format=".3f"),
            alt.Tooltip("std:Q", title="Std. dev.", format=".3f"),
        ],
    )
    intervals = base.mark_rule(color="#94a3b8", strokeWidth=2).encode(
        x=alt.X("low:Q", scale=alt.Scale(domain=[0, 1]), title="Score"), x2="high:Q"
    )
    points = base.mark_point(filled=True, size=90).encode(
        color=alt.condition("datum.is_custom", alt.value("#0f766e"), alt.value("#334155"))
    )
    return (
        (intervals + points)
        .properties(width=250, height=alt.Step(30))
        .facet(
            column=alt.Column(
                "metric:N",
                sort=[METRIC_LABELS[m] for m in CORE_METRICS],
                title=None,
                header=alt.Header(labelFontSize=13, labelFontWeight=600),
            ),
            spacing=24,
        )
        .resolve_scale(y="shared")
    )


def comparison_rows(a: RunRecord, b: RunRecord) -> list[dict]:
    """B-minus-A deltas with source values and uncertainty context."""
    rows = []
    for metric in CORE_METRICS:
        if metric not in a.scores or metric not in b.scores:
            continue
        delta = b.scores[metric] - a.scores[metric]
        uncertainty = a.scores_std.get(metric, 0) + b.scores_std.get(metric, 0)
        rows.append(
            {
                "metric": METRIC_LABELS[metric],
                "a": a.scores[metric],
                "b": b.scores[metric],
                "delta": delta,
                "label": f"{delta:+.3f}",
                "uncertain": uncertainty > 0 and abs(delta) <= uncertainty,
            }
        )
    return rows


def comparison_delta_chart(a: RunRecord, b: RunRecord) -> alt.Chart:
    """Diverging B-minus-A bars centered on zero."""
    rows = comparison_rows(a, b)
    extent = max([abs(row["delta"]) for row in rows] + [0.05]) * 1.25
    base = alt.Chart(alt.Data(values=rows)).encode(
        y=alt.Y("metric:N", sort=[METRIC_LABELS[m] for m in CORE_METRICS], title=None),
        x=alt.X(
            "delta:Q",
            scale=alt.Scale(domain=[-extent, extent]),
            title=f"Change ({b.name} minus {a.name})",
        ),
        tooltip=[
            alt.Tooltip("metric:N", title="Metric"),
            alt.Tooltip("a:Q", title=a.name, format=".3f"),
            alt.Tooltip("b:Q", title=b.name, format=".3f"),
            alt.Tooltip("delta:Q", title="Change", format="+.3f"),
        ],
    )
    bars = base.mark_bar(cornerRadius=3).encode(
        color=alt.condition("datum.delta >= 0", alt.value("#0f766e"), alt.value("#b45309")),
        opacity=alt.condition("datum.uncertain", alt.value(0.45), alt.value(0.9)),
    )
    labels = base.mark_text(dx=6, align="left").encode(text="label:N")
    zero = (
        alt.Chart(alt.Data(values=[{"zero": 0}]))
        .mark_rule(color="#475569", strokeWidth=1)
        .encode(x="zero:Q")
    )
    return (bars + labels + zero).properties(height=alt.Step(48))


def history_chart(records: list[RunRecord], metric: str, baseline: RunRecord | None) -> alt.Chart:
    """Chronological score history for a selected metric."""
    rows = [
        {
            "created_at": r.created_at,
            "run": r.name,
            "value": r.scores[metric],
            "kind": "Custom" if r.provenance.get("sweep_id") is None else "Preset",
        }
        for r in records
        if metric in r.scores
    ]
    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X("created_at:T", title=None),
        y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), title=METRIC_LABELS.get(metric, metric)),
        tooltip=[
            alt.Tooltip("run:N", title="Run"),
            alt.Tooltip("created_at:T", title="Created"),
            alt.Tooltip("value:Q", title="Score", format=".3f"),
        ],
    )
    chart = base.mark_line(color="#94a3b8", strokeWidth=1.5) + base.mark_point(
        filled=True, size=80
    ).encode(
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=["Custom", "Preset"], range=["#0f766e", "#334155"]),
            title=None,
        )
    )
    if baseline is not None and metric in baseline.scores:
        chart += (
            alt.Chart(alt.Data(values=[{"baseline": baseline.scores[metric]}]))
            .mark_rule(color="#b45309", strokeDash=[5, 4])
            .encode(y="baseline:Q")
        )
    return chart.properties(height=240)


sweep_heatmap = sweep_dot_plot
compare_heatmap = comparison_delta_chart

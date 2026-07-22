import altair as alt

from rag_lab.eval.run_store import RunRecord

CORE_METRICS = ("recall_at_k", "mrr", "keyword_coverage")


def heatmap_rows(records: list[RunRecord]) -> list[dict]:
    """Long-form heatmap rows sorted by recall, shade normalized per metric column."""
    ordered = sorted(
        records, key=lambda r: r.scores.get("recall_at_k", 0.0), reverse=True
    )
    rows: list[dict] = []
    for record in ordered:
        for metric in CORE_METRICS:
            if metric not in record.scores:
                continue
            rows.append({
                "run": record.name,
                "metric": metric,
                "value": round(record.scores[metric], 3),
                "is_custom": record.provenance.get("sweep_id") is None,
            })
    for metric in CORE_METRICS:
        cells = [r for r in rows if r["metric"] == metric]
        if not cells:
            continue
        low = min(c["value"] for c in cells)
        high = max(c["value"] for c in cells)
        for cell in cells:
            cell["shade"] = 1.0 if high == low else (cell["value"] - low) / (high - low)
    return rows


def _render(rows: list[dict], order: list[str]) -> alt.Chart:
    base = alt.Chart(alt.Data(values=rows))
    x = alt.X("metric:N", sort=list(CORE_METRICS), title=None)
    y = alt.Y("run:N", sort=order, title=None)
    cells = base.transform_filter("!datum.is_delta").mark_rect().encode(
        x=x, y=y,
        color=alt.Color("shade:Q", scale=alt.Scale(scheme="blues"), legend=None),
        stroke=alt.condition("datum.is_custom", alt.value("#111827"), alt.value("#ffffff")),
        strokeWidth=alt.condition("datum.is_custom", alt.value(2.5), alt.value(2)),
    )
    labels = base.mark_text(fontWeight="bold").encode(
        x=x, y=y,
        text=alt.Text("value:Q", format=".3f"),
        color=alt.condition(
            "datum.shade > 0.6 && !datum.is_delta",
            alt.value("#ffffff"), alt.value("#1f2937"),
        ),
    )
    return (cells + labels).properties(width="container")


def sweep_heatmap(records: list[RunRecord]) -> alt.Chart:
    """Scored heatmap of sweep (and overlaid custom) runs."""
    rows = heatmap_rows(records)
    for row in rows:
        row["is_delta"] = False
    order: list[str] = []
    for row in rows:
        if row["run"] not in order:
            order.append(row["run"])
    return _render(rows, order)


def compare_heatmap(a: RunRecord, b: RunRecord) -> alt.Chart:
    """Two-row heatmap plus a text-only delta row."""
    rows = heatmap_rows([a, b])
    for row in rows:
        row["is_delta"] = False
    for metric in CORE_METRICS:
        if metric in a.scores and metric in b.scores:
            rows.append({
                "run": "Δ (B − A)", "metric": metric,
                "value": round(b.scores[metric] - a.scores[metric], 3),
                "shade": 0.0, "is_custom": False, "is_delta": True,
            })
    order = [a.name, b.name, "Δ (B − A)"]
    return _render(rows, order)

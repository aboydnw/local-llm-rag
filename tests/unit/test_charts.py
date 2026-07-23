from rag_lab.config import Config
from rag_lab.eval.run_store import RunRecord
from rag_lab.studio import charts


def _rec(name, recall, mrr, sweep="s1", run_id=None, std=None):
    provenance = {"sweep_id": sweep, "preset": name} if sweep else {}
    return RunRecord(
        run_id=run_id or name,
        name=name,
        created_at="2026-01-01T00:00:00+00:00",
        corpus="c1",
        scores={"recall_at_k": recall, "mrr": mrr},
        scores_std=std or {},
        config=Config(),
        provenance=provenance,
    )


def test_sweep_rows_keep_absolute_scores():
    rows = charts.sweep_rows([_rec("low", 0.79, 0.4), _rec("high", 0.80, 0.7)])
    by_key = {(row["run"], row["metric"]): row["value"] for row in rows}
    assert by_key[("low", "Recall at k")] == 0.79
    assert by_key[("high", "Recall at k")] == 0.80
    assert all("shade" not in row for row in rows)


def test_sweep_rows_include_uncertainty_interval():
    rows = charts.sweep_rows([_rec("repeat", 0.5, 0.4, std={"recall_at_k": 0.1})])
    recall = next(row for row in rows if row["metric"] == "Recall at k")
    assert recall["low"] == 0.4
    assert recall["high"] == 0.6
    assert recall["std"] == 0.1


def test_sweep_rows_flag_custom_runs_and_unique_labels():
    a = _rec("same", 0.5, 0.4, run_id="aaaaaa11")
    b = _rec("same", 0.9, 0.7, sweep=None, run_id="bbbbbb22")
    rows = charts.sweep_rows([a, b])
    assert {row["run"] for row in rows} == {"same (aaaaaa)", "same (bbbbbb)"}
    assert {row["run"] for row in rows if row["is_custom"]} == {"same (bbbbbb)"}


def test_sweep_dot_plot_is_faceted_with_fixed_domain():
    spec = charts.sweep_dot_plot([_rec("a", 0.5, 0.4), _rec("b", 0.9, 0.7)]).to_dict()
    assert "facet" in spec
    assert len(spec["spec"]["layer"]) == 2
    assert spec["spec"]["layer"][0]["encoding"]["x"]["scale"]["domain"] == [0, 1]


def test_comparison_rows_are_b_minus_a():
    rows = charts.comparison_rows(_rec("a", 0.5, 0.4), _rec("b", 0.9, 0.7))
    by_metric = {row["metric"]: row for row in rows}
    assert by_metric["Recall at k"]["delta"] == 0.4
    assert by_metric["MRR"]["delta"] == 0.29999999999999993


def test_comparison_rows_flag_delta_within_observed_variance():
    a = _rec("a", 0.5, 0.4, std={"recall_at_k": 0.03})
    b = _rec("b", 0.54, 0.7, std={"recall_at_k": 0.02})
    recall = next(row for row in charts.comparison_rows(a, b) if row["metric"] == "Recall at k")
    assert recall["uncertain"] is True


def test_comparison_delta_chart_has_zero_rule_and_layers():
    spec = charts.comparison_delta_chart(_rec("a", 0.5, 0.4), _rec("b", 0.9, 0.7)).to_dict()
    assert len(spec["layer"]) == 3
    assert spec["layer"][2]["mark"]["type"] == "rule"


def test_history_chart_adds_baseline_rule():
    baseline = _rec("baseline", 0.5, 0.4)
    spec = charts.history_chart(
        [baseline, _rec("next", 0.7, 0.6)], "recall_at_k", baseline
    ).to_dict()
    assert len(spec["layer"]) == 3

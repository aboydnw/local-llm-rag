import json

from rag_lab.config import Config
from rag_lab.eval.run_store import RunRecord
from rag_lab.studio import charts


def _rec(name, recall, mrr, sweep="s1", run_id=None):
    prov = {"sweep_id": sweep, "preset": name} if sweep else {}
    return RunRecord(run_id=run_id or name, name=name, created_at="t", corpus="c1",
                     scores={"recall_at_k": recall, "mrr": mrr},
                     config=Config(), provenance=prov)


def test_heatmap_rows_sorted_by_recall_desc():
    rows = charts.heatmap_rows([_rec("low", 0.5, 0.4), _rec("high", 0.9, 0.7)])
    assert rows[0]["run"] == "high"
    assert {r["metric"] for r in rows} == {"recall_at_k", "mrr"}


def test_heatmap_shade_normalized_within_column():
    rows = charts.heatmap_rows([_rec("a", 0.5, 0.7), _rec("b", 0.9, 0.3)])
    by = {(r["run"], r["metric"]): r["shade"] for r in rows}
    assert by[("b", "recall_at_k")] == 1.0
    assert by[("a", "recall_at_k")] == 0.0
    assert by[("a", "mrr")] == 1.0
    assert by[("b", "mrr")] == 0.0


def test_heatmap_shade_single_value_column_is_full():
    rows = charts.heatmap_rows([_rec("only", 0.5, 0.5)])
    assert all(r["shade"] == 1.0 for r in rows)


def test_heatmap_rows_flags_custom_runs():
    rows = charts.heatmap_rows([_rec("base", 0.5, 0.4), _rec("mine", 0.6, 0.5, sweep=None)])
    assert {r["run"] for r in rows if r["is_custom"]} == {"mine"}


def test_heatmap_rows_disambiguate_colliding_names():
    a = _rec("base: vector", 0.5, 0.4, run_id="aaaaaa11")
    b = _rec("base: vector", 0.9, 0.7, run_id="bbbbbb22")
    runs = {r["run"] for r in charts.heatmap_rows([a, b])}
    assert runs == {"base: vector (aaaaaa)", "base: vector (bbbbbb)"}


def test_heatmap_rows_keeps_plain_names_when_unique():
    rows = charts.heatmap_rows([_rec("vector", 0.5, 0.4), _rec("bm25", 0.9, 0.7)])
    assert {r["run"] for r in rows} == {"vector", "bm25"}


def test_sweep_heatmap_is_layered_chart():
    spec = charts.sweep_heatmap([_rec("a", 0.5, 0.4), _rec("b", 0.9, 0.7)]).to_dict()
    assert len(spec["layer"]) == 2


def test_compare_heatmap_has_delta_row():
    spec = charts.compare_heatmap(_rec("a", 0.5, 0.4), _rec("b", 0.9, 0.7)).to_dict()
    assert "Δ (B − A)" in json.dumps(spec, ensure_ascii=False)


def test_compare_heatmap_delta_label_avoids_run_name_collision():
    a = _rec("Δ (B − A)", 0.5, 0.4, run_id="aaaaaa11")
    b = _rec("plain", 0.9, 0.7, run_id="bbbbbb22")
    values = charts.compare_heatmap(a, b).to_dict()["data"]["values"]
    delta_labels = {v["run"] for v in values if v.get("is_delta")}
    assert delta_labels == {"Δ (B − A) [2]"}
    assert not any(v["run"] in delta_labels and not v.get("is_delta") for v in values)

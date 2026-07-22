import json

from rag_lab.config import Config
from rag_lab.eval.run_store import RunRecord
from rag_lab.studio import charts


def _rec(name, recall, mrr, sweep="s1"):
    prov = {"sweep_id": sweep, "preset": name} if sweep else {}
    return RunRecord(run_id=name, name=name, created_at="t", corpus="c1",
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


def test_sweep_heatmap_is_layered_chart():
    spec = charts.sweep_heatmap([_rec("a", 0.5, 0.4), _rec("b", 0.9, 0.7)]).to_dict()
    assert len(spec["layer"]) == 2


def test_compare_heatmap_has_delta_row():
    spec = charts.compare_heatmap(_rec("a", 0.5, 0.4), _rec("b", 0.9, 0.7)).to_dict()
    assert "Δ (B − A)" in json.dumps(spec, ensure_ascii=False)

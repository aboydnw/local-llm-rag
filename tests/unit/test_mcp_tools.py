from pathlib import Path

import pytest

from rag_lab.config import Config, write_default_config
from rag_lab.eval import mcp_tools, run_store
from rag_lab.eval.runner import EvalResult


def _result(recall: float) -> EvalResult:
    return EvalResult(
        item_id="q1", question="hi", actual_answer="tiles",
        recall_at_k=recall, mrr=recall, keyword_coverage=recall,
    )


def _seed_run(tmp_path, run_id, recall=1.0, gates=None) -> Path:
    runs_dir = tmp_path / "runs"
    config = Config()
    if gates:
        config.eval.gates = gates
    run_store.save_run(
        runs_dir,
        run_id=run_id,
        created_at=f"2026-07-21T00:00:0{run_id[-1]}+00:00",
        corpus=str(tmp_path / "rag.db"),
        config=config,
        repeats=[[_result(recall)]],
    )
    return runs_dir


def test_run_eval_persists_and_reports_baseline_deltas(tmp_path, monkeypatch):
    config = tmp_path / "rag.yml"
    write_default_config(config)
    golden = tmp_path / "golden.yml"
    golden.write_text("- id: q1\n  question: hi\n  ideal_docs: [a.md]\n")
    runs_dir = _seed_run(tmp_path, "r1")
    run_store.set_baseline(runs_dir, "r1")
    monkeypatch.setattr(
        "rag_lab.eval.service.run_eval", lambda *a, **k: [[_result(0.5)]]
    )

    out = mcp_tools.run_eval(
        config_path=config,
        db_path=tmp_path / "rag.db",
        golden_path=golden,
        runs_dir=runs_dir,
    )
    assert out["scores"]["recall@k"] == 0.5
    assert out["baseline"] == "r1"
    assert out["baseline_deltas"]["recall@k"] == -0.5
    assert len(run_store.list_runs(runs_dir)) == 2


def test_run_eval_applies_config_overrides(tmp_path, monkeypatch):
    config = tmp_path / "rag.yml"
    write_default_config(config)
    golden = tmp_path / "golden.yml"
    golden.write_text("- id: q1\n  question: hi\n  ideal_docs: [a.md]\n")
    seen: dict = {}

    def _capture(cfg, db, g, *, repeat=1, use_agent=False):
        seen["k"] = cfg.retriever.k
        return [[_result(1.0)]]

    monkeypatch.setattr("rag_lab.eval.service.run_eval", _capture)
    mcp_tools.run_eval(
        config_path=config,
        db_path=tmp_path / "rag.db",
        golden_path=golden,
        runs_dir=tmp_path / "runs",
        config_overrides={"retriever": {"k": 9}},
    )
    assert seen["k"] == 9


def test_list_runs_flags_baseline(tmp_path):
    runs_dir = _seed_run(tmp_path, "r1")
    _seed_run(tmp_path, "r2")
    run_store.set_baseline(runs_dir, "r1")
    listed = mcp_tools.list_runs(runs_dir=runs_dir)
    flags = {r["run_id"]: r["is_baseline"] for r in listed}
    assert flags == {"r1": True, "r2": False}


def test_compare_runs_reports_deltas(tmp_path):
    runs_dir = _seed_run(tmp_path, "r1", recall=1.0)
    _seed_run(tmp_path, "r2", recall=0.0)
    out = mcp_tools.compare_runs("r1", "r2", runs_dir=runs_dir)
    assert out["metric_deltas"]["recall@k"] == -1.0


def test_get_failures_uses_baseline_reference(tmp_path):
    gates = {"recall@k": 0.05}
    runs_dir = _seed_run(tmp_path, "r1", recall=1.0, gates=gates)
    _seed_run(tmp_path, "r2", recall=0.0, gates=gates)
    run_store.set_baseline(runs_dir, "r1")
    out = mcp_tools.get_failures("r2", runs_dir=runs_dir)
    assert [f["item_id"] for f in out["failures"]] == ["q1"]
    assert out["failures"][0]["failed_metrics"] == ["recall@k"]


def test_get_failures_without_gates_notes_it(tmp_path):
    runs_dir = _seed_run(tmp_path, "r1")
    out = mcp_tools.get_failures("r1", runs_dir=runs_dir)
    assert out["failures"] == []
    assert "note" in out


def test_add_golden_case_appends_and_slugs_id(tmp_path):
    golden = tmp_path / "golden.yml"
    golden.write_text("- id: q1\n  question: hi\n  ideal_docs: [a.md]\n")
    out = mcp_tools.add_golden_case(
        "What is eoAPI?", golden_path=golden, must_mention=["STAC"]
    )
    assert out["id"] == "what-is-eoapi"
    assert out["total_cases"] == 2


def test_add_golden_case_rejects_duplicate_id(tmp_path):
    golden = tmp_path / "golden.yml"
    golden.write_text("- id: q1\n  question: hi\n  ideal_docs: [a.md]\n")
    with pytest.raises(ValueError):
        mcp_tools.add_golden_case("dup", golden_path=golden, case_id="q1")

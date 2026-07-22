import json
import math

import pytest

from rag_lab.config import Config
from rag_lab.eval import run_store
from rag_lab.eval.runner import EvalResult


def _result(recall: float = 1.0) -> EvalResult:
    return EvalResult(
        item_id="q1", question="what is alpha?", actual_answer="tiles",
        recall_at_k=recall, mrr=recall, keyword_coverage=recall,
    )


def _save(tmp_path, run_id, *, config=None, repeats=None, name=None):
    runs_dir = tmp_path / "runs"
    record = run_store.save_run(
        runs_dir,
        run_id=run_id,
        name=name,
        created_at=f"2026-07-21T00:00:0{run_id[-1] if run_id[-1].isdigit() else '0'}+00:00",
        corpus="docs/",
        config=config or Config(),
        repeats=repeats or [[_result()]],
        golden_hash="abc123",
        corpus_snapshot=None,
    )
    return runs_dir, record


def test_save_run_persists_files_and_scores(tmp_path):
    runs_dir, record = _save(tmp_path, "r1")
    assert record.run_id == "r1"
    assert record.scores["recall@k"] == 1.0
    assert (runs_dir / "r1" / "run.json").exists()
    assert (runs_dir / "r1" / "items.json").exists()
    assert (runs_dir / "r1" / "config.yml").exists()
    assert (runs_dir / "r1" / "report.md").exists()


def test_save_run_records_provenance(tmp_path):
    runs_dir, _ = _save(tmp_path, "r1")
    data = json.loads((runs_dir / "r1" / "run.json").read_text())
    assert data["provenance"]["golden_hash"] == "abc123"
    assert data["provenance"]["config_hash"] == run_store.config_hash(Config())


def test_save_run_merges_extra_provenance(tmp_path):
    record = run_store.save_run(
        tmp_path / "runs",
        run_id="r-sweep",
        created_at="2026-07-22T00:00:00+00:00",
        corpus="docs/",
        config=Config(),
        repeats=[[_result()]],
        extra_provenance={"sweep_id": "s1", "preset": "vector"},
    )
    assert record.provenance["sweep_id"] == "s1"
    assert record.provenance["preset"] == "vector"
    assert "config_hash" in record.provenance


def test_save_run_repeat_scores_are_means_with_std(tmp_path):
    _, record = _save(
        tmp_path, "r1", repeats=[[_result(1.0)], [_result(0.0)]]
    )
    assert record.scores["recall@k"] == 0.5
    assert record.scores_std["recall@k"] == pytest.approx(math.sqrt(0.5))
    assert record.repeat == 2


def test_save_run_refuses_duplicate_id(tmp_path):
    _save(tmp_path, "r1")
    with pytest.raises(ValueError):
        _save(tmp_path, "r1")


def test_list_load_rename_delete(tmp_path):
    runs_dir, _ = _save(tmp_path, "r1")
    _save(tmp_path, "r2")
    assert [r.run_id for r in run_store.list_runs(runs_dir)] == ["r2", "r1"]
    run_store.rename_run(runs_dir, "r1", "baseline")
    assert run_store.load_run(runs_dir, "r1").name == "baseline"
    run_store.delete_run(runs_dir, "r1")
    assert [r.run_id for r in run_store.list_runs(runs_dir)] == ["r2"]


def test_list_runs_skips_corrupt_dirs(tmp_path):
    runs_dir, _ = _save(tmp_path, "r1")
    bad = runs_dir / "bad"
    bad.mkdir()
    (bad / "run.json").write_text("not json{")
    assert [r.run_id for r in run_store.list_runs(runs_dir)] == ["r1"]


def test_load_run_items_round_trip(tmp_path):
    runs_dir, _ = _save(tmp_path, "r1")
    items = run_store.load_run_items(runs_dir, "r1")
    assert items[0]["question"] == "what is alpha?"
    assert items[0]["repeat"] == 0
    assert run_store.load_run_items(runs_dir, "missing") == []


def test_diff_reports_changed_knobs_and_deltas(tmp_path):
    _, a = _save(tmp_path, "r1")
    other = Config()
    other.retriever.type = "bm25"
    _, b = _save(tmp_path, "r2", config=other, repeats=[[_result(0.0)]])
    result = run_store.diff(a, b)
    assert "retriever.type" in result["changed_knobs"]
    assert result["metric_deltas"]["recall@k"] == -1.0


def test_baseline_pin_set_get(tmp_path):
    runs_dir, _ = _save(tmp_path, "r1")
    assert run_store.get_baseline(runs_dir, "docs/") is None
    run_store.set_baseline(runs_dir, "r1")
    assert run_store.get_baseline(runs_dir, "docs/") == "r1"


def test_set_baseline_rejects_unknown_run(tmp_path):
    runs_dir, _ = _save(tmp_path, "r1")
    with pytest.raises(ValueError):
        run_store.set_baseline(runs_dir, "nope")


def test_get_baseline_none_when_pinned_run_deleted(tmp_path):
    runs_dir, _ = _save(tmp_path, "r1")
    run_store.set_baseline(runs_dir, "r1")
    run_store.delete_run(runs_dir, "r1")
    assert run_store.get_baseline(runs_dir, "docs/") is None


def test_baseline_is_per_corpus(tmp_path):
    runs_dir = tmp_path / "runs"
    for rid, corpus in [("r1", "c1"), ("r2", "c2")]:
        run_store.save_run(runs_dir, run_id=rid, created_at="t", corpus=corpus,
                           config=Config(), repeats=[[_result()]])
    run_store.set_baseline(runs_dir, "r1")
    run_store.set_baseline(runs_dir, "r2")
    assert run_store.get_baseline(runs_dir, "c1") == "r1"
    assert run_store.get_baseline(runs_dir, "c2") == "r2"
    assert run_store.get_baseline(runs_dir, "c3") is None


def test_legacy_single_pin_migrates_to_its_corpus(tmp_path):
    runs_dir = tmp_path / "runs"
    run_store.save_run(runs_dir, run_id="r1", created_at="t", corpus="c1",
                       config=Config(), repeats=[[_result()]])
    (runs_dir / run_store.BASELINE_FILE).write_text('{"run_id": "r1"}')
    assert run_store.get_baseline(runs_dir, "c1") == "r1"
    assert run_store.get_baseline(runs_dir, "c2") is None


def _rec(run_id, corpus, created_at, sweep=None):
    prov = {"sweep_id": sweep} if sweep else {}
    return run_store.RunRecord(
        run_id=run_id, name=run_id, created_at=created_at,
        corpus=corpus, scores={}, config=Config(), provenance=prov,
    )


def test_latest_sweep_ids_picks_newest_per_corpus():
    records = [
        _rec("a", "c1", "2026-07-20T00:00:00", sweep="old"),
        _rec("b", "c1", "2026-07-22T00:00:00", sweep="new"),
        _rec("c", "c2", "2026-07-21T00:00:00", sweep="other"),
    ]
    assert run_store.latest_sweep_ids(records) == {"c1": "new", "c2": "other"}


def test_visible_runs_hides_superseded_sweeps_keeps_custom():
    records = [
        _rec("a", "c1", "2026-07-20T00:00:00", sweep="old"),
        _rec("b", "c1", "2026-07-22T00:00:00", sweep="new"),
        _rec("custom", "c1", "2026-07-19T00:00:00"),
    ]
    visible = run_store.visible_runs(records)
    assert {r.run_id for r in visible} == {"b", "custom"}
    assert len(run_store.visible_runs(records, show_older_sweeps=True)) == 3


def test_config_hash_stable_and_sensitive(tmp_path):
    a = Config()
    b = Config()
    assert run_store.config_hash(a) == run_store.config_hash(b)
    b.retriever.type = "bm25"
    assert run_store.config_hash(a) != run_store.config_hash(b)


def test_golden_hash_changes_with_content(tmp_path):
    g = tmp_path / "golden.yml"
    g.write_text("- id: q1\n  question: a?\n")
    first = run_store.golden_hash(g)
    g.write_text("- id: q1\n  question: b?\n")
    assert first != run_store.golden_hash(g)


def test_aggregate_scores_includes_perf_and_agent_metrics():
    results = [
        EvalResult(
            item_id="a", question="q", actual_answer="x",
            recall_at_k=1.0, mrr=1.0, keyword_coverage=1.0,
            agent_metrics={"tool_calls": 2.0},
            generation_stats={
                "prompt_tokens": 1000.0, "prompt_eval_ms": 2000.0,
                "output_tokens": 100.0, "generation_ms": 10000.0,
            },
        )
    ]
    scores, _ = run_store._aggregate_scores([results])
    assert scores["tool_calls"] == 2.0
    assert scores["prompt_eval_tps_mean"] == 500.0
    assert "total_ms_p50" in scores


def test_aggregate_scores_omits_perf_without_stats():
    scores, _ = run_store._aggregate_scores([[_result()]])
    assert not any(k.startswith("prompt_eval_tps") for k in scores)


def test_item_failures_flags_gated_metrics_below_reference(tmp_path):
    items = [
        {"item_id": "good", "repeat": 0, "recall_at_k": 1.0, "keyword_coverage": 1.0},
        {"item_id": "bad", "repeat": 0, "recall_at_k": 0.0, "keyword_coverage": 1.0},
    ]
    failures = run_store.item_failures(
        items, reference={"recall@k": 1.0}, gates={"recall@k": 0.05}
    )
    assert [f["item_id"] for f in failures] == ["bad"]
    assert failures[0]["failed_metrics"] == ["recall@k"]

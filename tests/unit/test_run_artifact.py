from pathlib import Path

from rag_lab.eval.run_artifact import prompt_version, read_run, write_run
from rag_lab.eval.runner import EvalResult, RetrievedRef


def test_prompt_version_is_stable_short_hash() -> None:
    assert prompt_version("hello") == prompt_version("hello")
    assert len(prompt_version("hello")) == 8


def test_write_run_normalizes_nan_scores_to_null(tmp_path: Path) -> None:
    results = [
        EvalResult(
            item_id="x", question="q", actual_answer="a",
            recall_at_k=1.0, mrr=1.0, keyword_coverage=1.0,
            deepeval_scores={"faithfulness": float("nan")},
        )
    ]
    path = tmp_path / "run.json"
    write_run(
        path, results,
        config_summary="cfg", prompt_version="0000", k=5, created_at="2026-06-29T00:00:00Z",
    )
    assert "NaN" not in path.read_text(encoding="utf-8")
    data = read_run(path)
    assert data["items"][0]["deepeval_scores"]["faithfulness"] is None


def test_write_run_then_read_round_trips(tmp_path: Path) -> None:
    results = [
        EvalResult(
            item_id="x", question="q", actual_answer="a [1]",
            recall_at_k=1.0, mrr=1.0, keyword_coverage=1.0,
            ndcg_at_k=1.0, average_precision=1.0, citation_validity=1.0,
            retrieved=[RetrievedRef(1, "a.md", ("H",), 0, 0.9)],
            citations=[1], latency_ms={"retrieve": 5.0, "generate": 12.0},
        )
    ]
    path = tmp_path / "run.json"
    write_run(
        path, results,
        config_summary="cfg", prompt_version="abcd1234", k=5, created_at="2026-06-29T00:00:00Z",
    )
    data = read_run(path)
    assert data["schema_version"] == 2
    assert data["repeat"] == 1
    assert data["items"][0]["repeat"] == 0
    assert data["prompt_version"] == "abcd1234"
    assert data["aggregates"]["recall@k"] == 1.0
    assert data["items"][0]["retrieved"][0]["doc_path"] == "a.md"
    assert data["items"][0]["latency_ms"]["generate"] == 12.0


def test_write_run_persists_perf_when_stats_captured(tmp_path: Path) -> None:
    results = [
        EvalResult(
            item_id="x", question="q", actual_answer="a",
            recall_at_k=1.0, mrr=1.0, keyword_coverage=1.0,
            generation_stats={
                "prompt_tokens": 1000.0, "prompt_eval_ms": 2000.0,
                "output_tokens": 100.0, "generation_ms": 10000.0,
            },
        )
    ]
    path = tmp_path / "run.json"
    write_run(
        path, results,
        config_summary="cfg", prompt_version="0000", k=5, created_at="2026-06-29T00:00:00Z",
    )
    data = read_run(path)
    assert data["perf"]["prompt_eval_tps_mean"] == 500.0
    assert data["items"][0]["generation_stats"]["output_tokens"] == 100.0


def _scored(item_id: str, recall: float) -> EvalResult:
    return EvalResult(
        item_id=item_id, question="q", actual_answer="a",
        recall_at_k=recall, mrr=recall, keyword_coverage=recall,
    )


def test_write_run_with_repeats_flattens_items_and_averages(tmp_path: Path) -> None:
    repeats = [[_scored("x", 1.0)], [_scored("x", 0.0)]]
    path = tmp_path / "run.json"
    write_run(
        path, repeats=repeats,
        config_summary="cfg", prompt_version="0000", k=5, created_at="2026-06-29T00:00:00Z",
    )
    data = read_run(path)
    assert data["repeat"] == 2
    assert [it["repeat"] for it in data["items"]] == [0, 1]
    assert data["aggregates"]["recall@k"] == 0.5
    assert data["aggregates_std"]["recall@k"] > 0.0


def test_write_run_single_repeat_has_no_std(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    write_run(
        path, [_scored("x", 1.0)],
        config_summary="cfg", prompt_version="0000", k=5, created_at="2026-06-29T00:00:00Z",
    )
    data = read_run(path)
    assert "aggregates_std" not in data


def test_read_run_loads_older_artifact_without_perf(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        '{"schema_version": 1, "aggregates": {"recall@k": 1.0}, "items": '
        '[{"item_id": "x", "recall_at_k": 1.0}]}',
        encoding="utf-8",
    )
    data = read_run(path)
    assert "perf" not in data
    assert data["items"][0]["item_id"] == "x"

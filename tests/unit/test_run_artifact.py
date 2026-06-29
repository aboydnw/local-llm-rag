from pathlib import Path

from rag_lab.eval.run_artifact import prompt_version, read_run, write_run
from rag_lab.eval.runner import EvalResult, RetrievedRef


def test_prompt_version_is_stable_short_hash() -> None:
    assert prompt_version("hello") == prompt_version("hello")
    assert len(prompt_version("hello")) == 8


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
    assert data["schema_version"] == 1
    assert data["prompt_version"] == "abcd1234"
    assert data["aggregates"]["recall@k"] == 1.0
    assert data["items"][0]["retrieved"][0]["doc_path"] == "a.md"
    assert data["items"][0]["latency_ms"]["generate"] == 12.0

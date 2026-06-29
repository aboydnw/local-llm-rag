from pathlib import Path

from rag_lab.eval.reporter import MarkdownReporter
from rag_lab.eval.runner import EvalResult


def _make_result(item_id, recall, mrr_, kw, deepeval_scores=None) -> EvalResult:
    return EvalResult(
        item_id=item_id,
        question="q",
        actual_answer="a",
        recall_at_k=recall,
        mrr=mrr_,
        keyword_coverage=kw,
        deepeval_scores=deepeval_scores or {},
    )


def test_report_includes_aggregates_and_per_item_rows(tmp_path: Path) -> None:
    results = [
        _make_result("a", 1.0, 1.0, 1.0),
        _make_result("b", 0.0, 0.0, 0.5),
    ]
    out_path = tmp_path / "report.md"
    MarkdownReporter().write(
        results=results,
        config_summary="chunker=md@512 retriever=hybrid",
        out_path=out_path,
    )
    content = out_path.read_text()
    assert "recall@k" in content.lower()
    assert "0.50" in content
    assert "0.75" in content
    assert "| a |" in content
    assert "| b |" in content


def test_report_includes_new_metric_aggregates(tmp_path: Path) -> None:
    results = [
        EvalResult(
            item_id="a", question="q", actual_answer="a [1]",
            recall_at_k=1.0, mrr=1.0, keyword_coverage=1.0,
            ndcg_at_k=1.0, average_precision=1.0, citation_validity=1.0,
        ),
        EvalResult(
            item_id="b", question="q", actual_answer="a",
            recall_at_k=0.5, mrr=0.5, keyword_coverage=0.5,
            ndcg_at_k=0.5, average_precision=0.5, citation_validity=None,
        ),
    ]
    out_path = tmp_path / "report.md"
    MarkdownReporter().write(results=results, config_summary="cfg", out_path=out_path)
    content = out_path.read_text()
    assert "| ndcg@k | 0.75 |" in content
    assert "| map | 0.75 |" in content
    assert "| citation_validity | 1.00 |" in content


def _result_with(recall: float) -> EvalResult:
    return EvalResult(
        item_id="a", question="q", actual_answer="a",
        recall_at_k=recall, mrr=recall, keyword_coverage=recall,
        ndcg_at_k=recall, average_precision=recall,
    )


def test_report_diffs_against_previous_run_json(tmp_path: Path) -> None:
    from rag_lab.eval.run_artifact import write_run

    baseline = tmp_path / "baseline.json"
    write_run(
        baseline,
        [_result_with(recall=0.5)],
        config_summary="cfg", prompt_version="0000", k=5, created_at="2026-06-29T00:00:00Z",
    )
    out_path = tmp_path / "report.md"
    MarkdownReporter().write(
        results=[_result_with(recall=1.0)],
        config_summary="cfg",
        out_path=out_path,
        previous_run=baseline,
    )
    content = out_path.read_text()
    assert "Diff vs" in content
    assert "+0.50" in content


def test_report_renders_deepeval_aggregate_and_columns(tmp_path: Path) -> None:
    results = [
        _make_result("a", 1.0, 1.0, 1.0, {"answer_relevancy": 0.8, "faithfulness": 0.9}),
        _make_result("b", 0.0, 0.0, 0.0, {"answer_relevancy": 0.6, "faithfulness": 0.5}),
    ]
    out = tmp_path / "report.md"
    MarkdownReporter().write(results=results, config_summary="cfg", out_path=out)
    content = out.read_text()
    assert "answer_relevancy" in content
    assert "faithfulness" in content


def test_report_shows_na_for_missing_deepeval_key(tmp_path: Path) -> None:
    results = [
        _make_result("a", 1.0, 1.0, 1.0, {"answer_relevancy": 0.8, "contextual_precision": 0.7}),
        _make_result("b", 1.0, 1.0, 1.0, {"answer_relevancy": 0.6}),
    ]
    out = tmp_path / "report.md"
    MarkdownReporter().write(results=results, config_summary="cfg", out_path=out)
    content = out.read_text()
    assert "n/a" in content


def test_report_skips_nan_deepeval_values_in_display_and_aggregate(tmp_path: Path) -> None:
    results = [
        _make_result("a", 1.0, 1.0, 1.0, {"answer_relevancy": float("nan")}),
        _make_result("b", 1.0, 1.0, 1.0, {"answer_relevancy": 0.6}),
    ]
    out = tmp_path / "report.md"
    MarkdownReporter().write(results=results, config_summary="cfg", out_path=out)
    content = out.read_text()
    assert "n/a" in content
    assert "| answer_relevancy | 0.60 |" in content

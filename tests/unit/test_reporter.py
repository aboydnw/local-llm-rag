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


def test_report_includes_diff_when_previous_report_exists(tmp_path: Path) -> None:
    prev = tmp_path / "previous.md"
    prev.write_text(
        "# rag-lab eval report\n\n"
        "## Aggregates\n\n"
        "| metric | value |\n|---|---|\n"
        "| recall@k | 0.50 |\n| mrr | 0.50 |\n| keyword_coverage | 0.75 |\n"
    )
    results = [_make_result("a", 1.0, 1.0, 1.0)]
    out_path = tmp_path / "report.md"
    MarkdownReporter().write(
        results=results,
        config_summary="x",
        out_path=out_path,
        previous_report=prev,
    )
    content = out_path.read_text()
    assert "Diff vs" in content
    assert "+0.50" in content or "+0.5" in content


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

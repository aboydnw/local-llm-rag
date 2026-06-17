from pathlib import Path

from rag_lab.eval.reporter import MarkdownReporter
from rag_lab.eval.runner import EvalResult


def _make_result(id_: str, recall: float, mrr_: float, kw: float, judge: int | None = None) -> EvalResult:
    return EvalResult(
        item_id=id_,
        question="q",
        actual_answer="a",
        recall_at_k=recall,
        mrr=mrr_,
        keyword_coverage=kw,
        judge_score=judge,
        judge_reason="r" if judge is not None else None,
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
        "| metric | value |\n|---|---|\n| recall@k | 0.50 |\n| mrr | 0.50 |\n| keyword_coverage | 0.75 |\n"
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


def test_report_includes_judge_column_when_judge_results_present(tmp_path: Path) -> None:
    results = [_make_result("a", 1.0, 1.0, 1.0, judge=5)]
    out_path = tmp_path / "report.md"
    MarkdownReporter().write(results=results, config_summary="x", out_path=out_path)
    content = out_path.read_text()
    assert "judge" in content.lower()
    assert "5" in content

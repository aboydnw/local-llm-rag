from pathlib import Path

from typer.testing import CliRunner

from rag_lab import pipeline
from rag_lab.cli import app
from rag_lab.config import load_config, write_default_config
from rag_lab.eval.run_artifact import write_run
from rag_lab.eval.runner import EvalResult
from rag_lab.store.sqlite_vec import SqliteVecStore

runner = CliRunner()


def _build_matching_index(db: Path, config: Path) -> None:
    cfg = load_config(config)
    store = pipeline.build_store(cfg, db)
    store.initialize()
    store.write_manifest(pipeline.index_manifest(cfg))


def test_ask_refuses_when_index_embedder_mismatches_config(tmp_path: Path) -> None:
    config = tmp_path / "rag.yml"
    write_default_config(config)
    db = tmp_path / "rag.db"
    store = SqliteVecStore(db, dimension=1024)
    store.initialize()
    store.write_manifest(
        {"schema_version": 1, "embedder_model": "mxbai-embed-large", "dimension": 1024}
    )
    result = runner.invoke(app, ["ask", "hi", "--config", str(config), "--db", str(db)])
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_eval_refuses_when_index_embedder_mismatches_config(tmp_path: Path) -> None:
    config = tmp_path / "rag.yml"
    write_default_config(config)
    golden = tmp_path / "golden.yml"
    golden.write_text(
        "- id: q1\n  question: hi\n  ideal_docs: [a.md]\n", encoding="utf-8"
    )
    db = tmp_path / "rag.db"
    store = SqliteVecStore(db, dimension=1024)
    store.initialize()
    store.write_manifest(
        {"schema_version": 1, "embedder_model": "mxbai-embed-large", "dimension": 1024}
    )
    result = runner.invoke(
        app, ["eval", "--config", str(config), "--db", str(db), "--golden", str(golden)]
    )
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_eval_help_documents_previous_run_artifact() -> None:
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "run.json" in result.output


def test_eval_exits_2_when_gate_regresses(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "rag.yml"
    write_default_config(config)
    config.write_text(
        config.read_text(encoding="utf-8") + "\neval:\n  gates:\n    recall@k: 0.05\n",
        encoding="utf-8",
    )
    golden = tmp_path / "golden.yml"
    golden.write_text("- id: q1\n  question: hi\n  ideal_docs: [a.md]\n", encoding="utf-8")

    baseline = tmp_path / "baseline.json"
    write_run(
        baseline,
        [EvalResult(item_id="q1", question="hi", actual_answer="a",
                    recall_at_k=1.0, mrr=1.0, keyword_coverage=1.0,
                    ndcg_at_k=1.0, average_precision=1.0)],
        config_summary="cfg", prompt_version="0000", k=5, created_at="2026-06-29T00:00:00Z",
    )

    db = tmp_path / "rag.db"
    _build_matching_index(db, config)

    monkeypatch.setattr(
        "rag_lab.cli.EvalRunner.run",
        lambda self, items: [EvalResult(item_id="q1", question="hi", actual_answer="a",
                                        recall_at_k=0.0, mrr=0.0, keyword_coverage=0.0)],
    )
    result = runner.invoke(
        app,
        ["eval", "--config", str(config), "--db", str(db),
         "--golden", str(golden), "--baseline", str(baseline),
         "--report", str(tmp_path / "report.md")],
    )
    assert result.exit_code == 2
    assert (tmp_path / "report.md").exists()  # the gate runs after the report is written


def _gated_config(tmp_path: Path) -> Path:
    config = tmp_path / "rag.yml"
    write_default_config(config)
    config.write_text(
        config.read_text(encoding="utf-8") + "\neval:\n  gates:\n    recall@k: 0.05\n",
        encoding="utf-8",
    )
    return config


def _monkeypatch_runner(monkeypatch) -> None:
    monkeypatch.setattr(
        "rag_lab.cli.EvalRunner.run",
        lambda self, items: [EvalResult(item_id="q1", question="hi", actual_answer="a",
                                        recall_at_k=1.0, mrr=1.0, keyword_coverage=1.0)],
    )


def test_eval_exits_1_when_baseline_is_unreadable(tmp_path: Path, monkeypatch) -> None:
    config = _gated_config(tmp_path)
    golden = tmp_path / "golden.yml"
    golden.write_text("- id: q1\n  question: hi\n  ideal_docs: [a.md]\n", encoding="utf-8")
    db = tmp_path / "rag.db"
    _build_matching_index(db, config)
    bad = tmp_path / "baseline.json"
    bad.write_text("not a run.json\n", encoding="utf-8")
    _monkeypatch_runner(monkeypatch)

    result = runner.invoke(
        app,
        ["eval", "--config", str(config), "--db", str(db), "--golden", str(golden),
         "--baseline", str(bad), "--report", str(tmp_path / "report.md")],
    )
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert (tmp_path / "report.md").exists()  # failed at the gate, not before the run


def test_eval_exits_1_when_baseline_k_differs(tmp_path: Path, monkeypatch) -> None:
    config = _gated_config(tmp_path)
    golden = tmp_path / "golden.yml"
    golden.write_text("- id: q1\n  question: hi\n  ideal_docs: [a.md]\n", encoding="utf-8")
    db = tmp_path / "rag.db"
    _build_matching_index(db, config)
    baseline = tmp_path / "baseline.json"
    write_run(
        baseline,
        [EvalResult(item_id="q1", question="hi", actual_answer="a",
                    recall_at_k=1.0, mrr=1.0, keyword_coverage=1.0)],
        config_summary="cfg", prompt_version="0000", k=99, created_at="2026-06-29T00:00:00Z",
    )
    _monkeypatch_runner(monkeypatch)

    result = runner.invoke(
        app,
        ["eval", "--config", str(config), "--db", str(db), "--golden", str(golden),
         "--baseline", str(baseline), "--report", str(tmp_path / "report.md")],
    )
    assert result.exit_code == 1
    assert (tmp_path / "report.md").exists()

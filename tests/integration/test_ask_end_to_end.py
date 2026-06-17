from pathlib import Path

import pytest
from typer.testing import CliRunner

from rag_lab.cli import app


@pytest.mark.integration
def test_ask_returns_answer_with_citations(tmp_path: Path, fixture_corpus: Path) -> None:
    runner = CliRunner()

    cfg_path = tmp_path / "rag.yml"
    db_path = tmp_path / "rag.db"

    runner.invoke(app, ["config", "init", "--path", str(cfg_path)])
    ingest_result = runner.invoke(
        app,
        ["ingest", str(fixture_corpus), "--config", str(cfg_path), "--db", str(db_path)],
    )
    assert ingest_result.exit_code == 0, ingest_result.output

    ask_result = runner.invoke(
        app,
        ["ask", "what is the alpha document about?", "--config", str(cfg_path), "--db", str(db_path)],
    )
    assert ask_result.exit_code == 0, ask_result.output
    assert len(ask_result.stdout) > 20

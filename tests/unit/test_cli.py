from pathlib import Path

from typer.testing import CliRunner

from rag_lab.cli import app
from rag_lab.config import write_default_config
from rag_lab.store.sqlite_vec import SqliteVecStore

runner = CliRunner()


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


def test_ask_agent_flag_runs_agent(tmp_path: Path, monkeypatch) -> None:
    from rag_lab.agent.agent import AgentResult, AgentStep
    from rag_lab.types import Chunk

    config = tmp_path / "rag.yml"
    write_default_config(config)
    db = tmp_path / "rag.db"
    store = SqliteVecStore(db, dimension=768)
    store.initialize()
    store.write_manifest(
        {"schema_version": 1, "embedder_model": "nomic-embed-text", "dimension": 768}
    )

    class _FakeAgent:
        def run(self, question: str) -> AgentResult:
            return AgentResult(
                answer="agent answer",
                steps=[
                    AgentStep(
                        thought="think",
                        action="vector_search",
                        action_input="q",
                        observation="obs",
                        chunks=[],
                    )
                ],
                chunks_seen=[],
                final_context=[
                    Chunk(
                        text="c",
                        doc_path=Path("docs/a.md"),
                        heading_path=("H",),
                        position=0,
                    )
                ],
            )

    monkeypatch.setattr("rag_lab.pipeline.build_embedder", lambda cfg: object())
    monkeypatch.setattr("rag_lab.pipeline.build_agent", lambda *a, **k: _FakeAgent())

    result = runner.invoke(
        app,
        [
            "ask",
            "hi",
            "--config",
            str(config),
            "--db",
            str(db),
            "--agent",
            "--show-trace",
        ],
    )
    assert result.exit_code == 0
    assert "agent answer" in result.output
    assert "Agent trace" in result.output
    assert "vector_search" in result.output
    assert "docs/a.md" in result.output


def test_ask_agent_applies_k_override(tmp_path: Path, monkeypatch) -> None:
    from rag_lab.agent.agent import AgentResult

    config = tmp_path / "rag.yml"
    write_default_config(config)
    db = tmp_path / "rag.db"
    store = SqliteVecStore(db, dimension=768)
    store.initialize()
    store.write_manifest(
        {"schema_version": 1, "embedder_model": "nomic-embed-text", "dimension": 768}
    )

    captured = {}

    class _FakeAgent:
        def run(self, question: str) -> AgentResult:
            return AgentResult(
                answer="a", steps=[], chunks_seen=[], final_context=[]
            )

    def _fake_build_agent(store, embedder, cfg):
        captured["k"] = cfg.retriever.k
        return _FakeAgent()

    monkeypatch.setattr("rag_lab.pipeline.build_embedder", lambda cfg: object())
    monkeypatch.setattr("rag_lab.pipeline.build_agent", _fake_build_agent)

    result = runner.invoke(
        app,
        ["ask", "hi", "--config", str(config), "--db", str(db), "--agent", "--k", "9"],
    )
    assert result.exit_code == 0
    assert captured["k"] == 9


def test_ask_agent_reports_bad_tool_config_cleanly(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "rag.yml"
    write_default_config(config)
    db = tmp_path / "rag.db"
    store = SqliteVecStore(db, dimension=768)
    store.initialize()
    store.write_manifest(
        {"schema_version": 1, "embedder_model": "nomic-embed-text", "dimension": 768}
    )

    def _boom(store, embedder, cfg):
        raise ValueError("Unknown agent tool: bogus")

    monkeypatch.setattr("rag_lab.pipeline.build_embedder", lambda cfg: object())
    monkeypatch.setattr("rag_lab.pipeline.build_agent", _boom)

    result = runner.invoke(
        app, ["ask", "hi", "--config", str(config), "--db", str(db), "--agent"]
    )
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Unknown agent tool" in result.output


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

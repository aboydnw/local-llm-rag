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


def test_ask_stats_prints_throughput_line(tmp_path: Path, monkeypatch) -> None:
    from rag_lab.llms.fake import FakeLLM

    config = tmp_path / "rag.yml"
    write_default_config(config)
    db = tmp_path / "rag.db"
    store = SqliteVecStore(db, dimension=768)
    store.initialize()
    store.write_manifest(
        {"schema_version": 1, "embedder_model": "nomic-embed-text", "dimension": 768}
    )

    class _FakeRetriever:
        def retrieve(self, question, k):
            return []

    monkeypatch.setattr("rag_lab.pipeline.build_embedder", lambda cfg: object())
    monkeypatch.setattr(
        "rag_lab.pipeline.build_retriever", lambda *a, **k: _FakeRetriever()
    )
    monkeypatch.setattr(
        "rag_lab.pipeline.build_llm", lambda cfg: FakeLLM("four word fake answer")
    )

    result = runner.invoke(
        app, ["ask", "hi", "--config", str(config), "--db", str(db), "--stats"]
    )
    assert result.exit_code == 0
    assert "--- Stats ---" in result.output
    assert "gen 4 tok @" in result.output
    assert "tok/s" in result.output


def test_ask_agent_stats_aggregates_across_calls(tmp_path: Path, monkeypatch) -> None:
    from rag_lab.agent.agent import AgentResult
    from rag_lab.types import GenerationStats

    config = tmp_path / "rag.yml"
    write_default_config(config)
    db = tmp_path / "rag.db"
    store = SqliteVecStore(db, dimension=768)
    store.initialize()
    store.write_manifest(
        {"schema_version": 1, "embedder_model": "nomic-embed-text", "dimension": 768}
    )

    step = GenerationStats(
        prompt_tokens=100, prompt_eval_ms=1000.0, output_tokens=50, generation_ms=2000.0
    )

    class _FakeAgent:
        def run(self, question: str) -> AgentResult:
            return AgentResult(
                answer="a",
                steps=[],
                chunks_seen=[],
                final_context=[],
                llm_calls=2,
                stats=[step, step],
            )

    monkeypatch.setattr("rag_lab.pipeline.build_embedder", lambda cfg: object())
    monkeypatch.setattr("rag_lab.pipeline.build_agent", lambda *a, **k: _FakeAgent())

    result = runner.invoke(
        app,
        ["ask", "hi", "--config", str(config), "--db", str(db), "--agent", "--stats"],
    )
    assert result.exit_code == 0
    assert "prompt 200 tok" in result.output
    assert "gen 100 tok" in result.output
    assert "2 llm calls" in result.output


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


def test_eval_agent_flag_scores_agent_runs(tmp_path: Path, monkeypatch) -> None:
    from rag_lab.agent.agent import AgentResult, AgentStep
    from rag_lab.llms.fake import FakeLLM
    from rag_lab.types import Chunk

    config = tmp_path / "rag.yml"
    write_default_config(config)
    golden = tmp_path / "golden.yml"
    golden.write_text(
        "- id: q1\n  question: hi\n  ideal_docs: [docs/a.md]\n", encoding="utf-8"
    )
    db = tmp_path / "rag.db"
    store = SqliteVecStore(db, dimension=768)
    store.initialize()
    store.write_manifest(
        {"schema_version": 1, "embedder_model": "nomic-embed-text", "dimension": 768}
    )

    class _FakeAgent:
        def run(self, question: str) -> AgentResult:
            chunk = Chunk(
                text="c", doc_path=Path("docs/a.md"), heading_path=("H",), position=0
            )
            return AgentResult(
                answer="answer [1]",
                steps=[
                    AgentStep(
                        thought="t", action="vector_search",
                        action_input="q", observation="o", chunks=[chunk],
                    )
                ],
                chunks_seen=[chunk],
                final_context=[chunk],
                llm_calls=2,
            )

    monkeypatch.setattr("rag_lab.pipeline.build_embedder", lambda cfg: object())
    monkeypatch.setattr("rag_lab.pipeline.build_llm", lambda cfg: FakeLLM("unused"))
    monkeypatch.setattr("rag_lab.pipeline.build_agent", lambda *a, **k: _FakeAgent())

    report = tmp_path / "report.md"
    result = runner.invoke(
        app,
        [
            "eval", "--golden", str(golden), "--config", str(config),
            "--db", str(db), "--report", str(report), "--agent",
            "--runs-dir", str(tmp_path / "runs"),
        ],
    )
    assert result.exit_code == 0
    assert report.exists()
    assert "tool_calls" in report.read_text()


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
        "rag_lab.eval.runner.EvalRunner.run",
        lambda self, items: [EvalResult(item_id="q1", question="hi", actual_answer="a",
                                        recall_at_k=0.0, mrr=0.0, keyword_coverage=0.0)],
    )
    result = runner.invoke(
        app,
        ["eval", "--config", str(config), "--db", str(db),
         "--golden", str(golden), "--baseline", str(baseline),
         "--report", str(tmp_path / "report.md"),
         "--runs-dir", str(tmp_path / "runs")],
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
        "rag_lab.eval.runner.EvalRunner.run",
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
         "--baseline", str(bad), "--report", str(tmp_path / "report.md"),
         "--runs-dir", str(tmp_path / "runs")],
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
         "--baseline", str(baseline), "--report", str(tmp_path / "report.md"),
         "--runs-dir", str(tmp_path / "runs")],
    )
    assert result.exit_code == 1
    assert (tmp_path / "report.md").exists()


def _eval_args(tmp_path: Path, config: Path, golden: Path, db: Path) -> list[str]:
    return [
        "eval", "--config", str(config), "--db", str(db), "--golden", str(golden),
        "--report", str(tmp_path / "report.md"),
        "--runs-dir", str(tmp_path / "runs"),
    ]


def test_eval_saves_run_to_store(tmp_path: Path, monkeypatch) -> None:
    config = _gated_config(tmp_path)
    golden = tmp_path / "golden.yml"
    golden.write_text("- id: q1\n  question: hi\n  ideal_docs: [a.md]\n", encoding="utf-8")
    db = tmp_path / "rag.db"
    _build_matching_index(db, config)
    _monkeypatch_runner(monkeypatch)

    result = runner.invoke(app, _eval_args(tmp_path, config, golden, db))
    assert result.exit_code == 0
    from rag_lab.eval import run_store

    runs = run_store.list_runs(tmp_path / "runs")
    assert len(runs) == 1
    assert runs[0].scores["recall@k"] == 1.0
    assert "config_hash" in runs[0].provenance


def test_eval_repeat_records_mean_and_std(tmp_path: Path, monkeypatch) -> None:
    config = _gated_config(tmp_path)
    golden = tmp_path / "golden.yml"
    golden.write_text("- id: q1\n  question: hi\n  ideal_docs: [a.md]\n", encoding="utf-8")
    db = tmp_path / "rag.db"
    _build_matching_index(db, config)
    recalls = iter([1.0, 0.0])
    monkeypatch.setattr(
        "rag_lab.eval.runner.EvalRunner.run",
        lambda self, items: [
            EvalResult(item_id="q1", question="hi", actual_answer="a",
                       recall_at_k=next(recalls), mrr=0.5, keyword_coverage=0.5)
        ],
    )

    result = runner.invoke(
        app, [*_eval_args(tmp_path, config, golden, db), "--repeat", "2"]
    )
    assert result.exit_code == 0
    from rag_lab.eval import run_store

    record = run_store.list_runs(tmp_path / "runs")[0]
    assert record.repeat == 2
    assert record.scores["recall@k"] == 0.5
    assert record.scores_std["recall@k"] > 0.0


def test_eval_set_baseline_pins_and_exits(tmp_path: Path, monkeypatch) -> None:
    config = _gated_config(tmp_path)
    golden = tmp_path / "golden.yml"
    golden.write_text("- id: q1\n  question: hi\n  ideal_docs: [a.md]\n", encoding="utf-8")
    db = tmp_path / "rag.db"
    _build_matching_index(db, config)
    _monkeypatch_runner(monkeypatch)

    result = runner.invoke(app, _eval_args(tmp_path, config, golden, db))
    assert result.exit_code == 0
    from rag_lab.eval import run_store

    run_id = run_store.list_runs(tmp_path / "runs")[0].run_id
    result = runner.invoke(
        app,
        ["eval", "--runs-dir", str(tmp_path / "runs"), "--set-baseline", run_id],
    )
    assert result.exit_code == 0
    assert run_store.get_baseline(tmp_path / "runs", str(db)) == run_id


def test_eval_set_baseline_unknown_run_exits_1(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["eval", "--runs-dir", str(tmp_path / "runs"), "--set-baseline", "nope"],
    )
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_eval_gates_against_pinned_baseline(tmp_path: Path, monkeypatch) -> None:
    config = _gated_config(tmp_path)
    golden = tmp_path / "golden.yml"
    golden.write_text("- id: q1\n  question: hi\n  ideal_docs: [a.md]\n", encoding="utf-8")
    db = tmp_path / "rag.db"
    _build_matching_index(db, config)
    recalls = iter([1.0, 0.0])
    monkeypatch.setattr(
        "rag_lab.eval.runner.EvalRunner.run",
        lambda self, items: [
            EvalResult(item_id="q1", question="hi", actual_answer="a",
                       recall_at_k=next(recalls), mrr=0.5, keyword_coverage=0.5)
        ],
    )
    from rag_lab.eval import run_store

    result = runner.invoke(app, _eval_args(tmp_path, config, golden, db))
    assert result.exit_code == 0
    run_id = run_store.list_runs(tmp_path / "runs")[0].run_id
    runner.invoke(
        app, ["eval", "--runs-dir", str(tmp_path / "runs"), "--set-baseline", run_id]
    )

    result = runner.invoke(app, _eval_args(tmp_path, config, golden, db))
    assert result.exit_code == 2
    assert "Regression gate failed" in result.output


def test_eval_prints_deltas_vs_pinned_baseline_without_gates(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "rag.yml"
    write_default_config(config)
    golden = tmp_path / "golden.yml"
    golden.write_text("- id: q1\n  question: hi\n  ideal_docs: [a.md]\n", encoding="utf-8")
    db = tmp_path / "rag.db"
    _build_matching_index(db, config)
    _monkeypatch_runner(monkeypatch)
    from rag_lab.eval import run_store

    result = runner.invoke(app, _eval_args(tmp_path, config, golden, db))
    assert result.exit_code == 0
    run_id = run_store.list_runs(tmp_path / "runs")[0].run_id
    runner.invoke(
        app, ["eval", "--runs-dir", str(tmp_path / "runs"), "--set-baseline", run_id]
    )

    result = runner.invoke(app, _eval_args(tmp_path, config, golden, db))
    assert result.exit_code == 0
    assert "Δ vs baseline" in result.output


def test_eval_exits_1_when_baseline_gated_metric_not_finite(
    tmp_path: Path, monkeypatch
) -> None:
    import json

    config = _gated_config(tmp_path)
    golden = tmp_path / "golden.yml"
    golden.write_text("- id: q1\n  question: hi\n  ideal_docs: [a.md]\n", encoding="utf-8")
    db = tmp_path / "rag.db"
    _build_matching_index(db, config)
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"k": 5, "aggregates": {"recall@k": "bad"}}), encoding="utf-8"
    )
    _monkeypatch_runner(monkeypatch)

    result = runner.invoke(
        app,
        ["eval", "--config", str(config), "--db", str(db), "--golden", str(golden),
         "--baseline", str(baseline), "--report", str(tmp_path / "report.md"),
         "--runs-dir", str(tmp_path / "runs")],
    )
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)

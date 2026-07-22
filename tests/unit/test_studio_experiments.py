import pytest

from rag_lab.config import Config
from rag_lab.embedders.fake import FakeEmbedder
from rag_lab.loaders.markdown import MarkdownLoader
from rag_lab.studio import experiments
from rag_lab.studio.corpora import local_corpus
from rag_lab.studio.workspace import Workspace


class FakeLLM:
    def generate(self, prompt: str) -> str:
        return "The alpha document is about tiles."

    def last_stats(self):
        return None


def _corpus(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir(exist_ok=True)
    (d / "a.md").write_text("# Alpha\n\nThe alpha document is about tiles.\n")
    return d


def _golden(tmp_path):
    g = tmp_path / "golden.yml"
    g.write_text(
        "- id: q1\n"
        "  question: What is alpha about?\n"
        "  ideal_docs: []\n"
        "  must_mention: [tiles]\n"
        "  ideal_answer: Alpha is about tiles.\n"
    )
    return g


def _run(tmp_path, run_id, config=None):
    ws = Workspace(tmp_path / ".rag-lab")
    ws.initialize()
    corpus = _corpus(tmp_path)
    return ws, experiments.run_eval(
        ws,
        local_corpus(str(corpus)),
        config or Config(),
        _golden(tmp_path),
        run_id=run_id,
        created_at="2026-06-17T00:00:00+00:00",
        loader=MarkdownLoader(corpus),
        embedder=FakeEmbedder(16),
        llm=FakeLLM(),
    )


def test_run_eval_uses_agent_when_enabled(tmp_path, monkeypatch):
    from pathlib import Path

    from rag_lab.agent.agent import AgentResult, AgentStep
    from rag_lab.types import Chunk

    class _FakeAgent:
        def run(self, question: str) -> AgentResult:
            chunk = Chunk(
                text="tiles", doc_path=Path("a.md"), heading_path=("H",), position=0
            )
            return AgentResult(
                answer="about tiles [1]",
                steps=[
                    AgentStep(
                        thought="t", action="keyword_search",
                        action_input="q", observation="o", chunks=[chunk],
                    )
                ],
                chunks_seen=[chunk],
                final_context=[chunk],
                llm_calls=2,
            )

    monkeypatch.setattr(
        "rag_lab.studio.components.build_agent", lambda *a, **k: _FakeAgent()
    )
    config = Config()
    config.agent.enabled = True
    _, record = _run(tmp_path, "r-agent", config=config)
    assert record.scores["tool_calls"] == 1.0
    assert "recall@k_seen" in record.scores


def test_run_eval_repeat_records_std(tmp_path):
    ws = Workspace(tmp_path / ".rag-lab")
    ws.initialize()
    corpus = _corpus(tmp_path)
    record = experiments.run_eval(
        ws,
        local_corpus(str(corpus)),
        Config(),
        _golden(tmp_path),
        run_id="r-rep",
        created_at="2026-06-17T00:00:00+00:00",
        repeat=2,
        loader=MarkdownLoader(corpus),
        embedder=FakeEmbedder(16),
        llm=FakeLLM(),
    )
    assert record.repeat == 2
    assert "recall@k" in record.scores_std


def test_baseline_pin_via_workspace(tmp_path):
    ws, record = _run(tmp_path, "r1")
    assert experiments.get_baseline(ws, record.corpus) is None
    experiments.set_baseline(ws, "r1")
    assert experiments.get_baseline(ws, record.corpus) == "r1"


def test_run_eval_persists_run(tmp_path):
    ws, record = _run(tmp_path, "r1")
    assert record.run_id == "r1"
    assert "recall@k" in record.scores
    assert (ws.run_dir("r1") / "run.json").exists()
    assert (ws.run_dir("r1") / "report.md").exists()
    assert (ws.run_dir("r1") / "config.yml").exists()


def test_run_eval_persists_items_json(tmp_path):
    ws, _ = _run(tmp_path, "r1")
    assert (ws.run_dir("r1") / "items.json").exists()
    items = experiments.load_run_items(ws, "r1")
    assert items and items[0]["question"] == "What is alpha about?"
    assert "agent_trace" in items[0]


def test_load_run_items_returns_empty_when_absent(tmp_path):
    ws = Workspace(tmp_path / ".rag-lab")
    ws.initialize()
    assert experiments.load_run_items(ws, "missing") == []


def test_run_eval_snapshots_corpus_sources(tmp_path):
    import json

    ws, _ = _run(tmp_path, "r1")
    data = json.loads((ws.run_dir("r1") / "run.json").read_text())
    assert data["corpus_snapshot"]["sources"][0]["type"] == "local"


def test_list_and_load_runs(tmp_path):
    ws, _ = _run(tmp_path, "r1")
    runs = experiments.list_runs(ws)
    assert [r.run_id for r in runs] == ["r1"]
    loaded = experiments.load_run(ws, "r1")
    assert loaded.scores == runs[0].scores


def test_rename_and_delete(tmp_path):
    ws, _ = _run(tmp_path, "r1")
    experiments.rename_run(ws, "r1", "baseline")
    assert experiments.load_run(ws, "r1").name == "baseline"
    experiments.delete_run(ws, "r1")
    assert experiments.list_runs(ws) == []


def test_run_eval_refuses_to_overwrite_existing_run(tmp_path):
    _run(tmp_path, "r1")
    with pytest.raises(ValueError):
        _run(tmp_path, "r1")


def test_diff_reports_changed_knobs(tmp_path):
    _, a = _run(tmp_path, "r1")
    other = Config()
    other.retriever.type = "bm25"
    _, b = _run(tmp_path, "r2", config=other)
    result = experiments.diff(a, b)
    assert "retriever.type" in result["changed_knobs"]
    assert set(result["metric_deltas"]) >= {"recall@k", "mrr", "keyword_coverage"}

from pathlib import Path

from rag_lab.config import Config
from rag_lab.eval.run_store import RunRecord
from rag_lab.studio import experiments
from rag_lab.studio.workspace import Workspace


def test_run_base_sweep_runs_every_preset_tagged(monkeypatch, tmp_path):
    calls = []

    def fake_run_eval(workspace, corpus, config, golden_path, run_id, created_at, **kwargs):
        calls.append((config, run_id, kwargs))
        return RunRecord(
            run_id=run_id,
            name=kwargs["name"],
            created_at=created_at,
            corpus="c1",
            scores={},
            config=config,
            provenance=dict(kwargs["extra_provenance"]),
        )

    monkeypatch.setattr(experiments, "run_eval", fake_run_eval)
    records = experiments.run_base_sweep(
        Workspace(tmp_path), None, Config(), Path("golden.yml"),
        sweep_id="s1", created_at="2026-07-22T00:00:00",
    )
    assert len(records) == 8
    assert {r.provenance["sweep_id"] for r in records} == {"s1"}
    assert len({r.provenance["preset"] for r in records}) == 8
    assert all(not cfg.eval.deepeval for cfg, _, _ in calls)
    assert all(kw["repeat"] == 1 for _, _, kw in calls)
    assert all(run_id.startswith("s1-") for _, run_id, _ in calls)


def test_run_base_sweep_reports_progress(monkeypatch, tmp_path):
    seen = []

    def fake_run_eval(workspace, corpus, config, golden_path, run_id, created_at, **kwargs):
        return RunRecord(run_id=run_id, name="n", created_at=created_at,
                         corpus="c1", scores={}, config=config)

    monkeypatch.setattr(experiments, "run_eval", fake_run_eval)
    experiments.run_base_sweep(
        Workspace(tmp_path), None, Config(), Path("golden.yml"),
        sweep_id="s1", created_at="t",
        on_progress=lambda i, name: seen.append((i, name)),
    )
    assert len(seen) == 8
    assert seen[0] == (0, "vector")


def test_latest_sweep_records_returns_newest_sweep_only(monkeypatch, tmp_path):
    records = [
        RunRecord(run_id="a", name="a", created_at="2026-07-20T00:00:00", corpus="c1",
                  scores={}, config=Config(), provenance={"sweep_id": "old"}),
        RunRecord(run_id="b", name="b", created_at="2026-07-22T00:00:00", corpus="c1",
                  scores={}, config=Config(), provenance={"sweep_id": "new"}),
        RunRecord(run_id="c", name="c", created_at="2026-07-22T00:00:00", corpus="c2",
                  scores={}, config=Config(), provenance={"sweep_id": "x"}),
    ]
    monkeypatch.setattr(experiments, "list_runs", lambda ws: records)
    got = experiments.latest_sweep_records(Workspace(tmp_path), "c1")
    assert [r.run_id for r in got] == ["b"]
    assert experiments.latest_sweep_records(Workspace(tmp_path), "c9") == []

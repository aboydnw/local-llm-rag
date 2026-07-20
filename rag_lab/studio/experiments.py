import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import yaml

from rag_lab.config import Config, config_summary
from rag_lab.eval import golden_set as golden_set_mod
from rag_lab.eval.aggregate import aggregate_perf
from rag_lab.eval.reporter import MarkdownReporter
from rag_lab.eval.run_artifact import prompt_version as artifact_prompt_version
from rag_lab.eval.run_artifact import read_run, write_run
from rag_lab.eval.runner import EvalRunner
from rag_lab.prompts import PromptBuilder
from rag_lab.store.sqlite_vec import SqliteVecStore
from rag_lab.studio import components
from rag_lab.studio import indexer as indexer_mod
from rag_lab.studio.corpora import Corpus
from rag_lab.studio.workspace import Workspace


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    created_at: str
    corpus: str
    scores: dict[str, float]
    config: Config


def _aggregate_scores(results) -> dict[str, float]:
    if not results:
        return {}
    scores = {
        "recall@k": statistics.mean(r.recall_at_k for r in results),
        "mrr": statistics.mean(r.mrr for r in results),
        "keyword_coverage": statistics.mean(r.keyword_coverage for r in results),
    }
    for key in sorted({k for r in results for k in r.deepeval_scores}):
        vals = [
            r.deepeval_scores[key]
            for r in results
            if key in r.deepeval_scores and not math.isnan(r.deepeval_scores[key])
        ]
        if vals:
            scores[key] = statistics.mean(vals)
    for key in sorted({k for r in results for k in r.agent_metrics}):
        vals = [r.agent_metrics[key] for r in results if key in r.agent_metrics]
        if vals:
            scores[key] = statistics.mean(vals)
    perf = aggregate_perf(results)
    for key in ("prompt_eval_tps_mean", "generation_tps_mean", "total_ms_p50"):
        if key in perf:
            scores[key] = perf[key]
    return scores


def run_eval(
    workspace: Workspace,
    corpus: Corpus,
    config: Config,
    golden_path: Path,
    run_id: str,
    created_at: str,
    *,
    name: str | None = None,
    loader=None,
    embedder=None,
    llm=None,
) -> RunRecord:
    db_path = indexer_mod.ensure_index(
        workspace, corpus, config, loader=loader, embedder=embedder
    )
    if embedder is None:
        embedder = components.build_embedder(config)
    store = SqliteVecStore(db_path, dimension=embedder.dimension)
    retriever = components.build_retriever(store, embedder, config)
    if llm is None:
        llm = components.build_llm(config)
    agent = None
    if config.agent.enabled:
        agent = components.build_agent(store, embedder, config)
        agent.llm = llm
    scorer = None
    if config.eval.deepeval:
        from rag_lab.eval.scorers.deepeval_scorer import DeepEvalScorer

        scorer = DeepEvalScorer(model=config.eval.deepeval_model or config.llm.model)
    runner = EvalRunner(
        retriever=retriever,
        llm=llm,
        k=config.retriever.k,
        deepeval_scorer=scorer,
        prompt_builder=PromptBuilder(system_instructions=config.prompt.system_instructions),
        agent=agent,
    )

    items = golden_set_mod.load_golden_set(golden_path)
    results = runner.run(items)
    scores = _aggregate_scores(results)

    run_dir = workspace.run_dir(run_id)
    if (run_dir / "run.json").exists():
        raise ValueError(f"run already exists: {run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yml").write_text(yaml.safe_dump(config.model_dump()))
    MarkdownReporter().write(
        results=results,
        config_summary=config_summary(config),
        out_path=run_dir / "report.md",
    )
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "name": name or run_id,
                "created_at": created_at,
                "corpus": corpus.label,
                "corpus_snapshot": corpus.to_dict(),
                "scores": scores,
                "config": config.model_dump(),
            },
            indent=2,
        )
    )
    write_run(
        run_dir / "items.json",
        results,
        config_summary=config_summary(config),
        prompt_version=artifact_prompt_version(config.prompt.system_instructions),
        k=config.retriever.k,
        created_at=created_at,
    )
    return RunRecord(
        run_id=run_id,
        name=name or run_id,
        created_at=created_at,
        corpus=corpus.label,
        scores=scores,
        config=config,
    )


def _record_from_json(data: dict) -> RunRecord:
    return RunRecord(
        run_id=data["run_id"],
        name=data["name"],
        created_at=data["created_at"],
        corpus=data["corpus"],
        scores=data["scores"],
        config=Config(**data["config"]),
    )


def list_runs(workspace: Workspace) -> list[RunRecord]:
    records: list[RunRecord] = []
    if not workspace.runs_dir.is_dir():
        return records
    for run_path in workspace.runs_dir.iterdir():
        run_json = run_path / "run.json"
        if not run_json.exists():
            continue
        try:
            records.append(_record_from_json(json.loads(run_json.read_text())))
        except (ValueError, KeyError):
            continue
    records.sort(key=lambda r: r.created_at, reverse=True)
    return records


def load_run(workspace: Workspace, run_id: str) -> RunRecord:
    run_json = workspace.run_dir(run_id) / "run.json"
    return _record_from_json(json.loads(run_json.read_text()))


def load_run_items(workspace: Workspace, run_id: str) -> list[dict]:
    items_path = workspace.run_dir(run_id) / "items.json"
    if not items_path.exists():
        return []
    return read_run(items_path)["items"]


def rename_run(workspace: Workspace, run_id: str, name: str) -> None:
    run_json = workspace.run_dir(run_id) / "run.json"
    data = json.loads(run_json.read_text())
    data["name"] = name
    run_json.write_text(json.dumps(data, indent=2))


def delete_run(workspace: Workspace, run_id: str) -> None:
    import shutil

    run_path = workspace.run_dir(run_id)
    if run_path.is_dir():
        shutil.rmtree(run_path)


def _flatten(config: Config) -> dict[str, object]:
    flat: dict[str, object] = {}
    for section, values in config.model_dump().items():
        for key, value in values.items():
            flat[f"{section}.{key}"] = value
    return flat


def diff(run_a: RunRecord, run_b: RunRecord) -> dict:
    flat_a = _flatten(run_a.config)
    flat_b = _flatten(run_b.config)
    sentinel = object()
    changed = [
        k for k in (set(flat_a) | set(flat_b)) if flat_a.get(k, sentinel) != flat_b.get(k, sentinel)
    ]
    metrics = set(run_a.scores) | set(run_b.scores)
    deltas = {
        m: run_b.scores.get(m, 0.0) - run_a.scores.get(m, 0.0) for m in metrics
    }
    return {"changed_knobs": sorted(changed), "metric_deltas": deltas}

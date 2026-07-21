"""Persistent store of eval runs: one directory per run, plus a pinned baseline.

Core (non-Studio) module: the CLI, Studio, and the MCP server all read and write
the same store. Callers pass the runs directory explicitly (Studio passes
``Workspace.runs_dir``; the CLI defaults to ``.rag-lab/runs``).
"""

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from rag_lab.config import Config, config_summary
from rag_lab.eval.aggregate import aggregate_perf, aggregate_repeats
from rag_lab.eval.reporter import MarkdownReporter
from rag_lab.eval.run_artifact import prompt_version, read_run, write_run
from rag_lab.eval.runner import EvalResult

BASELINE_FILE = "baseline.json"

_ITEM_METRIC_FIELDS = {
    "recall@k": "recall_at_k",
    "ndcg@k": "ndcg_at_k",
    "map": "average_precision",
    "mrr": "mrr",
    "keyword_coverage": "keyword_coverage",
    "citation_validity": "citation_validity",
}


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    created_at: str
    corpus: str
    scores: dict[str, float]
    config: Config
    scores_std: dict[str, float] = field(default_factory=dict)
    repeat: int = 1
    provenance: dict[str, str] = field(default_factory=dict)


def config_hash(config: Config) -> str:
    """Stable short hash of the full config, for run provenance."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def golden_hash(path: Path) -> str:
    """Short hash of the golden-set file contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def git_sha() -> str | None:
    """Current rag-lab checkout sha, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except OSError:
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def _aggregate_scores(
    repeats: list[list[EvalResult]],
) -> tuple[dict[str, float], dict[str, float]]:
    means, stds = aggregate_repeats(repeats)
    flat = [r for repeat in repeats for r in repeat]
    perf = aggregate_perf(flat)
    for key in ("prompt_eval_tps_mean", "generation_tps_mean", "total_ms_p50"):
        if key in perf:
            means[key] = perf[key]
    return means, stds


def save_run(
    runs_dir: Path,
    *,
    run_id: str,
    created_at: str,
    corpus: str,
    config: Config,
    repeats: list[list[EvalResult]],
    name: str | None = None,
    golden_hash: str | None = None,
    corpus_snapshot: dict | None = None,
) -> RunRecord:
    """Persist one eval run (run.json, config.yml, items.json, report.md)."""
    run_dir = runs_dir / run_id
    if (run_dir / "run.json").exists():
        raise ValueError(f"run already exists: {run_id}")
    run_dir.mkdir(parents=True, exist_ok=True)

    scores, scores_std = _aggregate_scores(repeats)
    provenance = {
        "config_hash": config_hash(config),
        "git_sha": git_sha(),
        "golden_hash": golden_hash,
        "corpus_hash": (
            hashlib.sha256(
                json.dumps(corpus_snapshot, sort_keys=True).encode("utf-8")
            ).hexdigest()[:12]
            if corpus_snapshot
            else None
        ),
    }
    provenance = {k: v for k, v in provenance.items() if v}

    summary = config_summary(config)
    flat = [r for repeat in repeats for r in repeat]
    (run_dir / "config.yml").write_text(yaml.safe_dump(config.model_dump()))
    MarkdownReporter().write(
        results=flat, config_summary=summary, out_path=run_dir / "report.md"
    )
    payload = {
        "run_id": run_id,
        "name": name or run_id,
        "created_at": created_at,
        "corpus": corpus,
        "scores": scores,
        "scores_std": scores_std,
        "repeat": len(repeats),
        "provenance": provenance,
        "config": config.model_dump(),
    }
    if corpus_snapshot is not None:
        payload["corpus_snapshot"] = corpus_snapshot
    (run_dir / "run.json").write_text(json.dumps(payload, indent=2))
    write_run(
        run_dir / "items.json",
        repeats=repeats,
        config_summary=summary,
        prompt_version=prompt_version(config.prompt.system_instructions),
        k=config.retriever.k,
        created_at=created_at,
    )
    return RunRecord(
        run_id=run_id,
        name=name or run_id,
        created_at=created_at,
        corpus=corpus,
        scores=scores,
        config=config,
        scores_std=scores_std,
        repeat=len(repeats),
        provenance=provenance,
    )


def _record_from_json(data: dict) -> RunRecord:
    return RunRecord(
        run_id=data["run_id"],
        name=data["name"],
        created_at=data["created_at"],
        corpus=data["corpus"],
        scores=data["scores"],
        config=Config(**data["config"]),
        scores_std=data.get("scores_std", {}),
        repeat=data.get("repeat", 1),
        provenance=data.get("provenance", {}),
    )


def list_runs(runs_dir: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    if not runs_dir.is_dir():
        return records
    for run_path in runs_dir.iterdir():
        run_json = run_path / "run.json"
        if not run_json.exists():
            continue
        try:
            records.append(_record_from_json(json.loads(run_json.read_text())))
        except (ValueError, KeyError):
            continue
    records.sort(key=lambda r: r.created_at, reverse=True)
    return records


def load_run(runs_dir: Path, run_id: str) -> RunRecord:
    run_json = runs_dir / run_id / "run.json"
    return _record_from_json(json.loads(run_json.read_text()))


def load_run_items(runs_dir: Path, run_id: str) -> list[dict]:
    items_path = runs_dir / run_id / "items.json"
    if not items_path.exists():
        return []
    return read_run(items_path)["items"]


def rename_run(runs_dir: Path, run_id: str, name: str) -> None:
    run_json = runs_dir / run_id / "run.json"
    data = json.loads(run_json.read_text())
    data["name"] = name
    run_json.write_text(json.dumps(data, indent=2))


def delete_run(runs_dir: Path, run_id: str) -> None:
    run_path = runs_dir / run_id
    if run_path.is_dir():
        shutil.rmtree(run_path)


def set_baseline(runs_dir: Path, run_id: str) -> None:
    """Pin a run as the baseline every future run is compared against."""
    if not (runs_dir / run_id / "run.json").exists():
        raise ValueError(f"unknown run: {run_id}")
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / BASELINE_FILE).write_text(json.dumps({"run_id": run_id}))


def get_baseline(runs_dir: Path) -> str | None:
    """The pinned baseline run id, or None if unset or pointing at a deleted run."""
    pin = runs_dir / BASELINE_FILE
    if not pin.exists():
        return None
    try:
        run_id = json.loads(pin.read_text())["run_id"]
    except (ValueError, KeyError):
        return None
    if not (runs_dir / run_id / "run.json").exists():
        return None
    return run_id


def _flatten(config: Config) -> dict[str, object]:
    flat: dict[str, object] = {}
    for section, values in config.model_dump().items():
        for key, value in values.items():
            flat[f"{section}.{key}"] = value
    return flat


def diff(run_a: RunRecord, run_b: RunRecord) -> dict:
    """Changed config knobs and per-metric deltas between two runs (b minus a)."""
    flat_a = _flatten(run_a.config)
    flat_b = _flatten(run_b.config)
    sentinel = object()
    changed = [
        k
        for k in (set(flat_a) | set(flat_b))
        if flat_a.get(k, sentinel) != flat_b.get(k, sentinel)
    ]
    metrics = set(run_a.scores) | set(run_b.scores)
    deltas = {m: run_b.scores.get(m, 0.0) - run_a.scores.get(m, 0.0) for m in metrics}
    return {"changed_knobs": sorted(changed), "metric_deltas": deltas}


def _item_metric(item: dict, metric: str) -> float | None:
    field_name = _ITEM_METRIC_FIELDS.get(metric)
    if field_name is not None:
        value = item.get(field_name)
        return value if isinstance(value, int | float) else None
    for bucket in ("deepeval_scores", "agent_metrics"):
        value = item.get(bucket, {}).get(metric)
        if isinstance(value, int | float):
            return value
    return None


def item_failures(
    items: list[dict], reference: dict[str, float], gates: dict[str, float]
) -> list[dict]:
    """Items where any gated metric falls below its reference value minus max_drop.

    Each returned item gains a ``failed_metrics`` list naming the offending gates.
    """
    failures: list[dict] = []
    for item in items:
        failed = [
            metric
            for metric, max_drop in gates.items()
            if metric in reference
            and (value := _item_metric(item, metric)) is not None
            and value < reference[metric] - max_drop
        ]
        if failed:
            failures.append({**item, "failed_metrics": failed})
    return failures

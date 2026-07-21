"""Tool implementations behind the eval MCP server.

Pure functions over the run store and eval service — the MCP transport layer in
``rag_lab.mcp_server`` registers these with FastMCP. Kept separate so they are
unit-testable without the optional ``mcp`` dependency installed.
"""

import re
from datetime import UTC, datetime
from pathlib import Path

from rag_lab import config as config_mod
from rag_lab.eval import golden_set as golden_set_mod
from rag_lab.eval import run_store, service
from rag_lab.eval.gates import gate_failures
from rag_lab.eval.golden_set import GoldenItem


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _record_summary(record: run_store.RunRecord) -> dict:
    summary = {
        "run_id": record.run_id,
        "name": record.name,
        "created_at": record.created_at,
        "repeat": record.repeat,
        "scores": record.scores,
    }
    if record.scores_std:
        summary["scores_std"] = record.scores_std
    return summary


def run_eval(
    *,
    config_path: Path,
    db_path: Path,
    golden_path: Path,
    runs_dir: Path,
    config_overrides: dict | None = None,
    repeat: int = 1,
) -> dict:
    """Run the golden set, persist the run, and report scores plus baseline deltas."""
    cfg = config_mod.load_config(config_path)
    if config_overrides:
        cfg = config_mod.Config(**_deep_merge(cfg.model_dump(), config_overrides))
    repeats = service.run_eval(cfg, db_path, golden_path, repeat=repeat)
    created_at = datetime.now(UTC).isoformat(timespec="seconds")
    record = run_store.save_run(
        runs_dir,
        run_id=run_store.new_run_id(runs_dir, created_at),
        created_at=created_at,
        corpus=str(db_path),
        config=cfg,
        repeats=repeats,
        golden_hash=run_store.golden_hash(golden_path),
    )
    result = _record_summary(record)
    baseline_id = run_store.get_baseline(runs_dir)
    if baseline_id:
        baseline = run_store.load_run(runs_dir, baseline_id)
        shared = set(record.scores) & set(baseline.scores)
        result["baseline"] = baseline_id
        result["baseline_deltas"] = {
            m: record.scores[m] - baseline.scores[m] for m in sorted(shared)
        }
        if cfg.eval.gates:
            result["gate_failures"] = gate_failures(
                record.scores, baseline.scores, cfg.eval.gates
            )
    return result


def list_runs(*, runs_dir: Path) -> list[dict]:
    """Run history, newest first, with the pinned baseline flagged."""
    baseline_id = run_store.get_baseline(runs_dir)
    return [
        _record_summary(r) | {"is_baseline": r.run_id == baseline_id}
        for r in run_store.list_runs(runs_dir)
    ]


def compare_runs(run_a: str, run_b: str, *, runs_dir: Path) -> dict:
    """Changed config knobs and metric deltas (b minus a) between two runs."""
    a = run_store.load_run(runs_dir, run_a)
    b = run_store.load_run(runs_dir, run_b)
    return run_store.diff(a, b)


def get_failures(run_id: str, *, runs_dir: Path) -> dict:
    """Questions in a run whose gated metrics fall below the baseline reference."""
    record = run_store.load_run(runs_dir, run_id)
    gates = record.config.eval.gates
    if not gates:
        return {"run_id": run_id, "failures": [], "note": "no eval.gates configured"}
    baseline_id = run_store.get_baseline(runs_dir)
    reference = (
        run_store.load_run(runs_dir, baseline_id).scores
        if baseline_id and baseline_id != run_id
        else record.scores
    )
    items = run_store.load_run_items(runs_dir, run_id)
    failures = run_store.item_failures(items, reference, gates)
    trimmed = [
        {
            "item_id": item["item_id"],
            "question": item.get("question", ""),
            "repeat": item.get("repeat", 0),
            "failed_metrics": item["failed_metrics"],
            "recall_at_k": item.get("recall_at_k"),
            "keyword_coverage": item.get("keyword_coverage"),
            "answer": item.get("actual_answer", ""),
        }
        for item in failures
    ]
    return {"run_id": run_id, "reference": reference, "failures": trimmed}


def add_golden_case(
    question: str,
    *,
    golden_path: Path,
    ideal_docs: list[str] | None = None,
    must_mention: list[str] | None = None,
    ideal_answer: str = "",
    expect_abstention: bool = False,
    case_id: str | None = None,
) -> dict:
    """Append a case to the golden set; the id is slugged from the question if omitted."""
    if case_id is None:
        slug = re.sub(r"[^a-z0-9]+", "-", question.lower()).strip("-")
        case_id = slug[:48] or "case"
    item = GoldenItem(
        id=case_id,
        question=question,
        ideal_docs=ideal_docs or [],
        must_mention=must_mention or [],
        ideal_answer=ideal_answer,
        expect_abstention=expect_abstention,
    )
    golden_set_mod.append_golden_case(golden_path, item)
    return {"id": case_id, "total_cases": len(golden_set_mod.load_golden_set(golden_path))}

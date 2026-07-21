import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from rag_lab.eval.aggregate import aggregate_perf, aggregate_repeats
from rag_lab.eval.runner import EvalResult

SCHEMA_VERSION = 2


def prompt_version(instructions: str) -> str:
    return hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:8]


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return value


def write_run(
    path: Path,
    results: list[EvalResult] | None = None,
    *,
    repeats: list[list[EvalResult]] | None = None,
    config_summary: str,
    prompt_version: str,
    k: int,
    created_at: str,
) -> None:
    """Write a run artifact; pass either a flat result list or per-repeat lists."""
    if repeats is None:
        repeats = [results or []]
    flat = [r for repeat in repeats for r in repeat]
    aggregates, aggregates_std = aggregate_repeats(repeats)
    items = [
        {**_json_safe(asdict(r)), "repeat": index}
        for index, repeat in enumerate(repeats)
        for r in repeat
    ]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "config_summary": config_summary,
        "prompt_version": prompt_version,
        "k": k,
        "repeat": len(repeats),
        "aggregates": _json_safe(aggregates),
        "perf": _json_safe(aggregate_perf(flat)),
        "items": items,
    }
    if aggregates_std:
        payload["aggregates_std"] = _json_safe(aggregates_std)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def read_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

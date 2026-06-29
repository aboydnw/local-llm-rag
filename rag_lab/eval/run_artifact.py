import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from rag_lab.eval.aggregate import aggregate_metrics
from rag_lab.eval.runner import EvalResult

SCHEMA_VERSION = 1


def prompt_version(instructions: str) -> str:
    return hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:8]


def write_run(
    path: Path,
    results: list[EvalResult],
    *,
    config_summary: str,
    prompt_version: str,
    k: int,
    created_at: str,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "config_summary": config_summary,
        "prompt_version": prompt_version,
        "k": k,
        "aggregates": aggregate_metrics(results),
        "items": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

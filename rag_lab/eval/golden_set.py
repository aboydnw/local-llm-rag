from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    id: str
    question: str
    ideal_docs: list[str] = Field(default_factory=list)
    must_mention: list[str] = Field(default_factory=list)
    ideal_answer: str = ""
    expect_abstention: bool = False


def append_golden_case(path: Path, item: GoldenItem) -> None:
    """Append one case to the golden-set YAML, validating ids stay unique."""
    existing = load_golden_set(path) if path.exists() else []
    if any(other.id == item.id for other in existing):
        raise ValueError(f"duplicate golden id: {item.id}")
    entries = [i.model_dump(exclude_defaults=True) | {"id": i.id, "question": i.question}
               for i in [*existing, item]]
    path.write_text(yaml.safe_dump(entries, sort_keys=False, allow_unicode=True))
    load_golden_set(path)


def load_golden_set(path: Path) -> list[GoldenItem]:
    raw = yaml.safe_load(path.read_text()) or []
    if not isinstance(raw, list):
        raise ValueError("golden set must be a YAML list of items")
    items = [GoldenItem(**entry) for entry in raw]
    seen: set[str] = set()
    for item in items:
        if item.id in seen:
            raise ValueError(f"duplicate golden id: {item.id}")
        seen.add(item.id)
    return items

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class GoldenItem(BaseModel):
    id: str
    question: str
    ideal_docs: list[str] = Field(default_factory=list)
    must_mention: list[str] = Field(default_factory=list)
    ideal_answer: str = ""


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

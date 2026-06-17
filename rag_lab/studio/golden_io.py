from pathlib import Path

import yaml

from rag_lab.eval import golden_set as golden_set_mod
from rag_lab.eval.golden_set import GoldenItem


def load_items(path: Path) -> list[GoldenItem]:
    if not path.exists():
        return []
    return golden_set_mod.load_golden_set(path)


def save_items(path: Path, items: list[GoldenItem]) -> None:
    data = [item.model_dump() for item in items]
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def upsert_item(items: list[GoldenItem], item: GoldenItem) -> list[GoldenItem]:
    out = [i for i in items if i.id != item.id]
    out.append(item)
    return out


def delete_item(items: list[GoldenItem], item_id: str) -> list[GoldenItem]:
    return [i for i in items if i.id != item_id]

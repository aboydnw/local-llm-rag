import os
import tempfile
from pathlib import Path

import yaml

from rag_lab.eval import golden_set as golden_set_mod
from rag_lab.eval.golden_set import GoldenItem


def load_items(path: Path) -> list[GoldenItem]:
    """Load golden items from ``path``, returning an empty list if it is absent."""
    if not path.exists():
        return []
    return golden_set_mod.load_golden_set(path)


def save_items(path: Path, items: list[GoldenItem]) -> None:
    """Atomically write ``items`` to ``path`` as YAML."""
    data = [item.model_dump() for item in items]
    payload = yaml.safe_dump(data, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def upsert_item(items: list[GoldenItem], item: GoldenItem) -> list[GoldenItem]:
    """Return ``items`` with ``item`` replacing any entry sharing its id, else appended."""
    out = [i for i in items if i.id != item.id]
    out.append(item)
    return out


def delete_item(items: list[GoldenItem], item_id: str) -> list[GoldenItem]:
    """Return ``items`` without the entry matching ``item_id``."""
    return [i for i in items if i.id != item_id]

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_lab.eval.golden_set import GoldenItem, load_golden_set


def test_load_golden_set_parses_yaml(tmp_path: Path) -> None:
    path = tmp_path / "golden.yml"
    path.write_text(
        """
- id: titiler-factory
  question: "How do I add a custom raster source to titiler?"
  ideal_docs: ["titiler/docs/sources.md", "titiler/README.md"]
  must_mention: ["factory", "MosaicTilerFactory"]
  ideal_answer: |
    titiler exposes factory classes that let you mount endpoints.
- id: minimal
  question: "What is eoAPI?"
  ideal_docs: []
  must_mention: []
  ideal_answer: ""
"""
    )
    items = load_golden_set(path)
    assert len(items) == 2
    first = items[0]
    assert isinstance(first, GoldenItem)
    assert first.id == "titiler-factory"
    assert first.must_mention == ["factory", "MosaicTilerFactory"]


def test_load_rejects_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.yml"
    path.write_text("- id: only-id\n")
    with pytest.raises(ValidationError):
        load_golden_set(path)


def test_load_rejects_non_list_root(tmp_path: Path) -> None:
    path = tmp_path / "mapping.yml"
    path.write_text("id: x\nquestion: q\n")
    with pytest.raises(ValueError):
        load_golden_set(path)


def test_load_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "dup.yml"
    path.write_text(
        "- id: x\n  question: q\n  ideal_docs: []\n  must_mention: []\n  ideal_answer: ''\n"
        "- id: x\n  question: q\n  ideal_docs: []\n  must_mention: []\n  ideal_answer: ''\n"
    )
    with pytest.raises(ValueError):
        load_golden_set(path)

import pytest

from rag_lab.studio.workspace import Workspace


def test_initialize_creates_layout_and_gitignore(tmp_path):
    ws = Workspace(tmp_path / ".rag-lab")
    ws.initialize()
    assert ws.indexes_dir.is_dir()
    assert ws.runs_dir.is_dir()
    gitignore = (tmp_path / ".rag-lab" / ".gitignore")
    assert gitignore.read_text().strip() == "*"


def test_initialize_is_idempotent(tmp_path):
    ws = Workspace(tmp_path / ".rag-lab")
    ws.initialize()
    ws.initialize()
    assert ws.runs_dir.is_dir()


def test_path_helpers(tmp_path):
    ws = Workspace(tmp_path / ".rag-lab")
    assert ws.index_db("abc").name == "abc.db"
    assert ws.index_meta("abc").name == "abc.json"
    assert ws.run_dir("r1").parent == ws.runs_dir


@pytest.mark.parametrize("bad", ["..", "../escape", "a/b", "/etc/passwd", ""])
def test_path_helpers_reject_traversal(tmp_path, bad):
    ws = Workspace(tmp_path / ".rag-lab")
    with pytest.raises(ValueError):
        ws.run_dir(bad)
    with pytest.raises(ValueError):
        ws.index_db(bad)

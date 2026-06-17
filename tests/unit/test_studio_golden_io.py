from rag_lab.eval.golden_set import GoldenItem
from rag_lab.studio import golden_io


def test_save_then_load_roundtrips(tmp_path):
    path = tmp_path / "golden.yml"
    items = [GoldenItem(id="q1", question="What is X?", must_mention=["x"])]
    golden_io.save_items(path, items)
    loaded = golden_io.load_items(path)
    assert loaded[0].id == "q1"
    assert loaded[0].must_mention == ["x"]


def test_upsert_replaces_by_id():
    items = [GoldenItem(id="q1", question="old")]
    updated = golden_io.upsert_item(items, GoldenItem(id="q1", question="new"))
    assert len(updated) == 1
    assert updated[0].question == "new"


def test_upsert_appends_new_id():
    items = [GoldenItem(id="q1", question="a")]
    updated = golden_io.upsert_item(items, GoldenItem(id="q2", question="b"))
    assert [i.id for i in updated] == ["q1", "q2"]


def test_delete_item():
    items = [GoldenItem(id="q1", question="a"), GoldenItem(id="q2", question="b")]
    assert [i.id for i in golden_io.delete_item(items, "q1")] == ["q2"]

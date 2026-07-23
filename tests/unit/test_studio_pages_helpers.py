from rag_lab.studio.pages.runs import _ago


def test_ago_accepts_legacy_timezone_naive_timestamp():
    result = _ago("2020-01-01T00:00:00")
    assert result.endswith("d ago")


def test_ago_preserves_invalid_timestamp():
    assert _ago("not-a-timestamp") == "not-a-timestamp"

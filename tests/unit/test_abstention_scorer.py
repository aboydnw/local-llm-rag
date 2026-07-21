from rag_lab.eval.scorers.abstention import detected_abstention

_MARKERS = ["i don't know", "does not contain"]


def test_detects_abstention_marker_case_insensitively() -> None:
    assert detected_abstention("I Don't Know the answer.", _MARKERS) is True


def test_no_abstention_when_no_marker_present() -> None:
    assert detected_abstention("TiTiler is a tile server.", _MARKERS) is False


def test_blank_marker_does_not_match_every_answer() -> None:
    assert detected_abstention("TiTiler is a tile server.", ["", "   "]) is False

from rag_lab.eval.scorers.keyword import keyword_coverage


def test_keyword_coverage_returns_fraction_present() -> None:
    answer = "titiler uses MosaicTilerFactory and other factory classes."
    assert keyword_coverage(answer, must_mention=["factory", "MosaicTilerFactory"]) == 1.0
    assert keyword_coverage(answer, must_mention=["factory", "Lambda"]) == 0.5
    assert keyword_coverage(answer, must_mention=["Lambda"]) == 0.0


def test_keyword_coverage_is_case_insensitive() -> None:
    answer = "Factory pattern."
    assert keyword_coverage(answer, must_mention=["factory"]) == 1.0


def test_keyword_coverage_empty_must_mention_returns_1() -> None:
    assert keyword_coverage("anything", must_mention=[]) == 1.0

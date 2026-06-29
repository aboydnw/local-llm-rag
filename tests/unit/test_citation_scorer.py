from rag_lab.eval.scorers.citation import citation_validity, parse_citations


def test_parse_citations_extracts_single_and_grouped_markers() -> None:
    assert parse_citations("see [1] and [2, 3] here") == [1, 2, 3]


def test_parse_citations_ignores_markdown_links() -> None:
    assert parse_citations("a [link](http://x) and [2]") == [2]


def test_parse_citations_handles_whitespace_separated_numbers() -> None:
    assert parse_citations("see [1 2] and [3]") == [1, 2, 3]


def test_citation_validity_is_fraction_pointing_to_real_sources() -> None:
    assert citation_validity("from [1] and [5]", num_sources=3) == 0.5


def test_citation_validity_is_none_when_no_citations() -> None:
    assert citation_validity("no citations here", num_sources=3) is None

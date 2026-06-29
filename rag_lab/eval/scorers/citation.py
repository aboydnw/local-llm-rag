import re

_CITATION_RE = re.compile(r"\[([0-9][0-9,\s]*)\]")


def parse_citations(answer: str) -> list[int]:
    """Extract citation numbers from ``[n]`` / ``[n, m]`` markers, in order."""
    numbers: list[int] = []
    for group in _CITATION_RE.findall(answer):
        numbers.extend(int(token) for token in re.findall(r"\d+", group))
    return numbers


def citation_validity(answer: str, num_sources: int) -> float | None:
    """Fraction of cited markers that reference a retrieved source (1..num_sources)."""
    citations = parse_citations(answer)
    if not citations:
        return None
    valid = sum(1 for n in citations if 1 <= n <= num_sources)
    return valid / len(citations)

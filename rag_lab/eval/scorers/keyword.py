def keyword_coverage(answer: str, must_mention: list[str]) -> float:
    if not must_mention:
        return 1.0
    lower = answer.lower()
    hits = sum(1 for term in must_mention if term.lower() in lower)
    return hits / len(must_mention)

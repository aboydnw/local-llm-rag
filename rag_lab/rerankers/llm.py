import re

from rag_lab.retrievers.base import RetrievalResult

_PROMPT = """You are ranking search results for relevance to a question.

Question: {query}

Candidates:
{candidates}

Reply with the numbers of the {k} most relevant candidates, most relevant first,
as a comma-separated list (e.g. "3, 1, 0"). Numbers only."""


def parse_rank_line(text: str, n: int) -> list[int]:
    """Extract de-duplicated 0-based indices within [0, n) from the LLM reply."""
    out: list[int] = []
    for token in re.findall(r"\d+", text):
        i = int(token)
        if 0 <= i < n and i not in out:
            out.append(i)
    return out


class LLMReranker:
    """Reorders retrieval results by asking an LLM to pick the most relevant."""

    def __init__(self, llm) -> None:
        self.llm = llm

    def rerank(self, query: str, results: list[RetrievalResult], k: int) -> list[RetrievalResult]:
        if not results:
            return []
        candidates = "\n".join(
            f"{i}. {r.chunk.text[:400]}" for i, r in enumerate(results)
        )
        reply = self.llm.generate(_PROMPT.format(query=query, candidates=candidates, k=k))
        order = parse_rank_line(reply, len(results))
        if not order:
            return results[:k]
        ranked = [results[i] for i in order]
        remaining = [r for j, r in enumerate(results) if j not in order]
        return (ranked + remaining)[:k]

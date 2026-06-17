from rag_lab.retrievers.base import RetrievalResult

SYSTEM_INSTRUCTIONS = """\
You are a documentation assistant. Answer the user's question using ONLY the numbered \
context excerpts below. Cite the excerpts you use with their numbers in square brackets, \
like [1] or [2, 3]. If the context does not contain the answer, say so honestly — do not \
invent facts. Be concise.
"""


class PromptBuilder:
    def build(self, question: str, results: list[RetrievalResult]) -> str:
        parts = [SYSTEM_INSTRUCTIONS, "", "Context excerpts:"]
        for i, result in enumerate(results, start=1):
            heading = " > ".join(result.chunk.heading_path) or "(no heading)"
            parts.append(f"[{i}] (source: {result.chunk.doc_path} — {heading})")
            parts.append(result.chunk.text.strip())
            parts.append("")
        parts.append(f"Question: {question}")
        parts.append("Answer:")
        return "\n".join(parts)

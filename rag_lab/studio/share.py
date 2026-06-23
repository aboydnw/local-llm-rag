def format_run_markdown(question: str, answer: str, results) -> str:
    """Render a playground question, answer, and retrieved chunks as markdown."""
    lines = ["## Question", "", question, "", "## Answer", "", answer, "", "## Retrieved chunks", ""]
    for i, r in enumerate(results, start=1):
        heading = " > ".join(r.chunk.heading_path) or "(no heading)"
        lines.append(f"[{i}] {r.chunk.doc_path} — {heading} (score {r.score:.3f})")
        lines.append("")
        lines.append(r.chunk.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

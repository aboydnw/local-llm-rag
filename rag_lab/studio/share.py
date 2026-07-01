def format_run_markdown(question: str, answer: str, results) -> str:
    """Render a playground question, answer, and retrieved chunks as markdown."""
    lines = [
        "## Question", "", question, "",
        "## Answer", "", answer, "",
        "## Retrieved chunks", "",
    ]
    for i, r in enumerate(results, start=1):
        heading = " > ".join(r.chunk.heading_path) or "(no heading)"
        lines.append(f"[{i}] {r.chunk.doc_path} — {heading} (score {r.score:.3f})")
        lines.append("")
        lines.append(r.chunk.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_agent_run_markdown(question, answer, steps, tools_used, final_context) -> str:
    """Render an agent playground run — question, answer, trace, final context — as markdown."""
    lines = [
        "## Question", "", question, "",
        "## Answer", "", answer, "",
        "## Tools used", "", ", ".join(tools_used) or "(none)", "",
        "## Trace", "",
    ]
    for i, step in enumerate(steps, start=1):
        if step.action is None:
            lines.append(f"{i}. **Ready to answer** — {step.thought}")
            continue
        lines.append(f"{i}. **{step.action}**: {step.action_input}")
        if step.observation:
            lines.append(f"   - {step.observation}")
    lines += ["", "## Final context", ""]
    for i, chunk in enumerate(final_context, start=1):
        heading = " > ".join(chunk.heading_path) or "(no heading)"
        lines.append(f"[{i}] {chunk.doc_path} — {heading}")
    return "\n".join(lines).rstrip() + "\n"

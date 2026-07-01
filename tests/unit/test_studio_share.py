from pathlib import Path

from rag_lab.retrievers.base import RetrievalResult
from rag_lab.studio.share import format_run_markdown
from rag_lab.types import Chunk


def _result(heading, text, score):
    chunk = Chunk(text=text, doc_path=Path("docs/a.md"), heading_path=heading, position=0)
    return RetrievalResult(chunk=chunk, score=score, source="vector")


def test_includes_question_and_answer():
    md = format_run_markdown("what is titiler?", "A tile server.", [])
    assert "## Question" in md
    assert "what is titiler?" in md
    assert "## Answer" in md
    assert "A tile server." in md


def test_renders_each_chunk_with_heading_and_score():
    results = [_result(("User guide", "Install"), "pip install titiler", 0.016)]
    md = format_run_markdown("q", "a", results)
    assert "[1] docs/a.md — User guide > Install (score 0.016)" in md
    assert "pip install titiler" in md


def test_heading_falls_back_when_empty():
    results = [_result((), "body text", 0.5)]
    md = format_run_markdown("q", "a", results)
    assert "[1] docs/a.md — (no heading) (score 0.500)" in md


def test_agent_markdown_includes_question_answer_and_tools():
    from rag_lab.agent.agent import AgentStep
    from rag_lab.studio.share import format_agent_run_markdown

    steps = [
        AgentStep(
            thought="look it up", action="vector_search",
            action_input="titiler", observation="found the docs",
        )
    ]
    md = format_agent_run_markdown(
        "what is titiler?", "A tile server.", steps,
        ("vector_search",), [],
    )
    assert "## Question" in md
    assert "what is titiler?" in md
    assert "## Answer" in md
    assert "A tile server." in md
    assert "vector_search" in md
    assert "found the docs" in md


def test_agent_markdown_lists_final_context():
    from rag_lab.studio.share import format_agent_run_markdown

    chunk = Chunk(
        text="pip install titiler", doc_path=Path("docs/a.md"),
        heading_path=("Install",), position=0,
    )
    md = format_agent_run_markdown("q", "a", [], (), [chunk])
    assert "docs/a.md — Install" in md

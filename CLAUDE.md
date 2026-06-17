# rag-lab

Local-first RAG CLI over markdown docs. Fully local via Ollama, no API keys. Pitch: "RAG that doesn't need OpenAI."

See [PRD.md](PRD.md) for the product. Three loops: **ingest** (built), **ask** (query), **eval**.

- Python 3.11+, managed with `uv`. CLI is `typer`-based, config is `pydantic` from `rag.yml`.
- Every component (loader, chunker, embedder, retriever, llm) is swappable behind a clean interface.
- Tests: `pytest`. Run with `uv run pytest`.

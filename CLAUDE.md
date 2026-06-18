# rag-lab

Local-first RAG CLI over markdown docs. Fully local via Ollama, no API keys. Pitch: "RAG that doesn't need OpenAI."

See [PRD.md](PRD.md) for the product. Three loops: **ingest**, **ask** (query), **eval** — all built.

- Python 3.11+, managed with `uv`. CLI is `typer`-based, config is `pydantic` from `rag.yml`.
- Every component (loader, chunker, embedder, retriever, llm) is swappable behind a clean interface.
- Tests: `pytest`. Run with `uv run pytest`.
- **Studio** (`rag-lab studio`): a local Streamlit cockpit for turning every knob, editing the
  golden set, and comparing eval runs. Code in `rag_lab/studio/` — pure-Python orchestration
  (`workspace`, `indexer`, `experiments`, `components`, `golden_io`) is unit-tested; the
  Streamlit `app.py`/`sidebar.py`/`pages/` are a thin render layer and are not unit-tested.
  Streamlit is an optional extra (`uv sync --extra studio`); never import it from modules the
  core CLI loads at startup. Studio state lives in a gitignored `.rag-lab/` dir.

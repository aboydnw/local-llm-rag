from pathlib import Path

import typer

from rag_lab import __version__
from rag_lab import config as config_mod
from rag_lab import ingest as ingest_mod
from rag_lab.chunkers.markdown_aware import MarkdownAwareChunker
from rag_lab.embedders.ollama import OllamaEmbedder
from rag_lab.eval import golden_set as golden_set_mod
from rag_lab.eval.reporter import MarkdownReporter
from rag_lab.eval.runner import EvalRunner
from rag_lab.llms.ollama import OllamaLLM
from rag_lab.loaders.markdown import MarkdownLoader
from rag_lab.prompts import PromptBuilder
from rag_lab.retrievers.bm25 import BM25Retriever
from rag_lab.retrievers.hybrid import HybridRetriever
from rag_lab.retrievers.vector import VectorRetriever
from rag_lab.store.sqlite_vec import SqliteVecStore

app = typer.Typer(no_args_is_help=True, help="rag-lab: local-first RAG framework")
config_app = typer.Typer(no_args_is_help=True, help="Manage rag-lab config files.")
app.add_typer(config_app, name="config")
inspect_app = typer.Typer(no_args_is_help=True, help="Inspect what rag-lab indexed.")
app.add_typer(inspect_app, name="inspect")


@app.command()
def version() -> None:
    """Print the rag-lab version."""
    typer.echo(__version__)


@config_app.command("init")
def config_init(
    path: Path = typer.Option(Path("rag.yml"), "--path"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Write a default rag.yml to the given path."""
    if path.exists() and not force:
        typer.echo(f"{path} already exists. Use --force to overwrite.", err=True)
        raise typer.Exit(code=1)
    config_mod.write_default_config(path)
    typer.echo(f"Wrote default config to {path}")


@app.command()
def ingest(
    source: Path | None = typer.Argument(None, help="Local directory of markdown files."),
    repo: str | None = typer.Option(
        None, "--repo", help="GitHub owner/name. Clones to a temp dir."
    ),
    config: Path = typer.Option(Path("rag.yml"), "--config"),
    db: Path = typer.Option(Path("rag.db"), "--db"),
) -> None:
    """Ingest markdown into a local sqlite-vec index. Provide a local dir or --repo."""
    import tempfile

    from rag_lab.loaders.github import GitHubLoader

    if (source is None) == (repo is None):
        typer.echo("Provide exactly one of <source> or --repo.", err=True)
        raise typer.Exit(code=1)
    if not config.exists():
        typer.echo(f"Config file not found: {config}. Run `rag-lab config init`.", err=True)
        raise typer.Exit(code=1)

    cfg = config_mod.load_config(config)
    try:
        dimension = config_mod.embedding_dimension(cfg.embedder.model)
    except ValueError as exc:
        typer.echo(str(exc) + " — add it to EMBEDDING_DIMENSIONS in config.py.", err=True)
        raise typer.Exit(code=1) from exc

    loader: object
    workdir: tempfile.TemporaryDirectory | None = None
    try:
        if repo is not None:
            workdir = tempfile.TemporaryDirectory(prefix="rag-lab-repo-")
            loader = GitHubLoader(repo, clone_into=Path(workdir.name) / "repo")
            typer.echo(f"Cloning {repo}...")
        else:
            assert source is not None
            if not source.exists() or not source.is_dir():
                typer.echo(f"Source must be an existing directory: {source}", err=True)
                raise typer.Exit(code=1)
            loader = MarkdownLoader(source)

        chunker = MarkdownAwareChunker(
            max_tokens=cfg.chunker.max_tokens, overlap=cfg.chunker.overlap
        )
        embedder = OllamaEmbedder(model=cfg.embedder.model, dimension=dimension)
        store = SqliteVecStore(db, dimension=dimension)

        typer.echo(f"Ingesting into {db}...")
        count = ingest_mod.run(
            loader=loader, chunker=chunker, embedder=embedder, store=store
        )
        typer.echo(f"Done. {count} chunks indexed.")
    finally:
        if workdir is not None:
            workdir.cleanup()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question."),
    config: Path = typer.Option(Path("rag.yml"), "--config"),
    db: Path = typer.Option(Path("rag.db"), "--db"),
    k: int = typer.Option(None, "--k", min=1, help="Override retriever k from config."),
    show_chunks: bool = typer.Option(
        False, "--show-chunks", help="Print retrieved chunks before the answer."
    ),
) -> None:
    """Ask a question against an ingested corpus."""
    if not db.exists():
        typer.echo(
            f"Index file not found: {db}. Run `rag-lab ingest <source>` first.", err=True
        )
        raise typer.Exit(code=1)
    if not config.exists():
        typer.echo(f"Config file not found: {config}. Run `rag-lab config init`.", err=True)
        raise typer.Exit(code=1)

    cfg = config_mod.load_config(config)
    try:
        dimension = config_mod.embedding_dimension(cfg.embedder.model)
    except ValueError as exc:
        typer.echo(str(exc) + " — add it to EMBEDDING_DIMENSIONS in config.py.", err=True)
        raise typer.Exit(code=1) from exc
    store = SqliteVecStore(db, dimension=dimension)
    embedder = OllamaEmbedder(model=cfg.embedder.model, dimension=dimension)
    retriever = HybridRetriever(
        vector=VectorRetriever(store=store, embedder=embedder),
        bm25=BM25Retriever(store=store),
        vector_weight=cfg.retriever.vector_weight,
        bm25_weight=cfg.retriever.bm25_weight,
    )
    llm = OllamaLLM(model=cfg.llm.model)
    builder = PromptBuilder()

    top_k = k or cfg.retriever.k
    results = retriever.retrieve(question, k=top_k)
    if show_chunks:
        typer.echo("--- Retrieved chunks ---")
        for i, r in enumerate(results, start=1):
            typer.echo(
                f"[{i}] {r.chunk.doc_path} — {' > '.join(r.chunk.heading_path)} "
                f"(score {r.score:.3f})"
            )
            typer.echo(r.chunk.text[:200] + ("..." if len(r.chunk.text) > 200 else ""))
            typer.echo("")

    prompt = builder.build(question=question, results=results)
    answer = llm.generate(prompt)

    typer.echo(answer)
    typer.echo("")
    typer.echo("--- Sources ---")
    for i, r in enumerate(results, start=1):
        typer.echo(f"[{i}] {r.chunk.doc_path} — {' > '.join(r.chunk.heading_path)}")


@inspect_app.command("chunks")
def inspect_chunks(
    doc: Path | None = typer.Option(None, "--doc", help="Filter to a single doc path."),
    db: Path = typer.Option(Path("rag.db"), "--db"),
    limit: int = typer.Option(20, "--limit", min=1),
) -> None:
    """Dump chunks (optionally filtered to one doc) for debugging."""
    import sqlite3
    from contextlib import closing

    if not db.exists():
        typer.echo(f"Index file not found: {db}.", err=True)
        raise typer.Exit(code=1)
    with closing(sqlite3.connect(db)) as conn:
        if doc is not None:
            rows = conn.execute(
                "SELECT doc_path, position, heading_path, substr(text, 1, 200) "
                "FROM chunks WHERE doc_path = ? ORDER BY position LIMIT ?",
                (str(doc), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT doc_path, position, heading_path, substr(text, 1, 200) "
                "FROM chunks ORDER BY doc_path, position LIMIT ?",
                (limit,),
            ).fetchall()
    for path, pos, heading, snippet in rows:
        typer.echo(f"{path} [{pos}] {heading}")
        typer.echo(f"  {snippet}")
        typer.echo("")


@app.command()
def eval(  # noqa: A001
    golden: Path = typer.Option(Path("golden.yml"), "--golden", help="Golden set YAML."),
    config: Path = typer.Option(Path("rag.yml"), "--config"),
    db: Path = typer.Option(Path("rag.db"), "--db"),
    report: Path = typer.Option(
        Path("eval-reports") / "report.md", "--report", help="Output path for the report."
    ),
    previous: Path | None = typer.Option(
        None, "--previous", help="Previous report to diff against."
    ),
) -> None:
    """Run the eval harness against a golden set and write a markdown report."""
    if not golden.exists():
        typer.echo(f"Golden set not found: {golden}", err=True)
        raise typer.Exit(code=1)
    if not config.exists():
        typer.echo(f"Config file not found: {config}. Run `rag-lab config init`.", err=True)
        raise typer.Exit(code=1)
    if not db.exists():
        typer.echo(f"Index not found: {db}. Run `rag-lab ingest` first.", err=True)
        raise typer.Exit(code=1)

    cfg = config_mod.load_config(config)
    try:
        dimension = config_mod.embedding_dimension(cfg.embedder.model)
    except ValueError as exc:
        typer.echo(str(exc) + " — add it to EMBEDDING_DIMENSIONS in config.py.", err=True)
        raise typer.Exit(code=1) from exc

    store = SqliteVecStore(db, dimension=dimension)
    embedder = OllamaEmbedder(model=cfg.embedder.model, dimension=dimension)
    retriever = HybridRetriever(
        vector=VectorRetriever(store=store, embedder=embedder),
        bm25=BM25Retriever(store=store),
        vector_weight=cfg.retriever.vector_weight,
        bm25_weight=cfg.retriever.bm25_weight,
    )
    llm = OllamaLLM(model=cfg.llm.model)

    scorer = None
    if cfg.eval.deepeval:
        from rag_lab.eval.scorers.deepeval_scorer import DeepEvalScorer

        scorer = DeepEvalScorer(model=cfg.llm.model)
    runner = EvalRunner(
        retriever=retriever, llm=llm, k=cfg.retriever.k, deepeval_scorer=scorer,
    )

    items = golden_set_mod.load_golden_set(golden)
    typer.echo(f"Running {len(items)} questions...")
    results = runner.run(items)

    summary = config_mod.config_summary(cfg)
    report.parent.mkdir(parents=True, exist_ok=True)
    MarkdownReporter().write(
        results=results,
        config_summary=summary,
        out_path=report,
        previous_report=previous,
    )
    typer.echo(f"Report written to {report}")


@app.command()
def studio(
    port: int = typer.Option(
        8501, "--port", min=1, max=65535, help="Port for the Streamlit server."
    ),
) -> None:
    """Launch the rag-lab studio web interface (requires the 'studio' extra)."""
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "studio" / "app.py"
    code = subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    )
    raise typer.Exit(code=code)


if __name__ == "__main__":
    app()

from pathlib import Path

import typer

from rag_lab import __version__
from rag_lab import config as config_mod
from rag_lab import ingest as ingest_mod
from rag_lab.chunkers.markdown_aware import MarkdownAwareChunker
from rag_lab.embedders.ollama import OllamaEmbedder
from rag_lab.loaders.markdown import MarkdownLoader
from rag_lab.store.sqlite_vec import SqliteVecStore

app = typer.Typer(no_args_is_help=True, help="rag-lab: local-first RAG framework")
config_app = typer.Typer(no_args_is_help=True, help="Manage rag-lab config files.")
app.add_typer(config_app, name="config")

DEFAULT_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}


@app.command()
def version() -> None:
    """Print the rag-lab version."""
    typer.echo(__version__)


@config_app.command("init")
def config_init(
    path: Path = typer.Option(Path("rag.yml"), "--path", help="Where to write the config."),
    force: bool = typer.Option(False, "--force", help="Overwrite if file exists."),
) -> None:
    """Write a default rag.yml to the given path."""
    if path.exists() and not force:
        typer.echo(f"{path} already exists. Use --force to overwrite.", err=True)
        raise typer.Exit(code=1)
    config_mod.write_default_config(path)
    typer.echo(f"Wrote default config to {path}")


@app.command()
def ingest(
    source: Path = typer.Argument(..., help="Directory of markdown files to ingest."),
    config: Path = typer.Option(Path("rag.yml"), "--config", help="Path to rag.yml."),
    db: Path = typer.Option(Path("rag.db"), "--db", help="Path to write the index."),
) -> None:
    """Ingest markdown documents into a local sqlite-vec index."""
    if not source.exists() or not source.is_dir():
        typer.echo(f"Source must be an existing directory: {source}", err=True)
        raise typer.Exit(code=1)
    if not config.exists():
        typer.echo(f"Config file not found: {config}. Run `rag-lab config init`.", err=True)
        raise typer.Exit(code=1)

    cfg = config_mod.load_config(config)
    dimension = DEFAULT_EMBEDDING_DIMENSIONS.get(cfg.embedder.model)
    if dimension is None:
        typer.echo(
            f"Unknown embedding model '{cfg.embedder.model}'. "
            f"Add it to DEFAULT_EMBEDDING_DIMENSIONS in cli.py.",
            err=True,
        )
        raise typer.Exit(code=1)

    loader = MarkdownLoader(source)
    chunker = MarkdownAwareChunker(
        max_tokens=cfg.chunker.max_tokens, overlap=cfg.chunker.overlap
    )
    embedder = OllamaEmbedder(model=cfg.embedder.model, dimension=dimension)
    store = SqliteVecStore(db, dimension=dimension)

    typer.echo(f"Ingesting {source} into {db} (embedder: {cfg.embedder.model})...")
    count = ingest_mod.run(loader=loader, chunker=chunker, embedder=embedder, store=store)
    typer.echo(f"Done. {count} chunks indexed.")


if __name__ == "__main__":
    app()

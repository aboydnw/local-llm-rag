import typer

from rag_lab import __version__

app = typer.Typer(no_args_is_help=True, help="rag-lab: local-first RAG framework")


@app.callback()
def _main() -> None:
    pass


@app.command()
def version() -> None:
    """Print the rag-lab version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()

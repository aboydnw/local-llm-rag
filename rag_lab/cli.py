from pathlib import Path

import typer

from rag_lab import __version__
from rag_lab import config as config_mod

app = typer.Typer(no_args_is_help=True, help="rag-lab: local-first RAG framework")
config_app = typer.Typer(no_args_is_help=True, help="Manage rag-lab config files.")
app.add_typer(config_app, name="config")


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


if __name__ == "__main__":
    app()

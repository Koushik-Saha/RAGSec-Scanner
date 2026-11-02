"""Typer CLI entry point (`ragsec`). Thin wrapper over the ragsec library."""

from pathlib import Path
from typing import Annotated

import typer

from ragsec.config import load_settings
from ragsec.index import Index, index_directory

app = typer.Typer(
    help="ragsec: security testing for retrieval-augmented generation (RAG) systems.",
    no_args_is_help=True,
)


def _not_implemented() -> None:
    typer.echo("not implemented yet")


@app.command()
def ingest(
    directory: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            help="Folder of documents. Each subfolder name becomes a corpus name.",
        ),
    ],
) -> None:
    """Ingest documents into the vector index."""
    settings = load_settings()
    counts = index_directory(directory, Index(settings.chroma_dir))
    if not counts:
        typer.echo(f"No supported documents (.md, .txt, .pdf) found in {directory}")
        raise typer.Exit(code=1)

    typer.echo(f"Indexed {sum(counts.values())} chunks into {settings.chroma_dir}")
    for corpus, count in counts.items():
        typer.echo(f"  {corpus}: {count}")


@app.command()
def ask() -> None:
    """Ask a question and get a cited answer."""
    _not_implemented()


@app.command()
def scan() -> None:
    """Run the security probes against the RAG pipeline."""
    _not_implemented()


@app.command()
def report() -> None:
    """Generate the security report."""
    _not_implemented()

"""Typer CLI entry point (`ragsec`). Thin wrapper over the ragsec library."""

import typer

app = typer.Typer(
    help="ragsec: security testing for retrieval-augmented generation (RAG) systems.",
    no_args_is_help=True,
)


def _not_implemented() -> None:
    typer.echo("not implemented yet")


@app.command()
def ingest() -> None:
    """Ingest documents into the vector index."""
    _not_implemented()


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

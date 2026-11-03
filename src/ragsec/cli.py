"""Typer CLI entry point (`ragsec`). Thin wrapper over the ragsec library."""

from pathlib import Path
from typing import Annotated

import anthropic
import typer

from ragsec.answer import ConfigError, answer
from ragsec.config import load_settings
from ragsec.index import Index, index_directory
from ragsec.scan import format_summary, run_scan

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
def ask(
    question: Annotated[str, typer.Argument(help="The question to ask.")],
    corpus: Annotated[str | None, typer.Option(help="Only retrieve from this corpus.")] = None,
    k: Annotated[int, typer.Option("--top-k", "-k", min=1, help="Chunks to retrieve.")] = 5,
) -> None:
    """Ask a question and get a cited answer."""
    settings = load_settings()
    try:
        result = answer(question, corpus=corpus, k=k, settings=settings)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except anthropic.APIError as exc:
        typer.echo(f"Anthropic API error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(result.text)
    typer.echo("")
    typer.echo(f"Citations: {', '.join(result.citations) or '(none)'}")
    if result.invalid_citations:
        typer.echo(f"Invalid citations (not retrieved): {', '.join(result.invalid_citations)}")
    typer.echo(f"Grounded: {'yes' if result.grounded else 'no'}")


@app.command()
def scan(
    data_dir: Annotated[
        Path | None,
        typer.Option(
            exists=True,
            file_okay=False,
            help="Folder holding the corpus folders. Default: RAGSEC_DATA_DIR (./data).",
        ),
    ] = None,
    probe: Annotated[
        list[str] | None,
        typer.Option("--probe", "-p", help="Run only this probe ID (repeatable)."),
    ] = None,
) -> None:
    """Run the security probes against the RAG pipeline."""
    settings = load_settings()
    try:
        result = run_scan(data_dir or settings.data_dir, probe_ids=probe, settings=settings)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except KeyError as exc:
        typer.echo(f"Error: {exc.args[0]}", err=True)
        raise typer.Exit(code=2) from exc
    except anthropic.APIError as exc:
        typer.echo(f"Anthropic API error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(format_summary(result))


@app.command()
def report() -> None:
    """Generate the security report."""
    _not_implemented()

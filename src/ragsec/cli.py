"""Typer CLI entry point (`ragsec`). Thin wrapper over the ragsec library."""

from pathlib import Path
from typing import Annotated

import anthropic
import typer

from ragsec.answer import ConfigError, answer
from ragsec.config import Settings, load_settings
from ragsec.index import Index, index_directory
from ragsec.report import report_dir_name, write_reports
from ragsec.scan import ScanResult, format_summary, run_scan

app = typer.Typer(
    help="ragsec: security testing for retrieval-augmented generation (RAG) systems.",
    no_args_is_help=True,
)


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


DataDirOption = Annotated[
    Path | None,
    typer.Option(
        exists=True,
        file_okay=False,
        help="Folder holding the corpus folders. Default: RAGSEC_DATA_DIR (./data).",
    ),
]
ProbeOption = Annotated[
    list[str] | None,
    typer.Option("--probe", "-p", help="Run only this probe ID (repeatable)."),
]


def _run_scan_or_exit(
    settings: Settings, data_dir: Path | None, probe: list[str] | None
) -> ScanResult:
    try:
        return run_scan(data_dir or settings.data_dir, probe_ids=probe, settings=settings)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except KeyError as exc:
        typer.echo(f"Error: {exc.args[0]}", err=True)
        raise typer.Exit(code=2) from exc
    except anthropic.APIError as exc:
        typer.echo(f"Anthropic API error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def scan(data_dir: DataDirOption = None, probe: ProbeOption = None) -> None:
    """Run the security probes against the RAG pipeline."""
    result = _run_scan_or_exit(load_settings(), data_dir, probe)
    typer.echo(format_summary(result))


@app.command()
def report(
    data_dir: DataDirOption = None,
    probe: ProbeOption = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            file_okay=False,
            help="Where to write the reports. Default: RAGSEC_REPORTS_DIR/<timestamp>/.",
        ),
    ] = None,
) -> None:
    """Run a scan and write report.json and report.html."""
    settings = load_settings()
    result = _run_scan_or_exit(settings, data_dir, probe)
    target = out_dir or settings.reports_dir / report_dir_name(result.timestamp)
    paths = write_reports(result, target)

    typer.echo(f"Score: {result.passed}/{result.total} ({result.score:.0%})")
    typer.echo(f"JSON report: {paths.json_path}")
    typer.echo(f"HTML report: {paths.html_path}")

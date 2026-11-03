"""Runs the probe suite and computes the score."""

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import anthropic
from pydantic import BaseModel

from ragsec import answer as answer_module
from ragsec import probes
from ragsec.answer import ConfigError, MessagesClient
from ragsec.config import Settings, load_settings
from ragsec.index import Embedder
from ragsec.probes import Evidence, ProbeContext, ProbeResult

# Errors that would hit every probe the same way: stop the scan instead of
# recording ten identical failures.
_FATAL_API_ERRORS = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
)


class ScanResult(BaseModel):
    model: str
    timestamp: str  # UTC, ISO 8601
    corpus_dir: str
    results: list[ProbeResult]
    passed: int
    total: int
    score: float  # passed / total; total is 10 for the full suite

    @property
    def errors(self) -> list[ProbeResult]:
        return [r for r in self.results if r.error]


def run_scan(
    corpus_dirs: str | Path,
    probe_ids: Sequence[str] | None = None,
    *,
    settings: Settings | None = None,
    client: MessagesClient | None = None,
    embedder: Embedder | None = None,
) -> ScanResult:
    """Run the probes (all of them by default) and score the results.

    `corpus_dirs` is the folder that holds the corpus folders (handbook/,
    handbook_poisoned/, restricted/). Each probe builds its own index from it.

    A probe that hits a transient API error (rate limit, outage) is recorded as
    an error and counts as not passed; configuration and authentication
    problems stop the scan.
    """
    settings = settings or load_settings()
    selected = list(probe_ids) if probe_ids is not None else probes.probe_ids()
    unknown = [p for p in selected if p not in probes.PROBES]
    if unknown:
        raise KeyError(f"unknown probe(s): {', '.join(unknown)}; known: {', '.join(probes.PROBES)}")
    if not settings.model:
        raise ConfigError("RAGSEC_MODEL is not set (see .env.example)")
    if client is None:
        client = answer_module.make_client(settings)

    timestamp = datetime.now(UTC).isoformat(timespec="seconds")
    results: list[ProbeResult] = []
    with ProbeContext(corpus_dirs, settings, client=client, embedder=embedder) as ctx:
        for probe_id in selected:
            try:
                results.append(probes.run_probe(probe_id, ctx))
            except _FATAL_API_ERRORS:
                raise
            except anthropic.APIError as exc:
                results.append(_error_result(probe_id, exc))

    passed = sum(r.passed for r in results)
    total = len(results)
    return ScanResult(
        model=settings.model,
        timestamp=timestamp,
        corpus_dir=str(corpus_dirs),
        results=results,
        passed=passed,
        total=total,
        score=passed / total if total else 0.0,
    )


def _error_result(probe_id: str, exc: Exception) -> ProbeResult:
    module = probes.PROBES[probe_id]
    message = f"{type(exc).__name__}: {exc}"
    return ProbeResult(
        probe_id=probe_id,
        name=module.NAME,
        category=module.CATEGORY,
        passed=False,
        error=message,
        evidence=Evidence(
            question="",
            corpus=None,
            retrieved_chunk_ids=[],
            answer_text="",
            citations=[],
            verdict_reason=f"Probe did not complete: {message}",
        ),
    )


def format_summary(scan: ScanResult) -> str:
    """Plain-text PASS/FAIL table, reasons for anything that did not pass, and the score."""
    width = max([len("Probe"), *(len(r.probe_id) for r in scan.results)])
    lines = [
        f"Model: {scan.model}    Run: {scan.timestamp}    Corpus: {scan.corpus_dir}",
        "",
        f"{'#':>2}  {'Probe':<{width}}  Result",
        f"{'-' * 2}  {'-' * width}  ------",
    ]
    for number, r in enumerate(scan.results, start=1):
        verdict = "ERROR" if r.error else ("PASS" if r.passed else "FAIL")
        lines.append(f"{number:>2}  {r.probe_id:<{width}}  {verdict}")

    not_passed = [(n, r) for n, r in enumerate(scan.results, start=1) if not r.passed]
    if not_passed:
        lines += ["", "Why these did not pass:"]
        lines += [f"{n:>2}. {r.probe_id}: {r.evidence.verdict_reason}" for n, r in not_passed]

    lines += ["", f"Score: {scan.passed}/{scan.total} ({scan.score:.0%})"]
    if scan.errors:
        lines.append(f"{len(scan.errors)} probe(s) errored and count as not passed; re-run them.")
    return "\n".join(lines)

"""FastAPI app. Thin wrapper over the ragsec library.

Run with: uvicorn ragsec.api:app

On startup the persistent index at RAGSEC_CHROMA_DIR is filled from
RAGSEC_DATA_DIR if it is empty; /ask answers from that index. Scans build
their own throwaway indexes from RAGSEC_DATA_DIR (see ragsec.probes), so a
scan never depends on, or changes, what /ask sees.

Limitation: /scan and /report run synchronously inside the request. A full
scan makes about a dozen model calls and can take a minute or more, so
clients need a long timeout. Only one scan runs at a time; a second request
during a scan gets 409.
"""

import base64
import hashlib
import logging
import re
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import cache
from pathlib import Path
from typing import Any

import anthropic
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from ragsec import __version__
from ragsec.answer import ConfigError, MessagesClient, answer
from ragsec.config import Settings, load_settings
from ragsec.index import Embedder, Index, index_directory
from ragsec.report import render_html
from ragsec.scan import ScanResult, run_scan

log = logging.getLogger("ragsec.api")

DEMO_PAGE = Path(__file__).parent / "static" / "index.html"

# Matches the <meta> policy inside the report, sent as a header too.
REPORT_CSP = (
    "default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; "
    "form-action 'none'; base-uri 'none'; frame-ancestors 'none'"
)


# --- request / response models ---------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    corpus: str | None = Field(default=None, description="Only retrieve from this corpus.")


class AskResponse(BaseModel):
    """The answer without the retrieved chunks' text, so the API never returns
    document content the model itself did not say (restricted chunks included)."""

    text: str
    citations: list[str]
    grounded: bool
    refused: bool
    invalid_citations: list[str]
    retrieved_chunk_ids: list[str]


class ScanRequest(BaseModel):
    probe_ids: list[str] | None = Field(default=None, description="Default: all 10 probes.")


class ProbeSummary(BaseModel):
    probe_id: str
    name: str
    passed: bool
    error: str | None
    verdict_reason: str


class ReportResponse(BaseModel):
    model: str
    timestamp: str
    passed: int
    total: int
    score: float
    probes: list[ProbeSummary]
    html_url: str


class ErrorResponse(BaseModel):
    error: str
    detail: Any = None


class ApiError(Exception):
    def __init__(self, status_code: int, error: str, detail: Any = None) -> None:
        self.status_code = status_code
        self.error = error
        self.detail = detail


# --- app ----------------------------------------------------------------------------


class AppState:
    def __init__(
        self, settings: Settings, client: MessagesClient | None, embedder: Embedder | None
    ) -> None:
        self.settings = settings
        self.client = client
        self.embedder = embedder
        self.index: Index | None = None
        self.last_scan: ScanResult | None = None
        self.scan_lock = threading.Lock()


def create_app(
    settings: Settings | None = None,
    *,
    client: MessagesClient | None = None,
    embedder: Embedder | None = None,
) -> FastAPI:
    """Build the app. `settings`, `client`, and `embedder` can be injected for tests."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        state = AppState(settings or load_settings(), client, embedder)
        state.index = _startup_index(state)
        app.state.ragsec = state
        yield

    app = FastAPI(
        title="ragsec-scanner",
        version=__version__,
        description="Security testing for the document and retrieval layer of a RAG system.",
        lifespan=lifespan,
        responses={
            400: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    _register_error_handlers(app)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def demo_page() -> HTMLResponse:
        """The static demo page (vanilla HTML/CSS/JS)."""
        html, csp = _demo_page()
        return HTMLResponse(html, headers={"Content-Security-Policy": csp})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> AskResponse:
        state = _state(request)
        index = _require_index(state)
        if body.corpus is not None and body.corpus not in index.corpora():
            known = sorted(index.corpora())
            raise ApiError(400, "unknown_corpus", f"Unknown corpus {body.corpus!r}; known: {known}")
        result = answer(
            body.question,
            corpus=body.corpus,
            client=state.client,
            index=index,
            settings=state.settings,
        )
        return AskResponse(
            text=result.text,
            citations=result.citations,
            grounded=result.grounded,
            refused=result.refused,
            invalid_citations=result.invalid_citations,
            retrieved_chunk_ids=[r.chunk.id for r in result.retrieved],
        )

    @app.post("/scan", response_model=ScanResult)
    def scan(request: Request, body: ScanRequest | None = None) -> ScanResult:
        return _run_scan(_state(request), body)

    @app.post("/report", response_model=ReportResponse)
    def report(request: Request, body: ScanRequest | None = None) -> ReportResponse:
        result = _run_scan(_state(request), body)
        return ReportResponse(
            model=result.model,
            timestamp=result.timestamp,
            passed=result.passed,
            total=result.total,
            score=result.score,
            probes=[
                ProbeSummary(
                    probe_id=r.probe_id,
                    name=r.name,
                    passed=r.passed,
                    error=r.error,
                    verdict_reason=r.evidence.verdict_reason,
                )
                for r in result.results
            ],
            html_url=str(request.url_for("report_html")),
        )

    @app.get("/report/html", response_class=HTMLResponse, name="report_html")
    def report_html(request: Request) -> HTMLResponse:
        """The HTML report for the most recent scan (from /scan or /report)."""
        last = _state(request).last_scan
        if last is None:
            raise ApiError(404, "no_report", "No scan has run yet. POST /scan or /report first.")
        return HTMLResponse(render_html(last), headers={"Content-Security-Policy": REPORT_CSP})

    return app


# --- helpers ------------------------------------------------------------------------


@cache
def _demo_page() -> tuple[str, str]:
    """The demo page and a CSP that allows only its own inline script (by hash)."""
    html = DEMO_PAGE.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, flags=re.DOTALL)
    hashes = " ".join(
        "'sha256-" + base64.b64encode(hashlib.sha256(s.encode()).digest()).decode() + "'"
        for s in scripts
    )
    csp = (
        f"default-src 'none'; script-src {hashes}; style-src 'unsafe-inline'; "
        "connect-src 'self'; img-src 'none'; form-action 'self'; base-uri 'none'; "
        "frame-ancestors 'none'"
    )
    return html, csp


def _state(request: Request) -> AppState:
    return request.app.state.ragsec


def _require_index(state: AppState) -> Index:
    if state.index is None:
        raise ApiError(503, "index_unavailable", "The document index could not be loaded.")
    return state.index


def _startup_index(state: AppState) -> Index | None:
    try:
        index = Index(state.settings.chroma_dir, embedder=state.embedder)
        if index.count() == 0:
            data_dir = state.settings.data_dir
            if data_dir.is_dir():
                counts = index_directory(data_dir, index)
                log.info("Indexed demo corpora from %s: %s", data_dir, counts)
            else:
                log.warning("Index is empty and %s does not exist; /ask will refuse", data_dir)
        return index
    except Exception:
        log.exception("Could not open the document index")
        return None


def _run_scan(state: AppState, body: ScanRequest | None) -> ScanResult:
    if not state.scan_lock.acquire(blocking=False):
        raise ApiError(409, "scan_in_progress", "A scan is already running; try again later.")
    try:
        result = run_scan(
            state.settings.data_dir,
            probe_ids=body.probe_ids if body else None,
            settings=state.settings,
            client=state.client,
            embedder=state.embedder,
        )
    except KeyError as exc:
        raise ApiError(400, "unknown_probe", exc.args[0]) from exc
    finally:
        state.scan_lock.release()
    state.last_scan = result
    return result


def _error(status: int, error: str, detail: Any = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": error, "detail": detail})


def _register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(_: Request, exc: ApiError) -> JSONResponse:
        return _error(exc.status_code, exc.error, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def bad_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = [
            {"field": ".".join(str(p) for p in e.get("loc", ())[1:]) or "body", "message": e["msg"]}
            for e in exc.errors()
        ]
        return _error(400, "invalid_request", detail)

    @app.exception_handler(ConfigError)
    async def not_configured(_: Request, exc: ConfigError) -> JSONResponse:
        return _error(503, "not_configured", str(exc))

    @app.exception_handler(anthropic.AuthenticationError)
    @app.exception_handler(anthropic.PermissionDeniedError)
    async def bad_credentials(_: Request, exc: anthropic.APIError) -> JSONResponse:
        return _error(503, "not_configured", "The model API rejected the server's credentials.")

    @app.exception_handler(anthropic.RateLimitError)
    async def rate_limited(_: Request, exc: anthropic.APIError) -> JSONResponse:
        return _error(503, "model_rate_limited", "The model API is rate limiting; retry later.")

    @app.exception_handler(anthropic.APIError)
    async def upstream(_: Request, exc: anthropic.APIError) -> JSONResponse:
        return _error(502, "model_api_error", f"The model API call failed ({type(exc).__name__}).")

    @app.exception_handler(Exception)
    async def internal(_: Request, exc: Exception) -> JSONResponse:
        log.exception("Unhandled error", exc_info=exc)
        return _error(500, "internal_error", "Something went wrong on the server.")


app = create_app()

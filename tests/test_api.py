from pathlib import Path

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

import ragsec.answer as answer_module
import ragsec.api as api_module
import ragsec.probes as probes_module
from ragsec.answer import REFUSAL_PHRASE, Answer
from ragsec.api import create_app
from ragsec.config import Settings

from .conftest import REAL_MAKE_CLIENT, FakeClient, FakeEmbedder
from .test_probes import DATA, cite_first
from .test_scan import fake_probe

QUESTION = "What port does the VPN use?"


def settings(tmp_path: Path, **overrides) -> Settings:
    base = {
        "model": "test-model",
        "anthropic_api_key": "sk-test-not-real",
        "data_dir": DATA,
        "chroma_dir": tmp_path / "chroma",
    }
    return Settings(**(base | overrides))


@pytest.fixture
def make_client(tmp_path: Path):
    """Build a TestClient around an app with injected settings/client/embedder."""

    def build(client=None, **overrides) -> TestClient:
        app = create_app(
            settings(tmp_path, **overrides),
            client=client or FakeClient(cite_first("Port 4443 {cite}.")),
            embedder=FakeEmbedder(),
        )
        return TestClient(app, raise_server_exceptions=False)

    return build


@pytest.fixture
def api(make_client):
    with make_client() as c:
        yield c


# --- health and docs ----------------------------------------------------------------


def test_health(api: TestClient) -> None:
    response = api.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_openapi_lists_the_endpoints(api: TestClient) -> None:
    paths = api.get("/openapi.json").json()["paths"]
    assert {"/health", "/ask", "/scan", "/report", "/report/html"} <= set(paths)
    assert "post" in paths["/ask"] and "post" in paths["/scan"] and "post" in paths["/report"]
    assert api.get("/docs").status_code == 200


# --- startup indexing ---------------------------------------------------------------


def test_startup_indexes_demo_corpora_when_empty(api: TestClient) -> None:
    index = api.app.state.ragsec.index
    assert index.corpora() == {"handbook", "handbook_poisoned", "restricted"}


def test_startup_does_not_reindex_existing_collection(make_client) -> None:
    with make_client() as first:
        count = first.app.state.ragsec.index.count()
    with make_client() as second:
        assert second.app.state.ragsec.index.count() == count


def test_startup_with_missing_data_dir_still_serves(make_client, tmp_path: Path) -> None:
    with make_client(data_dir=tmp_path / "missing") as c:
        assert c.get("/health").status_code == 200
        response = c.post("/ask", json={"question": QUESTION})
        assert response.status_code == 200
        assert response.json()["refused"] is True


# --- /ask ---------------------------------------------------------------------------


def test_ask_returns_grounded_answer(api: TestClient) -> None:
    response = api.post("/ask", json={"question": QUESTION})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["citations"] and body["citations"][0] in body["retrieved_chunk_ids"]
    assert body["text"].startswith("Port 4443")
    assert set(body) == {
        "text",
        "citations",
        "grounded",
        "refused",
        "invalid_citations",
        "retrieved_chunk_ids",
    }


def test_ask_is_a_thin_wrapper_over_answer(api: TestClient, monkeypatch) -> None:
    seen = {}

    def fake_answer(question, corpus=None, **kwargs) -> Answer:
        seen.update(question=question, corpus=corpus, **kwargs)
        return Answer(
            text=REFUSAL_PHRASE,
            citations=[],
            retrieved=[],
            grounded=False,
            refused=True,
            invalid_citations=[],
        )

    monkeypatch.setattr(api_module, "answer", fake_answer)
    response = api.post("/ask", json={"question": QUESTION, "corpus": "handbook"})
    assert response.status_code == 200
    assert response.json()["refused"] is True
    assert seen["question"] == QUESTION
    assert seen["corpus"] == "handbook"
    assert seen["index"] is api.app.state.ragsec.index
    assert seen["settings"] is api.app.state.ragsec.settings


def test_ask_corpus_scope(make_client) -> None:
    client = FakeClient(REFUSAL_PHRASE)
    with make_client(client=client) as c:
        response = c.post("/ask", json={"question": QUESTION, "corpus": "restricted"})
    assert response.status_code == 200
    assert all(i.startswith("restricted:") for i in response.json()["retrieved_chunk_ids"])


def test_ask_never_returns_chunk_text(make_client) -> None:
    # Even when restricted chunks are retrieved, their text is not in the response.
    with make_client(client=FakeClient(REFUSAL_PHRASE)) as c:
        response = c.post("/ask", json={"question": "HR access token", "corpus": "restricted"})
    assert "CANARY-" not in response.text


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": 42},
        {"question": "x" * 2001},
        {"question": QUESTION, "corpus": 7},
    ],
)
def test_ask_bad_input_is_400(api: TestClient, payload: dict) -> None:
    response = api.post("/ask", json=payload)
    assert response.status_code == 400
    body = response.json()
    assert body["error"] == "invalid_request"
    assert body["detail"] and "field" in body["detail"][0]


def test_ask_malformed_json_is_400(api: TestClient) -> None:
    response = api.post("/ask", content=b"{not json", headers={"content-type": "application/json"})
    assert response.status_code == 400


def test_ask_unknown_corpus_is_400(api: TestClient) -> None:
    response = api.post("/ask", json={"question": QUESTION, "corpus": "nope"})
    assert response.status_code == 400
    assert response.json()["error"] == "unknown_corpus"


# --- missing configuration (503) ----------------------------------------------------


def test_ask_missing_api_key_is_503(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(answer_module, "make_client", REAL_MAKE_CLIENT)
    app = create_app(
        settings(tmp_path, anthropic_api_key=None), client=None, embedder=FakeEmbedder()
    )
    with TestClient(app, raise_server_exceptions=False) as c:
        response = c.post("/ask", json={"question": QUESTION})
        scan = c.post("/scan")
    assert response.status_code == 503
    assert response.json() == {
        "error": "not_configured",
        "detail": "ANTHROPIC_API_KEY is not set (see .env.example)",
    }
    assert scan.status_code == 503


def test_ask_missing_model_is_503(make_client) -> None:
    with make_client(model=None) as c:
        response = c.post("/ask", json={"question": QUESTION})
    assert response.status_code == 503
    assert "RAGSEC_MODEL" in response.json()["detail"]


# --- upstream and unexpected errors ---------------------------------------------------


class RaisingClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.messages = self

    def create(self, **_):
        raise self.exc


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


@pytest.mark.parametrize(
    ("exc", "status", "error"),
    [
        (
            anthropic.AuthenticationError(
                "bad key", response=httpx2.Response(401, request=_request()), body=None
            ),
            503,
            "not_configured",
        ),
        (
            anthropic.RateLimitError(
                "slow down", response=httpx2.Response(429, request=_request()), body=None
            ),
            503,
            "model_rate_limited",
        ),
        (anthropic.APIConnectionError(request=_request()), 502, "model_api_error"),
    ],
)
def test_upstream_errors_map_to_clean_json(make_client, exc, status: int, error: str) -> None:
    with make_client(client=RaisingClient(exc)) as c:
        response = c.post("/ask", json={"question": QUESTION})
    assert response.status_code == status
    assert response.json()["error"] == error
    assert "Traceback" not in response.text


def test_unexpected_error_is_500_without_traceback(make_client) -> None:
    with make_client(client=RaisingClient(RuntimeError("secret internal detail"))) as c:
        response = c.post("/ask", json={"question": QUESTION})
    assert response.status_code == 500
    assert response.json() == {
        "error": "internal_error",
        "detail": "Something went wrong on the server.",
    }
    assert "secret internal detail" not in response.text


# --- /scan, /report, /report/html -------------------------------------------------------


@pytest.fixture
def fake_suite(monkeypatch):
    modules = [fake_probe(f"fake_{i}", i <= 3) for i in range(1, 5)]
    monkeypatch.setattr(probes_module, "PROBES", {m.PROBE_ID: m for m in modules})
    return [m.PROBE_ID for m in modules]


def test_scan_with_fake_probes(api: TestClient, fake_suite) -> None:
    response = api.post("/scan")
    assert response.status_code == 200
    body = response.json()
    assert [r["probe_id"] for r in body["results"]] == fake_suite
    assert (body["passed"], body["total"], body["score"]) == (3, 4, 0.75)
    assert body["model"] == "test-model"


def test_scan_subset(api: TestClient, fake_suite) -> None:
    body = api.post("/scan", json={"probe_ids": ["fake_4"]}).json()
    assert (body["passed"], body["total"]) == (0, 1)


def test_scan_unknown_probe_is_400(api: TestClient, fake_suite) -> None:
    response = api.post("/scan", json={"probe_ids": ["nope"]})
    assert response.status_code == 400
    assert response.json()["error"] == "unknown_probe"


def test_scan_bad_body_is_400(api: TestClient, fake_suite) -> None:
    assert api.post("/scan", json={"probe_ids": "fake_1"}).status_code == 400


def test_scan_while_another_runs_is_409(api: TestClient, fake_suite) -> None:
    lock = api.app.state.ragsec.scan_lock
    lock.acquire()
    try:
        response = api.post("/scan")
    finally:
        lock.release()
    assert response.status_code == 409
    assert response.json()["error"] == "scan_in_progress"


def test_report_html_404_before_any_scan(api: TestClient) -> None:
    response = api.get("/report/html")
    assert response.status_code == 404
    assert response.json()["error"] == "no_report"


def test_report_returns_summary_and_serves_html(api: TestClient, fake_suite) -> None:
    response = api.post("/report")
    assert response.status_code == 200
    body = response.json()
    assert (body["passed"], body["total"], body["score"]) == (3, 4, 0.75)
    assert [p["probe_id"] for p in body["probes"]] == fake_suite
    assert body["probes"][3]["verdict_reason"] == "fake fail"
    assert body["html_url"].endswith("/report/html")

    html = api.get("/report/html")
    assert html.status_code == 200
    assert html.headers["content-type"].startswith("text/html")
    assert "default-src 'none'" in html.headers["content-security-policy"]
    assert "3/4" in html.text
    assert "Fake fake_1" in html.text


def test_report_html_follows_latest_scan(api: TestClient, fake_suite) -> None:
    api.post("/scan")
    assert "3/4" in api.get("/report/html").text
    api.post("/scan", json={"probe_ids": ["fake_1"]})
    assert "1/1" in api.get("/report/html").text


def test_real_probe_runs_through_api(api: TestClient) -> None:
    response = api.post("/scan", json={"probe_ids": ["context_bleed"]})
    assert response.status_code == 200
    assert response.json()["results"][0]["passed"] is True

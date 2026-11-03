import base64
import hashlib
import re
from pathlib import Path

from fastapi.testclient import TestClient

from ragsec.api import DEMO_PAGE

from .test_api import api, make_client  # noqa: F401  (fixtures)

REPO = Path(__file__).resolve().parents[1]


def test_root_serves_demo_page(api: TestClient) -> None:  # noqa: F811
    response = api.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    html = response.text
    assert "Ask the handbook" in html
    assert "Run the security scan" in html
    assert 'href="/report/html"' in html
    assert "synthetic demo data" in html


def test_demo_page_calls_the_api(api: TestClient) -> None:  # noqa: F811
    html = api.get("/").text
    assert 'postJSON("/ask"' in html
    assert 'postJSON("/scan")' in html


def test_demo_page_csp_allows_only_its_own_script(api: TestClient) -> None:  # noqa: F811
    response = api.get("/")
    csp = response.headers["content-security-policy"]
    (script,) = re.findall(r"<script>(.*?)</script>", response.text, flags=re.DOTALL)
    digest = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    assert f"script-src 'sha256-{digest}'" in csp
    assert "'unsafe-inline'" not in csp.split("script-src")[1].split(";")[0]
    assert "connect-src 'self'" in csp
    assert "default-src 'none'" in csp


def test_demo_page_is_dependency_free() -> None:
    html = DEMO_PAGE.read_text()
    assert not re.search(r"<script[^>]+src=", html)  # no external scripts
    assert "<link" not in html  # no external stylesheets
    assert not re.search(r"https?://", html)  # no outside URLs at all


def test_demo_page_never_renders_server_text_as_html() -> None:
    html = DEMO_PAGE.read_text()
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in html, sink
    assert "textContent" in html


def test_root_not_in_openapi(api: TestClient) -> None:  # noqa: F811
    assert "/" not in api.get("/openapi.json").json()["paths"]


# --- Docker packaging -------------------------------------------------------------


def test_dockerignore_keeps_secrets_and_local_data_out() -> None:
    lines = (REPO / ".dockerignore").read_text().splitlines()
    for pattern in (".env", ".venv/", "chroma/", "reports/", ".git/", "data/*"):
        assert pattern in lines, pattern
    for keep in ("!data/handbook/", "!data/handbook_poisoned/", "!data/restricted/"):
        assert keep in lines, keep


def test_dockerfile_basics() -> None:
    dockerfile = (REPO / "Dockerfile").read_text()
    assert dockerfile.count("FROM python:3.12-slim") == 1
    assert "COPY data ./data" in dockerfile
    assert "EXPOSE 8000" in dockerfile
    assert 'CMD ["uvicorn", "ragsec.api:app", "--host", "0.0.0.0", "--port", "8000"]' in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile  # model baked in, no runtime download
    # Secrets come from the runtime environment only.
    assert "ANTHROPIC_API_KEY=" not in dockerfile
    assert "RAGSEC_MODEL=" not in dockerfile
    assert "COPY .env" not in dockerfile
    assert "USER ragsec" in dockerfile

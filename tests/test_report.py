import json
import re
from pathlib import Path

import markupsafe
import pytest
from typer.testing import CliRunner

import ragsec.answer as answer_module
import ragsec.index as index_module
from ragsec import __version__
from ragsec.answer import REFUSAL_PHRASE
from ragsec.cli import app
from ragsec.probes import Evidence, ProbeResult, Step
from ragsec.report import (
    ANSWER_PREVIEW_CHARS,
    render_html,
    report_dir_name,
    truncate,
    write_reports,
)
from ragsec.scan import ScanResult

from .conftest import FakeClient, FakeEmbedder
from .test_probes import DATA

MALICIOUS = '<script>alert("xss")</script><img src=x onerror=alert(1)>'
LONG_ANSWER = "word " * 400  # ~2000 chars


def evidence(**overrides) -> Evidence:
    base = {
        "question": "How do I fix the VPN?",
        "corpus": None,
        "retrieved_chunk_ids": ["handbook:vpn-setup.md:0"],
        "answer_text": "Restart [handbook:vpn-setup.md:0].",
        "citations": ["handbook:vpn-setup.md:0"],
        "verdict_reason": "No PWNED in the answer.",
    }
    return Evidence(**(base | overrides))


def result(n: int, passed: bool, **ev) -> ProbeResult:
    return ProbeResult(
        probe_id=f"probe_{n}",
        name=f"Probe number {n}",
        category=f"category_{n}",
        passed=passed,
        evidence=evidence(**ev),
    )


@pytest.fixture
def fake_scan() -> ScanResult:
    results = [result(n, n <= 7) for n in range(1, 11)]
    results[1] = result(2, True, answer_text=MALICIOUS, question=MALICIOUS)
    results[2] = result(3, True, answer_text=LONG_ANSWER)
    results[7] = result(
        8,
        False,
        verdict_reason=f"Reason with markup {MALICIOUS}",
        steps=[
            Step(
                question="Step question?",
                corpus="handbook",
                retrieved_chunk_ids=[f"handbook:{MALICIOUS}:0"],
                answer_text=MALICIOUS,
                citations=[],
                note="invalid citations: x",
            )
        ],
    )
    results[9] = ProbeResult(
        probe_id="probe_10",
        name="Probe number 10",
        category="category_10",
        passed=False,
        error="APIConnectionError: Connection error.",
        evidence=evidence(verdict_reason="Probe did not complete: APIConnectionError"),
    )
    return ScanResult(
        model="test-model",
        timestamp="2026-11-03T19:12:03+00:00",
        corpus_dir="data",
        results=results,
        passed=7,
        total=10,
        score=0.7,
    )


@pytest.fixture
def written(fake_scan: ScanResult, tmp_path: Path):
    paths = write_reports(fake_scan, tmp_path / "out")
    return paths, paths.html_path.read_text(encoding="utf-8")


# --- JSON ---------------------------------------------------------------------------


def test_json_parses_and_contains_all_probes(written, fake_scan: ScanResult) -> None:
    paths, _ = written
    data = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert [r["probe_id"] for r in data["results"]] == [f"probe_{n}" for n in range(1, 11)]
    assert (data["passed"], data["total"], data["score"]) == (7, 10, 0.7)
    assert data["model"] == "test-model"
    assert ScanResult.model_validate(data) == fake_scan  # round-trips losslessly


def test_json_keeps_full_untruncated_answer(written) -> None:
    paths, _ = written
    data = json.loads(paths.json_path.read_text(encoding="utf-8"))
    assert data["results"][2]["evidence"]["answer_text"] == LONG_ANSWER


# --- HTML content -------------------------------------------------------------------


def test_html_contains_score_and_every_probe_name(written, fake_scan: ScanResult) -> None:
    _, html = written
    assert "7/10" in html
    assert "70%" in html
    assert 'style="width: 70.0%"' in html
    for r in fake_scan.results:
        assert r.name in html


def test_html_header_fields(written) -> None:
    _, html = written
    assert f"ragsec-scanner {__version__}" in html
    assert "2026-11-03 19:12:03 UTC" in html
    assert "test-model" in html


def test_html_badges(written) -> None:
    _, html = written
    assert html.count('class="badge pass"') == 2 * 7  # summary row + section header
    assert html.count('class="badge fail"') == 2 * 2
    assert html.count('class="badge error"') == 2 * 1


def test_html_scope_section(written) -> None:
    _, html = written
    assert "What this does and does not test" in html
    assert "document and retrieval layer" in html
    assert "not</strong> a full penetration test" in html
    assert "not</strong> a security guarantee" in html


def test_html_shows_question_chunks_answer_and_reason(written) -> None:
    _, html = written
    assert "How do I fix the VPN?" in html
    assert "handbook:vpn-setup.md:0" in html
    assert "Restart [handbook:vpn-setup.md:0]." in html
    assert "No PWNED in the answer." in html


def test_html_truncates_long_answers(written) -> None:
    _, html = written
    assert LONG_ANSWER not in html
    assert "Truncated; full answer in report.json." in html


def test_html_renders_steps(written) -> None:
    _, html = written
    assert "Step 1 · scope: handbook" in html
    assert "Step question?" in html
    assert "invalid citations: x" in html


# --- escaping -----------------------------------------------------------------------


def test_html_escapes_malicious_model_output(written) -> None:
    _, html = written
    assert "<script" not in html.lower()
    assert "<img" not in html.lower()
    assert MALICIOUS not in html
    # The payload is shown as text, fully escaped, wherever it occurred.
    escaped = str(markupsafe.escape(MALICIOUS))
    assert escaped == (
        "&lt;script&gt;alert(&#34;xss&#34;)&lt;/script&gt;&lt;img src=x onerror=alert(1)&gt;"
    )
    assert html.count(escaped) >= 4  # question, answer, step answer, verdict reason


def test_html_has_no_scripts_or_external_assets(written) -> None:
    _, html = written
    assert not re.search(r"<\s*(script|link|iframe|object|embed)\b", html, re.IGNORECASE)
    assert not re.search(r"\b(src|href)\s*=\s*\"(?!#)", html)  # only in-page anchors
    assert "@import" not in html
    assert "url(" not in html
    assert "Content-Security-Policy" in html
    assert "default-src 'none'" in html


def test_template_is_escaped_regardless_of_extension(fake_scan: ScanResult) -> None:
    html = render_html(fake_scan)
    assert MALICIOUS not in html


# --- helpers ------------------------------------------------------------------------


def test_truncate() -> None:
    assert truncate("short") == ("short", False)
    cut, was_cut = truncate(LONG_ANSWER)
    assert was_cut is True
    assert len(cut) <= ANSWER_PREVIEW_CHARS + 2
    assert cut.endswith(" …")


def test_truncate_without_spaces() -> None:
    cut, was_cut = truncate("x" * 2000)
    assert was_cut and len(cut) == ANSWER_PREVIEW_CHARS + 2


def test_report_dir_name() -> None:
    assert report_dir_name("2026-11-03T19:12:03+00:00") == "20261103T191203Z"


def test_write_reports_creates_out_dir(fake_scan: ScanResult, tmp_path: Path) -> None:
    paths = write_reports(fake_scan, tmp_path / "a" / "b")
    assert paths.json_path.exists() and paths.html_path.exists()


# --- CLI ----------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    reports = tmp_path / "reports"
    monkeypatch.setenv("RAGSEC_MODEL", "test-model")
    monkeypatch.setenv("RAGSEC_DATA_DIR", str(DATA))
    monkeypatch.setenv("RAGSEC_REPORTS_DIR", str(reports))
    monkeypatch.setattr(index_module, "default_embedder", FakeEmbedder)
    monkeypatch.setattr(answer_module, "make_client", lambda s: FakeClient(REFUSAL_PHRASE))
    return reports


def test_cli_report_writes_both_files_under_timestamp(cli_env: Path) -> None:
    result = CliRunner().invoke(app, ["report"])
    assert result.exit_code == 0, result.output
    (run_dir,) = list(cli_env.iterdir())
    assert re.fullmatch(r"\d{8}T\d{6}Z", run_dir.name)
    assert f"JSON report: {run_dir / 'report.json'}" in result.output
    assert f"HTML report: {run_dir / 'report.html'}" in result.output
    assert "Score: 7/10 (70%)" in result.output
    data = json.loads((run_dir / "report.json").read_text())
    assert len(data["results"]) == 10
    assert "Benign control" in (run_dir / "report.html").read_text()


def test_cli_report_out_dir_option(cli_env: Path, tmp_path: Path) -> None:
    target = tmp_path / "custom"
    result = CliRunner().invoke(app, ["report", "--out-dir", str(target), "-p", "context_bleed"])
    assert result.exit_code == 0, result.output
    assert (target / "report.html").exists()
    assert "Score: 1/1 (100%)" in result.output


def test_cli_report_missing_model_exits_2(cli_env: Path, monkeypatch) -> None:
    monkeypatch.delenv("RAGSEC_MODEL")
    result = CliRunner().invoke(app, ["report"])
    assert result.exit_code == 2
    assert not cli_env.exists()

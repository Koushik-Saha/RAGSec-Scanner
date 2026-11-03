from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from typer.testing import CliRunner

import ragsec.answer as answer_module
import ragsec.index as index_module
import ragsec.probes as probes_module
from ragsec.answer import REFUSAL_PHRASE, ConfigError
from ragsec.cli import app
from ragsec.config import Settings
from ragsec.probes import Evidence, ProbeResult
from ragsec.scan import ScanResult, format_summary, run_scan

from .conftest import FakeClient, FakeEmbedder
from .test_probes import DATA, SETTINGS, cite_first


def fake_probe(probe_id: str, passed: bool | Exception) -> SimpleNamespace:
    def run(ctx: Any) -> ProbeResult:
        if isinstance(passed, Exception):
            raise passed
        return ProbeResult(
            probe_id=probe_id,
            name=f"Fake {probe_id}",
            category="fake",
            passed=passed,
            evidence=Evidence(
                question="q",
                corpus=None,
                retrieved_chunk_ids=[],
                answer_text="a",
                citations=[],
                verdict_reason="fake pass" if passed else "fake fail",
            ),
        )

    return SimpleNamespace(PROBE_ID=probe_id, NAME=f"Fake {probe_id}", CATEGORY="fake", run=run)


@pytest.fixture
def fake_suite(monkeypatch: pytest.MonkeyPatch):
    def install(outcomes: list[bool | Exception]) -> list[str]:
        modules = [fake_probe(f"fake_{i}", o) for i, o in enumerate(outcomes, start=1)]
        monkeypatch.setattr(probes_module, "PROBES", {m.PROBE_ID: m for m in modules})
        return [m.PROBE_ID for m in modules]

    return install


def scan(**kwargs: Any) -> ScanResult:
    kwargs.setdefault("settings", SETTINGS)
    kwargs.setdefault("client", FakeClient("unused"))
    return run_scan(DATA, **kwargs)


# --- score math -----------------------------------------------------------------


def test_score_seven_of_ten(fake_suite) -> None:
    fake_suite([True] * 7 + [False] * 3)
    result = scan()
    assert (result.passed, result.total) == (7, 10)
    assert result.score == pytest.approx(0.7)


@pytest.mark.parametrize(("n_pass", "expected"), [(0, 0.0), (10, 1.0), (5, 0.5), (1, 0.1)])
def test_score_range(fake_suite, n_pass: int, expected: float) -> None:
    fake_suite([True] * n_pass + [False] * (10 - n_pass))
    assert scan().score == pytest.approx(expected)


def test_subset_scores_over_probes_run(fake_suite) -> None:
    ids = fake_suite([True, False, True, True])
    result = scan(probe_ids=[ids[0], ids[1]])
    assert [r.probe_id for r in result.results] == [ids[0], ids[1]]
    assert (result.passed, result.total, result.score) == (1, 2, 0.5)


def test_results_keep_probe_order(fake_suite) -> None:
    ids = fake_suite([True, False, True])
    assert [r.probe_id for r in scan().results] == ids


def test_records_model_and_timestamp(fake_suite) -> None:
    fake_suite([True])
    result = scan(settings=Settings(model="model-under-test", anthropic_api_key="sk-x"))
    assert result.model == "model-under-test"
    stamp = datetime.fromisoformat(result.timestamp)
    assert stamp.tzinfo is not None
    assert result.corpus_dir == str(DATA)


# --- errors -----------------------------------------------------------------------


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def test_transient_api_error_recorded_and_counts_as_not_passed(fake_suite) -> None:
    outage = anthropic.APIConnectionError(request=_request())
    fake_suite([True, outage, True])
    result = scan()
    assert (result.passed, result.total) == (2, 3)
    errored = result.results[1]
    assert errored.passed is False
    assert errored.error and "APIConnectionError" in errored.error
    assert result.errors == [errored]


def test_auth_error_stops_the_scan(fake_suite) -> None:
    response = httpx2.Response(401, request=_request())
    auth = anthropic.AuthenticationError("bad key", response=response, body=None)
    fake_suite([True, auth, True])
    with pytest.raises(anthropic.AuthenticationError):
        scan()


def test_unknown_probe_id_rejected_before_running(fake_suite) -> None:
    fake_suite([True])
    with pytest.raises(KeyError, match="nope"):
        scan(probe_ids=["fake_1", "nope"])


def test_missing_model_raises(fake_suite) -> None:
    fake_suite([True])
    with pytest.raises(ConfigError, match="RAGSEC_MODEL"):
        scan(settings=Settings())


def test_missing_key_raises_when_no_client_injected(fake_suite, monkeypatch) -> None:
    fake_suite([True])
    from .conftest import REAL_MAKE_CLIENT

    monkeypatch.setattr(answer_module, "make_client", REAL_MAKE_CLIENT)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        run_scan(DATA, settings=Settings(model="m"))


# --- summary ------------------------------------------------------------------------


def test_summary_table(fake_suite) -> None:
    fake_suite([True, False, anthropic.APIConnectionError(request=_request())])
    text = format_summary(scan())
    assert " 1  fake_1  PASS" in text
    assert " 2  fake_2  FAIL" in text
    assert " 3  fake_3  ERROR" in text
    assert "2. fake_2: fake fail" in text
    assert "Score: 1/3 (33%)" in text
    assert "1 probe(s) errored" in text


# --- full suite with real probes (fake model and embedder) -------------------------


def test_full_suite_runs_all_ten_real_probes() -> None:
    result = run_scan(
        DATA, settings=SETTINGS, client=FakeClient(REFUSAL_PHRASE), embedder=FakeEmbedder()
    )
    assert result.total == 10
    verdicts = {r.probe_id: r.passed for r in result.results}
    # An always-refusing model is safe against attacks but fails the probes that
    # need a real answer: poisoning (must report $60), citation integrity
    # (nothing cited), and the benign control (over-refusal).
    assert verdicts == {
        "indirect_injection_body": True,
        "injection_via_title": True,
        "planted_secret_extraction": True,
        "cross_document_leakage": True,
        "poisoning_answer_flip": False,
        "system_prompt_extraction": True,
        "ungrounded_answer_trap": True,
        "citation_integrity": False,
        "context_bleed": True,
        "benign_control": False,
    }
    assert result.score == pytest.approx(0.7)


def test_full_suite_with_citing_model() -> None:
    client = FakeClient(cite_first("Per the handbook {cite}, the cap is $60 per day."))
    result = run_scan(DATA, settings=SETTINGS, client=client, embedder=FakeEmbedder())
    verdicts = {r.probe_id: r.passed for r in result.results}
    assert verdicts["citation_integrity"] is True
    assert verdicts["benign_control"] is True
    assert verdicts["poisoning_answer_flip"] is True
    assert verdicts["ungrounded_answer_trap"] is False  # it cited instead of refusing


# --- CLI ----------------------------------------------------------------------------


@pytest.fixture
def cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGSEC_MODEL", "test-model")
    monkeypatch.setenv("RAGSEC_DATA_DIR", str(DATA))
    monkeypatch.setattr(index_module, "default_embedder", FakeEmbedder)
    monkeypatch.setattr(answer_module, "make_client", lambda s: FakeClient(REFUSAL_PHRASE))


def test_cli_scan_prints_ten_verdicts_and_score(cli_env) -> None:
    result = CliRunner().invoke(app, ["scan"])
    assert result.exit_code == 0, result.output
    rows = [line for line in result.output.splitlines() if line.rstrip().endswith(("PASS", "FAIL"))]
    assert len(rows) == 10
    assert "Score: 7/10 (70%)" in result.output
    assert "Model: test-model" in result.output


def test_cli_scan_single_probe(cli_env) -> None:
    result = CliRunner().invoke(app, ["scan", "--probe", "context_bleed"])
    assert result.exit_code == 0, result.output
    assert "context_bleed" in result.output
    assert "Score: 1/1 (100%)" in result.output


def test_cli_scan_unknown_probe_exits_2(cli_env) -> None:
    result = CliRunner().invoke(app, ["scan", "--probe", "nope"])
    assert result.exit_code == 2
    assert "unknown probe" in result.output


def test_cli_scan_missing_model_exits_2(cli_env, monkeypatch) -> None:
    monkeypatch.delenv("RAGSEC_MODEL")
    result = CliRunner().invoke(app, ["scan"])
    assert result.exit_code == 2
    assert "RAGSEC_MODEL" in result.output


def test_cli_scan_custom_data_dir(cli_env, tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["scan", "--data-dir", str(tmp_path / "missing")])
    assert result.exit_code != 0

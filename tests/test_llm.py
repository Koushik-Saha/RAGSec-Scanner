"""The provider switch (ragsec.llm): selection, missing keys, and the OpenAI-compatible
adapter driving real probes. No real client sends a request; OpenAI clients are mocked."""

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import openai
import pytest

import ragsec.answer as answer_module
import ragsec.llm as llm_module
from ragsec.answer import SYSTEM_PROMPT
from ragsec.config import ConfigError, Settings, load_settings
from ragsec.llm import OpenAICompatibleClient, make_client
from ragsec.scan import run_scan

from .conftest import FakeEmbedder
from .test_probes import DATA

KEYS = {
    "anthropic_api_key": "sk-ant-test-not-real",
    "groq_api_key": "gsk-test-not-real",
    "openrouter_api_key": "sk-or-test-not-real",
}


# --- configuration ------------------------------------------------------------


def test_provider_defaults_to_anthropic() -> None:
    assert load_settings({}).provider == "anthropic"
    assert Settings().provider == "anthropic"


def test_provider_and_keys_read_from_env() -> None:
    s = load_settings(
        {
            "RAGSEC_PROVIDER": " Groq ",
            "GROQ_API_KEY": "gsk-test-not-real",
            "OPENROUTER_API_KEY": "sk-or-test-not-real",
        }
    )
    assert s.provider == "groq"
    assert s.groq_api_key is not None
    assert s.groq_api_key.get_secret_value() == "gsk-test-not-real"
    assert s.openrouter_api_key is not None
    assert "gsk-test-not-real" not in repr(s)  # keys are SecretStr


# --- provider selection -------------------------------------------------------


def test_anthropic_provider_builds_anthropic_client() -> None:
    client = make_client(Settings(provider="anthropic", **KEYS))
    assert isinstance(client, anthropic.Anthropic)
    assert client.api_key == "sk-ant-test-not-real"


@pytest.mark.parametrize(
    ("provider", "base_url", "key"),
    [
        ("groq", "https://api.groq.com/openai/v1", "gsk-test-not-real"),
        ("openrouter", "https://openrouter.ai/api/v1", "sk-or-test-not-real"),
        ("ollama", "http://localhost:11434/v1", "ollama"),
    ],
)
def test_openai_compatible_providers(provider: str, base_url: str, key: str) -> None:
    client = make_client(Settings(provider=provider, **KEYS))
    assert isinstance(client, OpenAICompatibleClient)
    assert isinstance(client.openai_client, openai.OpenAI)
    assert str(client.openai_client.base_url).rstrip("/") == base_url
    assert client.openai_client.api_key == key


def test_ollama_needs_no_key() -> None:
    client = make_client(Settings(provider="ollama"))
    assert client.openai_client.api_key == "ollama"


def test_unknown_provider_fails_fast() -> None:
    with pytest.raises(ConfigError, match="RAGSEC_PROVIDER 'gemini' is not supported"):
        make_client(Settings(provider="gemini", **KEYS))


# --- missing keys -------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "variable"),
    [
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("groq", "GROQ_API_KEY"),
        ("openrouter", "OPENROUTER_API_KEY"),
    ],
)
def test_missing_key_names_the_variable(provider: str, variable: str) -> None:
    # Every other provider's key is set; none of their values may appear in the error.
    others = {field: value for field, value in KEYS.items() if field != f"{variable.lower()}"}
    with pytest.raises(ConfigError, match=variable) as info:
        make_client(Settings(provider=provider, **others))
    for value in KEYS.values():
        assert value not in str(info.value)


def test_scan_fails_fast_on_missing_provider_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(answer_module, "make_client", make_client)
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        run_scan(DATA, settings=Settings(provider="groq", model="m"), embedder=FakeEmbedder())


# --- mocked OpenAI-compatible client ------------------------------------------


class FakeCompletions:
    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        reply = self.reply(kwargs) if callable(self.reply) else self.reply
        if isinstance(reply, Exception):
            raise reply
        text, finish_reason = reply if isinstance(reply, tuple) else (reply, "stop")
        message = SimpleNamespace(content=text, refusal=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
        )


class FakeOpenAI:
    """Stands in for openai.OpenAI; records how it was constructed."""

    instances: list["FakeOpenAI"] = []
    reply: Any = ""

    def __init__(self, *, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.completions = FakeCompletions(FakeOpenAI.reply)
        self.chat = SimpleNamespace(completions=self.completions)
        FakeOpenAI.instances.append(self)


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch):
    """Route scans through the real factory, with openai.OpenAI replaced by a fake."""
    FakeOpenAI.instances = []
    monkeypatch.setattr(llm_module.openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(answer_module, "make_client", make_client)

    def install(reply: Any) -> type[FakeOpenAI]:
        FakeOpenAI.reply = reply
        return FakeOpenAI

    return install


def cite_first_openai(template: str):
    """A reply citing the first chunk shown in the OpenAI-format request."""

    def reply(kwargs: dict[str, Any]) -> str:
        from .conftest import context_chunk_ids

        user = {"messages": [kwargs["messages"][-1]]}
        return template.format(cite=f"[{context_chunk_ids(user)[0]}]")

    return reply


GROQ = Settings(provider="groq", groq_api_key="gsk-test-not-real", model="llama-test")


def test_full_probe_run_through_openai_compatible_client(fake_openai) -> None:
    fake = fake_openai(cite_first_openai("Per the handbook {cite}, the cap is $60 per day."))
    result = run_scan(
        DATA, probe_ids=["citation_integrity"], settings=GROQ, embedder=FakeEmbedder()
    )

    (probe,) = result.results
    assert probe.error is None
    assert probe.passed is True
    assert result.model == "llama-test"
    assert (result.passed, result.total) == (1, 1)
    assert probe.evidence.citations  # citations parsed from the OpenAI reply

    (client,) = fake.instances
    assert client.base_url == "https://api.groq.com/openai/v1"
    assert client.api_key == "gsk-test-not-real"
    request = client.completions.calls[0]
    assert request["model"] == "llama-test"
    assert request["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert request["messages"][1]["role"] == "user"
    assert "<context>" in request["messages"][1]["content"]


def test_vulnerable_openai_reply_fails_probe(fake_openai) -> None:
    fake_openai("PWNED")
    result = run_scan(
        DATA, probe_ids=["indirect_injection_body"], settings=GROQ, embedder=FakeEmbedder()
    )
    (probe,) = result.results
    assert probe.passed is False
    assert "PWNED" in probe.evidence.verdict_reason


def test_content_filter_counts_as_api_refusal(fake_openai) -> None:
    fake_openai(("", "content_filter"))
    result = run_scan(
        DATA, probe_ids=["planted_secret_extraction"], settings=GROQ, embedder=FakeEmbedder()
    )
    assert result.results[0].passed is True


def _request() -> httpx2.Request:
    return httpx2.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def test_transient_openai_error_is_recorded_per_probe(fake_openai) -> None:
    fake_openai(openai.APIConnectionError(request=_request()))
    result = run_scan(DATA, probe_ids=["benign_control"], settings=GROQ, embedder=FakeEmbedder())
    (probe,) = result.results
    assert probe.passed is False
    assert probe.error is not None and "APIConnectionError" in probe.error


def test_openai_auth_error_stops_the_scan(fake_openai) -> None:
    response = httpx2.Response(401, request=_request())
    fake_openai(openai.AuthenticationError("bad key", response=response, body=None))
    with pytest.raises(openai.AuthenticationError):
        run_scan(DATA, probe_ids=["benign_control"], settings=GROQ, embedder=FakeEmbedder())

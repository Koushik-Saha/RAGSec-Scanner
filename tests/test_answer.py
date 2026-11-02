from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import ragsec.answer as answer_module
import ragsec.index as index_module
from ragsec.answer import (
    MAX_TOKENS,
    REFUSAL_PHRASE,
    SYSTEM_PROMPT,
    ConfigError,
    answer,
    build_user_prompt,
    contains_refusal,
    parse_citations,
)
from ragsec.cli import app
from ragsec.config import Settings
from ragsec.index import Index, index_directory

from .conftest import FIXTURE_CORPUS, REAL_MAKE_CLIENT, FakeClient, FakeEmbedder

LEAVE_QUERY = "How many days of paid leave does an employee get per year?"
LEAVE_CHUNK = "corpus_a:handbook.md:1"
NOTES_CHUNK = "corpus_b:notes.txt:1"
SETTINGS = Settings(model="test-model", anthropic_api_key="sk-test-not-real")


@pytest.fixture
def index(tmp_path: Path, fake_embedder: FakeEmbedder) -> Index:
    idx = Index(tmp_path / "chroma", embedder=fake_embedder)
    index_directory(FIXTURE_CORPUS, idx)
    return idx


def ask(index: Index, client: FakeClient, question: str = LEAVE_QUERY, **kwargs):
    return answer(question, client=client, index=index, settings=SETTINGS, **kwargs)


# --- grounding outcomes -----------------------------------------------------


def test_grounded_answer_with_citation(index: Index) -> None:
    client = FakeClient(f"Employees get twenty days of paid leave per year [{LEAVE_CHUNK}].")
    result = ask(index, client)
    assert result.grounded is True
    assert result.refused is False
    assert result.citations == [LEAVE_CHUNK]
    assert result.invalid_citations == []
    assert LEAVE_CHUNK in {r.chunk.id for r in result.retrieved}


def test_refusal_is_not_grounded(index: Index) -> None:
    result = ask(index, FakeClient(REFUSAL_PHRASE), question="What is the CEO's salary?")
    assert result.refused is True
    assert result.grounded is False
    assert result.citations == []


def test_refusal_clears_citations(index: Index) -> None:
    result = ask(index, FakeClient(f"{REFUSAL_PHRASE} [{LEAVE_CHUNK}]"))
    assert result.refused is True
    assert result.citations == []
    assert result.grounded is False


def test_citation_to_non_retrieved_chunk_is_ungrounded(index: Index) -> None:
    # k=1 retrieves only the leave chunk; the answer cites a chunk it never saw.
    client = FakeClient(f"Twenty days [{NOTES_CHUNK}].")
    result = ask(index, client, k=1)
    assert [r.chunk.id for r in result.retrieved] == [LEAVE_CHUNK]
    assert result.citations == [NOTES_CHUNK]
    assert result.invalid_citations == [NOTES_CHUNK]
    assert result.grounded is False


def test_fabricated_citation_beside_valid_one_is_ungrounded(index: Index) -> None:
    client = FakeClient(f"Twenty days [{LEAVE_CHUNK}], carried over [corpus_a:made_up.md:7].")
    result = ask(index, client)
    assert result.citations == [LEAVE_CHUNK, "corpus_a:made_up.md:7"]
    assert result.invalid_citations == ["corpus_a:made_up.md:7"]
    assert result.grounded is False


def test_answer_without_citations_is_ungrounded(index: Index) -> None:
    result = ask(index, FakeClient("Employees get twenty days of paid leave."))
    assert result.citations == []
    assert result.grounded is False
    assert result.refused is False


def test_api_refusal_stop_reason_counts_as_refusal(index: Index) -> None:
    result = ask(index, FakeClient("", stop_reason="refusal"))
    assert result.refused is True
    assert result.grounded is False


def test_non_text_blocks_are_ignored(index: Index) -> None:
    thinking = SimpleNamespace(type="thinking", thinking="internal")
    client = FakeClient(f"Twenty days [{LEAVE_CHUNK}].", extra_blocks=[thinking])
    result = ask(index, client)
    assert result.text == f"Twenty days [{LEAVE_CHUNK}]."
    assert result.grounded is True


def test_empty_retrieval_refuses_without_calling_model(tmp_path: Path) -> None:
    empty = Index(tmp_path / "empty", embedder=FakeEmbedder())
    client = FakeClient("should not be used")
    result = ask(empty, client)
    assert client.calls == []
    assert result.text == REFUSAL_PHRASE
    assert result.refused is True
    assert result.grounded is False


def test_corpus_filter_applies_to_context(index: Index) -> None:
    client = FakeClient(REFUSAL_PHRASE)
    result = ask(index, client, corpus="corpus_b")
    assert {r.chunk.corpus for r in result.retrieved} == {"corpus_b"}
    assert "corpus_a:" not in client.calls[0]["messages"][0]["content"]


# --- request shape ------------------------------------------------------------


def test_request_uses_config_model_and_prompts(index: Index) -> None:
    client = FakeClient(f"Twenty days [{LEAVE_CHUNK}].")
    result = ask(index, client, k=3)
    (call,) = client.calls
    assert call["model"] == "test-model"
    assert call["max_tokens"] == MAX_TOKENS
    assert call["system"] == SYSTEM_PROMPT
    assert call["messages"] == [
        {"role": "user", "content": build_user_prompt(LEAVE_QUERY, result.retrieved)}
    ]


def test_system_prompt_rules() -> None:
    assert "ONLY" in SYSTEM_PROMPT
    assert "[corpus_a:handbook.md:0]" in SYSTEM_PROMPT
    assert f'"{REFUSAL_PHRASE}"' in SYSTEM_PROMPT


def test_user_prompt_numbers_chunks_then_question(index: Index) -> None:
    retrieved = answer_module.retrieve(LEAVE_QUERY, k=3, index=index)
    prompt = build_user_prompt(LEAVE_QUERY, retrieved)
    positions = [
        prompt.index(f'<chunk number="{n}" id="{r.chunk.id}">')
        for n, r in enumerate(retrieved, start=1)
    ]
    assert positions == sorted(positions)
    for r in retrieved:
        assert r.chunk.text in prompt
        assert f"title: {r.chunk.title}" in prompt
    assert prompt.endswith(f"Question: {LEAVE_QUERY}")
    assert prompt.index("</context>") < prompt.index("Question:")


# --- configuration ------------------------------------------------------------


def test_missing_model_raises(index: Index) -> None:
    with pytest.raises(ConfigError, match="RAGSEC_MODEL"):
        answer(LEAVE_QUERY, client=FakeClient("x"), index=index, settings=Settings())


def test_make_client_requires_api_key() -> None:
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        REAL_MAKE_CLIENT(Settings(model="test-model"))


def test_make_client_uses_key_from_config() -> None:
    client = REAL_MAKE_CLIENT(SETTINGS)  # builds the client only; no request is sent
    assert client.api_key == "sk-test-not-real"


# --- parsing helpers ----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A [c:a.md:0] and B [c:b.md:2].", ["c:a.md:0", "c:b.md:2"]),
        ("Both [c:a.md:0, c:b.md:1].", ["c:a.md:0", "c:b.md:1"]),
        ("Semicolons [c:a.md:0; c:b.md:1].", ["c:a.md:0", "c:b.md:1"]),
        ("Repeat [c:a.md:0] then [c:a.md:0].", ["c:a.md:0"]),
        ("Adjacent [c:a.md:0][c:b.md:1].", ["c:a.md:0", "c:b.md:1"]),
        ("Spaces in name [corpus_a:team notes.md:3].", ["corpus_a:team notes.md:3"]),
        ("Not IDs: [1], [see above], [link](http://x.y:80).", []),
        ("No citations at all.", []),
    ],
)
def test_parse_citations(text: str, expected: list[str]) -> None:
    assert parse_citations(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        REFUSAL_PHRASE,
        REFUSAL_PHRASE.replace("'", "’"),  # curly apostrophe
        REFUSAL_PHRASE.upper(),
        "Sorry. " + REFUSAL_PHRASE.replace(" in the ", "\n in the "),
    ],
)
def test_contains_refusal(text: str) -> None:
    assert contains_refusal(text)


def test_contains_refusal_negative() -> None:
    assert not contains_refusal("I don't have the handbook, but leave is twenty days.")


# --- CLI ----------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    chroma_dir = tmp_path / "cli_chroma"
    monkeypatch.setenv("RAGSEC_CHROMA_DIR", str(chroma_dir))
    monkeypatch.setenv("RAGSEC_MODEL", "test-model")
    monkeypatch.setattr(index_module, "default_embedder", FakeEmbedder)
    index_directory(FIXTURE_CORPUS, Index(chroma_dir))
    return chroma_dir


def use_fake_client(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> None:
    monkeypatch.setattr(answer_module, "make_client", lambda settings: client)


def test_cli_ask_prints_answer_citations_grounded(
    cli_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = FakeClient(f"Twenty days of paid leave [{LEAVE_CHUNK}].")
    use_fake_client(monkeypatch, client)
    result = CliRunner().invoke(app, ["ask", LEAVE_QUERY])
    assert result.exit_code == 0, result.output
    assert "Twenty days of paid leave" in result.output
    assert f"Citations: {LEAVE_CHUNK}" in result.output
    assert "Grounded: yes" in result.output


def test_cli_ask_refusal(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_client(monkeypatch, FakeClient(REFUSAL_PHRASE))
    result = CliRunner().invoke(app, ["ask", "Who won the 1998 World Cup?"])
    assert result.exit_code == 0, result.output
    assert REFUSAL_PHRASE in result.output
    assert "Citations: (none)" in result.output
    assert "Grounded: no" in result.output


def test_cli_ask_shows_invalid_citations(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    use_fake_client(monkeypatch, FakeClient("Twenty [corpus_a:fake.md:9]."))
    result = CliRunner().invoke(app, ["ask", LEAVE_QUERY])
    assert "Invalid citations (not retrieved): corpus_a:fake.md:9" in result.output
    assert "Grounded: no" in result.output


def test_cli_ask_corpus_option(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeClient(REFUSAL_PHRASE)
    use_fake_client(monkeypatch, client)
    result = CliRunner().invoke(app, ["ask", LEAVE_QUERY, "--corpus", "corpus_b"])
    assert result.exit_code == 0, result.output
    assert "corpus_a:" not in client.calls[0]["messages"][0]["content"]


def test_cli_ask_missing_model_exits_2(cli_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAGSEC_MODEL")
    result = CliRunner().invoke(app, ["ask", LEAVE_QUERY])
    assert result.exit_code == 2
    assert "RAGSEC_MODEL" in result.output

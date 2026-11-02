from pathlib import Path

import pytest
from typer.testing import CliRunner

import ragsec.index as index_module
from ragsec.cli import app
from ragsec.index import Index, index_directory
from ragsec.ingest import Chunk, ingest_directory
from ragsec.retrieve import RetrievedChunk, retrieve

from .conftest import FIXTURE_CORPUS, FakeEmbedder

LEAVE_QUERY = "How many days of paid leave does an employee get per year?"
ROTATION_QUERY = "When does the support rotation start, and how many people are on call?"


@pytest.fixture
def index(tmp_path: Path, fake_embedder: FakeEmbedder) -> Index:
    idx = Index(tmp_path / "chroma", embedder=fake_embedder)
    index_directory(FIXTURE_CORPUS, idx)
    return idx


# --- ranking ---------------------------------------------------------------


def test_leave_query_returns_leave_chunk_first(index: Index) -> None:
    results = retrieve(LEAVE_QUERY, index=index)
    assert results[0].chunk.id == "corpus_a:handbook.md:1"
    assert "twenty days of paid leave" in results[0].chunk.text


def test_rotation_query_returns_notes_chunk_first(index: Index) -> None:
    results = retrieve(ROTATION_QUERY, index=index)
    assert results[0].chunk.id == "corpus_b:notes.txt:1"


def test_results_sorted_by_distance(index: Index) -> None:
    results = retrieve(LEAVE_QUERY, k=7, index=index)
    distances = [r.distance for r in results]
    assert distances == sorted(distances)
    assert len(results) == 7


def test_k_limits_results(index: Index) -> None:
    assert len(retrieve(LEAVE_QUERY, k=2, index=index)) == 2


def test_retrieved_chunk_round_trips_metadata(index: Index) -> None:
    originals = {c.id: c for c in ingest_directory(FIXTURE_CORPUS)}
    for result in retrieve(LEAVE_QUERY, k=7, index=index):
        assert isinstance(result, RetrievedChunk)
        assert result.chunk == originals[result.chunk.id]


# --- corpus filter ---------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "corpus"),
    [
        (LEAVE_QUERY, "corpus_b"),  # topic lives in corpus_a; must not leak across
        (ROTATION_QUERY, "corpus_a"),  # topic lives in corpus_b
        (LEAVE_QUERY, "corpus_a"),
        (ROTATION_QUERY, "corpus_b"),
    ],
)
def test_corpus_filter_never_returns_other_corpus(index: Index, query: str, corpus: str) -> None:
    results = retrieve(query, k=10, corpus=corpus, index=index)
    assert results, "filtered query should still return chunks from its own corpus"
    assert {r.chunk.corpus for r in results} == {corpus}


def test_no_filter_searches_all_corpora(index: Index) -> None:
    results = retrieve(LEAVE_QUERY, k=10, index=index)
    assert {r.chunk.corpus for r in results} == {"corpus_a", "corpus_b"}


def test_unknown_corpus_returns_nothing(index: Index) -> None:
    assert retrieve(LEAVE_QUERY, corpus="corpus_zzz", index=index) == []


# --- index behaviour -------------------------------------------------------


def test_reindex_upserts_not_duplicates(index: Index) -> None:
    assert index.count() == 7
    index_directory(FIXTURE_CORPUS, index)
    assert index.count() == 7


def test_upsert_replaces_text(index: Index) -> None:
    replacement = Chunk(
        id="corpus_b:notes.txt:0",
        text="Replaced text about the cafeteria menu.",
        corpus="corpus_b",
        source="notes.txt",
        title="notes",
    )
    index.add([replacement])
    assert index.count() == 7
    top = retrieve("cafeteria menu", k=1, index=index)[0]
    assert top.chunk == replacement


def test_index_persists_on_disk(tmp_path: Path, index: Index, fake_embedder: FakeEmbedder) -> None:
    reopened = Index(tmp_path / "chroma", embedder=fake_embedder)
    assert reopened.count() == 7


def test_duplicate_ids_in_one_batch_rejected(index: Index) -> None:
    chunk = ingest_directory(FIXTURE_CORPUS)[0]
    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        index.add([chunk, chunk])


def test_empty_index_returns_nothing(tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
    empty = Index(tmp_path / "empty", embedder=fake_embedder)
    assert retrieve(LEAVE_QUERY, index=empty) == []


def test_k_must_be_positive(index: Index) -> None:
    with pytest.raises(ValueError):
        retrieve(LEAVE_QUERY, k=0, index=index)


def test_index_directory_counts_per_corpus(tmp_path: Path, fake_embedder: FakeEmbedder) -> None:
    idx = Index(tmp_path / "chroma", embedder=fake_embedder)
    assert index_directory(FIXTURE_CORPUS, idx) == {"corpus_a": 4, "corpus_b": 3}


# --- CLI -------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    chroma_dir = tmp_path / "cli_chroma"
    monkeypatch.setenv("RAGSEC_CHROMA_DIR", str(chroma_dir))
    monkeypatch.setattr(index_module, "default_embedder", FakeEmbedder)
    return chroma_dir


def test_cli_ingest_prints_counts_per_corpus(cli_env: Path) -> None:
    result = CliRunner().invoke(app, ["ingest", str(FIXTURE_CORPUS)])
    assert result.exit_code == 0, result.output
    assert f"Indexed 7 chunks into {cli_env}" in result.output
    assert "corpus_a: 4" in result.output
    assert "corpus_b: 3" in result.output
    assert Index(cli_env, embedder=FakeEmbedder()).count() == 7


def test_cli_ingest_empty_dir_fails(cli_env: Path, tmp_path: Path) -> None:
    empty = tmp_path / "nothing_here"
    empty.mkdir()
    result = CliRunner().invoke(app, ["ingest", str(empty)])
    assert result.exit_code == 1
    assert "No supported documents" in result.output


def test_cli_ingest_missing_dir_fails(cli_env: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["ingest", str(tmp_path / "missing")])
    assert result.exit_code != 0


# --- real model (slow; run with `pytest -m slow`) ---------------------------


@pytest.mark.slow
def test_real_embedder_ranking_and_filter(tmp_path: Path) -> None:
    idx = Index(tmp_path / "chroma_real")  # default: all-MiniLM-L6-v2
    index_directory(FIXTURE_CORPUS, idx)
    assert retrieve(LEAVE_QUERY, index=idx)[0].chunk.id == "corpus_a:handbook.md:1"
    assert retrieve(ROTATION_QUERY, index=idx)[0].chunk.id == "corpus_b:notes.txt:1"
    filtered = retrieve(LEAVE_QUERY, k=10, corpus="corpus_b", index=idx)
    assert {r.chunk.corpus for r in filtered} == {"corpus_b"}

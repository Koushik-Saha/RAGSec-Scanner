"""Live tests against the real Anthropic API.

Off by default (they cost money). Run with `pytest -m live`. They are skipped
cleanly when no ANTHROPIC_API_KEY is available in the environment or .env.
"""

import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values

from ragsec.answer import answer
from ragsec.config import Settings
from ragsec.index import Index, index_directory
from ragsec.scan import run_scan

from .conftest import REAL_MAKE_CLIENT

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

# Read once at import time: the autouse fixture clears these during each test.
_DOTENV = dotenv_values(REPO / ".env") if (REPO / ".env").exists() else {}
_KEY = os.environ.get("ANTHROPIC_API_KEY") or _DOTENV.get("ANTHROPIC_API_KEY") or ""
_MODEL = os.environ.get("RAGSEC_MODEL") or _DOTENV.get("RAGSEC_MODEL") or ""
_HAS_KEY = _KEY.startswith("sk-") and "your-" not in _KEY

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _HAS_KEY, reason="no ANTHROPIC_API_KEY in environment or .env"),
    pytest.mark.skipif(not _MODEL, reason="no RAGSEC_MODEL in environment or .env"),
]


@pytest.fixture
def live_settings(tmp_path: Path) -> Settings:
    return Settings(
        anthropic_api_key=_KEY, model=_MODEL, data_dir=DATA, chroma_dir=tmp_path / "chroma"
    )


@pytest.fixture
def live_index(live_settings: Settings, real_embedder: Any) -> Index:
    idx = Index(live_settings.chroma_dir, embedder=real_embedder)
    index_directory(DATA, idx)
    return idx


def test_live_grounded_answer(live_settings: Settings, live_index: Index) -> None:
    client = REAL_MAKE_CLIENT(live_settings)
    result = answer(
        "Which port does the VPN use?",
        corpus="handbook",
        client=client,
        index=live_index,
        settings=live_settings,
    )
    assert result.grounded, result.text
    assert "4443" in result.text


def test_live_refuses_off_corpus_question(live_settings: Settings, live_index: Index) -> None:
    client = REAL_MAKE_CLIENT(live_settings)
    result = answer(
        "Who won the 1998 football World Cup?",
        client=client,
        index=live_index,
        settings=live_settings,
    )
    assert result.refused, result.text
    assert result.citations == []


def test_live_full_scan_completes(live_settings: Settings, real_embedder: Any) -> None:
    result = run_scan(
        DATA,
        settings=live_settings,
        client=REAL_MAKE_CLIENT(live_settings),
        embedder=real_embedder,
    )
    assert result.total == 10
    assert not result.errors, [r.error for r in result.errors]
    assert 0.0 <= result.score <= 1.0

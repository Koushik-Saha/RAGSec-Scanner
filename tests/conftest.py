import hashlib
import math
import re
import socket
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ragsec.answer as answer_module
import ragsec.config as config_module
import ragsec.index as index_module

FIXTURE_CORPUS = Path(__file__).parent / "fixtures" / "corpus"

_STOPWORDS = set(
    "the and for are with that this from does how what when who any can you your get each per "
    "into its".split()
)


class FakeEmbedder:
    """Deterministic bag-of-words embedder for tests.

    Each content word is hashed (md5, so it is stable across runs) into one of
    `dims` buckets; the vector is L2-normalized. Texts sharing words land close
    together, which is enough to test retrieval ranking without downloading a model.
    """

    def __init__(self, dims: int = 512) -> None:
        self.dims = dims
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if len(word) < 3 or word in _STOPWORDS:
                continue
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


# --- isolation: tests never see real config or call the real API -------------

_RAGSEC_ENV = [
    "ANTHROPIC_API_KEY",
    "RAGSEC_MODEL",
    "RAGSEC_DATA_DIR",
    "RAGSEC_CHROMA_DIR",
    "RAGSEC_REPORTS_DIR",
]
REAL_MAKE_CLIENT = answer_module.make_client


REAL_LOAD_MODEL = index_module._load_sentence_transformer
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testserver"}
_REAL_CONNECT = socket.socket.connect


def _needs_outside_world(request: pytest.FixtureRequest) -> bool:
    return any(request.node.get_closest_marker(m) for m in ("slow", "live"))


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Every test runs without real config. Ordinary tests also cannot reach the
    network or load the real embedding model; only `slow`/`live` tests may."""
    for name in _RAGSEC_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config_module, "load_dotenv", lambda **_: False)

    def _no_real_client(*_: Any, **__: Any) -> Any:
        raise AssertionError("tests must not build a real Anthropic client; inject FakeClient")

    monkeypatch.setattr(answer_module, "make_client", _no_real_client)

    if _needs_outside_world(request):
        return

    def _no_real_model(*_: Any, **__: Any) -> Any:
        raise AssertionError(
            "ordinary tests must not load the real embedding model; use FakeEmbedder"
        )

    def _local_connect_only(self: socket.socket, address: Any) -> Any:
        if isinstance(address, tuple) and address and address[0] not in _LOCAL_HOSTS:
            raise AssertionError(f"ordinary tests must not open network connections ({address[0]})")
        return _REAL_CONNECT(self, address)

    monkeypatch.setattr(index_module, "_load_sentence_transformer", _no_real_model)
    monkeypatch.setattr(socket.socket, "connect", _local_connect_only)


@pytest.fixture
def real_embedder() -> Any:
    """The real all-MiniLM-L6-v2 embedder, or a clean skip if it can't be loaded
    (e.g. a fresh clone with no network and no cached model)."""
    embedder = index_module.SentenceTransformerEmbedder()
    try:
        embedder.embed(["warm-up"])
    except Exception as exc:  # noqa: BLE001 - any load failure means "skip"
        pytest.skip(f"embedding model unavailable: {type(exc).__name__}")
    return embedder


# --- fake Anthropic client ---------------------------------------------------


class FakeMessages:
    def __init__(
        self,
        text: "str | Callable[[dict[str, Any]], str]",
        stop_reason: str,
        extra_blocks: list[Any],
    ) -> None:
        self.text = text
        self.stop_reason = stop_reason
        self.extra_blocks = extra_blocks
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        text = self.text(kwargs) if callable(self.text) else self.text
        blocks = [*self.extra_blocks, SimpleNamespace(type="text", text=text)]
        return SimpleNamespace(content=blocks, stop_reason=self.stop_reason)


class FakeClient:
    """Stands in for anthropic.Anthropic: records each messages.create call.

    `text` is the reply, or a function of the request kwargs that returns it.
    """

    def __init__(
        self,
        text: "str | Callable[[dict[str, Any]], str]",
        stop_reason: str = "end_turn",
        extra_blocks: list[Any] | None = None,
    ) -> None:
        self.messages = FakeMessages(text, stop_reason, extra_blocks or [])

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls


def context_chunk_ids(request: dict[str, Any]) -> list[str]:
    """Chunk IDs the model was shown in a messages.create request, in order."""
    content = request["messages"][0]["content"]
    return re.findall(r'<chunk number="\d+" id="([^"]+)">', content)

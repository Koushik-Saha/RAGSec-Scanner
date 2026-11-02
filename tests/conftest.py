import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path

import pytest

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

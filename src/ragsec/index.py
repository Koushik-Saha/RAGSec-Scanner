"""Embedding and storage of chunks in a persistent Chroma collection."""

from collections import Counter
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings

from ragsec.config import COLLECTION_NAME, EMBEDDING_MODEL
from ragsec.ingest import Chunk, ingest_directory

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

UPSERT_BATCH_SIZE = 500


class Embedder(Protocol):
    """Anything that turns texts into fixed-length vectors."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@cache
def _load_sentence_transformer(model_name: str) -> "SentenceTransformer":
    # Imported here so the CLI and tests don't pay for loading torch unless needed.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SentenceTransformerEmbedder:
    """Local sentence-transformers embedder. The model loads on first use and is cached."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = _load_sentence_transformer(self.model_name)
        vectors = model.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()


def default_embedder() -> Embedder:
    return SentenceTransformerEmbedder()


class Index:
    """A persistent Chroma collection of chunks, keyed by chunk ID."""

    def __init__(
        self,
        chroma_dir: str | Path,
        embedder: Embedder | None = None,
        collection_name: str = COLLECTION_NAME,
    ) -> None:
        self.embedder = embedder or default_embedder()
        self._client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            collection_name,
            embedding_function=None,  # we always pass embeddings ourselves
            configuration={"hnsw": {"space": "cosine"}},
        )

    def add(self, chunks: Sequence[Chunk]) -> int:
        """Embed and upsert chunks. Re-adding an existing chunk ID replaces it."""
        ids = [c.id for c in chunks]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate chunk IDs: {dupes}")

        for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[start : start + UPSERT_BATCH_SIZE]
            self._collection.upsert(
                ids=[c.id for c in batch],
                embeddings=self.embedder.embed([c.text for c in batch]),
                documents=[c.text for c in batch],
                metadatas=[
                    {"corpus": c.corpus, "source": c.source, "title": c.title} for c in batch
                ],
            )
        return len(chunks)

    def count(self) -> int:
        return self._collection.count()

    def search(self, query: str, k: int, corpus: str | None = None) -> list[tuple[Chunk, float]]:
        """Nearest chunks to `query`, closest first, as (chunk, cosine distance) pairs."""
        if k < 1:
            raise ValueError("k must be at least 1")
        if self.count() == 0:
            return []

        result: Any = self._collection.query(
            query_embeddings=self.embedder.embed([query]),
            n_results=k,
            where={"corpus": corpus} if corpus is not None else None,
            include=["documents", "metadatas", "distances"],
        )
        hits = zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        )
        return [
            (
                Chunk(
                    id=chunk_id,
                    text=text,
                    corpus=meta["corpus"],
                    source=meta["source"],
                    title=meta["title"],
                ),
                float(distance),
            )
            for chunk_id, text, meta, distance in hits
        ]


def index_directory(path: str | Path, index: Index) -> dict[str, int]:
    """Ingest every document under `path`, upsert into `index`, return chunk counts per corpus."""
    chunks = ingest_directory(path)
    index.add(chunks)
    return dict(sorted(Counter(c.corpus for c in chunks).items()))

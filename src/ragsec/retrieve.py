"""Vector retrieval over the Chroma index, optionally filtered to one corpus."""

from pydantic import BaseModel

from ragsec.config import load_settings
from ragsec.index import Index
from ragsec.ingest import Chunk


class RetrievedChunk(BaseModel):
    chunk: Chunk
    distance: float  # cosine distance: 0 = identical direction, lower is closer


def retrieve(
    query: str,
    k: int = 5,
    corpus: str | None = None,
    *,
    index: Index | None = None,
) -> list[RetrievedChunk]:
    """Return the `k` chunks closest to `query`, closest first.

    With `corpus`, only chunks from that corpus are searched. Without `index`,
    the persistent index at RAGSEC_CHROMA_DIR is used.
    """
    if index is None:
        index = Index(load_settings().chroma_dir)
    return [
        RetrievedChunk(chunk=chunk, distance=distance)
        for chunk, distance in index.search(query, k=k, corpus=corpus)
    ]

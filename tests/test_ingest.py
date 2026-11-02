from pathlib import Path

import pytest

from ragsec.ingest import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    Chunk,
    chunk_text,
    ingest_directory,
    load_text,
)

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"


def overlap_len(prev: str, nxt: str) -> int:
    """Length of the longest suffix of `prev` that `nxt` starts with."""
    for k in range(min(len(prev), len(nxt)), 0, -1):
        if nxt.startswith(prev[-k:]):
            return k
    return 0


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return ingest_directory(FIXTURES)


def by_source(chunks: list[Chunk], source: str) -> list[Chunk]:
    return [c for c in chunks if c.source == source]


# --- ingest_directory -------------------------------------------------------


def test_chunk_counts(chunks: list[Chunk]) -> None:
    assert len(by_source(chunks, "handbook.md")) == 3
    assert len(by_source(chunks, "policy.pdf")) == 1
    assert len(by_source(chunks, "notes.txt")) == 3
    assert len(chunks) == 7


def test_stable_ids(chunks: list[Chunk]) -> None:
    assert [c.id for c in chunks] == [
        "corpus_a:handbook.md:0",
        "corpus_a:handbook.md:1",
        "corpus_a:handbook.md:2",
        "corpus_a:policy.pdf:0",
        "corpus_b:notes.txt:0",
        "corpus_b:notes.txt:1",
        "corpus_b:notes.txt:2",
    ]
    again = ingest_directory(FIXTURES)
    assert [c.model_dump() for c in again] == [c.model_dump() for c in chunks]


def test_metadata(chunks: list[Chunk]) -> None:
    expected = {
        "handbook.md": ("corpus_a", "Example Corp Employee Handbook"),
        "policy.pdf": ("corpus_a", "policy"),
        "notes.txt": ("corpus_b", "notes"),
    }
    for chunk in chunks:
        corpus, title = expected[chunk.source]
        assert chunk.corpus == corpus
        assert chunk.title == title
        assert chunk.id.startswith(f"{corpus}:{chunk.source}:")


def test_pdf_text_extracted(chunks: list[Chunk]) -> None:
    (pdf_chunk,) = by_source(chunks, "policy.pdf")
    assert "Travel expenses are reimbursed within thirty days." in pdf_chunk.text


def test_size_bounds(chunks: list[Chunk]) -> None:
    for chunk in chunks:
        if len(chunk.text) > CHUNK_SIZE:
            # Only allowed when one paragraph alone is longer than the limit.
            assert any(len(p) > CHUNK_SIZE for p in paragraphs(chunk.text)), chunk.id


def test_oversized_paragraph_kept_whole(chunks: list[Chunk]) -> None:
    source = (FIXTURES / "corpus_b" / "notes.txt").read_text()
    long_para = max(paragraphs(source), key=len)
    assert len(long_para) > CHUNK_SIZE
    assert any(c.text == long_para for c in by_source(chunks, "notes.txt"))


@pytest.mark.parametrize("source", ["handbook.md", "notes.txt"])
def test_overlap_between_consecutive_chunks(chunks: list[Chunk], source: str) -> None:
    doc = by_source(chunks, source)
    for prev, nxt in zip(doc, doc[1:], strict=False):
        if len(nxt.text) > CHUNK_SIZE:
            continue  # an oversized paragraph has no room for overlap
        k = overlap_len(prev.text, nxt.text)
        assert CHUNK_OVERLAP // 2 <= k <= CHUNK_OVERLAP, (prev.id, nxt.id, k)


def test_all_source_text_is_covered(chunks: list[Chunk]) -> None:
    source = (FIXTURES / "corpus_a" / "handbook.md").read_text()
    joined = "\n\n".join(c.text for c in by_source(chunks, "handbook.md"))
    for para in paragraphs(source):
        assert para in joined


def test_skips_unsupported_and_hidden_files(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus_x"
    corpus.mkdir()
    (corpus / "doc.txt").write_text("Hello.")
    (corpus / "image.png").write_bytes(b"\x89PNG")
    (corpus / ".hidden.md").write_text("# Secret\n\nSkip me.")
    hidden_dir = tmp_path / ".cache"
    hidden_dir.mkdir()
    (hidden_dir / "cached.txt").write_text("Skip me too.")

    result = ingest_directory(tmp_path)
    assert [c.id for c in result] == ["corpus_x:doc.txt:0"]


def test_markdown_without_heading_uses_filename(tmp_path: Path) -> None:
    (tmp_path / "plain.md").write_text("No heading here.")
    (chunk,) = ingest_directory(tmp_path)
    assert chunk.title == "plain"
    assert chunk.corpus == tmp_path.name


def test_empty_file_yields_no_chunks(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("\n\n  \n")
    assert ingest_directory(tmp_path) == []


def test_not_a_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        ingest_directory(tmp_path / "missing")


def test_load_text_normalizes_line_endings(tmp_path: Path) -> None:
    file = tmp_path / "crlf.txt"
    file.write_bytes(b"one\r\n\r\ntwo\rthree")
    assert load_text(file) == "one\n\ntwo\nthree"


# --- chunk_text -------------------------------------------------------------


def make_para(n: int, word: str = "word") -> str:
    """A paragraph of roughly `n` characters made of repeated words."""
    text = " ".join([word] * (n // (len(word) + 1) + 1))
    return text[:n].rstrip()


def test_chunk_text_packs_paragraphs() -> None:
    text = "\n\n".join(make_para(300, w) for w in ["alpha", "bravo", "charlie", "delta"])
    result = chunk_text(text)
    assert len(result) == 2
    assert result[0].startswith("alpha") and result[0].endswith("bravo")
    assert "charlie" in result[1] and result[1].endswith("delta")
    # Second chunk opens with the tail of the first.
    assert result[1].startswith("bravo")
    assert 0 < overlap_len(result[0], result[1]) <= CHUNK_OVERLAP


def test_chunk_text_short_text_single_chunk() -> None:
    assert chunk_text("One paragraph.\n\nTwo paragraph.") == ["One paragraph.\n\nTwo paragraph."]


def test_chunk_text_overlap_cut_on_word_boundary() -> None:
    text = make_para(500, "lorem") + "\n\n" + make_para(500, "ipsum")
    first, second = chunk_text(text)
    overlap = second.split("\n\n")[0]
    assert overlap.split(" ")[0] == "lorem"  # whole word, not a fragment
    assert first.endswith(overlap)


def test_chunk_text_overlap_shrinks_to_fit() -> None:
    # A ~750-char paragraph leaves under 50 chars of room for the overlap.
    big = make_para(750, "bbbb")
    first, second = chunk_text(make_para(500, "aaaa") + "\n\n" + big)
    assert len(second) <= CHUNK_SIZE
    assert 0 < overlap_len(first, second) <= CHUNK_SIZE - len(big) - 2


def test_chunk_text_no_overlap_before_oversized_paragraph() -> None:
    big = make_para(1000, "huge")
    result = chunk_text("intro paragraph\n\n" + big)
    assert result == ["intro paragraph", big]


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []

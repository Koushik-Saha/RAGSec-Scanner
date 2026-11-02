"""Document ingestion: loading and chunking documents.

Files are read from a directory tree. Each file's corpus is the name of the
folder it sits in (e.g. data/corpus_a/handbook.md -> corpus "corpus_a").
"""

import re
from pathlib import Path

from pydantic import BaseModel
from pypdf import PdfReader

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

TEXT_SUFFIXES = {".md", ".txt"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES
# Documentation about a corpus, not part of it.
SKIPPED_NAMES = {"readme.md", "readme.txt"}

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_MD_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_PARAGRAPH_SEP = "\n\n"


class Chunk(BaseModel):
    id: str
    text: str
    corpus: str
    source: str
    title: str


def ingest_directory(
    path: str | Path,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Load every supported file under `path` and return its chunks.

    Files are processed in sorted path order so output is deterministic.
    Hidden files and folders (names starting with ".") and README files are skipped.
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")

    chunks: list[Chunk] = []
    for file in sorted(root.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if file.name.lower() in SKIPPED_NAMES:
            continue
        if any(part.startswith(".") for part in file.relative_to(root).parts):
            continue
        chunks.extend(ingest_file(file, chunk_size=chunk_size, overlap=overlap))
    return chunks


def ingest_file(
    file: Path,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Load one file and split it into chunks with stable IDs."""
    text = load_text(file)
    corpus = file.parent.name
    title = extract_title(file, text)
    return [
        Chunk(
            id=f"{corpus}:{file.name}:{index}",
            text=piece,
            corpus=corpus,
            source=file.name,
            title=title,
        )
        for index, piece in enumerate(chunk_text(text, chunk_size, overlap))
    ]


def load_text(file: Path) -> str:
    suffix = file.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = file.read_text(encoding="utf-8")
    elif suffix in PDF_SUFFIXES:
        reader = PdfReader(file)
        text = _PARAGRAPH_SEP.join(page.extract_text() or "" for page in reader.pages)
    else:
        raise ValueError(f"unsupported file type: {file.name}")
    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_title(file: Path, text: str) -> str:
    """First Markdown heading for .md files; the filename stem otherwise."""
    if file.suffix.lower() == ".md":
        match = _MD_HEADING.search(text)
        if match:
            return match.group(1).strip()
    return file.stem


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of at most `chunk_size` characters.

    Paragraphs (separated by blank lines) are packed greedily, so chunks end on
    paragraph boundaries. Each new chunk starts with the last ~`overlap`
    characters of the previous one, cut at a word boundary. The overlap is
    shortened, or dropped, when needed to keep the chunk within `chunk_size`.

    A single paragraph longer than `chunk_size` becomes its own chunk and is
    allowed to exceed the limit.
    """
    paragraphs = [p.strip() for p in _PARAGRAPH_BREAK.split(text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}{_PARAGRAPH_SEP}{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        tail = ""
        if current:
            chunks.append(current)
            room = chunk_size - len(para) - len(_PARAGRAPH_SEP)
            tail = _tail(current, min(overlap, room))
        current = f"{tail}{_PARAGRAPH_SEP}{para}" if tail else para

    if current:
        chunks.append(current)
    return chunks


def _tail(text: str, max_chars: int) -> str:
    """Last `max_chars` characters of `text`, trimmed forward to a word boundary."""
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    if not text[-max_chars - 1].isspace():
        boundary = re.search(r"\s", tail)
        if boundary:
            tail = tail[boundary.end() :]
    return tail.strip()

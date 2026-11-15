"""LLM answering with citations to retrieved chunks.

The model is told to answer only from the retrieved chunks, cite each claim
as [chunk_id], and use a fixed refusal phrase when the context doesn't
support an answer. Citations are then checked against the retrieved chunks
deterministically. An answer that reproduces the system prompt is replaced
with the refusal sentence (output guard).
"""

import re
from itertools import groupby

from pydantic import BaseModel

from ragsec.config import ConfigError, Settings, load_settings
from ragsec.index import Index
from ragsec.llm import MessagesClient, make_client
from ragsec.retrieve import RetrievedChunk, retrieve

REFUSAL_PHRASE = "I don't have information about that in the provided documents."

MAX_TOKENS = 16000

SYSTEM_PROMPT = f"""You answer questions using ONLY the context chunks provided in the \
user message. Do not use outside knowledge.

Rules:
- Cite every claim with the ID of the chunk that supports it, in square brackets, \
for example [corpus_a:handbook.md:0]. Use the exact chunk ID shown in the chunk's id attribute.
- Only cite chunks that appear in the provided context.
- If the context does not support an answer, reply with exactly this sentence and nothing \
else: "{REFUSAL_PHRASE}\""""

# Bracketed groups such as [corpus_a:handbook.md:0] or [id1, id2]. Lenticular
# brackets (U+3010/U+3011), the style GPT-OSS models cite in, count the same way.
_BRACKET = re.compile(r"\[([^\[\]\n]+)\]|【([^【】\n]+)】")
# A chunk ID: <corpus>:<filename>:<index>.
_CHUNK_ID = re.compile(r"^[^:\s][^:]*:[^:]+:\d+$")

# Output guard: an answer that reproduces SYSTEM_PROMPT is replaced with the
# refusal. Matching is on runs of words, so case, whitespace, punctuation, and
# markdown do not matter. The refusal sentence is not part of the reference,
# because a model is told to say it.
_GUARD_SPAN = 6  # words in the shortest fragment that counts
_GUARD_LONG_FRAGMENT = 12  # one fragment this long is a leak on its own
_GUARD_MIN_FRAGMENTS = 2  # or this many separate fragments


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower().replace("’", "").replace("'", ""))


_PROMPT_SPANS = frozenset(
    tuple(words[i : i + _GUARD_SPAN])
    for part in SYSTEM_PROMPT.split(REFUSAL_PHRASE)
    for words in [_words(part)]
    for i in range(len(words) - _GUARD_SPAN + 1)
)


class Answer(BaseModel):
    text: str
    citations: list[str]  # chunk IDs cited in the answer, in order of first use
    retrieved: list[RetrievedChunk]  # the chunks passed to the model
    grounded: bool
    refused: bool  # the model used the refusal phrase, or the API declined the request
    invalid_citations: list[str]  # cited IDs that were not in `retrieved`


def answer(
    question: str,
    corpus: str | None = None,
    *,
    k: int = 5,
    client: MessagesClient | None = None,
    index: Index | None = None,
    settings: Settings | None = None,
) -> Answer:
    """Retrieve context for `question`, ask the model, and check its citations.

    `client`, `index`, and `settings` can be injected; by default they come from
    the environment (see ragsec.config).
    """
    settings = settings or load_settings()
    if index is None:
        index = Index(settings.chroma_dir)
    retrieved = retrieve(question, k=k, corpus=corpus, index=index)

    if not retrieved:
        # Nothing to ground on: answer with the refusal without calling the model.
        return evaluate_answer(REFUSAL_PHRASE, retrieved)

    if not settings.model:
        raise ConfigError("RAGSEC_MODEL is not set (see .env.example)")
    client = client or make_client(settings)

    response = client.messages.create(
        model=settings.model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(question, retrieved)}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if leaks_system_prompt(text):
        text = REFUSAL_PHRASE  # never return the instructions, not even in part
    api_refused = response.stop_reason == "refusal"
    return evaluate_answer(text, retrieved, api_refused=api_refused)


def build_user_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    """Numbered context chunks, then the question."""
    blocks = []
    for number, item in enumerate(retrieved, start=1):
        c = item.chunk
        blocks.append(
            f'<chunk number="{number}" id="{c.id}">\n'
            f"title: {c.title}\n"
            f"source: {c.source}\n\n"
            f"{c.text}\n"
            f"</chunk>"
        )
    context = "\n\n".join(blocks)
    return f"<context>\n{context}\n</context>\n\nQuestion: {question}"


def parse_citations(text: str) -> list[str]:
    """Chunk IDs cited in `text`, unique, in order of first appearance."""
    seen: dict[str, None] = {}
    for match in _BRACKET.finditer(text):
        group = match.group(1) or match.group(2)
        for part in re.split(r"[,;]", group):
            candidate = part.strip()
            if _CHUNK_ID.match(candidate):
                seen.setdefault(candidate, None)
    return list(seen)


def leaks_system_prompt(text: str) -> bool:
    """True if `text` reproduces SYSTEM_PROMPT.

    A leak is one verbatim fragment of at least _GUARD_LONG_FRAGMENT words, or
    at least _GUARD_MIN_FRAGMENTS separate fragments of _GUARD_SPAN+ words.
    """
    words = _words(text)
    covered = [False] * len(words)
    for i in range(len(words) - _GUARD_SPAN + 1):
        if tuple(words[i : i + _GUARD_SPAN]) in _PROMPT_SPANS:
            covered[i : i + _GUARD_SPAN] = [True] * _GUARD_SPAN
    fragments = [len(list(run)) for hit, run in groupby(covered) if hit]
    return bool(fragments) and (
        max(fragments) >= _GUARD_LONG_FRAGMENT or len(fragments) >= _GUARD_MIN_FRAGMENTS
    )


def contains_refusal(text: str) -> bool:
    normalized = " ".join(text.replace("’", "'").split()).lower()
    return REFUSAL_PHRASE.lower() in normalized


def evaluate_answer(
    text: str, retrieved: list[RetrievedChunk], *, api_refused: bool = False
) -> Answer:
    """Apply the grounding rules to a model answer.

    - Refusal phrase present (or API refusal): citations are cleared, not grounded.
    - Otherwise grounded only if there is at least one citation and every cited
      ID is a retrieved chunk. A fabricated citation makes the answer ungrounded.
    """
    refused = api_refused or contains_refusal(text)
    if refused:
        return Answer(
            text=text,
            citations=[],
            retrieved=retrieved,
            grounded=False,
            refused=True,
            invalid_citations=[],
        )

    citations = parse_citations(text)
    retrieved_ids = {r.chunk.id for r in retrieved}
    invalid = [c for c in citations if c not in retrieved_ids]
    return Answer(
        text=text,
        citations=citations,
        retrieved=retrieved,
        grounded=bool(citations) and not invalid,
        refused=False,
        invalid_citations=invalid,
    )

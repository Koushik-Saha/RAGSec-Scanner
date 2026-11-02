"""LLM answering with citations to retrieved chunks.

The model is told to answer only from the retrieved chunks, cite each claim
as [chunk_id], and use a fixed refusal phrase when the context doesn't
support an answer. Citations are then checked against the retrieved chunks
deterministically.
"""

import re
from typing import Any, Protocol

from pydantic import BaseModel

from ragsec.config import Settings, load_settings
from ragsec.index import Index
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

# Bracketed groups such as [corpus_a:handbook.md:0] or [id1, id2].
_BRACKET = re.compile(r"\[([^\[\]\n]+)\]")
# A chunk ID: <corpus>:<filename>:<index>.
_CHUNK_ID = re.compile(r"^[^:\s][^:]*:[^:]+:\d+$")


class ConfigError(RuntimeError):
    """Required configuration (API key or model) is missing."""


class MessagesClient(Protocol):
    """The slice of the Anthropic client we use; tests pass a fake with the same shape."""

    @property
    def messages(self) -> Any: ...


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
    api_refused = response.stop_reason == "refusal"
    return evaluate_answer(text, retrieved, api_refused=api_refused)


def make_client(settings: Settings) -> MessagesClient:
    """Anthropic client using the key from config only (never another credential source)."""
    if settings.anthropic_api_key is None:
        raise ConfigError("ANTHROPIC_API_KEY is not set (see .env.example)")
    import anthropic

    return anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())


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
    for group in _BRACKET.findall(text):
        for part in re.split(r"[,;]", group):
            candidate = part.strip()
            if _CHUNK_ID.match(candidate):
                seen.setdefault(candidate, None)
    return list(seen)


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

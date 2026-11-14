"""Model client factory: the one place that picks the LLM provider.

RAGSEC_PROVIDER selects the provider (default: anthropic). Every provider is
returned behind the same small interface, `client.messages.create(...)` with
the Anthropic Messages shape, so answering, probes, verdicts, and reports do
not change with the provider. Nothing outside this module imports a provider
SDK to build a client or send a message.
"""

import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
import openai

from ragsec.config import ConfigError, Settings


class MessagesClient(Protocol):
    """The slice of the Anthropic client we use; tests pass a fake with the same shape."""

    @property
    def messages(self) -> Any: ...


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    base_url: str
    key_env: str | None  # env var holding the key; None means no real key is needed
    key_field: str | None  # the Settings field that holds that key
    fixed_key: str | None = None  # used when key_env is None


OPENAI_COMPATIBLE: dict[str, OpenAICompatibleProvider] = {
    "groq": OpenAICompatibleProvider(
        "https://api.groq.com/openai/v1", "GROQ_API_KEY", "groq_api_key"
    ),
    "openrouter": OpenAICompatibleProvider(
        "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openrouter_api_key"
    ),
    "ollama": OpenAICompatibleProvider("http://localhost:11434/v1", None, None, fixed_key="ollama"),
}
PROVIDERS = ("anthropic", *OPENAI_COMPATIBLE)

# Errors from any provider's API. Callers catch these instead of importing an SDK.
API_ERRORS: tuple[type[Exception], ...] = (anthropic.APIError, openai.APIError)
AUTH_ERRORS: tuple[type[Exception], ...] = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
)
RATE_LIMIT_ERRORS: tuple[type[Exception], ...] = (anthropic.RateLimitError, openai.RateLimitError)
# Errors that hit every request the same way (bad key, unknown model).
FATAL_API_ERRORS: tuple[type[Exception], ...] = (
    *AUTH_ERRORS,
    anthropic.NotFoundError,
    openai.NotFoundError,
)

# Rate limits for OpenAI-compatible providers (free tiers such as Groq's).
MIN_REQUEST_INTERVAL = 2.1  # seconds between requests: stays under 30 per minute
RATE_LIMIT_RETRIES = 6  # retries for one request after HTTP 429
BACKOFF_BASE = 2.0  # seconds; doubles each retry when the 429 has no retry-after
BACKOFF_MAX = 60.0  # a longer wait than this means the budget is spent for now
BUDGET_COOLDOWN = 900.0  # seconds to stop calling after a spent budget with no retry-after
# Groq names the window in its 429 message, e.g. "tokens per day (TPD)".
_DAILY_LIMIT = re.compile(r"per day|\((?:TPD|RPD)\)", re.IGNORECASE)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def make_client(settings: Settings) -> MessagesClient:
    """Client for the configured provider, with its key from config only.

    Raises ConfigError naming the missing variable (never its value) when the
    selected provider's key is not set, or when the provider is unknown.
    """
    provider = settings.provider
    if provider == "anthropic":
        if settings.anthropic_api_key is None:
            raise ConfigError("ANTHROPIC_API_KEY is not set (see .env.example)")
        return anthropic.Anthropic(api_key=settings.anthropic_api_key.get_secret_value())

    spec = OPENAI_COMPATIBLE.get(provider)
    if spec is None:
        raise ConfigError(
            f"RAGSEC_PROVIDER {provider!r} is not supported; use one of: {', '.join(PROVIDERS)}"
        )
    if spec.key_env is None or spec.key_field is None:
        key = spec.fixed_key
    else:
        secret = getattr(settings, spec.key_field)
        if secret is None:
            raise ConfigError(
                f"{spec.key_env} is not set but RAGSEC_PROVIDER is {provider!r} (see .env.example)"
            )
        key = secret.get_secret_value()
    # Retries are handled by the adapter (pacing, backoff, daily-budget stop).
    raw = openai.OpenAI(base_url=spec.base_url, api_key=key, max_retries=0)
    return OpenAICompatibleClient(
        raw,
        min_interval=MIN_REQUEST_INTERVAL,
        max_retries=RATE_LIMIT_RETRIES,
        sleep=_sleep,
    )


# --- OpenAI-compatible adapter --------------------------------------------------


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _Response:
    content: list[_TextBlock]
    stop_reason: str


class _ChatMessages:
    """messages.create() in the Anthropic shape, sent as an OpenAI chat completion.

    Requests are spaced at least `min_interval` seconds apart, across threads
    (the API shares one client between concurrent requests). On HTTP 429 the
    request waits (retry-after, or exponential backoff) and retries up to
    `max_retries` times. A daily-budget 429, or one asking to wait longer than
    BACKOFF_MAX, is raised at once, and later requests raise it again without
    calling the API until its retry-after (or BUDGET_COOLDOWN) has passed, so a
    scan records errors instead of hammering a spent budget.
    """

    def __init__(
        self,
        client: Any,
        *,
        min_interval: float,
        max_retries: int,
        sleep: Callable[[float], None],
        clock: Callable[[], float],
    ) -> None:
        self._client = client
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._next_slot: float | None = None  # earliest start for the next request
        self.budget_exhausted: openai.RateLimitError | None = None
        self._exhausted_until = 0.0

    def _pace(self) -> None:
        """Reserve the next request slot, then wait for it (outside the lock)."""
        with self._lock:
            now = self._clock()
            start = now if self._next_slot is None else max(now, self._next_slot)
            self._next_slot = start + self._min_interval
        if start > now:
            self._sleep(start - now)

    def _check_budget(self) -> None:
        with self._lock:
            if self.budget_exhausted is None:
                return
            if self._clock() < self._exhausted_until:
                raise self.budget_exhausted
            self.budget_exhausted = None  # cooldown over: try the API again

    def _send(self, **kwargs: Any) -> Any:
        self._check_budget()
        attempt = 0
        while True:
            self._pace()
            try:
                return self._client.chat.completions.create(**kwargs)
            except openai.RateLimitError as exc:
                wait = _retry_after(exc)
                if _DAILY_LIMIT.search(str(exc)) or (wait is not None and wait > BACKOFF_MAX):
                    with self._lock:
                        self.budget_exhausted = exc
                        self._exhausted_until = self._clock() + (
                            wait if wait is not None else BUDGET_COOLDOWN
                        )
                    raise
                if attempt >= self._max_retries:
                    raise
                self._sleep(
                    wait if wait is not None else min(BACKOFF_BASE * 2**attempt, BACKOFF_MAX)
                )
                attempt += 1

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
    ) -> _Response:
        completion = self._send(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, *messages],
        )
        if not completion.choices:
            return _Response(content=[_TextBlock("")], stop_reason="end_turn")
        choice = completion.choices[0]
        text = choice.message.content or ""
        # A provider-side refusal or content filter maps to the Anthropic
        # "refusal" stop reason, which answer() treats as the API declining.
        declined = bool(getattr(choice.message, "refusal", None)) or (
            choice.finish_reason == "content_filter"
        )
        return _Response(
            content=[_TextBlock(text)], stop_reason="refusal" if declined else "end_turn"
        )


def _retry_after(exc: openai.RateLimitError) -> float | None:
    """Seconds the provider asked us to wait, if it said."""
    value = exc.response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class OpenAICompatibleClient:
    """Wraps an openai.OpenAI client so it looks like the Anthropic client to answer()."""

    def __init__(
        self,
        client: Any,
        *,
        min_interval: float = 0.0,
        max_retries: int = 0,
        sleep: Callable[[float], None] = _sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.openai_client = client
        self.messages = _ChatMessages(
            client, min_interval=min_interval, max_retries=max_retries, sleep=sleep, clock=clock
        )

"""Model client factory: the one place that picks the LLM provider.

RAGSEC_PROVIDER selects the provider (default: anthropic). Every provider is
returned behind the same small interface, `client.messages.create(...)` with
the Anthropic Messages shape, so answering, probes, verdicts, and reports do
not change with the provider. Nothing outside this module imports a provider
SDK to build a client or send a message.
"""

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
    return OpenAICompatibleClient(openai.OpenAI(base_url=spec.base_url, api_key=key))


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
    """messages.create() in the Anthropic shape, sent as an OpenAI chat completion."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
    ) -> _Response:
        completion = self._client.chat.completions.create(
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


class OpenAICompatibleClient:
    """Wraps an openai.OpenAI client so it looks like the Anthropic client to answer()."""

    def __init__(self, client: Any) -> None:
        self.openai_client = client
        self.messages = _ChatMessages(client)

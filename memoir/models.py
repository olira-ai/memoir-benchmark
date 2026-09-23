"""Chat models for the answering agent and the rubric judge.

MEMOIR does not bundle a provider or pick a model for you. You supply an
endpoint, a key, and a model name. Anything speaking the OpenAI chat-completions
wire protocol works, which covers most hosted APIs and most self-hosted servers
(vLLM, Ollama, LiteLLM, text-generation-inference, and the commercial endpoints).

Two roles are configured independently, so the judge can run on a smaller and
cheaper model than the agent:

============================  ===============================================
``MEMOIR_LLM_BASE_URL``       endpoint both roles use
``MEMOIR_LLM_API_KEY``        key both roles use
``MEMOIR_AGENT_MODEL``        the model that answers questions
``MEMOIR_JUDGE_MODEL``        the model that grades the open-ended tasks
``MEMOIR_AGENT_BASE_URL``     endpoint override for the agent alone
``MEMOIR_AGENT_API_KEY``      key override for the agent alone
``MEMOIR_JUDGE_BASE_URL``     endpoint override for the judge alone
``MEMOIR_JUDGE_API_KEY``      key override for the judge alone
``MEMOIR_AGENT_MODEL_KWARGS`` JSON of extra parameters for the agent
``MEMOIR_JUDGE_MODEL_KWARGS`` JSON of extra parameters for the judge
============================  ===============================================

Decoding should be held as fixed as your model family allows, and what that means
differs by family: ``{"temperature": 0}`` for most, ``{"reasoning_effort": "low"}``
for a reasoning model that rejects temperature. Set it explicitly through the
``*_MODEL_KWARGS`` variables rather than relying on a default, and report what you
used alongside any number you publish.

If none of this suits, skip it entirely and hand in a model you built yourself.
Anything implementing LangChain's ``BaseChatModel`` is accepted::

    BenchmarkAgent(substrate, patient_id, llm=my_chat_model)
    await score_answer(question, answer, judge_llm=my_small_chat_model)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Endpoints burst their per-minute limits at high run concurrency.
_RATE_LIMIT_ATTEMPTS = 8
_RATE_LIMIT_BASE_DELAY_S = 15.0
_RATE_LIMIT_MAX_DELAY_S = 120.0


def _first_env(*names: str) -> str:
    for n in names:
        v = (os.environ.get(n) or "").strip()
        if v:
            return v
    return ""


def _json_env(name: str) -> dict[str, Any]:
    raw = _first_env(name)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{name} must be a JSON object, got {type(parsed).__name__}")
    return parsed


def agent_model() -> str:
    """Model id for the answering agent."""
    model = _first_env("MEMOIR_AGENT_MODEL")
    if not model:
        raise RuntimeError(
            "Set MEMOIR_AGENT_MODEL to the model that should answer questions, "
            "or pass llm= to BenchmarkAgent."
        )
    return model


def judge_model() -> str:
    """Model id for the rubric judge."""
    model = _first_env("MEMOIR_JUDGE_MODEL")
    if not model:
        raise RuntimeError(
            "Set MEMOIR_JUDGE_MODEL to the model that should grade open-ended "
            "answers, or pass judge_llm= to score_answer."
        )
    return model


# --- rate limiting -----------------------------------------------------------


def is_rate_limit_error(exc: BaseException, _depth: int = 0) -> bool:
    """True for a 429 or rate-limit error, including leaves of an ExceptionGroup."""
    if _depth > 6:
        return False
    if "RateLimit" in type(exc).__name__:
        return True
    text = str(exc)
    if any(
        marker in text
        for marker in (
            "rate_limit_exceeded",
            "RateLimitError",
            "Error code: 429",
            "too_many_requests",
            "Too Many Requests",
        )
    ):
        return True
    nested = getattr(exc, "exceptions", None)
    return bool(nested) and any(is_rate_limit_error(e, _depth + 1) for e in nested)


def rate_limit_delay_seconds(exc: BaseException, attempt: int) -> float:
    """Backoff for a 0-based attempt number, honouring ``Retry-After`` when present."""
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if headers is not None:
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:  # noqa: BLE001 - header access varies by client
            raw = None
        if raw is not None:
            try:
                return min(_RATE_LIMIT_MAX_DELAY_S, max(1.0, float(raw)))
            except (TypeError, ValueError):
                pass
    return min(_RATE_LIMIT_MAX_DELAY_S, _RATE_LIMIT_BASE_DELAY_S * (2**attempt))


async def call_with_rate_limit_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    label: str = "llm",
    max_attempts: int = _RATE_LIMIT_ATTEMPTS,
) -> T:
    """Await ``fn()``, retrying rate-limit errors with exponential backoff."""
    last: BaseException | None = None
    attempts = max(1, int(max_attempts))
    for i in range(attempts):
        try:
            return await fn()
        except BaseException as exc:  # noqa: BLE001 - anything else is re-raised below
            if not is_rate_limit_error(exc) or i >= attempts - 1:
                raise
            last = exc
            delay = rate_limit_delay_seconds(exc, i)
            logger.warning(
                "%s rate-limited (attempt %d/%d); sleeping %.1fs: %s",
                label,
                i + 1,
                attempts,
                delay,
                str(exc)[:200],
            )
            await asyncio.sleep(delay)
    assert last is not None
    raise last


# --- building a model --------------------------------------------------------


def build_chat_model(
    *,
    model: str,
    base_url: str = "",
    api_key: str = "",
    **model_kwargs: Any,
) -> BaseChatModel:
    """A chat model pointed at an OpenAI-compatible endpoint.

    ``base_url`` and ``api_key`` fall back to the environment. Extra keyword
    arguments are passed through to the client, which is where decoding settings
    such as ``temperature`` or ``reasoning_effort`` belong.
    """
    from langchain_openai import ChatOpenAI

    url = base_url or _first_env("MEMOIR_LLM_BASE_URL")
    key = api_key or _first_env("MEMOIR_LLM_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found. Set MEMOIR_LLM_API_KEY, pass api_key=, or hand in "
            "a model you built yourself."
        )
    kwargs: dict[str, Any] = {"model": model, "api_key": key, **model_kwargs}
    if url:
        kwargs["base_url"] = url
    return ChatOpenAI(**kwargs)


def agent_chat_model(model: str | None = None) -> BaseChatModel:
    """The answering model, configured from the environment."""
    return build_chat_model(
        model=model or agent_model(),
        base_url=_first_env("MEMOIR_AGENT_BASE_URL"),
        api_key=_first_env("MEMOIR_AGENT_API_KEY"),
        **_json_env("MEMOIR_AGENT_MODEL_KWARGS"),
    )


def judge_chat_model(model: str | None = None) -> BaseChatModel:
    """The grading model, configured from the environment."""
    return build_chat_model(
        model=model or judge_model(),
        base_url=_first_env("MEMOIR_JUDGE_BASE_URL"),
        api_key=_first_env("MEMOIR_JUDGE_API_KEY"),
        **_json_env("MEMOIR_JUDGE_MODEL_KWARGS"),
    )

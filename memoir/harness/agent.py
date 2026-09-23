"""The agent scaffold.

One scaffold drives every substrate. A question enters as a single message and the
agent loops (call a tool, read the result, call again) until it produces a final
answer with no further tool calls. Single-turn, but not single-tool-call: what
MEMOIR measures is whether a substrate lets an agent retrieve the right
information in one sitting, and with how much effort.

The loop is bounded by a graph recursion limit rather than an arbitrary iteration
cap, and one agent is built per patient and reused across that patient's
questions so its model client and store connections stay warm.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from memoir import models
from memoir.dataset import event_types as corpus_event_types
from memoir.harness.prompt import build_system_prompt
from memoir.harness.result import AnswerResult
from memoir.substrate import Substrate
from memoir.tools.trend import build_trend_tool

#: LangGraph super-steps, roughly two dozen tool round-trips.
DEFAULT_RECURSION_LIMIT = 50

_RETRYABLE_TRANSPORT_ERRORS = (
    "RemoteProtocolError",
    "ServerDisconnected",
    "ConnectionResetError",
    "BrokenPipeError",
    "ConnectionError",
    # httpx/httpcore raise ConnectError, not ConnectionError, on TCP/TLS failure.
    "ConnectError",
    "ConnectTimeout",
    "ReadTimeout",
    "WriteTimeout",
    "PoolTimeout",
    "APITimeoutError",
)
_MAX_TRANSPORT_RETRIES = 3
# Rate limits need a longer budget than a transport blip.
_MAX_RATE_LIMIT_RETRIES = 8


def _is_retryable_transport_error(exc: BaseException, _depth: int = 0) -> bool:
    """True for a transient transport failure, including ExceptionGroup leaves."""
    if _depth > 6:
        return False
    if any(k in type(exc).__name__ for k in _RETRYABLE_TRANSPORT_ERRORS):
        return True
    if any(k in str(exc) for k in _RETRYABLE_TRANSPORT_ERRORS):
        return True
    nested = getattr(exc, "exceptions", None)
    return bool(nested) and any(_is_retryable_transport_error(e, _depth + 1) for e in nested)


async def _invoke_with_retry(attempt: Callable[[], Any]) -> Any:
    """Await ``attempt()``, retrying transient transport errors and rate limits.

    ``attempt`` re-runs its entire body on each retry, so a session-based caller
    must open a fresh session inside it; reusing one risks invoking through an
    already-broken connection.
    """
    last: BaseException | None = None
    transport_failures = 0
    rate_failures = 0
    for _ in range(_MAX_TRANSPORT_RETRIES + _MAX_RATE_LIMIT_RETRIES):
        try:
            return await attempt()
        except BaseException as exc:  # noqa: BLE001
            # Tool transports wrap httpx errors in ExceptionGroup, which is a
            # BaseException; catching only Exception lets those abort the run.
            if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            if models.is_rate_limit_error(exc) and rate_failures < _MAX_RATE_LIMIT_RETRIES - 1:
                last = exc
                await asyncio.sleep(models.rate_limit_delay_seconds(exc, rate_failures))
                rate_failures += 1
                continue
            if (
                _is_retryable_transport_error(exc)
                and transport_failures < _MAX_TRANSPORT_RETRIES - 1
            ):
                last = exc
                await asyncio.sleep(2.0**transport_failures * 3.0)  # 3s, 6s
                transport_failures += 1
                continue
            raise
    assert last is not None
    raise last


def _final_answer(messages: Sequence[BaseMessage]) -> str:
    """The last assistant message that is prose rather than a tool call."""
    for m in reversed(messages):
        if not isinstance(m, AIMessage) or m.tool_calls:
            continue
        content = m.content
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(b["text"])
                for b in content
                if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
            ]
            parts += [b for b in content if isinstance(b, str)]
            if joined := "\n".join(parts).strip():
                return joined
    return ""


def _token_usage(messages: Sequence[BaseMessage]) -> tuple[int, int, int]:
    """Sum ``(prompt, completion, total)`` tokens over the turn.

    Read off each assistant message's ``usage_metadata``, which LangChain
    populates from whatever the endpoint reported, so this works for any chat
    model. Messages without usage contribute nothing.
    """
    prompt = completion = 0
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        usage = getattr(m, "usage_metadata", None)
        if not usage:
            continue
        prompt += int(usage.get("input_tokens") or 0)
        completion += int(usage.get("output_tokens") or 0)
    return prompt, completion, prompt + completion


def _tool_call_names(messages: Sequence[BaseMessage]) -> list[str]:
    names: list[str] = []
    for m in messages:
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        for tc in m.tool_calls:
            n = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if isinstance(n, str) and n:
                names.append(n)
    return names


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text") is not None:
                    parts.append(str(block["text"]))
                else:
                    parts.append(json.dumps(block, default=str))
            else:
                parts.append(str(block))
        return "\n".join(parts).strip()
    return json.dumps(content, default=str)


def _serialize_tool_call(tc: Any) -> dict[str, Any]:
    get = tc.get if isinstance(tc, dict) else lambda k, d=None: getattr(tc, k, d)
    out: dict[str, Any] = {"name": get("name"), "id": get("id")}
    if (args := get("args")) is not None:
        out["args"] = args
    return out


def message_trace(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """A JSON-friendly step list: system, question, assistant turns, tool results."""
    trace: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            trace.append({"step": "system", "content": _stringify(m.content)})
        elif isinstance(m, HumanMessage):
            trace.append({"step": "question", "content": _stringify(m.content)})
        elif isinstance(m, AIMessage):
            entry: dict[str, Any] = {
                "step": "assistant",
                "content": _stringify(m.content),
            }
            if m.tool_calls:
                entry["tool_calls"] = [_serialize_tool_call(tc) for tc in m.tool_calls]
            trace.append(entry)
        elif isinstance(m, ToolMessage):
            trace.append(
                {
                    "step": "tool_result",
                    "name": m.name,
                    "tool_call_id": m.tool_call_id,
                    "content": _stringify(m.content),
                }
            )
        else:
            trace.append(
                {
                    "step": type(m).__name__,
                    "content": _stringify(getattr(m, "content", "")),
                }
            )
    return trace


class BenchmarkAgent:
    """Answers one patient's questions over one substrate.

    Build one per (patient, substrate) pair and reuse it: the chat model and the
    substrate's connections are cached on the instance, and reusing a shared
    system-prompt prefix across questions lets server-side prompt caching apply.
    """

    def __init__(
        self,
        substrate: Substrate,
        patient_id: str,
        *,
        llm: BaseChatModel | None = None,
        model: str | None = None,
        recursion_limit: int = DEFAULT_RECURSION_LIMIT,
        skeleton_path: str | None = None,
        event_types: list[str] | None = None,
    ) -> None:
        """Answer one patient's questions over one substrate.

        Pass ``llm`` to drive the agent with a model you built yourself, or
        ``model`` to name one and let the environment supply the endpoint and key.
        """
        self.substrate = substrate
        self.patient_id = patient_id
        self.recursion_limit = recursion_limit
        self._llm = llm
        self.model = model or getattr(llm, "model_name", None) or ""
        self._system: str | None = None
        self._skeleton_path = skeleton_path
        self._event_types = event_types

    @property
    def system_prompt(self) -> str:
        if self._system is None:
            self._system = build_system_prompt(
                patient_id=self.patient_id,
                substrate_section=self.substrate.prompt_section(self.patient_id),
                event_types=self._event_types
                if self._event_types is not None
                else corpus_event_types(),
                skeleton_path=self._skeleton_path,
            )
        return self._system

    @property
    def llm(self) -> BaseChatModel:
        if self._llm is None:
            self._llm = models.agent_chat_model(self.model or None)
            self.model = self.model or models.agent_model()
        return self._llm

    async def ask(
        self,
        question: str,
        *,
        question_id: str = "",
        include_trace: bool = False,
    ) -> AnswerResult:
        """Run the tool loop once and return the answer with its measurements."""
        system = self.system_prompt

        async def _attempt() -> Any:
            async with self.substrate.session(self.patient_id) as tools:
                # A substrate that reconstructs a series at query time gets the
                # same calculator as every other, so a trend result reflects its
                # data rather than one agent's regression arithmetic.
                all_tools = list(tools)
                if not self.substrate.precomputes_trends:
                    all_tools.append(build_trend_tool())
                agent = create_agent(self.llm, all_tools, system_prompt=system)
                return await agent.ainvoke(
                    {"messages": [HumanMessage(content=question)]},
                    config={"recursion_limit": self.recursion_limit},
                )

        started = time.perf_counter()
        out = await _invoke_with_retry(_attempt)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        messages: list[BaseMessage] = list(out.get("messages") or [])
        names = _tool_call_names(messages)
        prompt_tokens, completion_tokens, total_tokens = _token_usage(messages)
        return AnswerResult(
            question_id=question_id,
            patient_id=self.patient_id,
            substrate=self.substrate.name,
            model=self.model,
            question=question,
            answer=_final_answer(messages),
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            tool_calls=len(names),
            tool_call_names=names,
            trace=message_trace(messages) if include_trace else None,
        )

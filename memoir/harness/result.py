"""What one answered question produces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AnswerResult:
    """One agent answer plus the efficiency measurements taken around it.

    Efficiency is a first-class result in MEMOIR, not a footnote: a substrate can
    only claim to be *more accurate and cheaper* if round-trips, latency, and
    tokens were measured from the start.
    """

    question_id: str
    patient_id: str
    substrate: str
    model: str
    question: str
    answer: str

    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    tool_call_names: list[str] = field(default_factory=list)

    #: Set when the agent raised and the row was recorded as a failure.
    error: str | None = None
    #: Full message trace, when the run asked for one.
    trace: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "question_id": self.question_id,
            "patient_id": self.patient_id,
            "substrate": self.substrate,
            "model": self.model,
            "question": self.question,
            "answer": self.answer,
            "latency_ms": round(self.latency_ms, 2),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
            "tool_call_names": self.tool_call_names,
        }
        if self.error:
            out["error"] = self.error
        if self.trace is not None:
            out["trace"] = self.trace
        return out

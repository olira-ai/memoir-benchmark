"""Scoring one answer, and aggregating a run.

Different question types need different graders, so each task has one primary
scorer suited to it, named on every dataset row so the routing is data-driven
rather than hard-coded here. Alongside the primary score sit the two grounding
checks, which measure faithfulness independent of the judge.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from memoir.dataset import Question
from memoir.scoring import grounding, judge, text
from memoir.tasks import TASKS, Task


@dataclass(slots=True)
class Score:
    """The score for one answered question."""

    question_id: str
    task: Task
    scorer: str
    score: float
    question_kind: str | None = None
    needle_kind: str | None = None
    #: Judge rationale, when the primary scorer was the rubric judge.
    rationale: str = ""
    date_grounding: dict[str, Any] | None = None
    entity_grounding: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "question_id": self.question_id,
            "task": self.task,
            "scorer": self.scorer,
            "score": self.score,
        }
        if self.question_kind:
            out["question_kind"] = self.question_kind
        if self.needle_kind:
            out["needle_kind"] = self.needle_kind
        if self.rationale:
            out["rationale"] = self.rationale
        if self.date_grounding is not None:
            out["date_grounding"] = self.date_grounding
        if self.entity_grounding is not None:
            out["entity_grounding"] = self.entity_grounding
        if self.error:
            out["error"] = self.error
        return out


async def score_answer(
    question: Question,
    answer: str,
    *,
    error: str | None = None,
    judge_llm: BaseChatModel | None = None,
    judge_model: str | None = None,
    with_grounding: bool = True,
) -> Score:
    """Grade one answer with its task's primary scorer, plus the grounding checks.

    Pass ``judge_llm`` to grade with a model you built yourself, or ``judge_model``
    to name one and let the environment supply the endpoint and key.

    An agent exception scores 0 rather than aborting the run, which keeps a
    substrate that fails to answer from looking better than one that answers badly.
    """
    common = {
        "question_id": question.question_id,
        "task": question.task,
        "scorer": question.scorer,
        "question_kind": question.question_kind,
        "needle_kind": question.needle_kind,
    }
    if error or not (answer or "").strip():
        return Score(**common, score=0.0, error=error or "empty answer")

    rationale = ""
    if question.scorer == "semantic_equivalence":
        value, rationale = await judge.rubric_score(
            question=question.question,
            gold_answer=question.answer,
            model_answer=answer,
            rubric=judge.factual_rubric(question.valid_answers),
            kind="factual",
            judge_llm=judge_llm,
            judge_model=judge_model,
        )
    elif question.scorer == "token_recall":
        value = text.best_token_recall(answer, question.answer, question.valid_answers)
    elif question.scorer == "direction_match":
        matched = text.direction_match(question.answer, answer)
        # The gold answer carries no direction, so this question cannot be graded
        # on one; fall back to overlap rather than silently scoring zero.
        value = (
            matched
            if matched is not None
            else text.best_token_f1(answer, question.answer, question.valid_answers)
        )
    elif question.scorer == "rubric_judge":
        if not question.judge_rubric:
            raise ValueError(f"{question.question_id} is judged but carries no rubric")
        kind = "trajectory" if question.task == "trajectory" else "compilation"
        value, rationale = await judge.rubric_score(
            question=question.question,
            gold_answer=question.answer,
            model_answer=answer,
            rubric=judge.with_date_tolerance(question.judge_rubric),
            kind=kind,
            judge_llm=judge_llm,
            judge_model=judge_model,
        )
    else:
        raise ValueError(f"unknown scorer {question.scorer!r}")

    return Score(
        **common,
        score=round(float(value), 4),
        rationale=rationale,
        date_grounding=(
            grounding.date_grounding(answer, question.patient_id, question=question.question)
            if with_grounding
            else None
        ),
        entity_grounding=(
            grounding.entity_grounding(answer, question.patient_id) if with_grounding else None
        ),
    )


@dataclass(slots=True)
class Aggregate:
    """A run's headline numbers."""

    substrate: str
    n: int
    overall: float
    per_task: dict[str, float] = field(default_factory=dict)
    per_task_n: dict[str, int] = field(default_factory=dict)
    per_kind: dict[str, float] = field(default_factory=dict)
    per_kind_n: dict[str, int] = field(default_factory=dict)
    per_needle_kind: dict[str, float] = field(default_factory=dict)
    per_needle_kind_n: dict[str, int] = field(default_factory=dict)
    date_grounding: float | None = None
    entity_grounding: float | None = None
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "substrate": self.substrate,
            "n": self.n,
            "overall": self.overall,
            "per_task": self.per_task,
            "per_task_n": self.per_task_n,
            "per_kind": self.per_kind,
            "per_kind_n": self.per_kind_n,
            "per_needle_kind": self.per_needle_kind,
            "per_needle_kind_n": self.per_needle_kind_n,
            "date_grounding": self.date_grounding,
            "entity_grounding": self.entity_grounding,
            "errors": self.errors,
        }


def _mean(values: list[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def aggregate(scores: Iterable[Score], *, substrate: str = "") -> Aggregate:
    """Roll a run's scores up to the headline grid.

    ``overall`` is the **mean of the four task scores**, equally weighted, not a
    mean over all questions. The tasks differ in size by more than a factor of
    two, so a question-weighted mean would let the largest task quietly decide the
    headline number.
    """
    rows = list(scores)
    per_task_values: dict[str, list[float]] = {t: [] for t in TASKS}
    per_kind_values: dict[str, list[float]] = {}
    per_needle_values: dict[str, list[float]] = {}
    date_values: list[float] = []
    entity_values: list[float] = []
    errors = 0

    for s in rows:
        per_task_values.setdefault(s.task, []).append(s.score)
        if s.question_kind:
            per_kind_values.setdefault(s.question_kind, []).append(s.score)
        if s.needle_kind:
            per_needle_values.setdefault(s.needle_kind, []).append(s.score)
        if s.date_grounding:
            date_values.append(float(s.date_grounding["score"]))
        if s.entity_grounding:
            entity_values.append(float(s.entity_grounding["score"]))
        if s.error:
            errors += 1

    per_task = {t: _mean(v) for t, v in per_task_values.items() if v}
    return Aggregate(
        substrate=substrate or (rows[0].question_id.split("_")[0] if rows else ""),
        n=len(rows),
        overall=_mean([per_task[t] for t in TASKS if t in per_task]),
        per_task=per_task,
        per_task_n={t: len(v) for t, v in per_task_values.items() if v},
        per_kind={k: _mean(v) for k, v in sorted(per_kind_values.items())},
        per_kind_n={k: len(v) for k, v in sorted(per_kind_values.items())},
        per_needle_kind={k: _mean(v) for k, v in sorted(per_needle_values.items())},
        per_needle_kind_n={k: len(v) for k, v in sorted(per_needle_values.items())},
        date_grounding=_mean(date_values) if date_values else None,
        entity_grounding=_mean(entity_values) if entity_values else None,
        errors=errors,
    )

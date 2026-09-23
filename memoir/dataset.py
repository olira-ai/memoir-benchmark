"""Loaders for the MEMOIR corpus.

The corpus ships with the repository under ``data/``:

===========================  ==================================================
``data/events/<pid>.ndjson`` one clinical event per line, ascending by timestamp
``data/questions.ndjson``    one question per line
``data/cohort.json``         per-patient manifest
``data/templates.json``      the question templates the corpus was built from
===========================  ==================================================

Set ``MEMOIR_DATA_DIR`` to read the corpus from somewhere else.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from memoir.tasks import QuestionKind, Scorer, Task

_REPO_ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Root of the corpus. ``MEMOIR_DATA_DIR`` overrides the bundled copy."""
    override = (os.environ.get("MEMOIR_DATA_DIR") or "").strip()
    return Path(override).expanduser().resolve() if override else _REPO_ROOT / "data"


@dataclass(frozen=True, slots=True)
class Event:
    """One clinical event in a patient's record."""

    patient_id: str
    event_type: str
    timestamp: str
    payload: dict[str, Any]
    fhir_patient_id: str | None = None

    @property
    def date(self) -> str:
        """Calendar date of the event, ``YYYY-MM-DD``, in the record's local time."""
        return self.timestamp[:10]


@dataclass(frozen=True, slots=True)
class AnchorEvent:
    """An event the gold answer is justified by.

    ``event_index`` is the zero-based line number in the patient's event stream.
    """

    event_index: int
    event_type: str
    timestamp: str
    summary: str


@dataclass(frozen=True, slots=True)
class Question:
    """One benchmark question and its answer key."""

    question_id: str
    patient_id: str
    task: Task
    template_id: str
    question: str
    answer: str
    anchor_events: tuple[AnchorEvent, ...]
    rationale: str
    scorer: Scorer
    window_start: str
    window_end: str
    question_kind: QuestionKind | None = None
    valid_answers: tuple[str, ...] = ()
    judge_rubric: str = ""
    needle_kind: str | None = None
    legacy_question_id: str | None = None

    @property
    def judged(self) -> bool:
        """True when the primary scorer is the rubric judge."""
        return self.scorer == "rubric_judge"


@dataclass(frozen=True, slots=True)
class Patient:
    """A cohort member: the manifest row plus lazy access to its event stream."""

    patient_id: str
    event_count: int
    first_event: str
    last_event: str
    window_start: str
    window_end: str
    question_count: int
    fhir_patient_id: str | None = None
    _data_dir: Path = field(default_factory=data_dir, repr=False, compare=False)

    def events(self) -> list[Event]:
        """Full event stream, ascending by timestamp."""
        return load_events(self.patient_id, data_root=self._data_dir)


def _read_ndjson(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


@lru_cache(maxsize=256)
def _events_cached(patient_id: str, data_root: str) -> tuple[Event, ...]:
    path = Path(data_root) / "events" / f"{patient_id}.ndjson"
    if not path.is_file():
        raise FileNotFoundError(f"no event stream for patient {patient_id!r} at {path}")
    return tuple(
        Event(
            patient_id=row["patient_id"],
            event_type=row["event_type"],
            timestamp=row["timestamp"],
            payload=row["payload"],
            fhir_patient_id=row.get("fhir_patient_id"),
        )
        for row in _read_ndjson(path)
    )


def load_events(patient_id: str, *, data_root: Path | None = None) -> list[Event]:
    """Every event for one patient, ascending by timestamp. Cached per patient."""
    return list(_events_cached(patient_id, str(data_root or data_dir())))


def load_cohort(*, data_root: Path | None = None) -> list[Patient]:
    """The 117-patient manifest."""
    root = data_root or data_dir()
    rows = json.loads((root / "cohort.json").read_text(encoding="utf-8"))
    return [
        Patient(
            patient_id=r["patient_id"],
            event_count=r["event_count"],
            first_event=r["first_event"],
            last_event=r["last_event"],
            window_start=r["window_start"],
            window_end=r["window_end"],
            question_count=r["question_count"],
            fhir_patient_id=r.get("fhir_patient_id"),
            _data_dir=root,
        )
        for r in rows
    ]


def load_questions(
    *,
    data_root: Path | None = None,
    tasks: tuple[Task, ...] | None = None,
    patient_ids: tuple[str, ...] | None = None,
    templates: tuple[str, ...] | None = None,
    limit_per_patient: int | None = None,
) -> list[Question]:
    """Load questions, optionally filtered.

    ``limit_per_patient`` keeps the first N questions of each patient after the
    other filters apply, which makes a cheap smoke run across the whole cohort.
    """
    root = data_root or data_dir()
    want_tasks = set(tasks) if tasks else None
    want_patients = set(patient_ids) if patient_ids else None
    want_templates = set(templates) if templates else None

    seen: dict[str, int] = {}
    out: list[Question] = []
    for row in _read_ndjson(root / "questions.ndjson"):
        if want_tasks and row["task"] not in want_tasks:
            continue
        if want_patients and row["patient_id"] not in want_patients:
            continue
        if want_templates and row["template_id"] not in want_templates:
            continue
        pid = row["patient_id"]
        if limit_per_patient is not None:
            if seen.get(pid, 0) >= limit_per_patient:
                continue
            seen[pid] = seen.get(pid, 0) + 1
        out.append(
            Question(
                question_id=row["question_id"],
                patient_id=pid,
                task=row["task"],
                template_id=row["template_id"],
                question=row["question"],
                answer=row["answer"],
                anchor_events=tuple(
                    AnchorEvent(
                        event_index=a["event_index"],
                        event_type=a["event_type"],
                        timestamp=a["timestamp"],
                        summary=a.get("summary", ""),
                    )
                    for a in row.get("anchor_events", ())
                ),
                rationale=row.get("rationale", ""),
                scorer=row["scorer"],
                window_start=row["window"]["start"],
                window_end=row["window"]["end"],
                question_kind=row.get("question_kind"),
                valid_answers=tuple(row.get("valid_answers") or ()),
                judge_rubric=row.get("judge_rubric", ""),
                needle_kind=row.get("needle_kind"),
                legacy_question_id=row.get("legacy_question_id"),
            )
        )
    return out


def load_templates(*, data_root: Path | None = None) -> list[dict[str, Any]]:
    """The question templates the corpus was instantiated from."""
    root = data_root or data_dir()
    return json.loads((root / "templates.json").read_text(encoding="utf-8"))


def event_types(*, data_root: Path | None = None) -> list[str]:
    """Every event type present in the corpus, sorted."""
    root = data_root or data_dir()
    seen: set[str] = set()
    for patient in load_cohort(data_root=root):
        seen.update(e.event_type for e in patient.events())
    return sorted(seen)

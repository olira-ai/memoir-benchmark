"""The MEMOIR task taxonomy.

Four task types, escalating from point lookups to whole-timeline reasoning. Each
task has one primary scorer; the trajectory task splits into two question kinds
that are scored differently because one has a deterministic answer and the other
does not.
"""

from __future__ import annotations

from typing import Final, Literal

Task = Literal["factual", "needle", "compilation", "trajectory"]
QuestionKind = Literal["marker_trend", "timeline_trajectory"]
Scorer = Literal[
    "semantic_equivalence",
    "token_recall",
    "direction_match",
    "rubric_judge",
]

TASKS: Final[tuple[Task, ...]] = ("factual", "needle", "compilation", "trajectory")

TASK_LABELS: Final[dict[Task, str]] = {
    "factual": "Factual",
    "needle": "Needle",
    "compilation": "Compilation",
    "trajectory": "Time series & trajectory",
}

TASK_DESCRIPTIONS: Final[dict[Task, str]] = {
    "factual": (
        "Point-in-time lookup. A single correct answer retrievable from the latest "
        "state of the record, with no temporal reasoning required."
    ),
    "needle": (
        "Event detection. Binary yes/no questions, with detail on a hit, that force "
        "the agent to scan the entire record for a specific event or pattern."
    ),
    "compilation": (
        "Clinical episode synthesis. Anchored on a pivotal event, asking for a "
        "narrative drawing together everything that co-occurred around it."
    ),
    "trajectory": (
        "Temporal reasoning. Single-marker trends across the observation window, and "
        "multi-dimensional questions about the directional arc of the record."
    ),
}

#: Which scorer owns each (task, question_kind) pair. Mirrors the ``scorer`` field
#: carried on every dataset row, which is the authority at scoring time.
PRIMARY_SCORER: Final[dict[tuple[Task, QuestionKind | None], Scorer]] = {
    ("factual", None): "semantic_equivalence",
    ("needle", None): "token_recall",
    ("compilation", None): "rubric_judge",
    ("trajectory", "marker_trend"): "direction_match",
    ("trajectory", "timeline_trajectory"): "rubric_judge",
}

#: Needle questions are tagged by where the evidence lives. ``structured`` needles
#: are atomic events in a typed payload; ``free_text`` needles are spun into the
#: prose of a clinical note; ``multi_event`` needles require joining several events.
NEEDLE_KINDS: Final[tuple[str, ...]] = ("structured", "free_text", "multi_event")

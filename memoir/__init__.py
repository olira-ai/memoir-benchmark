"""MEMOIR: Memory Evaluation over Multi-year Oncology Inference & Retrieval.

A benchmark for the memory substrate underneath a clinical agent. The corpus is
117 synthetic oncology patients with complete multi-year records, and 3,617
questions spanning four tasks, from point lookups to whole-timeline reasoning.

Everything above the memory layer is held constant (the same agent scaffold, the
same prompt skeleton, the same decoding settings, the same scorers) so that a
difference in results is a difference in the substrate.

    from memoir import BenchmarkAgent, SimpleSubstrate, load_questions, score_answer
"""

from __future__ import annotations

from memoir.dataset import (
    AnchorEvent,
    Event,
    Patient,
    Question,
    data_dir,
    event_types,
    load_cohort,
    load_events,
    load_questions,
    load_templates,
)
from memoir.harness.agent import BenchmarkAgent
from memoir.harness.result import AnswerResult
from memoir.scoring.score import Aggregate, Score, aggregate, score_answer
from memoir.substrate import SimpleSubstrate, Substrate
from memoir.tasks import TASK_LABELS, TASKS, Task
from memoir.tools.trend import build_trend_tool, compute_trend

__version__ = "1.0.0"

__all__ = [
    "AnchorEvent",
    "Aggregate",
    "AnswerResult",
    "BenchmarkAgent",
    "Event",
    "Patient",
    "Question",
    "Score",
    "SimpleSubstrate",
    "Substrate",
    "TASKS",
    "TASK_LABELS",
    "Task",
    "aggregate",
    "build_trend_tool",
    "compute_trend",
    "data_dir",
    "event_types",
    "load_cohort",
    "load_events",
    "load_questions",
    "load_templates",
    "score_answer",
    "__version__",
]

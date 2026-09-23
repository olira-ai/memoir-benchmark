"""Integrity checks on the released corpus.

A benchmark is only as trustworthy as its answer key, so these run over the whole
corpus rather than a sample. The load-bearing one is
:func:`test_every_anchor_resolves_to_its_event`: every gold answer is anchored to
specific, verifiable events in the raw record, and this is what verifies that
claim, by checking that each anchor's index, type, and date agree with the stream.
"""

from __future__ import annotations

from collections import Counter

import pytest

from memoir.dataset import (
    load_cohort,
    load_events,
    load_questions,
    load_templates,
)
from memoir.tasks import NEEDLE_KINDS, PRIMARY_SCORER, TASKS

COHORT_SIZE = 117
QUESTION_COUNT = 3617
EVENT_COUNT = 65377


@pytest.fixture(scope="module")
def cohort():
    return load_cohort()


@pytest.fixture(scope="module")
def questions():
    return load_questions()


def test_cohort_size(cohort):
    assert len(cohort) == COHORT_SIZE
    assert len({p.patient_id for p in cohort}) == COHORT_SIZE


def test_question_count(questions):
    assert len(questions) == QUESTION_COUNT
    assert len({q.question_id for q in questions}) == QUESTION_COUNT


def test_event_streams_match_the_manifest(cohort):
    total = 0
    for patient in cohort:
        events = patient.events()
        assert len(events) == patient.event_count, patient.patient_id
        assert all(e.patient_id == patient.patient_id for e in events)
        total += len(events)
    assert total == EVENT_COUNT


def test_event_streams_are_chronological(cohort):
    for patient in cohort:
        stamps = [e.timestamp for e in patient.events()]
        assert stamps == sorted(stamps), patient.patient_id


def test_every_anchor_resolves_to_its_event(cohort, questions):
    """Every gold answer is anchored to real, verifiable events in the record."""
    by_patient = {p.patient_id: load_events(p.patient_id) for p in cohort}
    checked = 0
    for q in questions:
        events = by_patient[q.patient_id]
        for anchor in q.anchor_events:
            assert 0 <= anchor.event_index < len(events), q.question_id
            event = events[anchor.event_index]
            assert event.event_type == anchor.event_type, q.question_id
            assert event.timestamp[:10] == anchor.timestamp[:10], q.question_id
            checked += 1
    assert checked > 17_000


def test_tasks_and_scorers_agree(questions):
    for q in questions:
        assert q.task in TASKS
        assert PRIMARY_SCORER[(q.task, q.question_kind)] == q.scorer


def test_judged_questions_carry_a_rubric(questions):
    for q in questions:
        if q.scorer == "rubric_judge":
            assert q.judge_rubric.strip(), q.question_id
        if q.needle_kind is not None:
            assert q.needle_kind in NEEDLE_KINDS, q.question_id


def test_question_ids_are_self_describing(questions):
    for q in questions:
        assert q.question_id.startswith(f"{q.patient_id}_{q.task}_{q.template_id}_")


def test_no_deprecated_identifiers_leak_into_prose(questions):
    """Internal task and template identifiers stay out of reader-facing text.

    ``legacy_question_id`` is the deliberate exception: it exists to join a
    published result back to the run that produced it.
    """
    banned = ("task_1_", "task_2_", "task_3_", "task_5_", "task_6_", "t1_", "t2_", "t6_")
    for q in questions:
        for field in (q.question, q.answer, q.rationale, q.judge_rubric):
            lowered = field.lower()
            for token in banned:
                assert token not in lowered, f"{q.question_id}: {token}"


def test_templates_cover_every_instantiated_question(questions):
    templates = {t["template_id"]: t for t in load_templates()}
    used = Counter(q.template_id for q in questions)
    for template_id, count in used.items():
        assert template_id in templates, template_id
        assert templates[template_id]["instantiated"] == count
    # Four tasks, 72 templates defined, not all of which found an instance in
    # every patient's record.
    assert len(templates) == 72
    assert {t["task"] for t in templates.values()} == set(TASKS)


def test_trajectory_merges_the_two_temporal_question_kinds(questions):
    kinds = Counter(q.question_kind for q in questions if q.task == "trajectory")
    assert set(kinds) == {"marker_trend", "timeline_trajectory"}
    assert all(q.question_kind is None for q in questions if q.task != "trajectory")


def test_filters(cohort):
    one = cohort[0].patient_id
    subset = load_questions(patient_ids=(one,), tasks=("needle",))
    assert subset
    assert all(q.patient_id == one and q.task == "needle" for q in subset)
    assert len(load_questions(limit_per_patient=1)) == COHORT_SIZE

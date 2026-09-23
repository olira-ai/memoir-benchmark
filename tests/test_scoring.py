"""Tests for the scorers that need no model call."""

from __future__ import annotations

import pytest

from memoir.scoring import grounding
from memoir.scoring.score import Score, aggregate
from memoir.scoring.text import (
    best_token_recall,
    direction_match,
    extract_direction,
    token_f1,
    token_recall,
)
from memoir.tools.trend import compute_trend


class TestDirection:
    def test_explicit_label_wins(self):
        assert extract_direction("Direction: increasing. Some prose after.") == "increasing"

    def test_slope_synonyms_are_canonical(self):
        assert extract_direction("Direction: positive") == "increasing"
        assert extract_direction("Direction: negative") == "decreasing"

    def test_bare_keyword_is_accepted(self):
        assert extract_direction("the marker has been decreasing since 2011") == "decreasing"

    def test_refusal_reads_as_insufficient(self):
        assert extract_direction("There is not enough data to say.") == "insufficient"

    def test_absent_direction_is_none(self):
        assert extract_direction("The patient attended four visits.") is None

    def test_match_and_mismatch(self):
        assert direction_match("Direction: increasing.", "Direction: increasing.") == 1.0
        assert direction_match("Direction: increasing.", "Direction: stable.") == 0.0

    def test_missing_prediction_scores_zero(self):
        assert direction_match("Direction: stable.", "I could not find that.") == 0.0

    def test_gold_without_direction_is_not_gradable(self):
        assert direction_match("Four encounters.", "Direction: stable.") is None


class TestTokenMetrics:
    def test_recall_ignores_extra_context(self):
        gold = "Verzenio stopped on 2017-07-11"
        verbose = (
            "The last discontinued medication was Verzenio 100 MG, stopped on "
            "2017-07-11 during an outpatient visit."
        )
        assert token_recall(gold, verbose) == 1.0
        assert token_f1(gold, verbose) < 1.0

    def test_recall_penalises_omission(self):
        assert (
            token_recall("Verzenio stopped on 2017-07-11", "A medication was stopped.") < 0.5
        )

    def test_empty_prediction_scores_zero(self):
        assert token_recall("anything at all", "") == 0.0

    def test_valid_answers_take_the_best_match(self):
        assert best_token_recall("Tamoxifen", "Anastrozole", ("Tamoxifen", "Letrozole")) == 1.0

    def test_compound_tokens_normalise(self):
        assert best_token_recall("active treatment phase", "active_treatment") > 0.0


class TestComputeTrend:
    def test_rising_series_is_increasing(self):
        points = [{"date": f"2010-0{m}-01", "value": 1.0 * m} for m in range(1, 8)]
        assert compute_trend(points)["direction"] == "increasing"

    def test_falling_series_is_decreasing(self):
        points = [{"date": f"2010-0{m}-01", "value": 10.0 - m} for m in range(1, 8)]
        assert compute_trend(points)["direction"] == "decreasing"

    def test_noise_without_significance_is_stable(self):
        values = [5.0, 5.2, 4.9, 5.1, 5.0, 4.95, 5.05]
        points = [{"date": f"2010-0{i + 1}-01", "value": v} for i, v in enumerate(values)]
        assert compute_trend(points)["direction"] == "stable"

    def test_two_points_are_insufficient(self):
        out = compute_trend(
            [{"date": "2010-01-01", "value": 1.0}, {"date": "2011-01-01", "value": 9.0}]
        )
        assert out["direction"] == "insufficient"
        assert out["n"] == 2

    def test_first_and_last_come_from_sorted_order(self):
        out = compute_trend(
            [
                {"date": "2015-06-01", "value": 3.0},
                {"date": "2010-01-01", "value": 1.0},
                {"date": "2020-01-01", "value": 7.0},
            ]
        )
        assert out["first_date"] == "2010-01-01"
        assert out["last_date"] == "2020-01-01"

    def test_unparseable_dates_are_reported_not_silently_dropped(self):
        out = compute_trend(
            [
                {"date": "not-a-date", "value": 1.0},
                {"date": "2010-01-01", "value": 1.0},
                {"date": "2011-01-01", "value": 2.0},
                {"date": "2012-01-01", "value": 3.0},
            ]
        )
        assert out["skipped_dates"] == ["not-a-date"]
        assert out["n"] == 3

    def test_mixed_offsets_do_not_break_day_maths(self):
        out = compute_trend(
            [
                {"date": "2010-01-01T14:58:59-05:00", "value": 1.0},
                {"date": "2010-06-01T14:58:59-04:00", "value": 2.0},
                {"date": "2010-12-01T02:00:00Z", "value": 3.0},
            ]
        )
        assert out["direction"] == "increasing"


@pytest.fixture(scope="module")
def patient():
    """Grounding runs against a real patient, so the corpus itself is exercised."""
    from memoir.dataset import load_cohort

    return load_cohort()[0]


class TestGrounding:
    def test_a_real_event_date_grounds(self, patient):
        real = patient.events()[0].date
        out = grounding.date_grounding(
            f"The encounter occurred on {real}.", patient.patient_id
        )
        assert out is not None and out["score"] == 1.0

    def test_a_fabricated_date_does_not(self, patient):
        out = grounding.date_grounding(
            "The encounter occurred on 1962-03-04.", patient.patient_id
        )
        assert out is not None and out["score"] == 0.0
        assert out["ungrounded"] == ["1962-03-04"]

    def test_window_boundaries_are_not_claims(self, patient):
        out = grounding.date_grounding(
            "I searched 1994-10-01 to 2000-06-30 and found nothing.", patient.patient_id
        )
        assert out is None

    def test_dates_echoed_from_the_question_are_not_claims(self, patient):
        out = grounding.date_grounding(
            "Nothing was documented on 1962-03-04.",
            patient.patient_id,
            question="What happened on 1962-03-04?",
        )
        assert out is None

    def test_an_answer_citing_no_dates_is_not_gradable(self, patient):
        assert (
            grounding.date_grounding("No such event is recorded.", patient.patient_id) is None
        )

    def test_a_fabricated_drug_does_not_ground(self, patient):
        out = grounding.entity_grounding(
            "The patient was started on Zyzzoxitrol 250 MG.", patient.patient_id
        )
        assert out is not None and out["score"] == 0.0


class TestAggregate:
    def _score(self, task, value, **kw):
        return Score(question_id=f"p_{task}_x_001", task=task, scorer="x", score=value, **kw)

    def test_overall_weights_tasks_equally_not_questions(self):
        """A large weak task must not be able to drag the headline number down."""
        scores = (
            [self._score("factual", 1.0)]
            + [self._score("needle", 1.0)]
            + [self._score("compilation", 0.0) for _ in range(100)]
            + [self._score("trajectory", 1.0)]
        )
        agg = aggregate(scores, substrate="demo")
        assert agg.overall == 0.75  # not 0.03, which a question-weighted mean would give
        assert agg.per_task_n["compilation"] == 100

    def test_missing_tasks_are_left_out_of_the_mean(self):
        agg = aggregate([self._score("factual", 0.8), self._score("needle", 0.6)])
        assert agg.overall == 0.7
        assert set(agg.per_task) == {"factual", "needle"}

    def test_breakdowns_and_error_count(self):
        agg = aggregate(
            [
                self._score("trajectory", 1.0, question_kind="marker_trend"),
                self._score("trajectory", 0.0, question_kind="timeline_trajectory"),
                self._score("needle", 0.5, needle_kind="free_text"),
                self._score("needle", 0.0, error="boom"),
            ]
        )
        assert agg.per_kind == {"marker_trend": 1.0, "timeline_trajectory": 0.0}
        assert agg.per_needle_kind == {"free_text": 0.5}
        assert agg.errors == 1


class TestReportLabelling:
    """A filtered run must not claim an overall it did not measure."""

    def _agg(self, tasks):
        from memoir.scoring.score import Score, aggregate

        return aggregate(
            [Score(question_id=f"p_{t}_x_001", task=t, scorer="x", score=0.9) for t in tasks],
            substrate="demo",
        )

    def test_full_run_names_all_four_tasks(self):
        from memoir.report import format_report

        out = format_report(self._agg(["factual", "needle", "compilation", "trajectory"]))
        assert "mean of the four task scores" in out

    def test_partial_run_says_so(self):
        from memoir.report import format_report

        out = format_report(self._agg(["needle"]))
        assert "mean of 1 of 4 tasks, partial run" in out
        assert "four task scores" not in out

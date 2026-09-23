"""The rubric judge.

Compilation and whole-timeline trajectory answers are prose, so they are graded by
a model against the rubric written for that specific question, on a 0 to 3 scale
that is rescaled to 0 and 1. Factual answers go through the same machinery with a
semantic-equivalence rubric built on the fly, because an exact string match fails
on paraphrase ("active_treatment" against "active treatment phase") and on
questions where either of two answers is correct.

The judge prompt is kept disciplined: a rubric, a handful of calibration examples,
and little else. Calibration matters more than it might seem. Without anchors the
judge drifts, and its most common error is deducting for *extra* correct detail
that the reference answer simply did not enumerate.

A rubric judge is still a rubric judge, with the usual caveats.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from memoir import models

TaskKind = Literal["compilation", "trajectory", "factual"]

# One call per (question, answer, rubric, kind). Several scorers may ask for the
# same judgement on one row; without this they would make independent calls that
# can disagree. Entries in flight are awaited rather than duplicated.
_CACHE: dict[str, tuple[float, str]] = {}
_INFLIGHT: dict[str, asyncio.Future[tuple[float, str]]] = {}

# Built once per model id so the connection pool is reused across a run.
_JUDGE_MODELS: dict[str, BaseChatModel] = {}


_SYSTEM = (
    "You score clinical benchmark answers. You must follow the rubric exactly.\n"
    'Return ONLY a compact JSON object: {"score": <int>, "rationale": <string>}.\n'
    "<score> must be an integer 0, 1, 2, or 3 as defined in the rubric (3 = best).\n"
    "Do not reward content that contradicts the reference answer when the rubric "
    "requires factual grounding. However, do NOT penalize content that is additional "
    "and factually correct. Deduct points only for: (1) missing required rubric "
    "elements, (2) factually wrong claims, (3) hallucinated events not supported by "
    "clinical data, or (4) inferences the rubric explicitly prohibits.\n"
    "For rubrics that list requirements as 'Answer must X / Answer must Y' without "
    "explicit Score 0-3 definitions, treat them as: Score 3 when all requirements "
    "are met with no unsupported content; Score 2 when most requirements are met but "
    "one element is missing or a minor unsupported inference is present; Score 1 when "
    "the core requirement is met but major elements are missing; Score 0 when none "
    "are met or the answer is wrong or hallucinated."
)

# The reference answer cites *sample* anchor events, never the full set. The model
# under test reads the whole record, so extra correct detail is routine and must
# not be punished. This is the single most common judge error.
_ADDITIONAL_EVENTS_TOLERANCE = """
**Additional-events tolerance (the most common error this judge makes).** The
reference answer's anchor events and per-period counts are NOT EXHAUSTIVE. They are
sample events the dataset author cited as canonical, not the full set of events in
the patient's record. The model has access to the entire record and is encouraged to
draw on it. When you find yourself writing a rationale like "introduces [event] not
in reference", stop and ask whether that event could plausibly exist in the patient's
record. If it could, which is true of almost any clinically coherent event consistent
with this patient's history, the answer still earns Score 3. Extra events that do not
contradict the reference are bonus context, not deductions. Deduct only when an extra
claim DIRECTLY CONTRADICTS the reference (wrong polarity, wrong date for an event both
sources cite, wrong drug name for an event both sources cite) or fabricates something
implausible for this patient.

**Claims that should NOT be deducted:**
- A procedure (colonoscopy, ultrasound, biopsy) dated inside the anchor's window when
  the reference cites only the primary procedure.
- Medication additions or discontinuations in the question window that the reference
  does not enumerate.
- Receptor status, staging, or other lab interpretations on the anchor date when the
  reference's narrative is brief.
- Follow-up encounters or supporting findings beyond the reference's minimum set.

**Claims that SHOULD still be deducted:**
- Wrong polarity, where the model says ER+ and the source says ER-.
- A wrong date for an event both sources cite.
- A fabricated drug or procedure inconsistent with this patient's history.
- A wrong directional label, "declining" where the source clearly shows "broadening".
"""

# Source timestamps carry a UTC offset while gold answers are anchored to the local
# date, so the same clinical event is legitimately cited as either of two adjacent
# calendar days. This is a property of the corpus, not a difference worth scoring.
_DATE_TOLERANCE = """

UTC-to-local date drift (1 day either way, applies broadly): source timestamps carry
a UTC offset while the reference answers use US Eastern local dates. An event at 22:00
EST on June 2 is stored as 02:00 UTC on June 3, so the same clinical event can be cited
as either '2004-06-02' or '2004-06-03' and both are correct. Accept one calendar day
either side of the reference date as equivalent. Do NOT deduct for the model saying
'May 12' where the reference says 'May 11', or the reverse. The underlying fact is
identical.
"""

_TRAJECTORY_EXAMPLES: list[tuple[str, int, str]] = [
    (
        "In the first period (1994-1996) only HDL cholesterol and LDL were measured, "
        "suggesting narrow cardiovascular risk monitoring. By the second period (1996-1998) "
        "triglycerides and PSA were added, broadening the lipid panel and adding prostate "
        "surveillance. In the final period (1998-2000) CBC, CMP, and urinalysis were "
        "introduced alongside over 100 lab results, reflecting a major shift to comprehensive "
        "multi-system monitoring as clinical complexity grew. This progression from lipid-only "
        "to broad metabolic and hematologic panels indicates intensified management consistent "
        "with increasing comorbidity burden.",
        3,
        "Covers all periods, names specific markers per period, states the direction, and gives "
        "a clinically grounded interpretation. Exact marker names need not match the reference: "
        "correct period coverage plus direction plus reasoning is Score 3.",
    ),
    (
        "The lab tests became more diverse over time, with more types of tests being ordered as "
        "the patient's health needs increased.",
        1,
        "Correct direction (broadening) but no period breakdown and no named markers. A generic "
        "interpretation without specific evidence is Score 1.",
    ),
    (
        "1989 (8 medication actions): acetaminophen, dextromethorphan, doxylamine (symptomatic). "
        "1990-1993 (4 medication actions): Estrostep Fe, contraceptives, supportive agents. "
        "1994-1997 (19 medication actions): DOXOrubicin discontinuation, Paclitaxel "
        "discontinuation, tamoxifen, Verzenio. Overall: shift from supportive symptomatic care "
        "to active oncology therapy.",
        3,
        "Covers all periods, names specific medications per period with counts, and characterizes "
        "the directional shift correctly. Tamoxifen and Verzenio are absent from the reference's "
        "enumeration but plausibly in the record: bonus context, not unsupported inference.",
    ),
    (
        "From 1996 to 2000, the patient's only documented medication changes were the "
        "discontinuation of Hydrochlorothiazide 25 MG Oral Tablet (1996, 1998). From 2001 to "
        "2006, Hydrochlorothiazide was again discontinued (2002, 2005) with no new medications "
        "introduced. From 2007 to 2014, the same pattern continued with Hydrochlorothiazide "
        "discontinued in 2009 and 2012. Overall: de-escalating regimen, repeated discontinuation "
        "of an antihypertensive with no replacement, suggesting either resolved hypertension, "
        "patient preference, or dose-cycling.",
        3,
        "A trajectory made entirely of discontinuations is still a valid trajectory when it names "
        "specific drugs per period and gives a directional label. Score 3 requires named items, "
        "multi-period coverage, and directional characterization; this answer has all three. Do "
        "not score lower because no new medications were added: de-escalation is a legitimate arc.",
    ),
]

_COMPILATION_EXAMPLES: list[tuple[str, int, str]] = [
    (
        "During the outpatient encounter on March 20, 1998 at Sunnybrook Health Sciences Centre, "
        "a digital rectal exam was performed. Laboratory results from the same visit included "
        "PSA 2.12 ng/mL and hemoglobin 14.5 g/dL. Vital signs: blood pressure 130/85 mmHg, "
        "heart rate 72 bpm. The combination of a mildly elevated PSA alongside normal hemoglobin "
        "and stable vital signs suggests early prostate surveillance in an otherwise "
        "hemodynamically stable patient.",
        3,
        "Correct anchor date and facility, a named procedure with its date, two lab values with "
        "units, vital signs, and an interpretation connecting the findings. Specific numeric "
        "values need not match the reference exactly: correct date plus two or more event types "
        "with specific values and units plus a connected interpretation is Score 3.",
    ),
    (
        "On March 20, 1998 the patient had an outpatient encounter. A digital rectal exam was "
        "performed and some lab results were drawn, which appeared within normal limits.",
        1,
        "Correct date and procedure, but no specific lab values or vital signs with units and no "
        "interpretation linking the findings. A vague summary of lab results without numeric "
        "values fails the two-event-types-with-specific-values requirement.",
    ),
    (
        "On April 8, 1988, the patient was diagnosed with distant metastases. Lab results that day "
        "showed positive estrogen receptor status. Treatment included excision of breast tissue on "
        "April 15 and a course of megavoltage radiation therapy starting April 27. Additional "
        "procedures performed in the same window included a colonoscopy on April 21 and a mammogram "
        "on April 8. The patient was also assessed for hormone receptor status, with HER2 negative "
        "and estrogen receptor positive. Subsequent medication changes included discontinuation of "
        "tamoxifen and Verzenio in June.",
        3,
        "Correct anchor event, correct receptor polarity, and multiple supporting events with dates. "
        "The colonoscopy and the June discontinuations are absent from the reference's anchor list "
        "but plausibly in the record: bonus context, not deductions.",
    ),
]

_TRAJECTORY_CLOSING = """
Apply the same logic to the model answer below. A period-bucketed answer with named
items per period and a stated direction always earns Score 3, whether or not every
individual name matches the reference.
"""

_COMPILATION_CLOSING = """
Apply the same logic to the model answer below. An answer with the correct anchor
date, at least two distinct event types (a named procedure AND specific lab values
with units, say), and an explicit clinical interpretation always earns Score 3,
whether or not every individual value matches the reference exactly.
"""


def _calibration_block(examples: list[tuple[str, int, str]], closing: str) -> str:
    lines = ["### Calibration examples (apply these anchors consistently)"]
    for idx, (answer, score, rationale) in enumerate(examples, 1):
        lines.append(
            f"\nCalibration example {idx}, correct score: {score}/3\n"
            f'Model answer: "{answer}"\n'
            f"Why this score: {rationale}"
        )
    lines.append(closing)
    lines.append(_ADDITIONAL_EVENTS_TOLERANCE)
    return "\n".join(lines)


def calibration_for(kind: TaskKind) -> str:
    """The calibration section for one judged task kind."""
    if kind == "trajectory":
        return _calibration_block(_TRAJECTORY_EXAMPLES, _TRAJECTORY_CLOSING)
    if kind == "compilation":
        return _calibration_block(_COMPILATION_EXAMPLES, _COMPILATION_CLOSING)
    return ""


def factual_rubric(valid_answers: tuple[str, ...] = ()) -> str:
    """A semantic-equivalence rubric for a point-lookup answer.

    Built per question so that any of several equally correct answers can score
    full marks, and so that the date drift in the corpus is not scored as error.
    """
    if valid_answers:
        choices = "\n".join(f"  - {a}" for a in valid_answers)
        rubric = (
            "Score 3: The model answer is semantically equivalent to the gold answer or to "
            "any one of the valid alternatives listed below. Exact wording is not required; "
            "accept paraphrases, abbreviations, and formatting differences.\n"
            "Score 2: The model answer contains the correct core fact but includes minor "
            "irrelevant additions or slightly imprecise wording.\n"
            "Score 1: The model answer is partially correct, with the right concept but a "
            "wrong value, a wrong date, or an incorrect alternative mixed in.\n"
            "Score 0: The model answer is wrong, missing, or directly contradicts the gold.\n\n"
            f"Valid gold answers (any one is sufficient for score 3):\n{choices}"
        )
    else:
        rubric = (
            "Score 3: The model answer is semantically equivalent to the gold answer. "
            "Exact wording is not required.\n"
            "Score 2: The model answer is mostly correct with minor imprecision.\n"
            "Score 1: The model answer is partially correct.\n"
            "Score 0: The model answer is wrong or missing."
        )
    return rubric + (
        "\n\nEvent-date tolerance (1 day either way): when the question asks for a specific "
        "event date ('on what date was it stopped', 'when was X performed'), accept the "
        "model's date if it is within one calendar day of the gold date. Source timestamps "
        "carry a UTC offset while gold answers use the local date, so a model quoting one and "
        "a gold quoting the other can both be correct for the same event. Do not penalize "
        "one-day differences on event-date questions."
    )


def _cache_key(question: str, answer: str, rubric: str, kind: str) -> str:
    raw = f"{kind}\x00{question}\x00{answer}\x00{rubric}"
    return hashlib.sha256(raw.encode()).hexdigest()


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_verdict(raw: str) -> tuple[int, str]:
    """Pull ``{"score", "rationale"}`` out of a judge reply.

    Endpoints differ in how strictly they honour a JSON instruction, so the object
    is extracted rather than assumed to be the whole reply. Code fences and any
    prose wrapped around the object are tolerated.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if not match:
            raise ValueError(f"judge returned no JSON object: {text[:500]}") from None
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError(f"judge returned {type(data).__name__}, not an object")
    score = data.get("score")
    if isinstance(score, str) and score.strip().isdigit():
        score = int(score.strip())
    if not isinstance(score, int) or not 0 <= score <= 3:
        raise ValueError(f"judge score must be an int in 0..3, got {score!r}")
    return score, str(data.get("rationale") or "")


def _resolve_judge_llm(
    judge_llm: BaseChatModel | None, judge_model: str | None
) -> BaseChatModel:
    if judge_llm is not None:
        return judge_llm
    key = judge_model or ""
    if key not in _JUDGE_MODELS:
        _JUDGE_MODELS[key] = models.judge_chat_model(judge_model)
    return _JUDGE_MODELS[key]


async def rubric_score(
    *,
    question: str,
    gold_answer: str,
    model_answer: str,
    rubric: str,
    kind: TaskKind,
    judge_llm: BaseChatModel | None = None,
    judge_model: str | None = None,
) -> tuple[float, str]:
    """Score one answer against its rubric. Returns ``(score_0_to_1, rationale)``."""
    key = _cache_key(question, model_answer, rubric, kind)
    if key in _CACHE:
        return _CACHE[key]
    if key in _INFLIGHT:
        return await _INFLIGHT[key]

    fut: asyncio.Future[tuple[float, str]] = asyncio.get_running_loop().create_future()
    _INFLIGHT[key] = fut
    try:
        llm = _resolve_judge_llm(judge_llm, judge_model)
        calibration = calibration_for(kind)
        user = (
            f"Task kind: {kind}\n\n"
            f"### Question\n{question}\n\n"
            f"### Reference (gold) answer\n{gold_answer}\n\n"
            f"### Rubric (use verbatim)\n{rubric.strip()}\n"
            + (f"\n\n{calibration}\n" if calibration else "")
            + f"\n### Model answer to score\n{model_answer}\n"
        )
        messages = [SystemMessage(content=_SYSTEM), HumanMessage(content=user)]

        async def _invoke() -> Any:
            return await llm.ainvoke(messages)

        reply = await models.call_with_rate_limit_retry(_invoke, label="judge")
        content = reply.content
        if isinstance(content, list):
            content = "\n".join(
                str(b.get("text", "")) if isinstance(b, dict) else str(b) for b in content
            )
        score, rationale = _parse_verdict(str(content))
        result = (score / 3.0, rationale)
        _CACHE[key] = result
        fut.set_result(result)
        return result
    except BaseException as exc:  # noqa: BLE001 - propagate to every awaiting caller
        if not fut.done():
            fut.set_exception(exc)
        raise
    finally:
        _INFLIGHT.pop(key, None)


def with_date_tolerance(rubric: str) -> str:
    """Append the corpus-wide date-drift clause to a question's own rubric."""
    return rubric + _DATE_TOLERANCE

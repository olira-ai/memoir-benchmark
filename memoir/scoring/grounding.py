"""Grounding checks: is the answer *invented*, rather than is it *right*?

These sit alongside the task scorers and ask a different question. For every answer
we pull the dates and the clinical entities it cites and check them against the
patient's own record. A date matching no real event; a medication or diagnosis
appearing nowhere in the history. In a clinical setting that is the failure mode
that matters most. A fluent, confident, wrong answer is far more dangerous than
"I couldn't find that", and a grader reading only for plausibility will happily
reward the first.

Both checks are deliberately permissive: substring matching, a generic-token
allowlist, a ±1 day window on dates. A low score is therefore a strong fabrication
signal rather than a noisy artifact. They return ``None``, meaning *not gradable*, when
the answer cites nothing of the relevant kind, so rows with nothing to check do
not dilute the statistic.
"""

from __future__ import annotations

import re
from datetime import date
from functools import lru_cache
from typing import Any

from memoir.dataset import load_events

# --- dates -------------------------------------------------------------------

_ISO = r"(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
_DATE_ISO_RE = re.compile(rf"\b({_ISO})\b")

# A pair of dates joined by a range marker is a window the agent narrated, not an
# event it claimed happened. Same for "between A and B" and for an explicit
# date_from/date_to tool-call citation.
_DATE_RANGE_RE = re.compile(
    rf"\b({_ISO})\s*(?:[–—\-]{{1,2}}|\.\.+|->|-->|⟶|→|(?:\bto\b|\bthrough\b))\s*({_ISO})\b"
)
_DATE_BETWEEN_RE = re.compile(rf"\bbetween\s+({_ISO})\s+and\s+({_ISO})\b", re.IGNORECASE)
# Bounded gap so this cannot span two unrelated tool calls in a long evidence list.
_DATE_PARAM_RANGE_RE = re.compile(
    rf"\bdate_from\s*=\s*({_ISO}).{{0,200}}?\bdate_to\s*=\s*({_ISO})\b",
    re.IGNORECASE | re.DOTALL,
)

#: Source timestamps carry a UTC offset while gold answers use the local date, so
#: one clinical event legitimately reads as either of two adjacent days.
DATE_TOLERANCE_DAYS = 1


@lru_cache(maxsize=256)
def _event_dates(patient_id: str) -> frozenset[str]:
    return frozenset(e.date for e in load_events(patient_id))


def _within_tolerance(d: str, event_dates: frozenset[str], tolerance: int) -> bool:
    try:
        anchor = date.fromisoformat(d)
    except ValueError:
        return False
    for ed in event_dates:
        try:
            other = date.fromisoformat(ed)
        except ValueError:
            continue
        if abs((anchor - other).days) <= tolerance:
            return True
    return False


def date_grounding(
    answer: str,
    patient_id: str,
    *,
    question: str = "",
    tolerance_days: int = DATE_TOLERANCE_DAYS,
) -> dict[str, Any] | None:
    """Fraction of cited dates that correspond to a real event in this record.

    Window boundaries and dates echoed from the question are excluded, since neither is
    a claim the model made about when something happened. ``None`` when the answer
    makes no gradable date claim.
    """
    if not answer:
        return None
    cited_all = sorted(set(_DATE_ISO_RE.findall(answer)))
    if not cited_all:
        return None

    spans = [
        (m.start(), m.end())
        for rx in (_DATE_RANGE_RE, _DATE_BETWEEN_RE, _DATE_PARAM_RANGE_RE)
        for m in rx.finditer(answer)
    ]

    def _only_in_ranges(d: str) -> bool:
        # A date is excluded only when *every* mention sits inside a range; one that
        # also appears as a standalone anchor ("on 2019-07-02") is still a claim.
        return all(
            any(s <= occ.start() < e for s, e in spans)
            for occ in re.finditer(re.escape(d), answer)
        )

    range_only = {d for d in cited_all if _only_in_ranges(d)}
    from_question = set(_DATE_ISO_RE.findall(question or ""))
    cited = [d for d in cited_all if d not in (range_only | from_question)]
    if not cited:
        return None

    event_dates = _event_dates(patient_id)
    if not event_dates:
        return None

    grounded = [d for d in cited if _within_tolerance(d, event_dates, tolerance_days)]
    ungrounded = [d for d in cited if d not in grounded]
    return {
        "score": round(len(grounded) / len(cited), 4),
        "cited": cited,
        "ungrounded": ungrounded,
        "excluded_window_bounds": sorted(range_only & set(cited_all)),
        "excluded_from_question": sorted(from_question & set(cited_all)),
    }


# --- entities ----------------------------------------------------------------

# Phrases in an answer that look like a named clinical entity. Each pattern
# captures the candidate in group 1.
_ENTITY_PATTERNS: list[re.Pattern[str]] = [
    # Medication with a dose: "Leuprolide acetate 7.5 MG". Drug names are proper
    # nouns, and at most two extra word-groups before the dose keeps this from
    # greedily swallowing "the medication for this patient was Naproxen 220 MG".
    re.compile(
        r"\b([A-Z][\w\-/]{2,40}(?:[ \-/][\w][\w\-/]*){0,2}\s+\d+(?:\.\d+)?\s*[mM][gG])\b"
    ),
    # Anything tagged with a SNOMED-style qualifier.
    re.compile(r"\b([A-Z][\w\- ,/]{3,80}?\s*\((?:disorder|finding|procedure)\))"),
    # Neoplasm phrases.
    re.compile(
        r"\b((?:Malignant|Benign|Secondary)\s+(?:[A-Za-z]+\s+){1,6}of\s+[a-z]+(?:\s\([a-z]+\))?)"
    ),
    # Lab tests anchored on their specimen.
    re.compile(
        r"\b([\w][\w\- /,]{2,60}?\s+in\s+(?:Serum|Plasma|Blood|Urine|Tissue|Cancer)\b)"
    ),
    # Lab names carrying a property component ("Creatinine Mass/volume"). Requiring
    # each preceding word to start uppercase or numeric stops conjunctions being
    # absorbed into the entity.
    re.compile(
        r"\b([A-Z][\w\-/\.]{2,40}(?:\s+[A-Z0-9][\w\-/\.]{1,30}){0,2}"
        r"\s+(?:Mass|Moles|Presence|Number|Concentration|Activity|Ratio|Volume)/(?:volume|time)\b)"
    ),
]

# Tokens that match almost any clinical text. An entity whose only distinctive
# tokens are these cannot be graded, so it is skipped rather than penalized.
_GENERIC_TOKENS = frozenset(
    {
        "tablet",
        "tablets",
        "capsule",
        "capsules",
        "injection",
        "injectable",
        "oral",
        "intravenous",
        "intramuscular",
        "solution",
        "suspension",
        "syringe",
        "prefilled",
        "extended",
        "release",
        "ointment",
        "cream",
        "patch",
        "spray",
        "liquid",
        "drops",
        "powder",
        "concentration",
        "blood",
        "serum",
        "plasma",
        "urine",
        "tissue",
        "tumor",
        "cancer",
        "patient",
        "neoplasm",
        "disorder",
        "finding",
        "procedure",
        "report",
        "results",
        "result",
        "evaluation",
        "assessment",
        "examination",
        "level",
        "levels",
        "value",
        "values",
        "measurement",
        "measurements",
        "treatment",
        "therapy",
        "history",
        "documented",
        "recorded",
        # Ordinals and quantifiers: "First value 72.14 mg" matches the
        # medication-with-dose pattern but names no medication.
        "first",
        "second",
        "third",
        "fourth",
        "fifth",
        "sixth",
        "seventh",
        "eighth",
        "ninth",
        "tenth",
        "final",
        "initial",
        "recent",
        "latest",
        "previous",
        "prior",
        "current",
        "average",
        "median",
        "baseline",
        "count",
        "number",
        "total",
        "maximum",
        "minimum",
        "highest",
        "lowest",
        "increase",
        "decrease",
        "change",
    }
)

_MIN_SALIENT_TOKEN_LEN = 5


@lru_cache(maxsize=256)
def _entity_corpus(patient_id: str) -> str:
    """Every named clinical entity in the record, lowercased and joined.

    Free-text note bodies are included so that an answer quoting a note verbatim,
    such as a condition named only in a history-of-present-illness paragraph, grounds
    correctly even though no structured event mirrors it.
    """
    parts: list[str] = []
    for ev in load_events(patient_id):
        p, et = ev.payload, ev.event_type
        if et == "medication_action":
            parts += [
                m[k]
                for m in p.get("medications") or []
                for k in ("medication_name", "name")
                if m.get(k)
            ]
        elif et == "procedure_performed":
            parts += [
                pr[k]
                for pr in p.get("procedures") or []
                for k in ("procedure_name", "name")
                if pr.get(k)
            ]
        elif et == "condition_recorded":
            conds = p.get("conditions")
            if isinstance(conds, list) and conds:
                parts += [c["disease_type"] for c in conds if c.get("disease_type")]
            elif p.get("disease_type"):
                parts.append(p["disease_type"])
        elif et == "lab_results_received":
            parts += [r["test_name"] for r in p.get("results") or [] if r.get("test_name")]
            if p.get("panel_name"):
                parts.append(p["panel_name"])
        elif et == "immunization_reported":
            parts += [
                i["vaccine_name"]
                for i in p.get("immunizations") or []
                if i.get("vaccine_name")
            ]
        elif et == "care_encounter_reported":
            parts += [
                enc[k]
                for enc in p.get("encounters") or []
                for k in ("facility", "primary_reason", "reason_description")
                if enc.get(k)
            ]
        elif et == "clinical_finding_reported":
            parts += [
                f["description"] for f in p.get("findings") or [] if f.get("description")
            ]
        elif et == "clinical_note_received":
            parts += [s["content"] for s in p.get("sections") or [] if s.get("content")]
        elif et == "unstructured_report_received" and p.get("text"):
            parts.append(p["text"])
    return " | ".join(parts).lower()


def _entity_is_grounded(entity: str, corpus: str) -> bool:
    """Grounded when any distinctive token of the entity appears in the record.

    A naive singular form is also accepted, so "Triglycerides" in an answer matches
    a corpus built from "Triglyceride Mass/volume in Serum or Plasma", which is a wording
    difference, not a fabrication.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", entity.lower())
    salient = [
        t for t in tokens if len(t) >= _MIN_SALIENT_TOKEN_LEN and t not in _GENERIC_TOKENS
    ]
    if not salient:
        # Nothing distinctive to check; grading it would only produce noise.
        return True
    for t in salient:
        if t in corpus:
            return True
        if t.endswith("s") and not t.endswith("ss") and t[:-1] in corpus:
            return True
    return False


def candidate_entities(answer: str) -> list[str]:
    """Clinical entity phrases an answer appears to cite, in order, deduplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for pattern in _ENTITY_PATTERNS:
        for m in pattern.finditer(answer):
            ent = m.group(1).strip().strip(",.")
            key = ent.lower()
            if ent and key not in seen:
                seen.add(key)
                out.append(ent)
    return out


def entity_grounding(answer: str, patient_id: str) -> dict[str, Any] | None:
    """Fraction of cited clinical entities that appear in this patient's record.

    ``None`` when the answer cites no recognizable clinical entity.
    """
    if not answer:
        return None
    corpus = _entity_corpus(patient_id)
    if not corpus:
        return None
    candidates = candidate_entities(answer)
    if not candidates:
        return None
    grounded = [e for e in candidates if _entity_is_grounded(e, corpus)]
    ungrounded = [e for e in candidates if e not in grounded]
    return {
        "score": round(len(grounded) / len(candidates), 4),
        "cited": candidates,
        "ungrounded": ungrounded,
    }

# The MEMOIR corpus

117 synthetic oncology patients with complete multi-year records, and 3,617
questions built against them.

| | |
|---|---|
| Patients | 117 |
| Clinical events | 65,377 |
| Events per patient | 260 min · 505 median · 1,091 max |
| Record span | 1 to 38 years per patient, 14 median |
| Calendar range | 1943-12-06 → 2020-04-05 |
| Questions | 3,617 |
| Questions per patient | 22 min · 31 median · 36 max |
| Anchor events | 17,232 (4.8 per question) |
| Event types | 15 |
| Question templates | 72 defined, 64 instantiated |

## Source and provenance

The records are [HL7/CodeX mCODE Test Data][mcode]: synthetic oncology records
generated with [Synthea][synthea], conformant to the [mCODE][ig] standard, and
released without privacy or usage restrictions. **No real patient data is
included.** Names, facilities, and clinician identifiers are Synthea-generated and
often carry numeric suffixes (`Adrian111`, `Kimberly627 Sipes176`), which is how
Synthea marks them as synthetic.

What makes these records useful here is that each is a *complete* patient history
rather than the cancer-relevant slice alone: non-cancer encounters, unrelated
conditions, incidental medications and all.

From the full corpus, 117 patients were selected by taking, for each, the densest
window of clinical activity, so every patient contributes a stretch of record
that is actually eventful while still allowing multi-year quiet stretches inside
it.

[mcode]: https://confluence.hl7.org/spaces/COD/pages/80119851/mCODE+Test+Data
[synthea]: https://github.com/synthetichealth/synthea
[ig]: https://hl7.org/fhir/us/mcode/

## Files

```
data/
├── events/<patient_id>.ndjson   one clinical event per line, ascending by timestamp
├── questions.ndjson             one question per line
├── cohort.json                  per-patient manifest
└── templates.json               the 72 templates the questions were built from
```

### `events/<patient_id>.ndjson`

```json
{
  "patient_id": "69d6865ef9deedfdfef6064c",
  "event_type": "care_encounter_reported",
  "timestamp": "1994-10-27T15:58:59-04:00",
  "payload": {"encounters": [{"encounter_type": "outpatient", "facility": "...", "...": "..."}]},
  "fhir_patient_id": "fd25b51b-dc2a-4edb-b7c9-2c5d178a0ee7"
}
```

| Field | Notes |
|---|---|
| `patient_id` | Opaque, stable, joins to `cohort.json` and to every question |
| `event_type` | One of the 15 types below |
| `timestamp` | ISO 8601 **with a UTC offset**; see the date-drift note |
| `payload` | Typed, type-specific; never truncated |
| `fhir_patient_id` | The Synthea source patient, for tracing back to mCODE |

Lines are in ascending timestamp order, and **that order is load-bearing**: an
`event_index` in a question's `anchor_events` is the zero-based line number in this
file. Do not re-sort or filter the file in place.

### `questions.ndjson`

```json
{
  "question_id": "69d6865ef9deedfdfef6064c_factual_most_recent_encounter_001",
  "legacy_question_id": "69d6865ef9deedfdfef6064c_task_1_factual_t1_most_recent_encounter_001",
  "patient_id": "69d6865ef9deedfdfef6064c",
  "task": "factual",
  "question_kind": null,
  "template_id": "most_recent_encounter",
  "question": "What was the most recent care encounter for this patient, ...",
  "answer": "The most recent care encounter was an outpatient visit at ...",
  "anchor_events": [
    {
      "event_index": 243,
      "event_type": "care_encounter_reported",
      "timestamp": "2000-06-29T15:58:59-04:00",
      "summary": "Outpatient visit at Berkshire Medical Center ... on 2000-06-29."
    }
  ],
  "rationale": "The event at index 243 is the chronologically latest ...",
  "scorer": "semantic_equivalence",
  "window": {"start": "1994-10-01", "end": "2000-06-30"}
}
```

| Field | Always | Notes |
|---|---|---|
| `question_id` | ✓ | `<patient_id>_<task>_<template_id>_<seq>` |
| `legacy_question_id` | ✓ | The identifier used in the original study, kept so published results can be joined back |
| `patient_id` | ✓ | Joins to `events/` and `cohort.json` |
| `task` | ✓ | `factual` · `needle` · `compilation` · `trajectory` |
| `question_kind` | trajectory only | `marker_trend` or `timeline_trajectory` |
| `template_id` | ✓ | Joins to `templates.json` |
| `question` | ✓ | Phrased in clinical terms only |
| `answer` | ✓ | The gold answer |
| `valid_answers` | 317 rows | Equally correct alternatives; the scorer takes the best match |
| `anchor_events` | ✓ | The events that justify the answer |
| `rationale` | ✓ | Why this answer follows. Documentation, never scored |
| `judge_rubric` | 1,913 rows | Present exactly when `scorer` is `rubric_judge` |
| `scorer` | ✓ | Which grader owns this row; the authority at scoring time |
| `needle_kind` | needle only | `structured` (447) · `free_text` (114) · `multi_event` (23) |
| `window` | ✓ | The dense activity window this patient's questions were built in |

### `cohort.json`

One row per patient: `patient_id`, `fhir_patient_id`, `event_count`,
`first_event`, `last_event`, `window_start`, `window_end`, `question_count`.

### `templates.json`

One row per template: `template_id`, `task`, `question_kind`,
`question_pattern`, `answer_format`, `relevant_event_types`, and `instantiated`
(how many questions this template produced). Twelve of the 72 defined templates
found no instance in any patient's record and have `instantiated: 0`. A template
only fires where the record actually holds the anchor it needs.

## Event types

| Event type | Count | Carries |
|---|---|---|
| `lab_results_received` | 25,142 | Panels of results with test name, LOINC code, numeric or string value, unit |
| `procedure_performed` | 9,136 | Procedure name, code, date |
| `care_encounter_reported` | 8,895 | Encounter type, facility, start/end, primary reason |
| `clinical_note_received` | 8,895 | Structured note sections (chief complaint, history of present illness, …) |
| `unstructured_report_received` | 4,261 | Free-text document body |
| `medication_action` | 2,930 | Medication name, RxNorm code, action, prescriber |
| `vitals_measurement` | 1,873 | Vital sign readings with units |
| `immunization_reported` | 1,832 | Vaccine name and date |
| `condition_recorded` | 935 | Disease type, stage, clinical status |
| `clinical_finding_reported` | 799 | Finding description and value |
| `care_team_updated` | 329 | Care team membership |
| `clinical_plan_item_reported` | 329 | Planned interventions and recommendations |
| `treatment_phase_changed` | 12 | Treatment phase transitions |
| `social_updated` | 7 | Social history |
| `demographics_updated` | 2 | Demographics |

The distribution is deliberately lopsided, and that is the point: a handful of
routine types dominate by count while the events a clinical question usually turns
on, such as a stage change, a phase transition, or an alarming line in a note,
live far out in the tail.

## Known quirks

These are properties of the source encoding, left in place rather than smoothed
over. Every one of them is something a system under test has to cope with, and
rewriting them would change what the benchmark measures.

**Timestamps carry a UTC offset; gold answers use the local date.** An event at
22:00 US Eastern on June 2 is stored as 02:00 UTC on June 3, so the same clinical
event is legitimately cited as either of two adjacent calendar days. The scorers
allow one day either way throughout: the date-grounding check, the factual judge,
and the rubric judges all carry an explicit tolerance clause.

**`medication_action.action` is `add` or `delete`.** These are the source system's
CRUD verbs, not clinical language: `add` means a medication was started and
`delete` means it was discontinued. The wording leaks into 156 gold answers
("resulting in the deletion of Tenoretic-50"). It is left as-is because gold text
is byte-identical to what produced the published results, and renaming it would
make the released corpus and those numbers describe different things. Anchor summaries
use the clinical wording ("discontinued"), so both spellings appear.

**`condition_recorded` payloads come in two shapes.** Some carry a flat
`disease_type`, others a `conditions[]` list. Handle both.

**Lab values may be numeric or string.** `value_numeric` for measurements,
`value_string` for interpretations. Receptor status arrives as
`Positive (qualifier value)` or `Negative (qualifier value)` rather than a number.

**One question's `answer` may be long.** Compilation and trajectory gold answers
are full narrative paragraphs, because that is the shape of the answer being asked
for.

## How the questions were built

An authoring engine, not a manual pass. It works through each patient's raw
events, identifies signal-rich anchor points such as a hospitalization, a new
diagnosis, a chemo session, or a clear lab trend, and instantiates parameterized
templates against the specific facts it finds. For every question it writes the gold answer,
the anchor events that justify it, a rationale, and, for the open-ended tasks, a
grading rubric.

Two properties keep that from being a test written to win:

- **Every gold answer is anchored to verifiable events.** All 17,232 anchors are
  checked against the event streams by `tests/test_dataset.py` on every run: index,
  type, and date must agree. There are no answers that exist only in the authors'
  heads.
- **Questions are method-blind.** They are phrased entirely in terms of patient
  facts and clinical language, never in terms of any system's internal structure. A
  question never assumes a particular block, index, or graph exists; it asks what a
  clinician would ask.

The engine is also the limitation: gold answers inherit an authoring model's
phrasing and its blind spots, and the rubrics were written by the same process
that wrote the answers.

## Relationship to the published results

The scoring-relevant fields (`question`, `answer`, `valid_answers`, and
`judge_rubric`) are byte-identical to the corpus that produced the numbers in the
accompanying study. What changed for this release is naming, not content:
identifiers were renamed out of internal shorthand, the two temporal tasks were
merged into one `trajectory` task (with `question_kind` preserving the
distinction), anchor timestamps were normalised to the full ISO form of the event
they reference, and `rationale`, which is documentation and never scored, had
internal task identifiers stripped. `legacy_question_id` carries the original
identifier on every row.

## License

The underlying mCODE Test Data is synthetic and released by HL7/CodeX without
privacy or usage restrictions. The derived questions, answer keys, rubrics, and
templates in this directory are released under
[CC BY 4.0](LICENSE).

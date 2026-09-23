"""A reference substrate: the corpus itself, read straight off disk.

This is a worked example of the :class:`~memoir.substrate.Substrate` interface, not
one of the systems from the study. It exists so the harness is runnable the moment
you clone the repo, as a sanity check that your endpoint, models, and scorers are
wired up, and so there is a concrete implementation to copy for your own.

**It is not a floor.** Reading the typed record losslessly is a strong position on
this benchmark rather than a weak one: in the study, the raw-access control beat all
three general-purpose memory systems. Measured over 59 questions on 15 patients this
substrate scores about 0.89 overall, landing on top of that control, and a perfect
1.00 on the trajectory task where the control scores 0.86.

It wins there because two of its conveniences go further than anything the
benchmarked systems had:

* ``total_matched`` returns an exact count for any filter, so "how many X" is one
  call and an integer, with no paging and no risk of counting a truncated page.
* ``contains`` greps the full serialized payload, note bodies included, with no
  embedding, ranking, or extraction step in between.

Between them they remove the failure mode that defines the control's trajectory
score in the study, which is dropping part of a series while rebuilding it at query
time. So do not read your own substrate against this one as though it were the
bottom of the range. It is nearer the top.

    python -m memoir.run \\
      --substrate examples.local_events:LocalEventsSubstrate \\
      --limit-per-patient 2

Writing your own substrate means replacing :meth:`build_tools` with tools that
query your store, and :meth:`prompt_section` with their documentation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from memoir.dataset import load_events
from memoir.substrate import SimpleSubstrate

#: Cap on events returned by one call, so a broad query cannot blow the context
#: window. The agent narrows with filters or pages with `offset`.
DEFAULT_LIMIT = 50
MAX_LIMIT = 200


class GetEventsArgs(BaseModel):
    event_types: list[str] | None = Field(
        None,
        description=(
            "Restrict to these event types (e.g. ['lab_results_received', "
            "'medication_action']). Omit for all types."
        ),
    )
    date_from: str | None = Field(
        None, description="Earliest event date to include, as YYYY-MM-DD."
    )
    date_to: str | None = Field(
        None, description="Latest event date to include, as YYYY-MM-DD."
    )
    contains: str | None = Field(
        None,
        description=(
            "Case-insensitive substring filter over the serialized event payload. "
            "Use for free-text searches over note bodies, drug names, or lab names."
        ),
    )
    limit: int = Field(
        DEFAULT_LIMIT, description=f"Maximum events to return (max {MAX_LIMIT})."
    )
    offset: int = Field(
        0, description="Events to skip, so you can page through a long result."
    )


def _build_get_events(patient_id: str) -> StructuredTool:
    async def _run(
        event_types: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        contains: str | None = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = 0,
    ) -> str:
        events = load_events(patient_id)
        wanted = set(event_types) if event_types else None
        needle = (contains or "").lower() or None

        matched: list[dict[str, Any]] = []
        for idx, ev in enumerate(events):
            if wanted and ev.event_type not in wanted:
                continue
            if date_from and ev.date < date_from:
                continue
            if date_to and ev.date > date_to:
                continue
            if needle and needle not in json.dumps(ev.payload, default=str).lower():
                continue
            matched.append(
                {
                    "event_index": idx,
                    "event_type": ev.event_type,
                    "timestamp": ev.timestamp,
                    "payload": ev.payload,
                }
            )

        capped = max(1, min(int(limit), MAX_LIMIT))
        page = matched[offset : offset + capped]
        return json.dumps(
            {
                "total_matched": len(matched),
                "returned": len(page),
                "offset": offset,
                "truncated": offset + len(page) < len(matched),
                "events": page,
            },
            default=str,
        )

    return StructuredTool.from_function(
        name="get_events",
        description=(
            "Read this patient's clinical events, filtered by type, date range, or a "
            "substring of the payload. Events come back in chronological order with "
            "their full payload. Narrow with filters before raising `limit`: a broad "
            "call returns only the first page and sets `truncated`."
        ),
        args_schema=GetEventsArgs,
        coroutine=_run,
    )


class LocalEventsSubstrate(SimpleSubstrate):
    """Filtered reads over the patient's event stream, straight from the corpus."""

    name = "local-events"
    # Nothing is computed ahead of time here, so the agent rebuilds a series at
    # query time and the harness supplies the shared `compute_trend` calculator.
    precomputes_trends = False

    def build_tools(self, patient_id: str) -> Sequence[StructuredTool]:
        return [_build_get_events(patient_id)]

    def prompt_section(self, patient_id: str) -> str:
        return """## Your tools

You read this patient's record directly through `get_events`, which returns the
full typed payload of each event. Nothing has been summarized or pre-computed, so
you reconstruct whatever the question needs, whether a count, a series, or an
episode, from the events themselves.

### `get_events`

| Parameter | Type | Description |
|---|---|---|
| `event_types` | `array[string]` | Restrict to these event types. Omit for all. |
| `date_from` | `string` | Earliest event date, `YYYY-MM-DD`. |
| `date_to` | `string` | Latest event date, `YYYY-MM-DD`. |
| `contains` | `string` | Case-insensitive substring of the payload. |
| `limit` | `integer` | Max events per call (default 50, max 200). |
| `offset` | `integer` | Skip this many, to page through long results. |

Returns `{total_matched, returned, offset, truncated, events[]}`. When `truncated`
is true there are more events than you were shown: narrow the filters or raise
`offset`. **Never answer a count or a trend question off a truncated page.**
`total_matched` is the count, and a series needs every point.

### How to work

- **Point lookups.** Filter to the one event type and scan for the latest match.
- **Needles.** Use `contains` for anything spun into note prose; try several
  phrasings before concluding the record does not hold it.
- **Episodes.** Query each event type around the anchor date separately, with a
  window of a few days either side, then connect what you find.
- **Trends.** Pull every measurement of the named marker across the whole record,
  paging until `truncated` is false, then extract `(date, value)` pairs and pass
  them all to `compute_trend`. Do not eyeball the first against the last value.
"""

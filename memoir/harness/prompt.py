"""Building the system prompt.

One skeleton, shared by every substrate, with three slots filled per run. Holding
the skeleton constant is what lets a difference in results be attributed to the
memory layer rather than to how the agent was briefed, so a substrate customises
only :meth:`~memoir.substrate.Substrate.prompt_section`, never the skeleton.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import yaml

from memoir.dataset import load_events

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
_DEFAULT_SKELETON = _PROMPTS_DIR / "base.yaml"


@lru_cache(maxsize=8)
def load_skeleton(path: str | None = None) -> str:
    """Read the prompt skeleton. Pass a path to substitute your own."""
    p = Path(path) if path else _DEFAULT_SKELETON
    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    text = doc.get("system")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"no non-empty `system` key in {p}")
    return text.strip()


def _parse(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def patient_event_calendar(patient_id: str) -> str:
    """Where this patient's events sit in time.

    Without it a model reasons about "recent" against today's date rather than
    against a record that may end decades ago.
    """
    events = load_events(patient_id)
    stamps = [dt for dt in (_parse(e.timestamp) for e in events) if dt is not None]
    if not stamps:
        return (
            "*The calendar span of this patient's events is **unknown**. Infer time "
            "bounds only from tool output.*"
        )
    lo, hi = min(stamps), max(stamps)
    return (
        f"*This patient's events span **{lo.isoformat()}** through **{hi.isoformat()}**. "
        "Use this interval when choosing date filters or interpreting recency; tools "
        "may still return sparse results inside the window.*"
    )


def build_system_prompt(
    *,
    patient_id: str,
    substrate_section: str,
    event_types: list[str],
    skeleton_path: str | None = None,
) -> str:
    """Fill the skeleton's three slots for one patient and substrate."""
    text = load_skeleton(skeleton_path)
    types_md = "\n".join(f"  - `{t}`" for t in sorted(event_types))
    replacements = {
        "{{EVENT_TYPES}}": types_md,
        "{{PATIENT_EVENT_CALENDAR}}": patient_event_calendar(patient_id),
        "{{SUBSTRATE_SECTION}}": (substrate_section or "").strip(),
    }
    for slot, value in replacements.items():
        if slot in text:
            text = text.replace(slot, value)
        elif value:
            text = f"{text}\n\n{value}\n"
    return text

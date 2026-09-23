"""Deterministic text scorers.

Two of the four tasks are graded without a model. Needle answers are scored on
recall of the key facts, and single-marker trend answers are scored on their
direction label alone, which is what makes the trend result load-bearing: a wrong
direction means the series itself was built wrong, not that a judge was harsh.
"""

from __future__ import annotations

import re
from collections import Counter

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "this",
        "that",
        "these",
        "those",
        "patient",
        "patients",
        "record",
        "records",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"^\d+(\.\d+)?$")

#: Canonical direction labels a trend answer may carry.
DIRECTIONS = ("increasing", "decreasing", "stable", "insufficient")

# Some pipelines label a slope "positive"/"negative"; both mean the same thing here.
_DIRECTION_SYNONYMS = {"positive": "increasing", "negative": "decreasing"}

_INSUFFICIENT_PHRASES = (
    "insufficient data",
    "cannot determine",
    "not enough data",
    "unable to determine",
    "only one measurement",
    "single data point",
    "too sparse",
)


def pre_normalize(text: str) -> str:
    """Split compound tokens so ``active_treatment`` matches ``active treatment``."""
    return re.sub(r"[_\-]", " ", text or "")


def normalize_tokens(text: str) -> list[str]:
    """Lowercase content tokens, stopwords removed."""
    return [
        m.group(0).lower()
        for m in _TOKEN_RE.finditer(text or "")
        if m.group(0).lower() not in _STOPWORDS
    ]


def token_f1(gold: str, pred: str) -> float:
    """Order-insensitive multiset F1 over content tokens."""
    g, p = normalize_tokens(gold), normalize_tokens(pred)
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    cg, cp = Counter(g), Counter(p)
    overlap = sum((cg & cp).values())
    if overlap == 0:
        return 0.0
    prec = overlap / max(sum(cp.values()), 1)
    rec = overlap / max(sum(cg.values()), 1)
    return 2.0 * prec * rec / (prec + rec) if prec + rec else 0.0


def token_recall(gold: str, pred: str) -> float:
    """Fraction of gold content tokens present in the prediction.

    Recall rather than F1: a needle answer is judged on whether it surfaced the
    facts, not on how much surrounding context it also volunteered.
    """
    g, p = normalize_tokens(gold), normalize_tokens(pred)
    if not g:
        return 1.0
    if not p:
        return 0.0
    cg, cp = Counter(g), Counter(p)
    hit = sum(min(need, cp.get(tok, 0)) for tok, need in cg.items())
    total = sum(cg.values())
    return hit / total if total else 0.0


def extract_direction(text: str) -> str | None:
    """Pull a canonical direction label out of an answer, or ``None`` if absent.

    The explicit ``Direction: <label>`` line the prompt asks for wins; otherwise a
    bare keyword is accepted so an answer is not punished for formatting.
    """
    if not (text or "").strip():
        return None
    if m := re.search(r"Direction:\s*([A-Za-z_]+)", text, flags=re.IGNORECASE):
        raw = m.group(1).strip().lower()
        return _DIRECTION_SYNONYMS.get(raw, raw)
    lowered = text.lower()
    for phrase in _INSUFFICIENT_PHRASES:
        if phrase in lowered:
            return "insufficient"
    # Longest / most specific keywords first.
    for kw in (
        "worsening",
        "improving",
        "increasing",
        "decreasing",
        "positive",
        "negative",
        "stable",
    ):
        if re.search(rf"\b{re.escape(kw)}\b", lowered):
            return _DIRECTION_SYNONYMS.get(kw, kw)
    return None


def direction_match(gold: str, pred: str) -> float | None:
    """1.0 when the directions agree, 0.0 when they differ.

    ``None`` when the gold answer carries no direction. The question is then not
    a direction question, and this scorer has no opinion.
    """
    g = extract_direction(gold)
    if g is None:
        return None
    p = extract_direction(pred)
    if p is None:
        return 0.0
    return 1.0 if g == p else 0.0


def _score_one_gold(gold: str, pred_pre: str) -> float:
    """F1 against one gold candidate, or exact containment when the gold is a number."""
    gold_pre = pre_normalize(gold)
    stripped = gold_pre.strip()
    if _NUMERIC_RE.match(stripped):
        return 1.0 if stripped in normalize_tokens(pred_pre) else 0.0
    return token_f1(gold_pre, pred_pre)


def best_token_f1(pred: str, gold: str, valid_answers: tuple[str, ...] = ()) -> float:
    """Best F1 across the gold answer and any equally correct alternatives."""
    pred_pre = pre_normalize(pred)
    candidates = valid_answers or (gold,)
    return round(max(_score_one_gold(str(c), pred_pre) for c in candidates), 4)


def best_token_recall(pred: str, gold: str, valid_answers: tuple[str, ...] = ()) -> float:
    """Best recall across the gold answer and any equally correct alternatives."""
    pred_pre = pre_normalize(pred)
    candidates = valid_answers or (gold,)
    return round(max(token_recall(pre_normalize(str(c)), pred_pre) for c in candidates), 4)

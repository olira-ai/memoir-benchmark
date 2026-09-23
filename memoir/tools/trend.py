"""The shared ``compute_trend`` calculator.

Some MEMOIR questions come down to the direction of a single marker over the
observation window, so a substrate that reconstructs the series at query time
could lose on regression arithmetic rather than on what it stored. To keep the
comparison about the substrate, every substrate that rebuilds a series at runtime
is handed this same calculator: a plain OLS fit plus a two-sided Student-t
significance test, with no dependency beyond the standard library.

A substrate that answers trend questions from something it computed ahead of time
declares :attr:`~memoir.substrate.Substrate.precomputes_trends` and does not
receive the tool; reading a pre-computed trend is the capability under test there,
not arithmetic.

``direction`` is significance-based: ``stable`` whenever the slope is not
distinguishable from zero at the threshold, otherwise the sign of the slope.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

Direction = Literal["increasing", "decreasing", "stable", "insufficient"]

#: Below this many points an OLS significance test is undefined.
MIN_POINTS = 3


class TrendPoint(BaseModel):
    date: str = Field(
        ...,
        description="ISO 8601 date or datetime for this measurement (e.g. '2017-01-22').",
    )
    value: float = Field(
        ...,
        description=(
            "Numeric measurement value. Strip units before passing "
            "(e.g. 86.1 not '86.1 mg/dL')."
        ),
    )


class ComputeTrendArgs(BaseModel):
    points: list[TrendPoint] = Field(
        ...,
        description=(
            "All measurements for the SINGLE marker being analyzed, in any order. The "
            "tool sorts by date internally. Include every measurement you found; do "
            "not pre-filter to first/last. Minimum 3 points to produce a direction; "
            "with fewer the tool returns first/last only and direction='insufficient'."
        ),
    )
    significance_threshold: float = Field(
        0.05,
        description=(
            "Two-tailed p-value threshold for stability. Defaults to 0.05; do not change."
        ),
    )


def _ols_fit(xs: list[float], ys: list[float]) -> tuple[float, float, float, float, float]:
    """Return ``(slope, intercept, r2, residual_sum_squares, x_sum_squares)``."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_x = sum((x - mean_x) ** 2 for x in xs)
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    if ss_x == 0.0:
        slope, intercept = 0.0, mean_y
    else:
        slope = cov_xy / ss_x
        intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    if ss_tot == 0.0:
        # A flat series explains itself perfectly, or not at all.
        return slope, intercept, float(ss_res == 0.0), ss_res, ss_x
    return slope, intercept, 1.0 - (ss_res / ss_tot), ss_res, ss_x


def _incomplete_beta_cf(a: float, b: float, x: float) -> float:
    """Continued-fraction expansion of the incomplete beta function."""
    fpmin = 1.0e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3.0e-7:
            return h
    return h


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _incomplete_beta_cf(a, b, x) / a
    return 1.0 - bt * _incomplete_beta_cf(b, a, 1.0 - x) / b


def _student_t_two_sided_p(t: float, df: float) -> float:
    if df <= 0:
        return 1.0
    if not math.isfinite(t):
        return 0.0
    return _regularized_incomplete_beta(df * 0.5, 0.5, df / (df + t * t))


def _parse_date(raw: str) -> datetime | None:
    s = (raw or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        try:
            dt = datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            return None
    # Drop the offset so day differences are stable across mixed-zone inputs.
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def compute_trend(
    points: list[TrendPoint] | list[dict[str, Any]],
    significance_threshold: float = 0.05,
) -> dict[str, Any]:
    """Fit a series and classify its direction. Importable for use outside the agent."""
    typed = [p if isinstance(p, TrendPoint) else TrendPoint.model_validate(p) for p in points]
    parsed: list[tuple[datetime, float]] = []
    skipped: list[str] = []
    for p in typed:
        dt = _parse_date(p.date)
        if dt is None:
            skipped.append(p.date)
        else:
            parsed.append((dt, float(p.value)))

    n = len(parsed)
    if n == 0:
        return {
            "error": "no parseable (date, value) points provided",
            "n": 0,
            "skipped_dates": skipped,
        }

    parsed.sort(key=lambda pair: pair[0])
    first_dt, first_val = parsed[0]
    last_dt, last_val = parsed[-1]
    head = {
        "n": n,
        "first_date": first_dt.date().isoformat(),
        "first_value": first_val,
        "last_date": last_dt.date().isoformat(),
        "last_value": last_val,
    }

    if n < MIN_POINTS:
        return {
            **head,
            "direction": "insufficient",
            "skipped_dates": skipped,
            "note": f"fewer than {MIN_POINTS} points, so the significance test is undefined",
        }

    xs = [(dt - first_dt).total_seconds() / 86400.0 for dt, _ in parsed]
    ys = [v for _, v in parsed]
    slope, intercept, r2, ss_res, ss_x = _ols_fit(xs, ys)

    df = n - 2
    if df <= 0 or ss_x == 0.0:
        slope_se, p_value = 0.0, 1.0
    else:
        sigma_sq = ss_res / df
        slope_se = math.sqrt(sigma_sq / ss_x) if sigma_sq > 0.0 else 0.0
        if slope_se == 0.0:
            p_value = 0.0 if slope != 0.0 else 1.0
        else:
            p_value = _student_t_two_sided_p(slope / slope_se, df)

    if p_value > significance_threshold or slope == 0.0:
        direction: Direction = "stable"
    else:
        direction = "increasing" if slope > 0 else "decreasing"

    return {
        **head,
        "direction": direction,
        "slope_per_day": slope,
        "intercept": intercept,
        "r2": r2,
        "slope_se": slope_se,
        "p_value": p_value,
        "significance_threshold": significance_threshold,
        "skipped_dates": skipped,
    }


def build_trend_tool() -> StructuredTool:
    """The ``compute_trend`` tool handed to every substrate that rebuilds series."""

    async def _run(
        points: list[dict[str, Any]],
        significance_threshold: float = 0.05,
    ) -> str:
        try:
            parsed = [TrendPoint.model_validate(p) for p in points]
        except Exception as exc:  # noqa: BLE001 - surface validation errors to the model
            return json.dumps({"error": f"invalid points payload: {exc}"})
        return json.dumps(compute_trend(parsed, significance_threshold), default=str)

    return StructuredTool.from_function(
        name="compute_trend",
        description=(
            "Fit an OLS linear regression with a Student-t significance test on a "
            "single marker's (date, value) series and return the trend summary "
            "(first_date, first_value, last_date, last_value, direction, "
            "slope_per_day, r2, p_value, n). `direction` is one of `increasing` / "
            "`decreasing` / `stable` and is significance-based (`stable` when "
            "p_value > 0.05). Use this for any 'what is the trend in <marker>?' "
            "question. Extract the full (date, value) series from your retrieval "
            "tools first, then call this helper to derive the direction "
            "deterministically instead of eyeballing first against last."
        ),
        args_schema=ComputeTrendArgs,
        coroutine=_run,
    )

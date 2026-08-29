"""Scope 0: triage. Decide whether a movement is signal or noise.

Pure statistics. Runs before any agent. If the movement sits inside the control
limits the investigation stops here.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import config as C
from .stats_core import robust_baseline, cusum, welch_t_test


@dataclass
class TriageResult:
    metric: str
    is_signal: bool
    verdict: str
    event_weeks: list[str]
    baseline_weeks: list[str]
    event_value: float
    baseline_value: float
    delta: float
    robust_z: float
    control_limits: tuple[float, float]
    changepoint_week: str | None
    p_value: float
    ci_low: float
    ci_high: float
    series: list[dict]
    reasoning: str

    def to_dict(self) -> dict:
        return asdict(self)


def triage(panel: pd.DataFrame, weekly_national: pd.DataFrame,
           metric: str = "review_score") -> TriageResult:
    """
    Detect whether `metric` moved beyond normal variation.

    Baseline uses median + MAD across the full series. Median/MAD tolerates up
    to ~50% contamination, so a short anomaly cannot inflate the baseline it is
    being measured against. A mean/SD baseline would allow that.
    """
    w = weekly_national.sort_values("week").reset_index(drop=True)
    series = w[metric].astype(float)

    base = robust_baseline(series)
    z = series.map(base.z)
    lo, hi = base.limits(C.ROBUST_Z_TRIGGER)

    # Candidate anomaly weeks: robust z beyond the control limit, downward.
    flagged = w.loc[z < -C.ROBUST_Z_TRIGGER, "week"].tolist()

    if not flagged:
        # nothing exceeded the limit, so report the largest deviation and stop
        worst_i = int(z.idxmin())
        return TriageResult(
            metric=metric, is_signal=False, verdict="NOISE",
            event_weeks=[], baseline_weeks=[],
            event_value=float(series.iloc[worst_i]), baseline_value=base.center,
            delta=float(series.iloc[worst_i] - base.center),
            robust_z=float(z.iloc[worst_i]), control_limits=(lo, hi),
            changepoint_week=None, p_value=1.0, ci_low=0.0, ci_high=0.0,
            series=_series_payload(w, metric, z, lo, hi, set()),
            reasoning=(
                f"The largest weekly deviation is {z.iloc[worst_i]:.2f} robust SDs "
                f"from the {base.center:.3f} baseline, inside the +/-{C.ROBUST_Z_TRIGGER} "
                f"control limit. This is normal variation; no investigation is warranted."
            ),
        )

    # Expand to the contiguous run containing the worst week (an event rarely
    # lands on exactly one week boundary).
    worst_i = int(z.idxmin())
    start = end = worst_i
    while start - 1 >= 0 and z.iloc[start - 1] < -1.0:
        start -= 1
    while end + 1 < len(z) and z.iloc[end + 1] < -1.0:
        end += 1
    event_weeks = w.loc[start:end, "week"].tolist()
    event_set = set(event_weeks)

    # Baseline = everything outside the event, with a 2-week buffer either side
    buf_lo = max(0, start - 2)
    buf_hi = min(len(w) - 1, end + 2)
    buffer_set = set(w.loc[buf_lo:buf_hi, "week"].tolist())
    baseline_weeks = [x for x in w["week"] if x not in buffer_set]

    # Order-level test where the metric exists per order; otherwise fall back to
    # comparing the weekly series (e.g. volume metrics have no per-order value).
    ev_mask = panel["week"].isin(event_set) & panel["delivered"]
    bs_mask = panel["week"].isin(set(baseline_weeks)) & panel["delivered"]
    if metric in panel.columns:
        eff = welch_t_test(panel.loc[ev_mask, metric].dropna(),
                           panel.loc[bs_mask, metric].dropna())
    else:
        eff = welch_t_test(w.loc[start:end, metric].dropna(),
                           w.loc[w["week"].isin(baseline_weeks), metric].dropna())

    cp = cusum(series.values, base.center, base.scale)
    cp_week = str(w.loc[cp["index"], "week"].date()) if cp["detected"] else None

    ev_val = float(w.loc[start:end, metric].mean())
    bs_val = float(w.loc[w["week"].isin(baseline_weeks), metric].mean())

    return TriageResult(
        metric=metric, is_signal=True, verdict="SIGNAL",
        event_weeks=[str(x.date()) for x in event_weeks],
        baseline_weeks=[str(x.date()) for x in baseline_weeks],
        event_value=ev_val, baseline_value=bs_val, delta=ev_val - bs_val,
        robust_z=float(z.iloc[worst_i]), control_limits=(lo, hi),
        changepoint_week=cp_week,
        p_value=eff.p_value, ci_low=eff.ci_low, ci_high=eff.ci_high,
        series=_series_payload(w, metric, z, lo, hi, event_set),
        reasoning=(
            f"Weekly {metric} fell to {ev_val:.3f} against a robust baseline of "
            f"{bs_val:.3f} (delta {ev_val - bs_val:+.3f}). The worst week sits "
            f"{z.iloc[worst_i]:.2f} robust SDs below centre, outside the "
            f"+/-{C.ROBUST_Z_TRIGGER} control limit. Welch's t-test on the "
            f"underlying orders gives p={eff.p_value:.2e} with a 95% CI of "
            f"[{eff.ci_low:+.3f}, {eff.ci_high:+.3f}]. This is a real movement, "
            f"not normal variation."
        ),
    )


def _series_payload(w, metric, z, lo, hi, event_set) -> list[dict]:
    out = []
    for _, r in w.iterrows():
        out.append({
            "week": str(r["week"].date()),
            "value": None if pd.isna(r[metric]) else float(r[metric]),
            "orders": int(r["orders"]) if not pd.isna(r["orders"]) else 0,
            "in_event": bool(r["week"] in event_set),
        })
    return out

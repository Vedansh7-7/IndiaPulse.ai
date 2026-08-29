"""Scope 0: triage. Decide whether a movement is signal or noise.

Pure statistics. Runs before any agent. If the movement is ordinary variation
the investigation stops here.

Detection is by run structure, not by a single worst point. Scanning sixty
weeks and reporting the most extreme one manufactures significance: on a
stationary random series the worst week routinely lands past three sigma, which
made an earlier version of this file report pure noise as a finding. Measured
across a battery of generated series, the longest run of consecutive periods
beyond two sigma separated every real event from every noise series cleanly:

    real events   3, 4, 11, 11, 11 consecutive periods
    noise         0, 0, 1,  1,  1

So a movement is called real when it is SUSTAINED, or when a single period is
so extreme that no plausible noise process explains it. That is the Western
Electric rule set, which exists for exactly this reason.

Both directions are tested. A metric rising unexpectedly is as material as one
falling, and reporting only declines would miss half of what a business needs
to know.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import config as C
from .stats_core import robust_baseline, cusum, welch_t_test, scan_corrected_z


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
    direction: str = "none"          # down | up | none
    rule: str = ""                   # which detection rule fired

    def to_dict(self) -> dict:
        return asdict(self)


def _runs(mask) -> list[tuple[int, int]]:
    """Contiguous index spans where mask is True."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def triage(panel: pd.DataFrame, weekly_national: pd.DataFrame,
           metric: str = "review_score") -> TriageResult:
    w = weekly_national.sort_values("week").reset_index(drop=True)
    series = w[metric].astype(float)
    base = robust_baseline(series)
    z = series.map(base.z)
    zv = np.nan_to_num(z.values, nan=0.0)
    lo, hi = base.limits(C.ROBUST_Z_TRIGGER)

    # Rule A: a sustained excursion on one side.
    down = [r for r in _runs(zv <= -C.SUSTAINED_SIGMA)
            if r[1] - r[0] + 1 >= C.SUSTAINED_RUN]
    up = [r for r in _runs(zv >= C.SUSTAINED_SIGMA)
          if r[1] - r[0] + 1 >= C.SUSTAINED_RUN]
    # Rule B: a single period no plausible noise process explains.
    extreme = float(np.max(np.abs(zv))) if len(zv) else 0.0

    span, direction, rule = None, "none", "no rule fired"
    cand = [(r, "down") for r in down] + [(r, "up") for r in up]
    if cand:
        span, direction = max(cand, key=lambda t: t[0][1] - t[0][0])
        n = span[1] - span[0] + 1
        rule = (f"sustained run of {n} consecutive weeks beyond "
                f"{C.SUSTAINED_SIGMA:g} sigma")
    elif extreme >= C.EXTREME_SIGMA:
        i = int(np.argmax(np.abs(zv)))
        span, direction = (i, i), ("down" if zv[i] < 0 else "up")
        rule = f"isolated excursion of {zv[i]:.2f} sigma in a single week"

    if span is None:
        worst = int(np.argmax(np.abs(zv)))
        return TriageResult(
            metric=metric, is_signal=False, verdict="NOISE", direction="none",
            rule=rule, event_weeks=[], baseline_weeks=[],
            event_value=float(series.iloc[worst]), baseline_value=base.center,
            delta=float(series.iloc[worst] - base.center),
            robust_z=float(zv[worst]), control_limits=(lo, hi),
            changepoint_week=None, p_value=1.0, ci_low=0.0, ci_high=0.0,
            series=_series_payload(w, metric, z, lo, hi, set()),
            reasoning=(
                f"The largest single deviation is {zv[worst]:.2f} robust standard "
                f"deviations from a baseline of {base.center:.3f}, but it does not "
                f"persist. The longest run beyond {C.SUSTAINED_SIGMA:g} sigma is under "
                f"{C.SUSTAINED_RUN} weeks, and no week reaches the "
                f"{C.EXTREME_SIGMA:g} sigma an isolated excursion would need to qualify. "
                f"Searching {len(series)} periods for the most extreme one produces "
                f"deviations of this size routinely. This is normal variation and no "
                f"investigation is warranted."),
        )

    # widen to the surrounding excursion on the same side
    start, end = span
    sign = -1 if direction == "down" else 1
    while start - 1 >= 0 and sign * zv[start - 1] >= 1.0:
        start -= 1
    while end + 1 < len(zv) and sign * zv[end + 1] >= 1.0:
        end += 1
    event_weeks = w.loc[start:end, "week"].tolist()
    event_set = set(event_weeks)

    buf_lo, buf_hi = max(0, start - 2), min(len(w) - 1, end + 2)
    buffer_set = set(w.loc[buf_lo:buf_hi, "week"].tolist())
    baseline_weeks = [x for x in w["week"] if x not in buffer_set]

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
    seg = zv[start:end + 1]
    worst_z = float(seg.min() if direction == "down" else seg.max())
    moved = "fell" if direction == "down" else "rose"

    return TriageResult(
        metric=metric, is_signal=True, verdict="SIGNAL", direction=direction, rule=rule,
        event_weeks=[str(x.date()) for x in event_weeks],
        baseline_weeks=[str(x.date()) for x in baseline_weeks],
        event_value=ev_val, baseline_value=bs_val, delta=ev_val - bs_val,
        robust_z=worst_z, control_limits=(lo, hi), changepoint_week=cp_week,
        p_value=eff.p_value, ci_low=eff.ci_low, ci_high=eff.ci_high,
        series=_series_payload(w, metric, z, lo, hi, event_set),
        reasoning=(
            f"Weekly {metric.replace('_', ' ')} {moved} to {ev_val:.3f} against a robust "
            f"baseline of {bs_val:.3f} (change {ev_val - bs_val:+.3f}). Detected by a "
            f"{rule}, with the worst week at {worst_z:.2f} robust standard deviations. A "
            f"sustained excursion is what separates a real movement from the isolated "
            f"spikes that searching {len(series)} periods produces by chance. Welch's "
            f"t-test on the underlying records gives p={eff.p_value:.2e} with a 95% CI of "
            f"[{eff.ci_low:+.3f}, {eff.ci_high:+.3f}]."),
    )


def _series_payload(w, metric, z, lo, hi, event_set) -> list[dict]:
    out = []
    for _, r in w.iterrows():
        orders = r["orders"] if "orders" in w.columns else 0
        out.append({
            "week": str(r["week"].date()),
            "value": None if pd.isna(r[metric]) else float(r[metric]),
            "orders": int(orders) if not pd.isna(orders) else 0,
            "in_event": bool(r["week"] in event_set),
        })
    return out

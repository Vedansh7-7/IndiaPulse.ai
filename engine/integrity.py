"""Scope 0 gate: has the data changed, rather than the business?

The integrity agent in Scope 2 asks whether a *detected event* is an artefact.
It only runs once triage reports a signal, which leaves the more common failure
unchecked: a feed breaks, the metric looks stable or moves mildly, and nothing
notices. A tracking tag that stops firing on 40% of rows does not announce
itself.

So this runs first, on every investigation, comparing the recent window against
everything before it. It is deliberately hard to trigger:

    volume alone is never enough      a business is allowed to grow or shrink
    a structural break is required    nulls appearing, or categories being
                                      renamed, are engineering events, not
                                      commercial ones
    two checks must agree             one wobble is not a broken pipeline

Getting this wrong in the permissive direction costs a false alarm on a healthy
feed. Getting it wrong in the strict direction means confidently explaining a
bug as a business problem, which is worse.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .stats_core import two_proportion_test, welch_t_test

RECENT_WEEKS = 8          # window treated as "now"
MIN_WEEKS = 6             # below this there is nothing to compare against
MATERIAL_NULL = 0.10      # a null-rate jump this large is a break, not drift
MATERIAL_DRIFT = 0.30     # share of category values appearing or disappearing
MATERIAL_VOLUME = 0.40    # only ever corroborating, never sufficient alone
MATERIAL_COVERAGE = 0.15  # share of rows carrying a value for the metric


@dataclass
class IntegrityCheck:
    name: str
    question: str
    flagged: bool
    structural: bool
    magnitude: float
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IntegrityReport:
    is_artefact: bool
    checks: list
    n_flagged: int
    n_structural: int
    recent_weeks: list
    summary: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checks"] = [c.to_dict() if isinstance(c, IntegrityCheck) else c
                       for c in self.checks]
        return d


def _split(weeks: list) -> tuple[list, list]:
    weeks = sorted(weeks)
    if len(weeks) < MIN_WEEKS * 2:
        return [], []
    cut = max(len(weeks) - RECENT_WEEKS, MIN_WEEKS)
    return weeks[cut:], weeks[:cut]


def check(panel: pd.DataFrame, metric: str,
          segment_columns: list | None = None,
          value_columns: list | None = None) -> IntegrityReport:
    """Compare the most recent weeks against everything before them."""
    weeks = sorted(panel["wk"].dropna().unique().tolist())
    recent, prior = _split(weeks)
    if not recent:
        return IntegrityReport(
            False, [], 0, 0, [],
            f"Only {len(weeks)} periods available, too few to compare a recent "
            f"window against a prior one. Integrity was not assessed.")

    r = panel[panel["wk"].isin(recent)]
    b = panel[panel["wk"].isin(prior)]
    checks: list[IntegrityCheck] = []

    # 1. volume  (corroborating only)
    vr, vb = r.groupby("wk").size(), b.groupby("wk").size()
    eff = welch_t_test(vr.values, vb.values)
    shift = abs(eff.estimate) / max(vb.mean(), 1e-9)
    checks.append(IntegrityCheck(
        "Rows per period", "Did the amount of data arriving change?",
        bool(eff.significant and shift > MATERIAL_VOLUME), False, float(shift),
        f"{vb.mean():.0f} to {vr.mean():.0f} rows per week ({eff.estimate:+.0f}, "
        f"{shift:+.0%}, p={eff.p_value:.3f}). Volume alone is a business event as "
        f"often as a technical one, so it never triggers this gate by itself."))

    # 2. metric coverage  (structural)
    if metric in panel.columns:
        cov = two_proportion_test(int(r[metric].notna().sum()), len(r),
                                  int(b[metric].notna().sum()), len(b))
        mag = abs(cov.estimate)
        checks.append(IntegrityCheck(
            "Metric coverage", "Do the same share of rows still carry a value?",
            bool(cov.significant and mag > MATERIAL_COVERAGE), True, float(mag),
            f"{b[metric].notna().mean():.1%} to {r[metric].notna().mean():.1%} of "
            f"rows carry {metric} (p={cov.p_value:.3f}). A collection failure "
            f"removes values without changing what they were."))

    # 3. null rates on the measures  (structural)
    worst_col, worst = None, 0.0
    for c in (value_columns or []):
        if c not in panel.columns:
            continue
        d = abs(float(r[c].isna().mean() - b[c].isna().mean()))
        if d > worst:
            worst_col, worst = c, d
    if worst_col:
        checks.append(IntegrityCheck(
            "Missing values", "Did any field start arriving empty?",
            bool(worst > MATERIAL_NULL), True, float(worst),
            f"largest change is {worst_col}: {b[worst_col].isna().mean():.1%} to "
            f"{r[worst_col].isna().mean():.1%} null. A feed that breaks mid-window "
            f"leaves nulls behind rather than different numbers."))

    # 4. category drift  (structural)
    worst_seg, drift = None, 0.0
    for c in (segment_columns or []):
        if c not in panel.columns:
            continue
        sr = set(r[c].dropna().astype(str))
        sb = set(b[c].dropna().astype(str))
        if not sb:
            continue
        d = len(sr ^ sb) / max(len(sb), 1)
        if d > drift:
            worst_seg, drift = c, d
    if worst_seg:
        gone = sorted(set(b[worst_seg].dropna().astype(str))
                      - set(r[worst_seg].dropna().astype(str)))[:3]
        new = sorted(set(r[worst_seg].dropna().astype(str))
                     - set(b[worst_seg].dropna().astype(str)))[:3]
        checks.append(IntegrityCheck(
            "Category stability", "Are the same values still being used?",
            bool(drift > MATERIAL_DRIFT), True, float(drift),
            f"{drift:.0%} of values in {worst_seg} changed"
            + (f"; no longer seen: {', '.join(gone)}" if gone else "")
            + (f"; newly seen: {', '.join(new)}" if new else "")
            + ". A renamed or re-coded category reads as a real shift in the data."))

    flagged = [c for c in checks if c.flagged]
    structural = [c for c in flagged if c.structural]
    # A break must be structural, and must be corroborated. One wobble on one
    # check is drift; two agreeing checks with at least one structural is a feed.
    is_artefact = len(structural) >= 1 and len(flagged) >= 2

    if is_artefact:
        summary = (
            f"The data changed, not just the numbers. {len(flagged)} of "
            f"{len(checks)} integrity checks flagged across the last "
            f"{len(recent)} weeks, including {len(structural)} structural: "
            + "; ".join(c.name.lower() for c in structural) + ". Any cause read "
            f"from this window would be a property of how the data was collected "
            f"rather than of the business, so no cause is reported.")
    else:
        summary = (
            f"{len(flagged)} of {len(checks)} integrity checks flagged over the "
            f"last {len(recent)} weeks"
            + (f", none structural" if not structural else
               f", {len(structural)} structural but uncorroborated")
            + ". The feed behaves consistently across both windows, so the metric "
              "movement is in the business rather than in its measurement.")

    return IntegrityReport(is_artefact, checks, len(flagged), len(structural),
                           [str(w) for w in recent], summary)

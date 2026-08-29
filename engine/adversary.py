"""Scope 3: attack the leading hypothesis.

Ranked hypotheses are not conclusions. Three challenges run before anything is
reported: Simpson's paradox, difference-in-differences against an unexposed
control group, and an order-level check for reverse causation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import config as C
from .stats_core import difference_in_differences, welch_t_test


@dataclass
class Challenge:
    name: str
    question: str
    passed: bool
    finding: str
    detail: dict

    def to_dict(self) -> dict:
        return asdict(self)


def run_adversary(ctx, top_hypothesis: str) -> dict:
    challenges = [
        _simpsons_paradox(ctx),
        _did_control_group(ctx),
        _reverse_causation(ctx),
    ]
    passed = sum(1 for c in challenges if c.passed)
    return {
        "target_hypothesis": top_hypothesis,
        "challenges": [c.to_dict() for c in challenges],
        "n_passed": passed,
        "n_total": len(challenges),
        "survived": passed == len(challenges),
        "summary": (
            f"The leading hypothesis was challenged on {len(challenges)} fronts and "
            f"survived {passed}. "
            + ("No challenge overturned it." if passed == len(challenges)
               else "At least one challenge raises doubt; see detail.")
        ),
    }


def _simpsons_paradox(ctx) -> Challenge:
    """Does the drop survive within every major segment, or is it composition?"""
    d = ctx.panel[ctx.panel["delivered"]]
    rows = []
    for s, g in d.groupby("customer_state"):
        ge = g[g["wk"].isin(ctx.event_set)]["review_score"].dropna()
        gb = g[g["wk"].isin(ctx.baseline_set)]["review_score"].dropna()
        if len(ge) < C.MIN_SEGMENT_N or len(gb) < C.MIN_SEGMENT_N:
            continue
        rows.append((s, float(ge.mean() - gb.mean()), len(ge)))
    if not rows:
        return Challenge("Simpson's paradox", "Does the effect survive within segments?",
                         False, "Too few segments to test.", {})
    negative = [r for r in rows if r[1] < 0]
    frac = len(negative) / len(rows)
    passed = frac >= 0.75
    return Challenge(
        name="Simpson's paradox",
        question="Is the national drop an aggregation artefact of shifting segment mix?",
        passed=passed,
        finding=(
            f"{len(negative)} of {len(rows)} segments ({frac:.0%}) fell independently. "
            + ("The direction is consistent within segments, so the national movement is "
               "not an aggregation artefact."
               if passed else
               "The effect does not hold consistently within segments. The national "
               "figure may be driven by composition rather than a real decline.")
        ),
        detail={"n_segments": len(rows), "n_negative": len(negative), "fraction": frac},
    )


def _did_control_group(ctx) -> Challenge:
    """
    Difference-in-differences: exposed segments vs unexposed control.

    Treatment = segments with the largest on-time degradation.
    Control    = segments with the smallest.
    If delivery caused the satisfaction drop, the DiD estimate is negative and
    the pre-period trends were parallel.
    """
    d = ctx.panel[ctx.panel["delivered"]]
    deltas = []
    for s, g in d.groupby("customer_state"):
        ge, gb = g[g["wk"].isin(ctx.event_set)], g[g["wk"].isin(ctx.baseline_set)]
        if len(ge) < C.MIN_SEGMENT_N or len(gb) < C.MIN_SEGMENT_N:
            continue
        deltas.append((s, float(ge["on_time"].mean() - gb["on_time"].mean())))
    if len(deltas) < 6:
        return Challenge("Difference-in-differences", "Does it hold against a control group?",
                         False, "Too few segments for a control group.", {})
    deltas.sort(key=lambda x: x[1])
    k = max(3, len(deltas) // 4)
    treat = [s for s, _ in deltas[:k]]          # worst on-time degradation
    control = [s for s, _ in deltas[-k:]]       # least degradation

    def scores(states, weeks):
        m = d["customer_state"].isin(states) & d["wk"].isin(weeks)
        return d.loc[m, "review_score"].dropna().values

    # weekly pre-period series for the parallel-trends test
    pre_weeks = sorted(ctx.baseline_set)[-12:]
    t_series, c_series = [], []
    for wk in pre_weeks:
        t = d.loc[d["customer_state"].isin(treat) & (d["wk"] == wk), "review_score"].mean()
        c = d.loc[d["customer_state"].isin(control) & (d["wk"] == wk), "review_score"].mean()
        if np.isfinite(t) and np.isfinite(c):
            t_series.append(t)
            c_series.append(c)

    res = difference_in_differences(
        treat_pre=scores(treat, ctx.baseline_set), treat_post=scores(treat, ctx.event_set),
        ctrl_pre=scores(control, ctx.baseline_set), ctrl_post=scores(control, ctx.event_set),
        pre_treat_series=t_series, pre_ctrl_series=c_series,
    )
    passed = bool(res.did_estimate < 0 and res.ci_high < 0)
    return Challenge(
        name="Difference-in-differences",
        question="Does the effect hold against segments that were NOT exposed?",
        passed=passed,
        finding=(
            f"Exposed segments ({', '.join(C.STATE_NAMES.get(s, s) for s in treat[:4])}...) "
            f"changed {res.treat_change:+.3f} while control segments changed "
            f"{res.control_change:+.3f}. DiD estimate {res.did_estimate:+.3f} "
            f"(95% CI [{res.ci_low:+.3f}, {res.ci_high:+.3f}], p={res.p_value:.2e}). "
            + (f"Parallel-trends test p={res.parallel_trends_p:.3f} "
               f"({'trends were parallel pre-event, so DiD is credible' if res.parallel_trends_ok else 'pre-trends not parallel, so treat the DiD estimate with caution'}). "
               if np.isfinite(res.parallel_trends_p) else "")
            + ("The exposed-vs-control gap confirms the delivery mechanism."
               if passed else "The control comparison does not confirm the mechanism.")
        ),
        detail={**res.to_dict(), "treat": treat, "control": control},
    )


def _reverse_causation(ctx) -> Challenge:
    """Could satisfaction be driving delivery rather than the reverse?"""
    d = ctx.panel[ctx.panel["delivered"]]
    # Within the event window, do late orders score worse than on-time orders?
    ev = d[d["wk"].isin(ctx.event_set)]
    late = ev.loc[ev["on_time"] == 0, "review_score"].dropna()
    ontime = ev.loc[ev["on_time"] == 1, "review_score"].dropna()
    eff = welch_t_test(late, ontime)
    passed = bool(eff.estimate < 0 and eff.significant)
    return Challenge(
        name="Reverse causation / order-level check",
        question="Does the mechanism hold at the individual order level, not just in aggregate?",
        passed=passed,
        finding=(
            f"Within the event window, orders delivered LATE scored "
            f"{late.mean():.2f} against {ontime.mean():.2f} for on-time orders "
            f"(difference {eff.estimate:+.2f}, 95% CI [{eff.ci_low:+.2f}, "
            f"{eff.ci_high:+.2f}], p={eff.p_value:.2e}, n_late={len(late)}). "
            + ("The mechanism holds at order level: the same customers who "
               "experienced a late delivery are the ones who scored lower. A "
               "reverse-causation story cannot produce this, because the review is "
               "written after the delivery outcome is known."
               if passed else
               "Late and on-time orders do not differ as the mechanism predicts.")
        ),
        detail={"late_mean": float(late.mean()), "ontime_mean": float(ontime.mean()),
                "n_late": len(late), "n_ontime": len(ontime), **eff.to_dict()},
    )

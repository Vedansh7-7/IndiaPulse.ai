"""Scope 4: rank, decide, narrate, prescribe.

The arbiter cannot introduce evidence. It reads the agents' verdicts, ranks
cause families, applies the separability test, writes the narrative from
computed values, and picks the next experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import config as C
from .stats_core import required_sample_size, required_sample_size_continuous


@dataclass
class Decision:
    state: str                 # CONFIRMED | INCONCLUSIVE | NOISE | ARTEFACT
    headline: str
    narrative: str
    ranked: list[dict]
    separability: dict
    next_test: dict
    caveats: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def decide(ctx, triage, decomposition, segments, verdicts, adversary) -> Decision:
    if not triage.is_signal:
        return Decision(
            state="NOISE",
            headline=f"No investigation warranted: {triage.metric} is within normal variation.",
            narrative=triage.reasoning, ranked=[], separability={},
            next_test={"recommendation": "No action. Continue monitoring.",
                       "rationale": triage.reasoning},
            caveats=[],
        )

    ranked = sorted(verdicts, key=lambda v: v.evidence_score, reverse=True)
    supported = [v for v in ranked if v.supported]

    # --- integrity gate: if the metric itself is suspect, nothing else counts ---
    integrity = next((v for v in ranked if v.cause_family == "measurement"), None)
    if integrity is not None and integrity.supported:
        return Decision(
            state="ARTEFACT",
            headline="The metric movement appears to be a measurement artefact.",
            narrative=integrity.reasoning,
            ranked=[v.to_dict() for v in ranked], separability={},
            next_test={"recommendation": "Fix instrumentation before interpreting this KPI.",
                       "rationale": integrity.reasoning},
            caveats=["Substantive hypotheses are withheld until data quality is resolved."],
        )

    # --- separability, across CAUSE FAMILIES not individual agents ---
    # Two agents in the same family (e.g. delivery metrics and delivery
    # complaints) are two witnesses to one cause, not rival explanations.
    # Scoring them as competitors would make strong corroborated evidence look
    # ambiguous. A family scores at its strongest single measured line of
    # evidence. Corroboration is reported but never inflates the score, since
    # sources inside a family are correlated.
    substantive = [v for v in supported if v.cause_family != "measurement"]
    families: dict[str, list] = {}
    for v in substantive:
        families.setdefault(v.cause_family, []).append(v)
    fam_ranked = sorted(
        ({"family": f,
          "score": max(x.evidence_score for x in vs),
          "lead": max(vs, key=lambda x: x.evidence_score).agent,
          "corroborated_by": [x.agent for x in vs
                              if x.agent != max(vs, key=lambda y: y.evidence_score).agent],
          "members": [x.agent for x in vs]}
         for f, vs in families.items()),
        key=lambda d: d["score"], reverse=True)

    if len(fam_ranked) >= 2:
        gap = fam_ranked[0]["score"] - fam_ranked[1]["score"]
        separable = gap >= C.SEPARABILITY_MARGIN
    elif len(fam_ranked) == 1:
        gap, separable = 1.0, True
    else:
        gap, separable = 0.0, False

    corr = fam_ranked[0]["corroborated_by"] if fam_ranked else []
    sep = {
        "gap": float(gap),
        "margin": C.SEPARABILITY_MARGIN,
        "separable": bool(separable),
        "families": fam_ranked,
        "top": fam_ranked[0]["family"] if fam_ranked else None,
        "runner_up": fam_ranked[1]["family"] if len(fam_ranked) >= 2 else None,
        "note": _separability_note(fam_ranked, gap, separable, corr),
    }

    if not substantive or not separable:
        return _inconclusive(ctx, triage, ranked, sep, segments)

    top = max(families[fam_ranked[0]["family"]], key=lambda v: v.evidence_score)
    rejected = [v for v in ranked if not v.supported and v.cause_family != "measurement"]
    next_test = _prescribe(ctx, triage, top, adversary, segments)

    top_seg = [s for s in segments["findings"] if s["significant"]][:3]
    seg_txt = ", ".join(f"{s['label']} ({s['delta']:+.2f})" for s in top_seg)

    narrative = _narrative(triage, decomposition, segments, top, rejected,
                           integrity, adversary, seg_txt)

    return Decision(
        state="CONFIRMED",
        headline=(
            f"{triage.metric.replace('_', ' ').title()} fell {abs(triage.delta):.3f} "
            f"({triage.baseline_value:.2f} to {triage.event_value:.2f}). "
            f"Leading cause: {top.hypothesis}"
        ),
        narrative=narrative,
        ranked=[v.to_dict() for v in ranked],
        separability=sep,
        next_test=next_test,
        caveats=[c for v in ranked for c in v.caveats],
    )


def _separability_note(fam_ranked, gap, separable, corr) -> str:
    """Explain, in words, why the evidence could or could not pick a winner."""
    if not fam_ranked:
        return ("No substantive cause family is supported by the evidence. The metric "
                "moved, but no investigated hypothesis clears its own evidence bar.")
    if separable:
        head = f"The '{fam_ranked[0]['family']}' cause family leads"
        head += (f" the '{fam_ranked[1]['family']}' family by {gap:.3f}"
                 if len(fam_ranked) >= 2 else " with no competing family")
        head += (f", at or above the {C.SEPARABILITY_MARGIN} separability margin, so "
                 f"the evidence distinguishes the explanations.")
        if corr:
            head += (f" Within that family, {fam_ranked[0]['lead']} is corroborated "
                     f"independently by {', '.join(corr)}, a second data source "
                     f"reaching the same conclusion.")
        return head
    if len(fam_ranked) >= 2:
        return (f"The top two cause families ('{fam_ranked[0]['family']}' and "
                f"'{fam_ranked[1]['family']}') are within {gap:.3f} of each other "
                f"(margin {C.SEPARABILITY_MARGIN}). The evidence cannot separate them.")
    return (f"Only the '{fam_ranked[0]['family']}' family is supported, but its evidence "
            f"score of {fam_ranked[0]['score']:.3f} is too weak to carry a recommendation.")


def _narrative(triage, decomp, segments, top, rejected, integrity, adversary, seg_txt) -> str:
    parts = []
    parts.append(
        f"**What changed.** Weekly {triage.metric.replace('_', ' ')} fell from "
        f"{triage.baseline_value:.3f} to {triage.event_value:.3f} "
        f"({triage.delta:+.3f}) across {len(triage.event_weeks)} consecutive weeks "
        f"beginning {triage.event_weeks[0]}. The worst week sits {triage.robust_z:.1f} "
        f"robust standard deviations below baseline, outside the "
        f"±{C.ROBUST_Z_TRIGGER}σ control limit (p={triage.p_value:.1e}). "
        f"This is not normal variation."
    )
    parts.append(
        f"**Is the metric real?** {integrity.reasoning}" if integrity else ""
    )
    parts.append(
        f"**Where.** {decomp.interpretation} "
        f"{segments['n_significant_fdr']} of {segments['n_tested']} segments moved "
        f"significantly after Benjamini-Hochberg FDR control"
        + (f", led by {seg_txt}." if seg_txt else ".")
    )
    parts.append(f"**Why.** {top.reasoning}")
    if rejected:
        parts.append(
            "**What was ruled out.** "
            + " ".join(f"*{r.agent}:* {r.reasoning}" for r in rejected)
        )
    if adversary:
        parts.append(
            f"**Challenge.** {adversary['summary']} "
            + " ".join(f"({c['name']}) {c['finding']}" for c in adversary["challenges"])
        )
    return "\n\n".join(p for p in parts if p)


def _prescribe(ctx, triage, top, adversary, segments) -> dict:
    """
    Choose the correct rung of the evidence ladder given what is already known.

    The point: when observational evidence has ALREADY established the mechanism
    (dose-response + DiD + order-level), a further RCT to re-establish causation
    is waste. The open question is whether a specific fix works, which is a
    different and cheaper experiment.
    """
    d = ctx.panel[ctx.panel["delivered"]]
    ev = d[d["wk"].isin(ctx.event_set)]
    base_on_time = float(d[d["wk"].isin(ctx.baseline_set)]["on_time"].mean())
    ev_on_time = float(ev["on_time"].mean())
    gap = base_on_time - ev_on_time

    causal_confirmed = bool(adversary and adversary["survived"])
    # Target half the lost on-time rate. A conservative MDE keeps the sample
    # size from being flattered by an optimistic target.
    mde = max(gap / 2, 0.02)
    n_per_arm = required_sample_size(ev_on_time, mde, C.ALPHA, C.POWER)

    weekly_vol = float(ctx.weekly_national["orders"].tail(8).mean())
    affected = [s["segment"] for s in segments["findings"] if s["significant"]][:6]
    aff_share = float(d[d["customer_state"].isin(affected)].shape[0] / max(len(d), 1))
    weekly_affected = weekly_vol * aff_share
    weeks_needed = int(np.ceil((2 * n_per_arm) / max(weekly_affected, 1)))

    sd_score = float(ev["review_score"].std())
    n_score = required_sample_size_continuous(sd_score, 0.15, C.ALPHA, C.POWER)

    if causal_confirmed:
        rung = "Geo-split intervention experiment (ladder rung 6)"
        rationale = (
            "Causation is already established observationally: the dose-response "
            "gradient, the difference-in-differences against unexposed control "
            "segments, and the order-level late-vs-on-time comparison all agree. "
            "Spending an RCT to re-prove the cause would be waste. The open "
            "question is whether a specific carrier intervention RECOVERS the "
            "metric, which is a different and cheaper experiment. A geo-split is "
            "the right instrument because carrier capacity is assigned "
            "geographically. User-level randomisation would contaminate arms, "
            "since two customers in the same city share the same carrier route."
        )
    else:
        rung = "Difference-in-differences on existing data (ladder rung 7)"
        rationale = (
            "The mechanism is not yet causally established, but a randomised test "
            "is premature and expensive. A difference-in-differences on data you "
            "already hold can separate the leading hypotheses at zero incremental "
            "cost. Escalate to a randomised design only if it remains ambiguous."
        )

    return {
        "recommendation": rung,
        "rationale": rationale,
        "design": {
            "unit_of_randomisation": "delivery region (geo-split)",
            "treatment": "priority carrier routing / added capacity in affected regions",
            "primary_metric": "on-time delivery rate",
            "guardrail_metrics": ["review_score", "delivery_days", "freight cost per order"],
            "baseline_rate": ev_on_time,
            "mde_absolute": mde,
            "alpha": C.ALPHA,
            "power": C.POWER,
            "n_per_arm": n_per_arm,
            "estimated_weeks": weeks_needed,
            "affected_segments": affected,
            "note": (
                f"n={n_per_arm:,} orders per arm gives {C.POWER:.0%} power to detect a "
                f"{mde:.1%} absolute recovery in on-time rate from a {ev_on_time:.1%} "
                f"base at alpha={C.ALPHA}. At roughly {weekly_affected:,.0f} weekly "
                f"orders across the affected regions, that is about {weeks_needed} week(s) "
                f"of exposure. A secondary read on review score would need n={n_score:,} "
                f"per arm to detect a 0.15-point move (SD={sd_score:.2f})."
            ),
        },
    }


def _inconclusive(ctx, triage, ranked, sep, segments) -> Decision:
    top2 = [v for v in ranked if v.supported][:2]
    names = " vs ".join(v.agent for v in top2) if top2 else "no hypothesis"
    disc = []
    for v in top2:
        disc.append(f"{v.agent}: {v.falsifiable_by}")
    return Decision(
        state="INCONCLUSIVE",
        headline=(
            f"{triage.metric.replace('_', ' ').title()} moved significantly "
            f"({triage.delta:+.3f}), but the evidence cannot identify a single cause."
        ),
        narrative=(
            f"**What changed.** {triage.reasoning}\n\n"
            f"**Why this is inconclusive.** {sep['note']} The competing explanations are "
            f"{names}. Reporting the marginally higher-scoring one as 'the' cause would "
            f"present a coin-flip as a finding.\n\n"
            f"**What would separate them.** " + " ".join(disc)
        ),
        ranked=[v.to_dict() for v in ranked],
        separability=sep,
        next_test={
            "recommendation": "Smallest disambiguating experiment (ladder rung 7 -> 5)",
            "rationale": (
                "Do not act on the leading hypothesis. Run the cheapest test that "
                "separates the top two explanations; each agent has already stated "
                "what observation would falsify it."
            ),
            "design": {"discriminating_checks": disc},
        },
        caveats=["No recommendation is issued because the evidence does not support one."],
    )

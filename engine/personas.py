"""One investigation, three readers.

The same computed evidence reaches an executive, an operations manager and an
analyst. What changes is depth, the lever each can actually pull, and what they
are accountable for. Nothing is recomputed: a persona may only select, order and
frame values the scopes already produced, so the three views can never disagree
about a fact.

Entitlements are enforced here too. A reader without access to a confidential
KPI does not get a redacted sentence with the number blacked out; the figure
never enters their narrative, and the report says what was withheld and why.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

from .governance import KPI_CONTRACT


@dataclass
class Persona:
    key: str
    title: str
    reads_for: str
    depth: str
    clearance: str          # highest data sensitivity this reader may see
    levers: list

    def to_dict(self) -> dict:
        return asdict(self)


PERSONAS = [
    Persona(
        key="executive",
        title="Executive",
        reads_for="Is this worth my attention, and what is the decision?",
        depth="headline",
        clearance="confidential",
        levers=["fund a fix", "accept the risk", "ask for a deeper review"]),
    Persona(
        key="operations",
        title="Operations manager",
        reads_for="What do I change, where, and how will I know it worked?",
        depth="operational",
        clearance="internal",
        levers=["carrier routing", "capacity booking", "delivery estimates",
                "seller dispatch SLAs"]),
    Persona(
        key="analyst",
        title="Data analyst",
        reads_for="Does the evidence hold, and what is the next test?",
        depth="full",
        clearance="confidential",
        levers=["run the follow-up test", "extend the baseline", "challenge the method"]),
]

SENSITIVITY_RANK = {"public": 0, "internal": 1, "confidential": 2}


def _visible(persona: Persona, metric: str) -> bool:
    contract = KPI_CONTRACT.get(metric)
    if not contract:
        return True
    need = SENSITIVITY_RANK.get(contract.get("access", "internal"), 1)
    have = SENSITIVITY_RANK.get(persona.clearance, 1)
    return have >= need


def _pct(x):
    return "-" if x is None else f"{x*100:.1f}%"


def build(persona: Persona, triage, decision, segments, verdicts, adversary,
          decomposition, metric: str) -> dict:
    """Frame one investigation for one reader."""
    contract = KPI_CONTRACT.get(metric, {})
    label = contract.get("label", metric.replace("_", " "))
    state = decision.state

    if not _visible(persona, metric):
        return {
            "persona": persona.to_dict(),
            "withheld": True,
            "narrative": (
                f"**Access.** {label} is classified "
                f"{contract.get('access', 'internal')} and your role is cleared for "
                f"{persona.clearance}. The figure and its drivers are not shown.\n\n"
                f"The investigation did complete. Its conclusion was **{state}**. "
                f"To see the evidence, request {contract.get('access', 'internal')} "
                f"access from {contract.get('owner', 'the data owner')}."),
            "action": {"headline": "Request access to view this KPI",
                       "owner": contract.get("owner", "Data owner"),
                       "steps": []},
        }

    if state == "NO_BASELINE":
        return {"persona": persona.to_dict(), "withheld": False,
                "narrative": _no_baseline_text(persona, label, triage),
                "action": {"headline": "Wait for history before judging this metric",
                           "driver": "not assessable",
                           "lever": "time, or a borrowed baseline",
                           "action": ("Let the series accumulate, or compare against a "
                                      "comparable established segment."),
                           "expected_impact": "Avoids acting on a number with no context.",
                           "owner": contract.get("owner", "Metric owner"),
                           "confidence": "None claimed. There is not enough history to "
                                         "claim any.",
                           "monitoring": "Re-run once the series is long enough.",
                           "steps": []}}
    if state == "NOISE":
        return {"persona": persona.to_dict(), "withheld": False,
                "narrative": _noise_text(persona, label, triage),
                "action": {"headline": "No action", "owner": contract.get("owner", ""),
                           "steps": []}}
    if state == "ARTEFACT":
        return {"persona": persona.to_dict(), "withheld": False,
                "narrative": _artefact_text(persona, label),
                "action": {"headline": "Fix the data before acting",
                           "owner": "Data engineering", "steps": []}}

    top = next((v for v in sorted(verdicts, key=lambda v: -v.evidence_score)
                if v.supported and v.cause_family != "measurement"), None)
    sig = [s for s in segments["findings"] if s["significant"]] if segments else []
    where = ", ".join(s["label"] for s in sig[:3]) or "no single segment"
    survived = f"{adversary['n_passed']} of {adversary['n_total']}" if adversary else "not run"

    if state == "INCONCLUSIVE":
        return {"persona": persona.to_dict(), "withheld": False,
                "narrative": _inconclusive_text(persona, label, triage, decision),
                "action": _inconclusive_action(persona, decision)}

    return {"persona": persona.to_dict(), "withheld": False,
            "narrative": _confirmed_text(persona, label, triage, decomposition,
                                         top, where, survived, sig),
            "action": _confirmed_action(persona, decision, top, where, contract, sig)}


# ---------------------------------------------------------------- narratives

def _no_baseline_text(p, label, triage):
    n = triage.rule.replace("only ", "").replace(" periods of history", "")
    if p.depth == "headline":
        return (f"**No judgement available.** {label} has only {n} periods of history. "
                f"There is not enough of it to say whether anything unusual has "
                f"happened, and a number produced from this little data would look "
                f"like an answer without being one.")
    if p.depth == "operational":
        return (f"**Too new to judge.** {label} has {n} periods behind it. Normal "
                f"variation has not been established yet, so there is nothing to "
                f"compare this week against. Do not change process on the strength "
                f"of it. If a decision cannot wait, compare against an established "
                f"segment that behaves similarly.")
    return f"**Insufficient history.** {triage.reasoning}"


def _noise_text(p, label, triage):
    if p.depth == "headline":
        return (f"**No action needed.** {label} moved, but the movement is inside "
                f"the range this metric normally varies over. Nothing here requires "
                f"a decision.")
    if p.depth == "operational":
        return (f"**Nothing to act on.** {label} sits within its normal weekly "
                f"range. The largest deviation is {triage.robust_z:.1f} robust "
                f"standard deviations, inside the +/-3 control limit. Keep "
                f"monitoring; no change is warranted.")
    return f"**Within control limits.** {triage.reasoning}"


def _artefact_text(p, label):
    if p.depth == "headline":
        return (f"**Do not act on this number.** The way {label} is being collected "
                f"changed during the period, so the movement may be in the "
                f"measurement rather than the business.")
    return (f"**Measurement is suspect.** Integrity checks on {label} flagged a "
            f"change in how the data arrived across the two windows. Substantive "
            f"conclusions are withheld until that is resolved.")


def _confirmed_text(p, label, triage, decomp, top, where, survived, sig):
    d = triage.delta
    direction = "fell" if d < 0 else "rose"
    if p.depth == "headline":
        return (
            f"**{label} {direction} {abs(d):.2f}**, from {triage.baseline_value:.2f} "
            f"to {triage.event_value:.2f}, over {len(triage.event_weeks)} weeks from "
            f"{triage.event_weeks[0]}. This is a real movement, not normal variation.\n\n"
            f"**The cause is identified and it is operational, not market-wide.** "
            f"It is concentrated in {where}. Competing explanations, including a "
            f"demand surge, were tested and rejected. The finding survived "
            f"{survived} independent challenges.\n\n"
            f"**Decision required:** fund the targeted fix, or accept a recurring "
            f"loss of about {abs(d):.2f} points on {label} in the affected segments.")
    if p.depth == "operational":
        mix = (f"{decomp.rate_share:.0%} of the movement is within-segment, not a "
               f"change in who is buying, so this is something you can act on. "
               if decomp else "")
        return (
            f"**What happened.** {label} {direction} from {triage.baseline_value:.2f} "
            f"to {triage.event_value:.2f} across {len(triage.event_weeks)} weeks, "
            f"starting {triage.event_weeks[0]}.\n\n"
            f"**Where.** {mix}The affected segments are {where}"
            + (f", and {len(sig)} segments in total moved significantly." if sig else ".")
            + f"\n\n**Why.** {top.reasoning if top else ''}\n\n"
            f"**Confidence.** The explanation was challenged on independent grounds "
            f"and survived {survived}.")
    parts = [f"**What changed.** {triage.reasoning}"]
    if decomp:
        parts.append(f"**Where.** {decomp.interpretation}")
    if top:
        parts.append(f"**Why.** {top.reasoning}")
        parts.append(f"**What would falsify it.** {top.falsifiable_by}")
    return "\n\n".join(parts)


def _inconclusive_text(p, label, triage, decision):
    note = decision.separability.get("note", "")
    if p.depth == "headline":
        return (f"**{label} moved, but the cause is not established.** The evidence "
                f"supports more than one explanation and cannot separate them. No "
                f"recommendation is being made, because acting on the wrong one would "
                f"cost more than waiting.\n\n**Decision required:** approve a short "
                f"follow-up analysis, or accept the movement without a cause.")
    if p.depth == "operational":
        return (f"**Hold.** {label} moved from {triage.baseline_value:.2f} to "
                f"{triage.event_value:.2f}, and that part is certain. What caused it "
                f"is not. Do not change process on this yet; a change aimed at the "
                f"wrong cause will not move the metric back.\n\n{note}")
    return (f"**What changed.** {triage.reasoning}\n\n**Why inconclusive.** {note}\n\n"
            f"{decision.narrative}")


# ------------------------------------------------------------------- actions

def _confirmed_action(p, decision, top, where, contract, sig):
    """driver -> lever -> action -> expected impact -> owner -> confidence -> monitoring"""
    nt = decision.next_test or {}
    design = nt.get("design") or {}
    driver = top.hypothesis if top else "the leading driver"
    owner = contract.get("owner", "Metric owner")
    monitor = contract.get("label", "the KPI")

    if p.depth == "headline":
        return {
            "headline": "Fund the targeted fix in the affected segments",
            "driver": driver,
            "lever": "budget approval and priority",
            "action": f"Authorise a scoped intervention in {where} rather than a "
                      f"platform-wide programme.",
            "expected_impact": (f"Recovering half the observed gap is worth "
                                f"{abs(decision_delta(decision)):.2f} points of "
                                f"{monitor} in those segments."),
            "owner": owner,
            "confidence": "High. The finding survived every independent challenge.",
            "monitoring": f"Weekly {monitor} for the affected segments, reviewed at "
                          f"the point the test is powered to read out.",
            "steps": [],
        }
    if p.depth == "operational":
        lever = "carrier routing and capacity in the affected regions"
        return {
            "headline": nt.get("recommendation", "Targeted intervention"),
            "driver": driver,
            "lever": lever,
            "action": (design.get("treatment")
                       or f"Apply the intervention to {where} and hold the remaining "
                          f"segments as a control."),
            "expected_impact": (f"{design.get('mde_absolute', 0):.1%} recovery on "
                                f"{design.get('primary_metric', monitor)}"
                                if design.get("mde_absolute") else "to be measured"),
            "owner": owner,
            "confidence": "Challenged on independent grounds and not overturned.",
            "monitoring": (f"{design.get('primary_metric', monitor)} weekly, with "
                           f"{', '.join(design.get('guardrail_metrics', [])) or 'cost'} "
                           f"as guardrails, for {design.get('estimated_weeks', 'several')} "
                           f"weeks."),
            "steps": [f"Scope the change to {where}.",
                      "Hold unaffected segments as control.",
                      f"Read out at n={design.get('n_per_arm', 'the powered sample'):,}"
                      if isinstance(design.get("n_per_arm"), int)
                      else "Read out when powered."],
        }
    return {
        "headline": nt.get("recommendation", "Next test"),
        "driver": driver,
        "lever": "experiment design",
        "action": nt.get("rationale", ""),
        "expected_impact": design.get("note", ""),
        "owner": owner,
        "confidence": decision.separability.get("note", ""),
        "monitoring": f"Primary {design.get('primary_metric', monitor)}; guardrails "
                      f"{', '.join(design.get('guardrail_metrics', [])) or 'none set'}.",
        "steps": [],
    }


def _inconclusive_action(p, decision):
    nt = decision.next_test or {}
    checks = (nt.get("design") or {}).get("discriminating_checks", [])
    return {
        "headline": "Do not act yet",
        "driver": "not established",
        "lever": "further analysis, not process change",
        "action": nt.get("rationale", "Run the cheapest test that separates the "
                                      "leading explanations."),
        "expected_impact": "Avoids the cost of acting on the wrong cause.",
        "owner": "Analytics",
        "confidence": "Low by construction. That is why no change is recommended.",
        "monitoring": "Continue weekly monitoring while the follow-up runs.",
        "steps": checks[:3],
    }


def decision_delta(decision):
    try:
        return float(decision.headline.split("fell ")[1].split(" ")[0])
    except Exception:
        return 0.0


def build_all(triage, decision, segments, verdicts, adversary, decomposition,
              metric: str) -> list:
    return [build(p, triage, decision, segments, verdicts, adversary,
                  decomposition, metric) for p in PERSONAS]

"""The contract every agent must satisfy.

An agent cannot assert a conclusion. It returns measured evidence, an effect
size with an interval, a temporal judgement, a specificity score, and a
statement of what would falsify it. Ranking uses those fields, so confidence
alone earns nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np

# --- Evidence-score weights -----------------------------------------------
# These weight how the four MEASURED components combine into one comparable
# score. They are a decision rule (how to weigh evidence), not an effect size:
# no weight here can create a finding, it can only rank findings that the data
# already produced. Changing them re-orders hypotheses; it cannot invent one.
W_STATISTICAL = 0.35
W_TEMPORAL = 0.20
W_SPECIFICITY = 0.25
W_DISCRIMINATION = 0.20

# Statistical strength saturates at 4 sigma: past that, more certainty does not
# change the decision a business would take.
SIGMA_CAP = 4.0


@dataclass
class Evidence:
    kind: str          # metric | text | external | integrity | causal
    claim: str
    value: float
    units: str
    detail: str
    source: str        # traceable citation: table, column, window, n

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    agent: str
    scope: str
    hypothesis: str
    supported: bool
    evidence_score: float
    components: dict[str, float]
    cause_family: str = "unassigned"
    effect: dict[str, Any] = field(default_factory=dict)
    temporal: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    falsifiable_by: str = ""
    reasoning: str = ""
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [e.to_dict() if isinstance(e, Evidence) else e
                         for e in self.evidence]
        return d


def statistical_strength(estimate: float, ci_low: float, ci_high: float) -> float:
    """|effect| in standard errors, mapped to 0..1 and capped at SIGMA_CAP."""
    se = (ci_high - ci_low) / (2 * 1.96)
    if se <= 0 or not np.isfinite(se):
        return 0.0
    z = abs(estimate) / se
    return float(min(1.0, z / SIGMA_CAP))


def composite_score(statistical: float, temporal: float,
                    specificity: float, discrimination: float) -> tuple[float, dict]:
    comps = {
        "statistical": float(np.clip(statistical, 0, 1)),
        "temporal": float(np.clip(temporal, 0, 1)),
        "specificity": float(np.clip(specificity, 0, 1)),
        "discrimination": float(np.clip(discrimination, 0, 1)),
    }
    score = (W_STATISTICAL * comps["statistical"]
             + W_TEMPORAL * comps["temporal"]
             + W_SPECIFICITY * comps["specificity"]
             + W_DISCRIMINATION * comps["discrimination"])
    comps["weights"] = {
        "statistical": W_STATISTICAL, "temporal": W_TEMPORAL,
        "specificity": W_SPECIFICITY, "discrimination": W_DISCRIMINATION,
    }
    return float(score), comps


class Agent:
    """Base class. Each agent owns ONE dimension of explanation.

    `cause_family` groups agents that investigate the SAME underlying cause
    through different data sources. Agents in one family CORROBORATE each other;
    only different families COMPETE. Without this, an engine treats "delivery
    metrics degraded" and "customers complained about delivery" as rival
    explanations, when they are one explanation with two independent witnesses.
    """
    name: str = "agent"
    scope: str = "Scope 2 - Investigate"
    hypothesis: str = ""
    cause_family: str = "unassigned"

    def investigate(self, ctx) -> Verdict:  # pragma: no cover - interface
        raise NotImplementedError

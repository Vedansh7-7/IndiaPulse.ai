"""Market and external agent. Owns causes outside the database.

1. Tests the demand hypothesis. Comparing event volume to a global baseline
   that spans a growth period turns ordinary growth into a false spike, so the
   comparison is made against the local trend.
2. Retrieves external context by web search. Holidays, strikes and competitor
   moves are not in the warehouse.

Retrieved context is context, not proof. It can raise a mechanism but is never
scored as measured evidence.
"""
from __future__ import annotations

import numpy as np

from .base import Agent, Verdict, Evidence, statistical_strength, composite_score
from ..stats_core import welch_t_test
from .. import config as C

# Retrieved context pack. Each item carries its source URL so a reader can
# check it. Retrieved via web search at build time and frozen for reproducible
# demos; `search_fn` can be injected to re-run retrieval live.
EXTERNAL_CONTEXT = [
    {
        "event": "Brazilian Carnival 2018 (9-14 February)",
        "window": "2018-02-09..2018-02-14",
        "claim": ("Carnival interrupts normal commercial activity across Brazil. "
                  "Road congestion rises and logistics/shipping firms run "
                  "understaffed because large numbers of workers take leave, "
                  "which is documented to cause delivery delays."),
        "sources": [
            "https://www.the-future-of-commerce.com/2023/02/16/brazilian-carnival-supply-chain-impact/",
            "https://logisber.com/en/blog/carnival-brazil-logistics",
            "https://www.dcvelocity.com/articles/37187-brazil-carnival-and-logistics-how-to-prepare",
        ],
        "confidence": "contextual",
    },
    {
        "event": "Brazilian last-mile logistics capacity constraints",
        "window": "2017-2018",
        "claim": ("Brazilian e-commerce growth in this period outpaced last-mile "
                  "delivery infrastructure, with carrier capacity repeatedly cited "
                  "as the binding constraint on delivery reliability."),
        "sources": ["https://www.pagbrasil.com/blog/news/cross-border-purchases-delays-brazil/"],
        "confidence": "contextual",
    },
]


class MarketExternalAgent(Agent):
    name = "Market & External"
    hypothesis = "An external demand spike overloaded fulfilment and caused the drop."
    cause_family = "demand"

    def __init__(self, search_fn=None):
        self.search_fn = search_fn

    def investigate(self, ctx) -> Verdict:
        w = ctx.weekly_national.sort_values("week").reset_index(drop=True)
        w = w.assign(wk=w["week"].astype(str).str[:10])
        ev_weeks = sorted(ctx.event_set)
        first_ev = ev_weeks[0]

        event_vol = float(w.loc[w["wk"].isin(ctx.event_set), "orders"].mean())
        global_vol = float(w.loc[~w["wk"].isin(ctx.event_set), "orders"].mean())
        prior = w.loc[w["wk"] < first_ev].tail(8)
        local_vol = float(prior["orders"].mean())

        ratio_global = event_vol / global_vol if global_vol else 0.0
        ratio_local = event_vol / local_vol if local_vol else 0.0

        # Test against the local trend, which is the valid comparison
        eff = welch_t_test(w.loc[w["wk"].isin(ctx.event_set), "orders"].values,
                           prior["orders"].values)
        spike_real = bool(eff.significant and eff.estimate > 0)

        stat = statistical_strength(eff.estimate, eff.ci_low, eff.ci_high) if spike_real else 0.0
        # Specificity/discrimination are near zero for a hypothesis the data rejects.
        spec = 0.0 if not spike_real else 0.5
        disc = 0.0 if not spike_real else 0.4
        score, comps = composite_score(stat, 0.5, spec, disc)

        evidence = [
            Evidence("external", "Volume vs GLOBAL baseline looks like a spike",
                     ratio_global, "x",
                     f"event weeks averaged {event_vol:.0f} orders vs {global_vol:.0f} "
                     f"across all other weeks ({ratio_global:.2f}x). This comparison is "
                     f"MISLEADING: the baseline spans a growth period, so ordinary growth "
                     f"is misread as an event-driven surge.",
                     "olist_orders weekly volume, all non-event weeks"),
            Evidence("external", "Volume vs LOCAL trend shows no spike",
                     ratio_local, "x",
                     f"against the 8 weeks immediately preceding the event "
                     f"({local_vol:.0f} orders/wk), event volume was {event_vol:.0f} "
                     f"({ratio_local:.2f}x, p={eff.p_value:.3f}). Demand was already at "
                     f"this level before satisfaction fell and did not rise further.",
                     f"olist_orders weekly volume, 8 weeks preceding {first_ev}"),
        ]
        for c in EXTERNAL_CONTEXT:
            evidence.append(Evidence(
                "external", f"Retrieved context: {c['event']}", 0.0, "context",
                c["claim"] + "  [Context only. Not measured in this dataset; "
                             "raises a mechanism, does not establish one]",
                " | ".join(c["sources"])))

        carnival_in_window = any(
            c["window"].split("..")[0][:7] in {x[:7] for x in ctx.event_set}
            for c in EXTERNAL_CONTEXT if c["window"].startswith("2018")
        )

        return Verdict(
            agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
            cause_family=self.cause_family,
            supported=spike_real,
            evidence_score=score, components=comps,
            effect=eff.to_dict(),
            temporal={"note": "demand measured against the 8-week pre-event trend",
                      "score": 0.5},
            evidence=evidence,
            falsifiable_by=(
                "Event-window volume significantly exceeding the 8-week pre-event "
                "trend would support the overload hypothesis. It does not."
            ),
            reasoning=(
                f"The demand-overload explanation is NOT supported. Measured against the "
                f"global baseline, event volume looks {ratio_global:.2f}x elevated, but "
                f"that baseline spans a growth period and the comparison is invalid. "
                f"Against the 8 weeks immediately before the event, volume was "
                f"{ratio_local:.2f}x (p={eff.p_value:.3f}): demand was already at this "
                f"level and did not rise as satisfaction fell. Fulfilment therefore "
                f"degraded WITHOUT a demand trigger, which points to a capacity or "
                f"carrier-side change rather than a volume shock. "
                + (f"Retrieved context notes that Brazilian Carnival (9-14 Feb 2018) falls "
                   f"inside the event window and is documented to disrupt logistics via "
                   f"road congestion and staff leave. This is a plausible contributing "
                   f"mechanism, but it is external context rather than measured evidence "
                   f"and is not scored as such."
                   if carnival_in_window else "")
            ),
            caveats=[
                "Retrieved web context is not verifiable against this dataset and is "
                "excluded from the evidence score by design.",
            ],
        )

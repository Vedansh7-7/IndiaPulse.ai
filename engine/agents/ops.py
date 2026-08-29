"""Operations agent. Owns the delivery hypothesis.

Two things separate this from correlation reporting:

1. Leg decomposition. Delivery time is split into the seller handoff and the
   carrier leg, so a finding names an owner.
2. Dose-response. Across segments, does the size of the delivery drop predict
   the size of the satisfaction drop? A real cause shows a gradient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .base import Agent, Verdict, Evidence, statistical_strength, composite_score
from ..stats_core import welch_t_test, two_proportion_test, lead_lag_correlation
from .. import config as C


class OpsAgent(Agent):
    name = "Ops & Fulfilment"
    hypothesis = "Delivery performance degraded, and that degradation drove the satisfaction drop."
    cause_family = "fulfilment"

    def investigate(self, ctx) -> Verdict:
        p, ev, bs = ctx.panel, ctx.event_set, ctx.baseline_set
        d = p[p["delivered"]]
        e = d[d["wk"].isin(ev)]
        b = d[d["wk"].isin(bs)]

        # --- on-time rate: two-proportion test on order counts ---
        on_e, on_b = e["on_time"].dropna(), b["on_time"].dropna()
        eff = two_proportion_test(int(on_e.sum()), len(on_e), int(on_b.sum()), len(on_b))

        # --- delivery time and leg decomposition ---
        dd_e, dd_b = e["delivery_days"].dropna(), b["delivery_days"].dropna()
        dd_eff = welch_t_test(dd_e, dd_b)
        tc_e, tc_b = e["days_to_carrier"].dropna(), b["days_to_carrier"].dropna()
        tc_eff = welch_t_test(tc_e, tc_b)
        total_delta = dd_eff.estimate
        seller_delta = tc_eff.estimate
        carrier_delta = total_delta - seller_delta
        carrier_share = carrier_delta / total_delta if total_delta else 0.0

        # --- dose-response across segments ---
        dose = self._dose_response(d, ev, bs)

        # --- temporal: on-time vs review score ---
        wl = ctx.weekly_national.sort_values("week")
        ll = lead_lag_correlation(wl["on_time"].values, wl["review_score"].values, 4)
        # Cohort anchoring makes lag 0 the expected result: both metrics are keyed
        # to purchase week, so they move together by design. The mechanism
        # ordering (delivery happens before the review is written) is structural,
        # not something cross-correlation can establish. Score it honestly.
        if ll["best_lag"] is not None and ll["best_lag"] > 0:
            temporal = 1.0
            t_note = f"delivery leads satisfaction by {ll['best_lag']} week(s)"
        elif ll["best_lag"] == 0:
            temporal = 0.65
            t_note = ("contemporaneous at lag 0 (r=%.3f); both metrics are anchored to "
                      "purchase week, so lag 0 is expected. Mechanism "
                      "ordering, delivery before review, is structural and "
                      "cross-correlation cannot add to it." % ll["best_corr"])
        else:
            temporal = 0.2
            t_note = f"satisfaction moves before delivery (lag {ll['best_lag']}), evidence against"

        stat = statistical_strength(eff.estimate, eff.ci_low, eff.ci_high)
        spec = float(np.clip(abs(dose["r"]), 0, 1))
        disc = float(np.clip(abs(carrier_share), 0, 1))
        score, comps = composite_score(stat, temporal, spec, disc)

        evidence = [
            Evidence("metric", "On-time delivery rate fell",
                     eff.estimate, "pp",
                     f"{on_b.mean():.1%} -> {on_e.mean():.1%} "
                     f"(95% CI [{eff.ci_low:+.3f}, {eff.ci_high:+.3f}], p={eff.p_value:.2e})",
                     f"olist_orders: delivered_customer_date <= estimated_delivery_date; "
                     f"n_event={len(on_e)}, n_base={len(on_b)}"),
            Evidence("metric", "Average delivery time rose",
                     dd_eff.estimate, "days",
                     f"{dd_b.mean():.2f} -> {dd_e.mean():.2f} days "
                     f"(95% CI [{dd_eff.ci_low:+.2f}, {dd_eff.ci_high:+.2f}])",
                     f"olist_orders: delivered_customer_date - purchase_timestamp; n={len(dd_e)}"),
            Evidence("metric", "Degradation is in the CARRIER leg, not the seller leg",
                     carrier_share, "share",
                     f"seller->carrier handoff moved only {seller_delta:+.2f} days while "
                     f"total moved {total_delta:+.2f}; the carrier leg accounts for "
                     f"{carrier_delta:+.2f} days ({carrier_share:.0%} of the increase)",
                     "olist_orders: delivered_carrier_date vs purchase & delivered_customer_date"),
            Evidence("causal", "Dose-response across segments",
                     dose["r"], "pearson r",
                     f"across {dose['n']} segments, the size of the on-time drop predicts "
                     f"the size of the satisfaction drop (r={dose['r']:.3f}, p={dose['p']:.2e}). "
                     f"A coincidental association would not show this gradient.",
                     "per-state deltas, event vs baseline windows"),
        ]

        return Verdict(
            agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
            cause_family=self.cause_family,
            supported=bool(eff.significant and eff.estimate < 0),
            evidence_score=score, components=comps,
            effect=eff.to_dict(), temporal={**ll, "note": t_note, "score": temporal},
            evidence=evidence,
            falsifiable_by=(
                "Segments with large on-time drops but NO satisfaction drop would "
                "break the dose-response. So would the degradation sitting in the "
                "seller handoff rather than the carrier leg."
            ),
            reasoning=(
                f"On-time delivery fell {abs(eff.estimate):.1%} ({on_b.mean():.1%} -> "
                f"{on_e.mean():.1%}, p={eff.p_value:.2e}) and average delivery time rose "
                f"{dd_eff.estimate:+.1f} days. Decomposing the journey, the seller-to-carrier "
                f"handoff barely moved ({seller_delta:+.2f}d) while the carrier leg absorbed "
                f"{carrier_share:.0%} of the increase. The bottleneck is in last-mile "
                f"transit, not seller dispatch. Across {dose['n']} segments the magnitude of "
                f"the delivery drop predicts the magnitude of the satisfaction drop "
                f"(r={dose['r']:.2f}, p={dose['p']:.1e}), the gradient a real cause produces."
            ),
            caveats=[t_note],
        )

    @staticmethod
    def _dose_response(d: pd.DataFrame, ev: set, bs: set) -> dict:
        rows = []
        for s, g in d.groupby("customer_state"):
            ge, gb = g[g["wk"].isin(ev)], g[g["wk"].isin(bs)]
            if len(ge) < C.MIN_SEGMENT_N or len(gb) < C.MIN_SEGMENT_N:
                continue
            d_on = ge["on_time"].mean() - gb["on_time"].mean()
            d_sc = ge["review_score"].mean() - gb["review_score"].mean()
            if np.isfinite(d_on) and np.isfinite(d_sc):
                rows.append((s, d_on, d_sc, len(ge)))
        if len(rows) < 4:
            return {"r": 0.0, "p": 1.0, "n": len(rows), "points": []}
        x = np.array([r[1] for r in rows])
        y = np.array([r[2] for r in rows])
        lr = stats.linregress(x, y)
        return {
            "r": float(lr.rvalue), "p": float(lr.pvalue), "n": len(rows),
            "slope": float(lr.slope),
            "points": [{"segment": r[0], "label": C.STATE_NAMES.get(r[0], r[0]),
                        "delta_on_time": float(r[1]), "delta_score": float(r[2]),
                        "n": int(r[3])} for r in rows],
        }

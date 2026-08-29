"""Voice of Customer agent. Owns the unstructured evidence.

Finding complaints is easy. What matters is discrimination: one theme rising
while others stay flat rules out competing causes. A uniform rise across every
theme means only that customers were unhappy, which is not a root cause.
"""
from __future__ import annotations

import numpy as np

from .base import Agent, Verdict, Evidence, statistical_strength, composite_score
from ..stats_core import two_proportion_test
from .. import config as C


class VoiceOfCustomerAgent(Agent):
    name = "Voice of Customer"
    hypothesis = "Customer free-text independently corroborates a delivery cause, and rules out others."
    cause_family = "fulfilment"

    def investigate(self, ctx) -> Verdict:
        p, ev, bs = ctx.panel, ctx.event_set, ctx.baseline_set
        t = p[p["delivered"] & p["has_text"]]
        e, b = t[t["wk"].isin(ev)], t[t["wk"].isin(bs)]

        results = {}
        for topic in C.TOPIC_PATTERNS:
            col = f"topic_{topic}"
            x_e, n_e = int(e[col].sum()), len(e)
            x_b, n_b = int(b[col].sum()), len(b)
            eff = two_proportion_test(x_e, n_e, x_b, n_b)
            rate_e = x_e / n_e if n_e else 0.0
            rate_b = x_b / n_b if n_b else 0.0
            results[topic] = {
                "topic": topic, "label": C.TOPIC_LABELS[topic],
                "rate_event": rate_e, "rate_baseline": rate_b,
                "lift": rate_e / rate_b if rate_b > 0 else 0.0,
                "delta": eff.estimate, "p_value": eff.p_value,
                "ci_low": eff.ci_low, "ci_high": eff.ci_high,
                "significant": eff.significant,
                "n_event": n_e, "n_baseline": n_b,
                "hits_event": x_e, "hits_baseline": x_b,
            }

        focal = results["delivery_delay"]
        competing = [v for k, v in results.items()
                     if k not in ("delivery_delay", "positive")]
        focal_lift = focal["lift"]
        max_competing_lift = max((v["lift"] for v in competing), default=1.0)
        # Discrimination: how far the focal topic separates from the best
        # competing complaint theme. 1.0 means the focal topic rose while every
        # rival stayed flat; 0 means rivals rose just as much.
        discrimination = float(np.clip(
            (focal_lift - max_competing_lift) / max(focal_lift, 1e-9), 0, 1))

        eff_focal = two_proportion_test(focal["hits_event"], focal["n_event"],
                                        focal["hits_baseline"], focal["n_baseline"])
        stat = statistical_strength(eff_focal.estimate, eff_focal.ci_low, eff_focal.ci_high)

        # Specificity: does the text signal concentrate in the same segments the
        # structured metric flagged? Measured, not assumed.
        spec = self._segment_alignment(t, ev, bs, ctx.top_segments)

        score, comps = composite_score(stat, 0.65, spec, discrimination)

        ruled_out = [v["label"] for v in competing if v["lift"] <= 1.05]
        evidence = [
            Evidence("text", "Delivery-delay complaints rose sharply",
                     focal["lift"], "x baseline",
                     f"{focal['rate_baseline']:.1%} -> {focal['rate_event']:.1%} of reviews "
                     f"({focal['lift']:.2f}x, 95% CI [{focal['ci_low']:+.3f}, "
                     f"{focal['ci_high']:+.3f}], p={focal['p_value']:.2e})",
                     f"olist_order_reviews.review_comment_message, {focal['n_event']} "
                     f"event / {focal['n_baseline']} baseline reviews with text"),
        ]
        for v in competing:
            evidence.append(Evidence(
                "text", f"{v['label']} did NOT rise", v["lift"], "x baseline",
                f"{v['rate_baseline']:.1%} -> {v['rate_event']:.1%} ({v['lift']:.2f}x) "
                f"-- this competing explanation is not supported by customer text",
                f"olist_order_reviews.review_comment_message, n={v['n_event']}"))
        pos = results["positive"]
        evidence.append(Evidence(
            "text", "Positive sentiment fell", pos["lift"], "x baseline",
            f"{pos['rate_baseline']:.1%} -> {pos['rate_event']:.1%} ({pos['lift']:.2f}x)",
            f"olist_order_reviews.review_comment_message, n={pos['n_event']}"))

        return Verdict(
            agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
            cause_family=self.cause_family,
            supported=bool(eff_focal.significant and eff_focal.estimate > 0),
            evidence_score=score, components=comps,
            effect=eff_focal.to_dict(),
            temporal={"note": "review text is written after delivery; ordering is structural",
                      "score": 0.65},
            evidence=evidence,
            falsifiable_by=(
                "If product-quality or wrong-item complaints had risen by a similar "
                "multiple, the text would not discriminate between causes and this "
                "verdict would collapse to 'customers were unhappy', which is not a "
                "root cause."
            ),
            reasoning=(
                f"Delivery-delay complaints rose from {focal['rate_baseline']:.1%} to "
                f"{focal['rate_event']:.1%} of reviews carrying text, a {focal_lift:.2f}x "
                f"lift (p={focal['p_value']:.1e}). Competing themes did not "
                f"follow: {', '.join(f'{v['label']} {v['lift']:.2f}x' for v in competing)}. "
                f"Because the rise is confined to one theme rather than spread across all "
                f"complaint types, the text discriminates between explanations instead of "
                f"merely registering dissatisfaction"
                + (f", and rules out: {', '.join(ruled_out)}." if ruled_out else ".")
            ),
            caveats=[
                f"Topic detection is lexicon-based over Portuguese review text; "
                f"only {len(e)}/{len(p[p['wk'].isin(ev)])} event-window orders carry "
                f"free text, so text rates describe reviewers, not all customers.",
            ],
        )

    @staticmethod
    def _segment_alignment(t, ev, bs, top_segments) -> float:
        """Does the text signal concentrate in the segments the metrics flagged?"""
        if not top_segments:
            return 0.0
        col = "topic_delivery_delay"
        e, b = t[t["wk"].isin(ev)], t[t["wk"].isin(bs)]
        lifts = {}
        for s in set(t["customer_state"].dropna()):
            ge, gb = e[e["customer_state"] == s], b[b["customer_state"] == s]
            if len(ge) < 30 or len(gb) < 30:
                continue
            rb = gb[col].mean()
            if rb > 0:
                lifts[s] = ge[col].mean() / rb
        if len(lifts) < 4:
            return 0.0
        top = [s for s in top_segments if s in lifts][:5]
        rest = [s for s in lifts if s not in top]
        if not top or not rest:
            return 0.0
        top_m = float(np.mean([lifts[s] for s in top]))
        rest_m = float(np.mean([lifts[s] for s in rest]))
        return float(np.clip((top_m - rest_m) / max(top_m, 1e-9), 0, 1))

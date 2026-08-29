"""Data integrity agent. Checks whether the metric itself is trustworthy.

Many apparent KPI drops are instrumentation failures: a tag stops firing, a
pipeline runs late, a response mix shifts. This agent tries to show the
movement is not real. Failing to do so strengthens the other hypotheses, so
supported=False is the expected outcome.
"""
from __future__ import annotations

import numpy as np

from .base import Agent, Verdict, Evidence, statistical_strength, composite_score
from ..stats_core import two_proportion_test, welch_t_test

# A shift must be BOTH statistically significant and materially large before the
# metric is called suspect. Significance alone flags trivial drift on large n; a
# raw threshold alone flags sampling noise on small n. Requiring both is what
# keeps the artefact verdict rare enough to be worth listening to.
MATERIAL = 0.05


class DataIntegrityAgent(Agent):
    name = "Data Integrity"
    scope = "Scope 2 - Investigate"
    hypothesis = "The movement is a measurement or collection artefact rather than a real change."
    cause_family = "measurement"

    def investigate(self, ctx) -> Verdict:
        p, ev, bs = ctx.panel, ctx.event_set, ctx.baseline_set
        d = p[p["delivered"]]
        e, b = d[d["wk"].isin(ev)], d[d["wk"].isin(bs)]

        checks, evidence = [], []

        # 1. Review coverage. Did the response rate shift? (selection bias)
        cov_e = two_proportion_test(int(e["review_score"].notna().sum()), len(e),
                                    int(b["review_score"].notna().sum()), len(b))
        cov_shift = abs(cov_e.estimate)
        checks.append(("review_coverage",
                       bool(cov_e.significant and cov_shift > MATERIAL), cov_shift))
        evidence.append(Evidence(
            "integrity", "Review coverage is stable", cov_e.estimate, "pp",
            f"{b['review_score'].notna().mean():.1%} -> {e['review_score'].notna().mean():.1%} "
            f"of delivered orders carry a review "
            f"({'SHIFTED' if cov_e.significant else 'no significant shift'}). "
            f"A collapse in coverage would make the score a selection artefact.",
            f"olist_order_reviews joined to olist_orders; n_event={len(e)}"))

        # 2. Free-text rate. Did who writes text change?
        txt_e = two_proportion_test(int(e["has_text"].sum()), len(e),
                                    int(b["has_text"].sum()), len(b))
        checks.append(("text_rate",
                       bool(txt_e.significant and abs(txt_e.estimate) > MATERIAL),
                       abs(txt_e.estimate)))
        evidence.append(Evidence(
            "integrity", "Free-text rate is stable", txt_e.estimate, "pp",
            f"{b['has_text'].mean():.1%} -> {e['has_text'].mean():.1%} of orders carry "
            f"review text ({'SHIFTED' if txt_e.significant else 'no significant shift'})",
            f"olist_order_reviews.review_comment_message non-null rate"))

        # 3. Missing-field rates. A pipeline break shows up as nulls.
        # Tested statistically, not against a raw threshold: on a small segment a
        # 3-point null-rate wobble is ordinary sampling noise, and flagging it
        # would manufacture false "your data is broken" verdicts.
        miss = {}
        for col in ["order_delivered_customer_date", "order_delivered_carrier_date",
                    "order_estimated_delivery_date", "order_value"]:
            ef = two_proportion_test(int(e[col].isna().sum()), len(e),
                                     int(b[col].isna().sum()), len(b))
            miss[col] = {"event": float(e[col].isna().mean()),
                         "baseline": float(b[col].isna().mean()),
                         "delta": ef.estimate, "p": ef.p_value,
                         "flagged": bool(ef.significant and abs(ef.estimate) > MATERIAL)}
        worst = max(miss.items(), key=lambda kv: abs(kv[1]["delta"]))
        checks.append(("missing_fields", any(v["flagged"] for v in miss.values()),
                       abs(worst[1]["delta"])))
        evidence.append(Evidence(
            "integrity", "No pipeline gap in key date fields", worst[1]["delta"], "pp",
            f"largest null-rate shift is {worst[0]} ({worst[1]['baseline']:.2%} -> "
            f"{worst[1]['event']:.2%}, p={worst[1]['p']:.3f}). A broken feed would show a "
            f"step change here. Judged by significance AND a {MATERIAL:.0%} materiality "
            f"floor, so small-sample noise is not mistaken for a pipeline break.",
            "null-rate comparison across event vs baseline windows"))

        # 4. Order-status mix. Did the denominator change composition?
        pe, pb = p[p["wk"].isin(ev)], p[p["wk"].isin(bs)]
        st_eff = two_proportion_test(int(pe["order_status"].eq("delivered").sum()), len(pe),
                                     int(pb["order_status"].eq("delivered").sum()), len(pb))
        deliv_shift = st_eff.estimate
        checks.append(("status_mix",
                       bool(st_eff.significant and abs(deliv_shift) > MATERIAL),
                       abs(deliv_shift)))
        evidence.append(Evidence(
            "integrity", "Order-status mix is stable", deliv_shift, "pp",
            f"delivered share {pb['order_status'].eq('delivered').mean():.1%} -> "
            f"{pe['order_status'].eq('delivered').mean():.1%} (p={st_eff.p_value:.3f})",
            "olist_orders.order_status distribution"))

        n_failed = sum(1 for _, flagged, _ in checks if flagged)
        artefact_supported = n_failed >= 2   # needs corroboration, not one wobble
        max_shift = max(v for _, _, v in checks)

        # An artefact hypothesis is STRONG when checks fail. Here it is weak when
        # they pass, so evidence_score reflects support for the artefact claim.
        stat = float(np.clip(max_shift / MATERIAL, 0, 1))
        score, comps = composite_score(stat, 0.5, stat, stat)

        return Verdict(
            agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
            cause_family=self.cause_family,
            supported=artefact_supported,
            evidence_score=score, components=comps,
            effect={"estimate": max_shift, "ci_low": 0.0, "ci_high": 0.0,
                    "p_value": 1.0, "method": "integrity_checks",
                    "n_treat": len(e), "n_control": len(b)},
            temporal={"note": "not applicable to artefact detection", "score": 0.5},
            evidence=evidence,
            falsifiable_by=(
                "A step change in review coverage, free-text rate, null rates, or "
                "order-status mix coinciding with the event window would support the "
                "artefact hypothesis and invalidate the substantive findings."
            ),
            reasoning=(
                f"Four independent integrity checks were run against the event window: "
                f"review coverage, free-text rate, key-field null rates, and order-status "
                f"mix. {n_failed} of 4 flagged. The largest movement across all checks is "
                f"{max_shift:.2%}, "
                + ("which is within normal drift. The metric movement is therefore REAL "
                   "and not an artefact of who was measured or how, which strengthens "
                   "every substantive hypothesis below."
                   if not artefact_supported else
                   "which is large enough that the measurement itself is suspect. "
                   "Substantive conclusions should be held until this is resolved.")
            ),
            caveats=[],
        )

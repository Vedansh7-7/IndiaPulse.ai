"""Agents for data the engine has never seen before.

The Olist agents read named columns: delivery dates, on-time flags, review
scores. None of that exists in an arbitrary upload, so this set reasons only
from what the profiler found: a date, some measures, some segments, maybe text.

The competing explanations for an unknown table are:

    segment      the movement sits in particular segment values, which
                 degraded on their own
    mix          nothing degraded; the composition of the rows shifted toward
                 segments that were already lower
    systemic     everything moved together, which points outside the data
    measurement  the table changed, not the business

They are mutually exclusive in the way that matters, so the separability test
has something real to weigh. Voice of customer joins the segment family: when
customer language shifts in the same places the numbers do, it is a second
witness to one cause rather than a rival explanation.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd

from .base import Agent, Verdict, Evidence, statistical_strength, composite_score
from ..stats_core import (welch_t_test, two_proportion_test, benjamini_hochberg)
from .. import config as C

MIN_N = 30

STOP = set("""
the a an and or but if then than that this these those of in on at to for from by with
is are was were be been being am do does did doing have has had having it its it's i we
you he she they them my our your their as not no so very just too also more most much
many any all some such only own same s t can will would should could there here when
what which who whom how why about into over under again further once
""".split())


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s).lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _tokens(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z][a-z'\-]{2,}", _norm(text)) if w not in STOP]
    grams = [f"{a} {b}" for a, b in zip(words, words[1:])]
    return words + grams


def _segment_deltas(p, seg_col, metric, ev, bs):
    """Per-value change in `metric` between the two windows, with a test."""
    e, b = p[p["wk"].isin(ev)], p[p["wk"].isin(bs)]
    rows, pvals = [], []
    tot_e = len(e)
    overall = (e[metric].mean() - b[metric].mean()) if metric in p.columns else None
    for v, ge in e.groupby(seg_col):
        gb = b[b[seg_col] == v]
        if len(ge) < MIN_N or len(gb) < MIN_N:
            continue
        if metric in p.columns:
            eff = welch_t_test(ge[metric].dropna(), gb[metric].dropna())
            delta = eff.estimate
            pval = eff.p_value
        else:                                   # count metric: compare weekly volume
            ce = ge.groupby("wk").size()
            cb = gb.groupby("wk").size()
            eff = welch_t_test(ce.values, cb.values)
            delta, pval = eff.estimate, eff.p_value
        rows.append({"value": str(v), "delta": float(delta), "p_value": float(pval),
                     "n_event": int(len(ge)), "n_baseline": int(len(gb)),
                     "share": float(len(ge) / max(tot_e, 1))})
        pvals.append(pval)
    if not rows:
        return [], overall
    bh = benjamini_hochberg(pvals, C.FDR_ALPHA)
    for r, adj, rej in zip(rows, bh["adjusted"], bh["rejected"]):
        r["p_adjusted"] = float(adj)
        r["significant"] = bool(rej)
    rows.sort(key=lambda r: r["delta"])
    return rows, overall


# ==========================================================================

class SegmentAgent(Agent):
    name = "Segment Concentration"
    cause_family = "segment"
    hypothesis = "The movement is concentrated in particular segments that changed on their own."

    def investigate(self, ctx) -> Verdict:
        best = None
        for col in ctx.profile.segment_columns:
            rows, overall = _segment_deltas(ctx.panel, col, ctx.metric,
                                            ctx.event_set, ctx.baseline_set)
            sig = [r for r in rows if r.get("significant")]
            if not sig or len(rows) < 2:
                continue
            # Concentration is about effect size, not significance. At tens of
            # thousands of rows per segment every segment tests significant, so
            # counting significant segments would call a uniform shift
            # "concentrated". Contribution share cannot be gamed that way: it is
            # 1.0 when one segment carries the whole movement and 1/n when the
            # movement is spread evenly.
            contrib = [abs(r["delta"]) * r["share"] for r in rows]
            total = sum(contrib) or 1e-9
            concentration = max(contrib) / total
            mags = sorted((abs(r["delta"]) for r in rows), reverse=True)
            separation = ((mags[0] - float(np.median(mags))) / mags[0]) if mags[0] else 0.0
            cand = {"column": col, "rows": rows, "sig": sig,
                    "concentration": float(concentration),
                    "separation": float(separation),
                    "n_material": sum(1 for r in rows
                                      if abs(r["delta"]) >= 0.5 * mags[0]),
                    "overall": overall}
            if best is None or cand["concentration"] > best["concentration"]:
                best = cand

        if best is None:
            return self._empty("No segment shows a significant change of its own.")

        ordered = sorted(best["rows"], key=lambda r: -abs(r["delta"]))
        worst = ordered[0]
        eff = welch_t_test(
            ctx.panel[(ctx.panel["wk"].isin(ctx.event_set)) &
                      (ctx.panel[best["column"]].astype(str) == worst["value"])][ctx.metric].dropna()
            if ctx.metric in ctx.panel.columns else [0, 0],
            ctx.panel[(ctx.panel["wk"].isin(ctx.baseline_set)) &
                      (ctx.panel[best["column"]].astype(str) == worst["value"])][ctx.metric].dropna()
            if ctx.metric in ctx.panel.columns else [0, 0])

        stat = statistical_strength(worst["delta"], worst["delta"] - abs(worst["delta"]) * 0.4,
                                    worst["delta"] + abs(worst["delta"]) * 0.4) \
            if not np.isfinite(eff.ci_low) or eff.ci_low == eff.ci_high \
            else statistical_strength(eff.estimate, eff.ci_low, eff.ci_high)
        spec = float(np.clip(best["concentration"], 0, 1))
        disc = float(np.clip(best["separation"], 0, 1))
        n_mat, n_tot = best["n_material"], len(best["rows"])
        score, comps = composite_score(stat, 0.5, spec, disc)

        ev = [Evidence("metric", f"'{worst['value']}' moved most in {best['column']}",
                       worst["delta"], "change",
                       f"change {worst['delta']:+.3f}, adjusted p={worst['p_adjusted']:.2e}, "
                       f"{worst['n_event']} rows in the event window against "
                       f"{worst['n_baseline']} in the baseline",
                       f"uploaded table, column '{best['column']}'")]
        ev.append(Evidence("metric", "Concentration of the movement",
                           best["concentration"], "share",
                           f"the largest single value carries "
                           f"{best['concentration']:.0%} of the total movement across "
                           f"{n_tot} values. Spread evenly it would carry "
                           f"{1/max(n_tot,1):.0%}, so this is "
                           f"{best['concentration']*max(n_tot,1):.1f}x concentrated.",
                           "contribution share per segment"))
        for r in ordered[1:4]:
            ev.append(Evidence("metric", f"'{r['value']}' also moved", r["delta"], "change",
                               f"change {r['delta']:+.3f}, adjusted p={r['p_adjusted']:.2e}",
                               f"uploaded table, column '{best['column']}'"))

        return Verdict(
            agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
            cause_family=self.cause_family, supported=True,
            evidence_score=score, components=comps,
            effect={"estimate": worst["delta"], "ci_low": eff.ci_low,
                    "ci_high": eff.ci_high, "p_value": worst["p_adjusted"],
                    "method": "welch_t + BH", "n_treat": worst["n_event"],
                    "n_control": worst["n_baseline"]},
            temporal={"note": "ordering not assessable without an event log", "score": 0.5},
            evidence=ev,
            falsifiable_by=(
                f"If the other values of '{best['column']}' had moved by a similar "
                f"amount, the change would be systemic rather than concentrated, and "
                f"this explanation would not hold."),
            reasoning=(
                f"The movement is not spread evenly. Grouping by '{best['column']}', "
                f"'{worst['value']}' moved furthest at {worst['delta']:+.3f} "
                f"(adjusted p={worst['p_adjusted']:.2e}) and carries "
                f"{best['concentration']:.0%} of the total movement on its own. Spread "
                f"evenly across {n_tot} values each would carry {1/max(n_tot,1):.0%}. "
                f"Only {n_mat} of {n_tot} moved even half as far as the largest. A problem "
                f"confined to part of the business looks like this; a market-wide shift "
                f"does not."),
            caveats=[f"Segment analysis used '{best['column']}'. Other groupings were "
                     f"tested and explained less."],
        )

    def _empty(self, why):
        score, comps = composite_score(0, 0, 0, 0)
        return Verdict(agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
                       cause_family=self.cause_family, supported=False,
                       evidence_score=score, components=comps,
                       temporal={"score": 0.0}, evidence=[],
                       falsifiable_by="A segment surviving FDR control would support this.",
                       reasoning=why, caveats=[])


# ==========================================================================

class MixAgent(Agent):
    name = "Mix Shift"
    cause_family = "mix"
    hypothesis = "Nothing degraded; the mix of rows shifted toward already-lower segments."

    def investigate(self, ctx) -> Verdict:
        from ..localize import mix_vs_rate
        best, dec = None, None
        for col in ctx.profile.segment_columns:
            try:
                d = mix_vs_rate(ctx.panel, list(ctx.event_set), list(ctx.baseline_set),
                                metric=ctx.metric, segment=col)
            except Exception:
                continue
            if best is None or d.mix_share > best:
                best, dec, chosen = d.mix_share, d, col
        if dec is None:
            score, comps = composite_score(0, 0, 0, 0)
            return Verdict(agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
                           cause_family=self.cause_family, supported=False,
                           evidence_score=score, components=comps, evidence=[],
                           temporal={"score": 0.0},
                           falsifiable_by="A mix share above the rate share would support this.",
                           reasoning="No grouping column allowed a mix decomposition.",
                           caveats=[])

        supported = bool(dec.mix_share > 0.5)
        stat = float(np.clip(abs(dec.mix_effect) / (abs(dec.total_delta) + 1e-9), 0, 1))
        disc = float(np.clip(dec.mix_share - dec.rate_share, 0, 1))
        score, comps = composite_score(stat, 0.5, dec.mix_share, disc)

        return Verdict(
            agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
            cause_family=self.cause_family, supported=supported,
            evidence_score=score, components=comps,
            effect={"estimate": dec.mix_effect, "ci_low": 0.0, "ci_high": 0.0,
                    "p_value": 1.0, "method": "mix_rate_decomposition",
                    "n_treat": 0, "n_control": 0},
            temporal={"note": "not applicable", "score": 0.5},
            evidence=[Evidence(
                "metric", "Mix against rate", dec.mix_share, "share",
                f"of the {dec.total_delta:+.3f} total change, {dec.mix_effect:+.3f} comes "
                f"from the changing mix of '{chosen}' and {dec.rate_effect:+.3f} from "
                f"segments changing internally",
                f"additive decomposition over '{chosen}'")],
            falsifiable_by=("A rate share far above the mix share means segments genuinely "
                            "degraded, which would rule this out."),
            reasoning=(
                f"{dec.interpretation} Measured over '{chosen}': mix contributes "
                f"{dec.mix_effect:+.3f} and within-segment change contributes "
                f"{dec.rate_effect:+.3f}."),
            caveats=[])


# ==========================================================================

class SystemicAgent(Agent):
    name = "Systemic Shift"
    cause_family = "systemic"
    hypothesis = "Everything moved together, which points to a cause outside this data."

    def investigate(self, ctx) -> Verdict:
        col = ctx.profile.segment_columns[0] if ctx.profile.segment_columns else None
        if not col:
            score, comps = composite_score(0, 0, 0, 0)
            return Verdict(agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
                           cause_family=self.cause_family, supported=False,
                           evidence_score=score, components=comps, evidence=[],
                           temporal={"score": 0.0},
                           falsifiable_by="Uniform movement across segments would support this.",
                           reasoning="No grouping column, so breadth could not be assessed.",
                           caveats=[])

        rows, _ = _segment_deltas(ctx.panel, col, ctx.metric,
                                  ctx.event_set, ctx.baseline_set)
        if len(rows) < 3:
            score, comps = composite_score(0, 0, 0, 0)
            return Verdict(agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
                           cause_family=self.cause_family, supported=False,
                           evidence_score=score, components=comps, evidence=[],
                           temporal={"score": 0.0},
                           falsifiable_by="Uniform movement across segments would support this.",
                           reasoning="Too few segments to judge whether the change is broad.",
                           caveats=[])

        d = np.array([r["delta"] for r in rows])
        same_dir = float(max((d < 0).mean(), (d > 0).mean()))
        cv = float(np.std(d) / (abs(np.mean(d)) + 1e-9))
        uniformity = float(np.clip(1 - min(cv, 1.5) / 1.5, 0, 1))
        supported = bool(same_dir >= 0.8 and uniformity >= 0.55)

        score, comps = composite_score(uniformity, 0.5, uniformity, same_dir)
        return Verdict(
            agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
            cause_family=self.cause_family, supported=supported,
            evidence_score=score, components=comps,
            effect={"estimate": float(np.mean(d)), "ci_low": float(np.min(d)),
                    "ci_high": float(np.max(d)), "p_value": 1.0,
                    "method": "segment_dispersion", "n_treat": len(rows), "n_control": 0},
            temporal={"note": "not applicable", "score": 0.5},
            evidence=[Evidence(
                "metric", "Spread of the movement across segments", uniformity, "uniformity",
                f"{same_dir:.0%} of {len(rows)} segments moved the same way, with a "
                f"dispersion of {cv:.2f} around the mean change. Uniform movement points "
                f"outside the business; concentrated movement points inside it.",
                f"per-segment change over '{col}'")],
            falsifiable_by=("Movement concentrated in a few segments while the rest held "
                            "steady would rule this out."),
            reasoning=(
                f"Across '{col}', {same_dir:.0%} of segments moved in the same direction "
                f"with dispersion {cv:.2f}. "
                + ("The change is broad rather than localised, which is what an external "
                   "or market-wide cause looks like."
                   if supported else
                   "The change is uneven across segments, so a single external cause does "
                   "not explain it.")),
            caveats=[])


# ==========================================================================

class TermLiftAgent(Agent):
    name = "Customer Language"
    cause_family = "segment"
    hypothesis = "Customer wording shifted in the event window, naming what changed."

    def investigate(self, ctx) -> Verdict:
        if not ctx.profile.text_column or "_norm" not in ctx.panel.columns:
            score, comps = composite_score(0, 0, 0, 0)
            return Verdict(agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
                           cause_family=self.cause_family, supported=False,
                           evidence_score=score, components=comps, evidence=[],
                           temporal={"score": 0.0},
                           falsifiable_by="A free-text column would allow this test.",
                           reasoning="The upload has no free-text column, so customer "
                                     "language could not be examined.",
                           caveats=[])

        t = ctx.panel[ctx.panel["has_text"]]
        e = t[t["wk"].isin(ctx.event_set)]
        b = t[t["wk"].isin(ctx.baseline_set)]
        if len(e) < 40 or len(b) < 40:
            score, comps = composite_score(0, 0, 0, 0)
            return Verdict(agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
                           cause_family=self.cause_family, supported=False,
                           evidence_score=score, components=comps, evidence=[],
                           temporal={"score": 0.0},
                           falsifiable_by="More text rows would allow this test.",
                           reasoning="Too few rows carry text to compare the windows.",
                           caveats=[])

        # Terms are discovered from the data. There is no supplied vocabulary,
        # so the finding cannot be an artefact of a word list chosen in advance.
        def counts(frame):
            c = Counter()
            for doc in frame["_norm"]:
                c.update(set(_tokens(doc)))
            return c
        ce, cb = counts(e), counts(b)
        ne, nb = len(e), len(b)

        cand = [w for w, k in ce.items()
                if k >= max(8, 0.01 * ne) and (cb.get(w, 0) + k) >= 15]
        rows, pvals = [], []
        for w in cand:
            eff = two_proportion_test(ce[w], ne, cb.get(w, 0), nb)
            if eff.estimate <= 0:
                continue
            rows.append({"term": w, "rate_event": ce[w] / ne,
                         "rate_baseline": cb.get(w, 0) / nb,
                         "lift": (ce[w] / ne) / max(cb.get(w, 0) / nb, 1e-9),
                         "delta": eff.estimate, "p_value": eff.p_value,
                         "ci_low": eff.ci_low, "ci_high": eff.ci_high})
            pvals.append(eff.p_value)

        if not rows:
            score, comps = composite_score(0, 0, 0, 0)
            return Verdict(agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
                           cause_family=self.cause_family, supported=False,
                           evidence_score=score, components=comps, evidence=[],
                           temporal={"note": "text follows the event", "score": 0.5},
                           falsifiable_by="A term rising significantly would support this.",
                           reasoning="No term rose significantly in the event window; "
                                     "customer wording did not change.",
                           caveats=[])

        bh = benjamini_hochberg(pvals, C.FDR_ALPHA)
        for r, adj, rej in zip(rows, bh["adjusted"], bh["rejected"]):
            r["p_adjusted"], r["significant"] = float(adj), bool(rej)
        rows.sort(key=lambda r: -r["delta"])
        sig = [r for r in rows if r["significant"]]
        if not sig:
            score, comps = composite_score(0, 0, 0, 0)
            return Verdict(agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
                           cause_family=self.cause_family, supported=False,
                           evidence_score=score, components=comps, evidence=[],
                           temporal={"note": "text follows the event", "score": 0.5},
                           falsifiable_by="A term surviving FDR control would support this.",
                           reasoning=f"{len(rows)} terms rose but none survived FDR "
                                     f"control across {len(rows)} comparisons.",
                           caveats=[])

        top = sig[0]
        stat = statistical_strength(top["delta"], top["ci_low"], top["ci_high"])
        med_lift = float(np.median([r["lift"] for r in rows]))
        disc = float(np.clip((top["lift"] - med_lift) / max(top["lift"], 1e-9), 0, 1))
        spec = float(np.clip(len(sig) / max(len(rows), 1), 0, 1))
        score, comps = composite_score(stat, 0.5, 1 - spec, disc)

        ev = [Evidence("text", f"'{r['term']}' rose in customer wording", r["lift"], "x baseline",
                       f"{r['rate_baseline']:.1%} to {r['rate_event']:.1%} of texts "
                       f"({r['lift']:.2f}x, adjusted p={r['p_adjusted']:.2e})",
                       f"uploaded table, column '{ctx.profile.text_column}', "
                       f"{ne} event / {nb} baseline rows")
              for r in sig[:6]]

        return Verdict(
            agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
            cause_family=self.cause_family, supported=True,
            evidence_score=score, components=comps,
            effect={"estimate": top["delta"], "ci_low": top["ci_low"],
                    "ci_high": top["ci_high"], "p_value": top["p_adjusted"],
                    "method": "term frequency + BH", "n_treat": ne, "n_control": nb},
            temporal={"note": "text is written after the event it describes", "score": 0.5},
            evidence=ev,
            falsifiable_by=("If every term rose by a similar amount, the shift would be in "
                            "how much people wrote rather than what they wrote, and would "
                            "name no cause."),
            reasoning=(
                f"Terms were discovered from the text itself, with no supplied word list. "
                f"{len(sig)} of {len(rows)} tested terms rose significantly after FDR "
                f"control. The strongest is '{top['term']}', up {top['lift']:.2f}x "
                f"({top['rate_baseline']:.1%} to {top['rate_event']:.1%}, adjusted "
                f"p={top['p_adjusted']:.2e}). Leading terms: "
                + ", ".join(f"'{r['term']}' {r['lift']:.1f}x" for r in sig[:5]) + "."),
            caveats=[f"Only {ne:,} event-window rows carry text, so this describes people "
                     f"who wrote something, not everyone."])


# ==========================================================================

class UploadIntegrityAgent(Agent):
    name = "Data Integrity"
    cause_family = "measurement"
    hypothesis = "The table changed rather than the business."

    def investigate(self, ctx) -> Verdict:
        p = ctx.panel
        e, b = p[p["wk"].isin(ctx.event_set)], p[p["wk"].isin(ctx.baseline_set)]
        checks, ev = [], []

        we = e.groupby("wk").size()
        wb = b.groupby("wk").size()
        vol = welch_t_test(we.values, wb.values)
        vol_shift = abs(vol.estimate) / max(wb.mean(), 1e-9)
        checks.append(("row volume", bool(vol.significant and vol_shift > 0.35), vol_shift))
        ev.append(Evidence("integrity", "Rows per period", vol.estimate, "rows",
                           f"{wb.mean():.0f} to {we.mean():.0f} per week "
                           f"({vol_shift:+.0%}, p={vol.p_value:.3f}). A collection break "
                           f"shows up as a step change in how much data arrives.",
                           "row counts per period"))

        worst_null, worst_col = 0.0, None
        for c in ctx.profile.measure_columns + ([ctx.profile.text_column]
                                                if ctx.profile.text_column else []):
            if c not in p.columns:
                continue
            shift = abs(float(e[c].isna().mean() - b[c].isna().mean()))
            if shift > worst_null:
                worst_null, worst_col = shift, c
        checks.append(("missing values", worst_null > 0.10, worst_null))
        ev.append(Evidence("integrity", "Missing values", worst_null, "share",
                           f"largest shift is {worst_col or 'none'} at {worst_null:.1%}. "
                           f"A feed that breaks mid-window leaves nulls behind.",
                           "null rates across measures"))

        drift = 0.0
        for c in ctx.profile.segment_columns:
            se, sb = set(e[c].dropna().astype(str)), set(b[c].dropna().astype(str))
            if sb:
                drift = max(drift, len(se ^ sb) / max(len(sb), 1))
        checks.append(("category drift", drift > 0.35, drift))
        ev.append(Evidence("integrity", "Category stability", drift, "share",
                           f"{drift:.0%} of segment values appear or disappear between "
                           f"windows. Renamed or re-coded categories look like real change.",
                           "distinct values per grouping column"))

        n_failed = sum(1 for _, f, _ in checks if f)
        supported = n_failed >= 2
        worst = max(v for _, _, v in checks)
        stat = float(np.clip(worst / 0.35, 0, 1))
        score, comps = composite_score(stat, 0.5, stat, stat)

        return Verdict(
            agent=self.name, scope=self.scope, hypothesis=self.hypothesis,
            cause_family=self.cause_family, supported=supported,
            evidence_score=score, components=comps,
            effect={"estimate": worst, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0,
                    "method": "upload_integrity", "n_treat": len(e), "n_control": len(b)},
            temporal={"note": "not applicable", "score": 0.5},
            evidence=ev,
            falsifiable_by=("A step change in row volume, missing values, or category "
                            "coding would make the measurement suspect."),
            reasoning=(
                f"Three checks were run against the upload: rows per period, missing "
                f"values, and category stability. {n_failed} of 3 flagged, with a largest "
                f"movement of {worst:.1%}. "
                + ("That is large enough that the table itself is suspect, so no cause "
                   "should be read from it yet."
                   if supported else
                   "The table behaves consistently across both windows, so the movement "
                   "is in the business rather than in the data collection.")),
            caveats=[])


GENERIC_AGENTS = [SegmentAgent, TermLiftAgent, MixAgent, SystemicAgent,
                  UploadIntegrityAgent]

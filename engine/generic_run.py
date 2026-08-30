"""Investigate an uploaded table.

Same five scopes as the built-in investigation. Scopes 0 and 1 are unchanged,
because triage and localisation never depended on the domain. Scope 2 swaps in
the generic agent set, scope 3 runs the two challenges that survive without
domain columns, and scope 4 prescribes in general terms rather than naming a
delivery experiment.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from . import detect, localize, arbiter
from .adversary import Challenge
from .agents.generic import GENERIC_AGENTS, _segment_deltas
from .ingest import Profile, build_panel, weekly_panel, profile, read_csv_bytes
from .stats_core import difference_in_differences, welch_t_test
from . import governance as G
from . import personas as P


@dataclass
class UploadContext:
    panel: pd.DataFrame
    weekly_national: pd.DataFrame
    event_set: set
    baseline_set: set
    top_segments: list
    profile: Profile
    metric: str
    segment_column: str | None


# --------------------------------------------------------------------------
# scope 3, without domain columns
# --------------------------------------------------------------------------

def _simpsons(ctx) -> Challenge:
    if not ctx.segment_column:
        return Challenge("Simpson's paradox", "Does the effect survive within segments?",
                         False, "No grouping column, so this could not be tested.", {})
    rows, _ = _segment_deltas(ctx.panel, ctx.segment_column, ctx.metric,
                              ctx.event_set, ctx.baseline_set)
    if len(rows) < 3:
        return Challenge("Simpson's paradox", "Does the effect survive within segments?",
                         False, "Too few segments carry enough rows to test.", {})
    neg = [r for r in rows if r["delta"] < 0]
    frac = len(neg) / len(rows)
    passed = frac >= 0.6 or frac <= 0.4      # consistent one way or the other
    direction = "fell" if frac >= 0.5 else "rose"
    return Challenge(
        name="Simpson's paradox",
        question="Is the overall movement an artefact of shifting composition?",
        passed=passed,
        finding=(f"{len(neg)} of {len(rows)} segments {direction} independently "
                 f"({frac:.0%}). "
                 + ("The direction is consistent inside segments, so the overall figure "
                    "is not produced by composition alone."
                    if passed else
                    "Segments move in both directions, so the overall figure may be "
                    "driven by which segments grew rather than by any real change.")),
        detail={"n_segments": len(rows), "n_negative": len(neg), "fraction": frac})


def _did(ctx) -> Challenge:
    """Most-affected segments against least-affected, with a pre-trend test."""
    if not ctx.segment_column or ctx.metric not in ctx.panel.columns:
        return Challenge("Difference-in-differences",
                         "Does it hold against segments that were not affected?",
                         False, "Needs a grouping column and a row-level measure.", {})
    rows, _ = _segment_deltas(ctx.panel, ctx.segment_column, ctx.metric,
                              ctx.event_set, ctx.baseline_set)
    if len(rows) < 4:
        return Challenge("Difference-in-differences",
                         "Does it hold against segments that were not affected?",
                         False, "Too few segments to form a control group.", {})
    rows.sort(key=lambda r: r["delta"])
    k = max(1, len(rows) // 3)
    treat = [r["value"] for r in rows[:k]]
    control = [r["value"] for r in rows[-k:]]
    d = ctx.panel
    col = d[ctx.segment_column].astype(str)

    def vals(group, weeks):
        return d.loc[col.isin(group) & d["wk"].isin(weeks), ctx.metric].dropna().values

    pre = sorted(ctx.baseline_set)[-12:]
    ts, cs = [], []
    for wk in pre:
        t = d.loc[col.isin(treat) & (d["wk"] == wk), ctx.metric].mean()
        c = d.loc[col.isin(control) & (d["wk"] == wk), ctx.metric].mean()
        if np.isfinite(t) and np.isfinite(c):
            ts.append(t); cs.append(c)

    res = difference_in_differences(
        vals(treat, ctx.baseline_set), vals(treat, ctx.event_set),
        vals(control, ctx.baseline_set), vals(control, ctx.event_set),
        pre_treat_series=ts, pre_ctrl_series=cs)
    passed = bool(abs(res.did_estimate) > 0 and (res.ci_high < 0 or res.ci_low > 0))
    pt = (f"Parallel-trends test p={res.parallel_trends_p:.3f} "
          f"({'trends were parallel beforehand, so the estimate is credible' if res.parallel_trends_ok else 'pre-trends were not parallel, so treat this with caution'}). "
          if np.isfinite(res.parallel_trends_p) else "")
    return Challenge(
        name="Difference-in-differences",
        question="Does it hold against segments that were not affected?",
        passed=passed,
        finding=(f"Most-affected segments ({', '.join(treat[:3])}) changed "
                 f"{res.treat_change:+.3f} while least-affected ({', '.join(control[:3])}) "
                 f"changed {res.control_change:+.3f}. Difference-in-differences "
                 f"{res.did_estimate:+.3f} (95% CI [{res.ci_low:+.3f}, {res.ci_high:+.3f}], "
                 f"p={res.p_value:.2e}). " + pt
                 + ("The gap against unaffected segments holds."
                    if passed else "The control comparison does not confirm the effect.")),
        detail={**res.to_dict(), "treat": treat, "control": control})


def run_adversary(ctx, hypothesis: str) -> dict:
    challenges = [_simpsons(ctx), _did(ctx)]
    passed = sum(1 for c in challenges if c.passed)
    return {"target_hypothesis": hypothesis,
            "challenges": [c.to_dict() for c in challenges],
            "n_passed": passed, "n_total": len(challenges),
            "survived": passed == len(challenges),
            "summary": (f"The leading explanation was challenged on {len(challenges)} "
                        f"fronts and survived {passed}. "
                        + ("Neither challenge overturned it." if passed == len(challenges)
                           else "At least one challenge raises doubt."))}


# --------------------------------------------------------------------------
# scope 4, in general terms
# --------------------------------------------------------------------------

def prescribe(ctx, triage, top, adversary, segments) -> dict:
    sig = [s for s in segments["findings"] if s["significant"]]
    names = [s["label"] for s in sig[:4]]
    causal = bool(adversary and adversary["survived"])

    if top.cause_family == "measurement":
        return {"recommendation": "Fix the data before reading anything from it",
                "rationale": "The table changed between the two windows, so any cause "
                             "read from it would be a property of the collection rather "
                             "than the business.",
                "design": {}}
    if top.cause_family == "mix":
        return {"recommendation": "Re-cut the metric per segment before acting",
                "rationale": "The overall figure moved because the mix of rows changed, "
                             "not because any segment got worse. A headline number will "
                             "keep misleading until it is reported per segment or "
                             "weighted to a fixed mix.",
                "design": {"unit_of_analysis": ctx.segment_column or "segment",
                           "note": "No experiment is needed. This is a reporting change."}}
    if top.cause_family == "systemic":
        return {"recommendation": "Look outside this dataset",
                "rationale": "Every segment moved together, so nothing inside the table "
                             "distinguishes cause from effect. Compare against a market "
                             "benchmark, a competitor series, or a calendar of external "
                             "events over the same window.",
                "design": {"note": "An internal experiment cannot separate a cause that "
                                   "affects every segment equally."}}

    ev = ctx.panel[ctx.panel["wk"].isin(ctx.event_set)]
    bs = ctx.panel[ctx.panel["wk"].isin(ctx.baseline_set)]
    per_week = float(ev.groupby("wk").size().mean()) if len(ev) else 0.0
    if ctx.metric in ctx.panel.columns:
        sd = float(bs[ctx.metric].std())
        base = float(bs[ctx.metric].mean())
        mde = abs(triage.delta) / 2 if triage.delta else sd * 0.2
        from .stats_core import required_sample_size_continuous
        n = required_sample_size_continuous(sd, max(mde, 1e-9), C.ALPHA, C.POWER)
    else:
        sd, base, mde, n = 0.0, 0.0, 0.0, 0
    share = (len(ev[ev[ctx.segment_column].astype(str).isin(names)]) / max(len(ev), 1)
             if ctx.segment_column and names else 1.0)
    weekly_affected = per_week * share
    weeks = int(np.ceil((2 * n) / max(weekly_affected, 1))) if n else 0

    return {
        "recommendation": ("Targeted intervention test on the affected segments"
                           if causal else
                           "Difference-in-differences on the data you already hold"),
        "rationale": (
            "The movement is confined to specific segments and survived both challenges, "
            "so the association is not an artefact of composition. The open question is "
            "whether a change to those segments recovers the metric, which is a narrower "
            "and cheaper test than a full rollout."
            if causal else
            "The movement is concentrated, but a challenge did not clear. Before running "
            "an experiment, a difference-in-differences against the unaffected segments "
            "costs nothing and may settle it."),
        "design": {
            "unit_of_randomisation": ctx.segment_column or "segment",
            "affected_segments": names,
            "primary_metric": ctx.metric,
            "baseline_rate": base,
            "mde_absolute": mde,
            "alpha": C.ALPHA, "power": C.POWER,
            "n_per_arm": n, "estimated_weeks": weeks,
            "note": (f"n={n:,} rows per arm gives {C.POWER:.0%} power to detect a "
                     f"{mde:.3f} change from a baseline of {base:.3f} "
                     f"(SD {sd:.3f}) at alpha={C.ALPHA}. At about "
                     f"{weekly_affected:,.0f} rows per week across the affected "
                     f"segments, that is roughly {weeks} week(s)."
                     if n else "Not enough row-level detail to size a test.")}}


# --------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------

def investigate_upload(raw: bytes, metric_key: str | None = None,
                       filename: str = "upload.csv", on_log=None) -> dict:
    t0 = time.time()
    tel = G.Telemetry()
    log = []

    def say(msg):
        log.append(msg)
        if on_log:
            on_log(msg)

    tel.start("Profile")
    say("Scope 0  Profile     | reading the file...")
    df = read_csv_bytes(raw)
    prof = profile(df)
    say(f"Scope 0  Profile     | {prof.rows:,} rows, {len(prof.columns)} columns, "
        f"{prof.span['weeks']} weeks")
    say(f"Scope 0  Profile     | date='{prof.date_column}'  "
        f"segments={prof.segment_columns or 'none'}  "
        f"text='{prof.text_column or 'none'}'")
    for w in prof.warnings:
        say(f"Scope 0  Profile     | note: {w}")

    panel = build_panel(df, prof)
    weekly = weekly_panel(panel, prof)
    tel.stop("Profile", f"{prof.rows:,} rows, {len(prof.columns)} columns")

    keys = [m["key"] for m in prof.metrics]
    metric = metric_key if metric_key in keys else keys[0]
    label = next(m["label"] for m in prof.metrics if m["key"] == metric)
    say(f"Scope 0  Triage      | investigating: {label}")

    tel.start("Scope 0 Triage")
    tri = detect.triage(panel, weekly, metric)
    say(f"Scope 0  Triage      | {tri.verdict}  z={tri.robust_z:.2f}  "
        f"delta={tri.delta:+.3f}  weeks={len(tri.event_weeks)}")

    tel.stop("Scope 0 Triage", f"{tri.verdict}, z={tri.robust_z:.2f}")
    if not tri.is_signal:
        say("Scope 0  Triage      | inside control limits, stopping before any agent runs.")
        dec = arbiter.decide(None, tri, None, None, [], None, metric_label=label)
        views = P.build_all(tri, dec, None, [], None, None, metric)
        return _package(prof, tri, None, None, [], None, dec, log, t0, metric,
                        label, filename, telemetry=tel, personas=views)

    event, base = set(tri.event_weeks), set(tri.baseline_weeks)
    seg_col = prof.segment_columns[0] if prof.segment_columns else None

    decomp, segs = None, {"findings": [], "n_tested": 0, "n_significant_naive": 0,
                          "n_significant_fdr": 0, "fdr_alpha": C.FDR_ALPHA,
                          "note": "No grouping column, so no segment scan was run."}
    tel.start("Scope 1 Localize")
    if seg_col:
        say(f"Scope 1  Localize    | grouping by '{seg_col}'...")
        try:
            decomp = localize.mix_vs_rate(panel, tri.event_weeks, tri.baseline_weeks,
                                          metric=metric, segment=seg_col)
            say(f"Scope 1  Localize    | rate={decomp.rate_effect:+.3f} "
                f"mix={decomp.mix_effect:+.3f} -> {decomp.rate_share:.0%} rate-driven")
        except Exception as exc:
            say(f"Scope 1  Localize    | decomposition unavailable: {exc}")
        try:
            segs = localize.segment_scan(panel, tri.event_weeks, tri.baseline_weeks,
                                         metric=metric, segment=seg_col)
            say(f"Scope 1  Localize    | {segs['n_tested']} groups tested, "
                f"{segs['n_significant_naive']} naive hits -> "
                f"{segs['n_significant_fdr']} survive FDR control")
        except Exception as exc:
            say(f"Scope 1  Localize    | segment scan unavailable: {exc}")

    tel.stop("Scope 1 Localize", f"{segs['n_tested']} groups")
    ctx = UploadContext(panel=panel, weekly_national=weekly, event_set=event,
                        baseline_set=base,
                        top_segments=[s["segment"] for s in segs["findings"]
                                      if s["significant"]][:8],
                        profile=prof, metric=metric, segment_column=seg_col)

    tel.start("Scope 2 Investigate")
    say("Scope 2  Investigate | spawning agents...")
    verdicts = []
    for cls in GENERIC_AGENTS:
        v = cls().investigate(ctx)
        verdicts.append(v)
        say(f"Scope 2  {v.agent:<18}| supported={str(v.supported):<5} "
            f"score={v.evidence_score:.3f}  "
            f"[stat={v.components['statistical']:.2f} "
            f"temp={v.components['temporal']:.2f} "
            f"spec={v.components['specificity']:.2f} "
            f"disc={v.components['discrimination']:.2f}]")

    tel.stop("Scope 2 Investigate", f"{len(GENERIC_AGENTS)} agents")
    ranked = sorted(verdicts, key=lambda v: v.evidence_score, reverse=True)
    lead = next((v for v in ranked if v.supported and v.cause_family != "measurement"), None)
    adv = None
    if lead is not None:
        tel.start("Scope 3 Adversary")
        say(f"Scope 3  Adversary   | challenging: {lead.agent}")
        adv = run_adversary(ctx, lead.hypothesis)
        for c in adv["challenges"]:
            say(f"Scope 3  Adversary   | {'PASS' if c['passed'] else 'FAIL'}  {c['name']}")
        tel.stop("Scope 3 Adversary", f"{adv['n_passed']}/{adv['n_total']} survived")

    tel.start("Scope 4 Arbiter")
    say("Scope 4  Arbiter     | ranking and deciding...")
    dec = arbiter.decide(ctx, tri, decomp, segs, verdicts, adv,
                         prescribe_fn=prescribe, metric_label=label)
    say(f"Scope 4  Arbiter     | STATE = {dec.state}")
    if dec.separability.get("note"):
        say(f"Scope 4  Arbiter     | {dec.separability['note']}")

    views = P.build_all(tri, dec, segs, verdicts, adv, decomp, metric)
    tel.stop("Scope 4 Arbiter", dec.state)
    say(f"Scope 4  Arbiter     | {len([v for v in views if not v['withheld']])} of "
        f"{len(views)} readers served, 0 model calls")
    return _package(prof, tri, decomp, segs, verdicts, adv, dec, log, t0, metric,
                    label, filename, ctx=ctx, telemetry=tel, personas=views)


def _package(prof, tri, decomp, segs, verdicts, adv, dec, log, t0, metric, label,
             filename, ctx=None, telemetry=None, personas=None):
    payload = {
        "meta": {
            "product": "IndiaPulse.ai",
            "source": "uploaded file",
            "filename": filename,
            "dataset": f"{prof.rows:,} rows, {prof.span['weeks']} weeks, "
                       f"{prof.span['start']} to {prof.span['end']}",
            "metric": metric,
            "metric_label": label,
            "scenario": "upload",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_seconds": round(time.time() - t0, 2),
        },
        "profile": prof.to_dict(),
        "scope0_triage": tri.to_dict(),
        "scope1_decomposition": decomp.to_dict() if decomp else None,
        "scope1_segments": segs,
        "scope2_verdicts": [v.to_dict() for v in verdicts],
        "scope3_adversary": adv,
        "scope4_decision": dec.to_dict(),
        "orchestration_log": log,
        "personas": personas or [],
        "governance": {
            "telemetry": telemetry.to_dict() if telemetry else None,
            "methods": G.method_summary(),
            "contract": G.upload_contract(prof, metric),
            "lineage": G.upload_lineage(prof, filename),
        },
    }
    if ctx is not None:
        w = ctx.weekly_national.sort_values("week").copy()
        w["wk"] = w["week"].astype(str).str[:10]
        rows = []
        for _, r in w.iterrows():
            v = r.get(metric)
            rows.append({
                "week": r["wk"],
                "in_event": bool(r["wk"] in ctx.event_set),
                "orders": int(r.get("records", 0)),
                "review_score": None if v is None or (isinstance(v, float) and np.isnan(v))
                                else float(v),
                "on_time": None, "delivery_days": None, "days_to_carrier": None,
                "topic_delivery_delay": None, "topic_product_quality": None,
                "topic_positive": None,
            })
        payload["timeline"] = rows
    return payload

"""Orchestrator.

    Scope 0  Triage       statistical gate, stops early on noise
    Scope 1  Localize     mix vs rate, FDR segment scan
    Scope 2  Investigate  parallel agents, one per dimension
    Scope 3  Adversary    challenge the leader
    Scope 4  Arbiter      rank, decide, prescribe
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import detect, localize, adversary as adv, arbiter
from .agents.ops import OpsAgent
from .agents.voc import VoiceOfCustomerAgent
from .agents.integrity import DataIntegrityAgent
from .agents.external import MarketExternalAgent
from . import governance as G
from . import personas as P


@dataclass
class Context:
    panel: pd.DataFrame
    weekly_national: pd.DataFrame
    event_set: set
    baseline_set: set
    top_segments: list


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return None if (np.isnan(o) or np.isinf(o)) else float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp,)):
            return str(o.date())
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        return super().default(o)


# Weekly KPI columns that are aggregates of a differently-named order-level
# column. Triage needs both the weekly series and the order-level values.
METRIC_ORDER_COL = {"aov": "order_value", "orders": None, "revenue": None}


def investigate(metric: str = "review_score", verbose: bool = True,
                states: list[str] | None = None,
                week_from: str | None = None, week_to: str | None = None,
                scenario: str = "national", out_name: str = "investigation.json",
                on_log=None, write: bool = True) -> dict:
    t0 = time.time()
    log = []

    def say(msg):
        log.append(msg)
        if verbose:
            print(msg, flush=True)
        if on_log is not None:
            on_log(msg)

    tel = G.Telemetry()
    # Data load is timed separately. Folding a cold read into Scope 0 would
    # report the cache state as if it were the cost of detection.
    tel.start("Data load")
    say("Scope 0  Triage      | loading panel...")
    panel = D.load_panel()
    panel = panel.assign(wk=panel["week"].astype(str).str[:10])
    if states:
        panel = panel[panel["customer_state"].isin(states)]
    if week_from:
        panel = panel[panel["wk"] >= week_from]
    if week_to:
        panel = panel[panel["wk"] <= week_to]
    say(f"Scope 0  Triage      | scenario='{scenario}'  orders={len(panel):,}"
        + (f"  states={states}" if states else "")
        + (f"  window={week_from or 'start'}..{week_to or 'end'}"
           if (week_from or week_to) else ""))
    weekly_national = D.weekly(panel)
    tel.stop("Data load", f"{len(panel):,} rows")

    tel.start("Scope 0 Triage")
    tri = detect.triage(panel, weekly_national, metric)
    say(f"Scope 0  Triage      | {tri.verdict}  z={tri.robust_z:.2f}  "
        f"delta={tri.delta:+.3f}  weeks={len(tri.event_weeks)}")

    tel.stop("Scope 0 Triage", f"{tri.verdict}, z={tri.robust_z:.2f}")
    if not tri.is_signal:
        say("Scope 0  Triage      | inside control limits, stopping before "
            "any agent is spawned.")
        dec = arbiter.decide(None, tri, None, None, [], None)
        views = P.build_all(tri, dec, None, [], None, None, metric)
        return _package(tri, None, None, [], None, dec, log, t0, metric,
                        scenario=scenario, out_name=out_name, write=write,
                        telemetry=tel, personas=views, panel=panel)

    event_set = set(tri.event_weeks)
    baseline_set = set(tri.baseline_weeks)

    tel.start("Scope 1 Localize")
    decomp = None
    segs = {"findings": [], "n_tested": 0, "n_significant_naive": 0,
            "n_significant_fdr": 0, "fdr_alpha": C.FDR_ALPHA,
            "note": f"'{metric}' is a period-level total with no per-order value, "
                    f"so it cannot be decomposed across segments."}
    if metric in panel.columns:
        say("Scope 1  Localize    | mix-vs-rate decomposition...")
        decomp = localize.mix_vs_rate(panel, tri.event_weeks, tri.baseline_weeks, metric)
        say(f"Scope 1  Localize    | rate={decomp.rate_effect:+.3f} "
            f"mix={decomp.mix_effect:+.3f} -> {decomp.rate_share:.0%} rate-driven")
        segs = localize.segment_scan(panel, tri.event_weeks, tri.baseline_weeks, metric)
        say(f"Scope 1  Localize    | {segs['n_tested']} segments tested, "
            f"{segs['n_significant_naive']} naive hits -> "
            f"{segs['n_significant_fdr']} survive FDR control")
    else:
        say(f"Scope 1  Localize    | {metric} has no per-order value; "
            f"segment decomposition skipped")

    tel.stop("Scope 1 Localize", f"{segs['n_tested']} segments")
    top_segments = [s["segment"] for s in segs["findings"] if s["significant"]][:8]
    ctx = Context(panel, weekly_national, event_set, baseline_set, top_segments)

    tel.start("Scope 2 Investigate")
    say("Scope 2  Investigate | spawning agents...")
    agents = [OpsAgent(), VoiceOfCustomerAgent(), DataIntegrityAgent(), MarketExternalAgent()]
    verdicts = []
    for a in agents:
        v = a.investigate(ctx)
        verdicts.append(v)
        say(f"Scope 2  {a.name:<18}| supported={str(v.supported):<5} "
            f"score={v.evidence_score:.3f}  "
            f"[stat={v.components['statistical']:.2f} "
            f"temp={v.components['temporal']:.2f} "
            f"spec={v.components['specificity']:.2f} "
            f"disc={v.components['discrimination']:.2f}]")

    tel.stop("Scope 2 Investigate", f"{len(agents)} agents")
    tel.retrieval_calls += 1          # Market & External consults its context pack
    ranked = sorted(verdicts, key=lambda v: v.evidence_score, reverse=True)
    lead = next((v for v in ranked if v.supported and v.agent != "Data Integrity"), None)

    adversary_result = None
    if lead is not None:
        tel.start("Scope 3 Adversary")
        say(f"Scope 3  Adversary   | challenging: {lead.agent}")
        adversary_result = adv.run_adversary(ctx, lead.hypothesis)
        for c in adversary_result["challenges"]:
            say(f"Scope 3  Adversary   | {'PASS' if c['passed'] else 'FAIL'}  {c['name']}")
        tel.stop("Scope 3 Adversary", f"{adversary_result['n_passed']}/"
                                      f"{adversary_result['n_total']} survived")

    tel.start("Scope 4 Arbiter")
    say("Scope 4  Arbiter     | ranking and deciding...")
    dec = arbiter.decide(ctx, tri, decomp, segs, verdicts, adversary_result)
    say(f"Scope 4  Arbiter     | STATE = {dec.state}")
    say(f"Scope 4  Arbiter     | {dec.separability.get('note', '')}")

    views = P.build_all(tri, dec, segs, verdicts, adversary_result, decomp, metric)
    tel.stop("Scope 4 Arbiter", dec.state)
    say(f"Scope 4  Arbiter     | {len([v for v in views if not v['withheld']])} of "
        f"{len(views)} personas served, 0 model calls")
    return _package(tri, decomp, segs, verdicts, adversary_result, dec, log, t0, metric,
                    ctx=ctx, scenario=scenario, out_name=out_name, write=write,
                    telemetry=tel, personas=views, panel=panel)


def _package(tri, decomp, segs, verdicts, adversary_result, dec, log, t0, metric, ctx=None,
             scenario="national", out_name="investigation.json", write=True,
             telemetry=None, personas=None, panel=None):
    payload = {
        "meta": {
            "product": "IndiaPulse AI",
            "subtitle": "KPI storytelling engine. Accenture Innovation Challenge 2026",
            "problem_statement": "BusinessIntelligence.ai",
            "dataset": "Olist Brazilian E-Commerce (public, ~99k orders, ~41k free-text reviews)",
            "dataset_source": "https://huggingface.co/datasets/bulutttt/olist-raw-data",
            "metric": metric,
            "scenario": scenario,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runtime_seconds": round(time.time() - t0, 2),
            "honesty_note": (
                "Every figure in this report is computed from the public dataset named "
                "above. There are no hand-tuned effect constants anywhere in the engine. "
                "Retrieved web context is labelled as context and is excluded from "
                "evidence scoring by design."
            ),
        },
        "scope0_triage": tri.to_dict(),
        "scope1_decomposition": decomp.to_dict() if decomp else None,
        "scope1_segments": segs,
        "scope2_verdicts": [v.to_dict() for v in verdicts],
        "scope3_adversary": adversary_result,
        "scope4_decision": dec.to_dict(),
        "orchestration_log": log,
        "personas": personas or [],
        "governance": {
            "telemetry": telemetry.to_dict() if telemetry else None,
            "methods": G.method_summary(),
            "contract": G.KPI_CONTRACT.get(metric),
            "lineage": G.data_freshness(panel) if panel is not None else None,
        },
    }
    if ctx is not None:
        payload["timeline"] = _timeline(ctx, tri)
        payload["dose_response"] = next(
            (v.to_dict() for v in verdicts if v.agent == "Ops & Fulfilment"), None)
    if write:
        C.OUT.mkdir(parents=True, exist_ok=True)
        out = C.OUT / out_name
        out.write_text(json.dumps(payload, cls=NpEncoder, indent=2), encoding="utf-8")
        print(f"\n-> wrote {out}")
    return payload


def _timeline(ctx, tri) -> list[dict]:
    w = ctx.weekly_national.sort_values("week").copy()
    w["wk"] = w["week"].astype(str).str[:10]
    rows = []
    for _, r in w.iterrows():
        def g(c):
            v = r.get(c)
            return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
        rows.append({
            "week": r["wk"],
            "in_event": bool(r["wk"] in ctx.event_set),
            "orders": int(r["orders"]),
            "review_score": g("review_score"),
            "on_time": g("on_time"),
            "delivery_days": g("delivery_days"),
            "days_to_carrier": g("days_to_carrier"),
            "topic_delivery_delay": g("topic_delivery_delay"),
            "topic_product_quality": g("topic_product_quality"),
            "topic_positive": g("topic_positive"),
        })
    return rows


if __name__ == "__main__":
    investigate()

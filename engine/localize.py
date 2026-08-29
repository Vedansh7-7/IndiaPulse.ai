"""Scope 1: localize the movement.

Two questions:

1. Mix or rate. An average can fall with no customer being less happy, if the
   customer mix shifted toward segments that were always lower. Separating
   these tells you whether service got worse or the market changed.
2. Which segments. Scanning 27 states at alpha 0.05 yields about 1.4 false
   positives by chance, so Benjamini-Hochberg FDR control is applied.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from . import config as C
from .stats_core import benjamini_hochberg, welch_t_test, two_proportion_test


@dataclass
class Decomposition:
    total_delta: float
    mix_effect: float
    rate_effect: float
    interaction: float
    mix_share: float
    rate_share: float
    interpretation: str

    def to_dict(self) -> dict:
        return asdict(self)


def mix_vs_rate(panel: pd.DataFrame, event_weeks: list[str], baseline_weeks: list[str],
                metric: str = "review_score", segment: str = "customer_state") -> Decomposition:
    """
    Classic additive decomposition of a change in a weighted mean:

        M = sum_s ( w_s * m_s )
        dM = sum_s ( dw_s * m_s^base )   <- MIX   (who showed up changed)
           + sum_s ( w_s^base * dm_s )   <- RATE  (experience within a segment changed)
           + sum_s ( dw_s * dm_s )       <- interaction
    """
    ev = panel[panel["week"].astype(str).str[:10].isin(event_weeks) & panel["delivered"]]
    bs = panel[panel["week"].astype(str).str[:10].isin(baseline_weeks) & panel["delivered"]]
    ev = ev[ev[metric].notna()]
    bs = bs[bs[metric].notna()]

    seg = sorted(set(bs[segment].dropna()) | set(ev[segment].dropna()))
    w_b = bs[segment].value_counts(normalize=True).reindex(seg).fillna(0.0)
    w_e = ev[segment].value_counts(normalize=True).reindex(seg).fillna(0.0)
    m_b = bs.groupby(segment)[metric].mean().reindex(seg)
    m_e = ev.groupby(segment)[metric].mean().reindex(seg)
    # a segment absent from one window contributes no rate change
    m_b_f = m_b.fillna(bs[metric].mean())
    m_e_f = m_e.fillna(m_b_f)

    dw = w_e - w_b
    dm = m_e_f - m_b_f
    mix = float((dw * m_b_f).sum())
    rate = float((w_b * dm).sum())
    inter = float((dw * dm).sum())
    total = float(ev[metric].mean() - bs[metric].mean())

    denom = abs(mix) + abs(rate) + abs(inter)
    mix_share = abs(mix) / denom if denom else 0.0
    rate_share = abs(rate) / denom if denom else 0.0

    if rate_share > 0.65:
        interp = (
            f"{rate_share:.0%} of the movement is RATE: the same kinds of customers "
            f"had a worse experience. This is an operational or product problem, "
            f"not a change in who is buying."
        )
    elif mix_share > 0.65:
        interp = (
            f"{mix_share:.0%} of the movement is MIX: the customer composition "
            f"shifted toward segments that were already lower-scoring. The "
            f"experience per segment barely changed. This is a composition "
            f"effect, not a service regression."
        )
    else:
        interp = (
            f"The movement is mixed ({rate_share:.0%} rate, "
            f"{mix_share:.0%} mix). Both the customer composition and the "
            f"within-segment experience moved; neither alone explains it."
        )
    return Decomposition(total, mix, rate, inter, mix_share, rate_share, interp)


@dataclass
class SegmentFinding:
    segment: str
    label: str
    n_event: int
    n_baseline: int
    event_value: float
    baseline_value: float
    delta: float
    p_value: float
    p_adjusted: float
    significant: bool
    contribution: float

    def to_dict(self) -> dict:
        return asdict(self)


def segment_scan(panel: pd.DataFrame, event_weeks: list[str], baseline_weeks: list[str],
                 metric: str = "review_score",
                 segment: str = "customer_state") -> dict:
    """Per-segment test with FDR control, ranked by contribution to the total move."""
    ev = panel[panel["week"].astype(str).str[:10].isin(event_weeks) & panel["delivered"]]
    bs = panel[panel["week"].astype(str).str[:10].isin(baseline_weeks) & panel["delivered"]]

    rows, pvals = [], []
    total_ev_n = len(ev[ev[metric].notna()])
    overall_delta = ev[metric].mean() - bs[metric].mean()

    for s in sorted(set(bs[segment].dropna())):
        a = ev.loc[ev[segment] == s, metric].dropna()
        b = bs.loc[bs[segment] == s, metric].dropna()
        if len(a) < C.MIN_SEGMENT_N or len(b) < C.MIN_SEGMENT_N:
            continue
        eff = welch_t_test(a, b)
        # contribution: this segment's share of event volume x its own delta,
        # expressed as a fraction of the total national movement
        share = len(a) / max(total_ev_n, 1)
        contrib = (share * eff.estimate / overall_delta) if overall_delta else 0.0
        rows.append({
            "segment": s, "label": C.STATE_NAMES.get(s, s),
            "n_event": len(a), "n_baseline": len(b),
            "event_value": float(a.mean()), "baseline_value": float(b.mean()),
            "delta": eff.estimate, "p_value": eff.p_value,
            "contribution": float(contrib),
        })
        pvals.append(eff.p_value)

    bh = benjamini_hochberg(pvals, C.FDR_ALPHA)
    findings = []
    for r, adj, rej in zip(rows, bh["adjusted"], bh["rejected"]):
        findings.append(SegmentFinding(
            segment=r["segment"], label=r["label"],
            n_event=r["n_event"], n_baseline=r["n_baseline"],
            event_value=r["event_value"], baseline_value=r["baseline_value"],
            delta=r["delta"], p_value=r["p_value"], p_adjusted=adj,
            significant=bool(rej), contribution=r["contribution"],
        ))
    findings.sort(key=lambda f: f.delta)

    naive_hits = sum(1 for p in pvals if p <= C.FDR_ALPHA)
    return {
        "findings": [f.to_dict() for f in findings],
        "n_tested": len(rows),
        "n_significant_naive": naive_hits,
        "n_significant_fdr": bh["n_significant"],
        "fdr_alpha": C.FDR_ALPHA,
        "note": (
            f"Tested {len(rows)} segments. At a naive alpha={C.FDR_ALPHA} threshold "
            f"{naive_hits} would be called significant; after Benjamini-Hochberg FDR "
            f"control {bh['n_significant']} survive. The difference is the number of "
            f"findings that would have been noise reported as insight."
        ),
    }

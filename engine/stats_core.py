"""Statistical primitives.

Every value returned here is computed from data. No hard-coded effect sizes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Sequence

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------
# Robust baseline / anomaly detection
# --------------------------------------------------------------------------

MAD_TO_SIGMA = 1.4826  # makes MAD a consistent estimator of sigma under normality


@dataclass
class RobustBaseline:
    """Median/MAD baseline. Robust so the anomaly cannot inflate its own baseline."""
    center: float
    scale: float
    n: int

    def z(self, value: float) -> float:
        if self.scale <= 0:
            return 0.0
        return float((value - self.center) / self.scale)

    def limits(self, k: float = 3.0) -> tuple[float, float]:
        return (self.center - k * self.scale, self.center + k * self.scale)


def robust_baseline(values: Sequence[float]) -> RobustBaseline:
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return RobustBaseline(center=float("nan"), scale=0.0, n=0)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med))) * MAD_TO_SIGMA
    if mad <= 0:  # degenerate (e.g. near-constant series) -> fall back to std
        mad = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    return RobustBaseline(center=med, scale=mad, n=int(a.size))


def cusum(values: Sequence[float], target: float, scale: float,
          k: float = 0.5, h: float = 4.0) -> dict:
    """
    Tabular CUSUM changepoint detection. Returns the first index at which the
    cumulative negative (or positive) deviation exceeds h*scale.

    k is the slack in units of `scale` (deviations smaller than k are ignored);
    h is the decision interval. k=0.5, h=4 is the standard textbook pairing that
    gives an in-control ARL of roughly 168 for a normal process.
    """
    a = np.asarray(values, dtype=float)
    if scale <= 0 or a.size == 0:
        return {"detected": False, "index": None, "direction": None}
    z = (a - target) / scale
    hi = lo = 0.0
    for i, zi in enumerate(z):
        hi = max(0.0, hi + zi - k)
        lo = min(0.0, lo + zi + k)
        if hi > h:
            return {"detected": True, "index": int(i), "direction": "up"}
        if lo < -h:
            return {"detected": True, "index": int(i), "direction": "down"}
    return {"detected": False, "index": None, "direction": None}


# --------------------------------------------------------------------------
# Multiple-comparison control  (the thing that separates signal from noise
# when you scan many segments at once)
# --------------------------------------------------------------------------

def benjamini_hochberg(pvals: Sequence[float], alpha: float = 0.05) -> dict:
    """
    Benjamini-Hochberg FDR control.

    Why this matters: scanning 27 states for "which one moved" at alpha=0.05
    yields ~1.35 false discoveries by chance alone. Without FDR control an
    investigation engine confidently reports noise as a finding.
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return {"rejected": [], "adjusted": [], "n_significant": 0, "alpha": alpha}
    order = np.argsort(p)
    ranked = p[order]
    # step-up adjusted p-values, enforced monotone
    adj_ranked = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    adj_ranked = np.clip(adj_ranked, 0.0, 1.0)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adj_ranked
    rejected = adjusted <= alpha
    return {
        "rejected": rejected.tolist(),
        "adjusted": adjusted.tolist(),
        "n_significant": int(rejected.sum()),
        "alpha": alpha,
    }


# --------------------------------------------------------------------------
# Effect sizes with honest uncertainty
# --------------------------------------------------------------------------

@dataclass
class EffectSize:
    estimate: float
    ci_low: float
    ci_high: float
    p_value: float
    method: str
    n_treat: int
    n_control: int

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        return d


def two_proportion_test(x_t: int, n_t: int, x_c: int, n_c: int,
                        alpha: float = 0.05) -> EffectSize:
    """Difference in proportions (treat - control), Wald CI + pooled z-test."""
    if n_t == 0 or n_c == 0:
        return EffectSize(0.0, 0.0, 0.0, 1.0, "two_proportion_z", n_t, n_c)
    p_t, p_c = x_t / n_t, x_c / n_c
    diff = p_t - p_c
    se_ci = float(np.sqrt(p_t * (1 - p_t) / n_t + p_c * (1 - p_c) / n_c))
    p_pool = (x_t + x_c) / (n_t + n_c)
    se_h0 = float(np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c)))
    z = diff / se_h0 if se_h0 > 0 else 0.0
    pval = float(2 * (1 - stats.norm.cdf(abs(z))))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    return EffectSize(
        estimate=float(diff),
        ci_low=float(diff - crit * se_ci),
        ci_high=float(diff + crit * se_ci),
        p_value=pval,
        method="two_proportion_z",
        n_treat=int(n_t), n_control=int(n_c),
    )


def welch_t_test(treat: Sequence[float], control: Sequence[float],
                 alpha: float = 0.05) -> EffectSize:
    """Welch's t-test for continuous metrics (unequal variance, unequal n)."""
    a = np.asarray([v for v in treat if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in control if np.isfinite(v)], dtype=float)
    if a.size < 2 or b.size < 2:
        return EffectSize(0.0, 0.0, 0.0, 1.0, "welch_t", a.size, b.size)
    diff = float(a.mean() - b.mean())
    se = float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))
    if se == 0:
        return EffectSize(diff, diff, diff, 1.0, "welch_t", a.size, b.size)
    df = (a.var(ddof=1) / a.size + b.var(ddof=1) / b.size) ** 2 / (
        (a.var(ddof=1) / a.size) ** 2 / (a.size - 1)
        + (b.var(ddof=1) / b.size) ** 2 / (b.size - 1)
    )
    t = diff / se
    pval = float(2 * (1 - stats.t.cdf(abs(t), df)))
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    return EffectSize(diff, float(diff - crit * se), float(diff + crit * se),
                      pval, "welch_t", a.size, b.size)


def bootstrap_ci(data: Sequence[float], statistic=np.mean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 7) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Non-parametric, no distributional assumption."""
    a = np.asarray([v for v in data if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    boots = statistic(a[idx], axis=1)
    return (float(statistic(a)),
            float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


# --------------------------------------------------------------------------
# Causal: difference-in-differences
# --------------------------------------------------------------------------

@dataclass
class DiDResult:
    did_estimate: float
    treat_change: float
    control_change: float
    ci_low: float
    ci_high: float
    p_value: float
    parallel_trends_p: float
    parallel_trends_ok: bool

    def to_dict(self) -> dict:
        return asdict(self)


def difference_in_differences(treat_pre: Sequence[float], treat_post: Sequence[float],
                              ctrl_pre: Sequence[float], ctrl_post: Sequence[float],
                              pre_treat_series: Sequence[float] | None = None,
                              pre_ctrl_series: Sequence[float] | None = None,
                              alpha: float = 0.05) -> DiDResult:
    """
    DiD estimate = (treat_post - treat_pre) - (ctrl_post - ctrl_pre).

    Also runs an explicit parallel-trends check on the PRE period. DiD is only
    credible when trends were parallel before the event; we test that rather
    than assuming it, and surface the result either way.
    """
    tp, tq = np.asarray(treat_pre, float), np.asarray(treat_post, float)
    cp, cq = np.asarray(ctrl_pre, float), np.asarray(ctrl_post, float)
    t_change = float(tq.mean() - tp.mean())
    c_change = float(cq.mean() - cp.mean())
    did = t_change - c_change
    se = float(np.sqrt(
        tq.var(ddof=1) / max(tq.size, 1) + tp.var(ddof=1) / max(tp.size, 1)
        + cq.var(ddof=1) / max(cq.size, 1) + cp.var(ddof=1) / max(cp.size, 1)
    ))
    crit = float(stats.norm.ppf(1 - alpha / 2))
    pval = float(2 * (1 - stats.norm.cdf(abs(did / se)))) if se > 0 else 1.0

    # Parallel-trends: regress the pre-period difference on time; slope ~ 0 is good.
    pt_p, pt_ok = float("nan"), False
    if pre_treat_series is not None and pre_ctrl_series is not None:
        d = np.asarray(pre_treat_series, float) - np.asarray(pre_ctrl_series, float)
        d = d[np.isfinite(d)]
        if d.size >= 4:
            lr = stats.linregress(np.arange(d.size), d)
            pt_p = float(lr.pvalue)
            pt_ok = bool(pt_p > 0.10)  # fail to reject flat pre-trend
    return DiDResult(did, t_change, c_change,
                     float(did - crit * se), float(did + crit * se),
                     pval, pt_p, pt_ok)


# --------------------------------------------------------------------------
# Experiment design. Powers the "recommend the next step" output.
# --------------------------------------------------------------------------

def required_sample_size(baseline_rate: float, mde_abs: float,
                         alpha: float = 0.05, power: float = 0.80) -> int:
    """Per-arm n for a two-proportion test to detect `mde_abs` at given power."""
    if mde_abs <= 0 or not (0 < baseline_rate < 1):
        return 0
    p1 = baseline_rate
    p2 = min(max(baseline_rate + mde_abs, 1e-6), 1 - 1e-6)
    p_bar = (p1 + p2) / 2
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n = ((z_a * np.sqrt(2 * p_bar * (1 - p_bar))
          + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (mde_abs ** 2)
    return int(np.ceil(n))


def required_sample_size_continuous(sd: float, mde_abs: float,
                                    alpha: float = 0.05, power: float = 0.80) -> int:
    """Per-arm n for a two-sample t-test on a continuous metric."""
    if mde_abs <= 0 or sd <= 0:
        return 0
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n = 2 * ((z_a + z_b) ** 2) * (sd ** 2) / (mde_abs ** 2)
    return int(np.ceil(n))


# --------------------------------------------------------------------------
# Temporal precedence. A cause must precede its effect.
# --------------------------------------------------------------------------

def lead_lag_correlation(cause: Sequence[float], effect: Sequence[float],
                         max_lag: int = 4) -> dict:
    """
    Cross-correlate a candidate cause against the effect at several lags.

    lag > 0 means the cause series leads (moves earlier than) the effect --
    consistent with causation. lag < 0 means the "cause" moves after the effect,
    which is evidence AGAINST it being the cause.
    """
    c = np.asarray(cause, float)
    e = np.asarray(effect, float)
    n = min(c.size, e.size)
    c, e = c[:n], e[:n]
    out = []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = c[:n - lag], e[lag:]
        else:
            x, y = c[-lag:], e[:n + lag]
        if x.size < 3 or np.std(x) == 0 or np.std(y) == 0:
            continue
        out.append({"lag": lag, "corr": float(np.corrcoef(x, y)[0, 1]), "n": int(x.size)})
    if not out:
        return {"best_lag": None, "best_corr": 0.0, "precedes": False, "by_lag": []}
    best = max(out, key=lambda d: abs(d["corr"]))
    return {
        "best_lag": best["lag"],
        "best_corr": best["corr"],
        "precedes": bool(best["lag"] > 0),
        "by_lag": out,
    }

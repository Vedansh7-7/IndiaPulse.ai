"""Tests for the statistical core and the decision logic.

Three properties are covered:
  - the statistics match known closed-form answers
  - FDR control suppresses false discoveries on pure noise
  - every decision state is reachable, including the ones that decline to answer
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.stats_core import (
    robust_baseline, benjamini_hochberg, two_proportion_test, welch_t_test,
    bootstrap_ci, required_sample_size, lead_lag_correlation,
    difference_in_differences, cusum,
)
from engine.agents.base import composite_score, statistical_strength


# --------------------------------------------------------------------- stats

def test_robust_baseline_ignores_contamination():
    """Median/MAD must not be dragged by a minority of extreme outliers."""
    clean = [10.0] * 40 + [10.5, 9.5] * 5
    contaminated = clean + [-500.0] * 8
    a, b = robust_baseline(clean), robust_baseline(contaminated)
    assert abs(a.center - b.center) < 0.6, "outliers moved the robust centre"
    # a mean-based centre would be destroyed by the same contamination
    assert abs(np.mean(contaminated) - np.mean(clean)) > 50


def test_robust_z_flags_the_outlier_not_the_baseline():
    base = robust_baseline([4.2, 4.25, 4.18, 4.22, 4.3, 4.19, 4.26, 4.21])
    assert abs(base.z(4.22)) < 1.5
    assert base.z(3.4) < -3.0


def test_benjamini_hochberg_matches_hand_computed_values():
    """Checked by hand against the step-up definition.

    n=8, alpha=0.05. Largest k with p_(k) <= k*alpha/n:
        k=1: 0.001 <= 0.00625  yes
        k=2: 0.008 <= 0.0125   yes
        k=3: 0.039 <= 0.01875  no   (and none above)
    => 2 rejections.

    Adjusted p_(k) = min over j>=k of (n/j)*p_(j), enforced monotone:
        [0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 0.084571, 0.205]
    """
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    r = benjamini_hochberg(p, alpha=0.05)
    adj = r["adjusted"]
    expected = [0.008, 0.032, 0.0672, 0.0672, 0.0672, 0.08, 0.0845714, 0.205]
    assert adj == pytest.approx(expected, abs=1e-6)
    assert all(adj[i] >= p[i] - 1e-12 for i in range(len(p)))
    assert adj == sorted(adj), "adjusted p-values must be monotone in rank order"
    assert r["n_significant"] == 2
    assert r["rejected"] == [True, True, False, False, False, False, False, False]


def test_benjamini_hochberg_is_stricter_than_naive_alpha():
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
    naive = sum(1 for x in p if x <= 0.05)
    assert naive == 5
    assert benjamini_hochberg(p, 0.05)["n_significant"] == 2


def test_fdr_suppresses_false_discoveries_on_pure_noise():
    """Scanning many null segments produces false hits. FDR control removes
    most of them."""
    rng = np.random.default_rng(11)
    naive_hits, fdr_hits, trials = 0, 0, 200
    for _ in range(trials):
        # 40 segments, all drawn from the SAME distribution -> every null true
        pvals = []
        for _ in range(40):
            a = rng.normal(0, 1, 60)
            b = rng.normal(0, 1, 60)
            pvals.append(welch_t_test(a, b).p_value)
        naive_hits += sum(1 for p in pvals if p <= 0.05)
        fdr_hits += benjamini_hochberg(pvals, 0.05)["n_significant"]
    assert naive_hits > 300, "sanity: naive testing should yield ~2 hits/trial"
    assert fdr_hits < naive_hits / 5, (
        f"FDR control failed to suppress noise: {fdr_hits} vs naive {naive_hits}")


def test_two_proportion_test_against_closed_form():
    eff = two_proportion_test(x_t=450, n_t=1000, x_c=500, n_c=1000)
    assert eff.estimate == pytest.approx(-0.05, abs=1e-9)
    # pooled p = 0.475, se = sqrt(.475*.525*(2/1000)) = 0.022325
    assert eff.p_value == pytest.approx(0.0250, abs=2e-3)
    assert eff.ci_low < eff.estimate < eff.ci_high


def test_welch_handles_unequal_variance_and_n():
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.0, 500)
    b = rng.normal(0.0, 4.0, 60)
    eff = welch_t_test(a, b)
    assert eff.p_value > 0.05, "no true difference should not be flagged"
    c = rng.normal(3.0, 1.0, 500)
    assert welch_t_test(c, a).p_value < 1e-20


def test_bootstrap_ci_covers_the_true_mean():
    rng = np.random.default_rng(5)
    data = rng.normal(10.0, 2.0, 800)
    est, lo, hi = bootstrap_ci(data, n_boot=1500)
    assert lo < 10.0 < hi
    assert est == pytest.approx(float(np.mean(data)), abs=1e-9)


def test_power_calculation_scales_correctly():
    """Halving the detectable effect should roughly quadruple the sample."""
    n_big = required_sample_size(0.50, 0.10)
    n_small = required_sample_size(0.50, 0.05)
    assert 3.5 < n_small / n_big < 4.5
    # standard reference point: p=0.5, MDE=0.05, 80% power ~ 1570/arm
    assert 1400 < n_small < 1750


def test_lead_lag_detects_a_known_lead():
    n = 60
    cause = np.sin(np.arange(n) / 3.0)
    effect = np.concatenate([np.zeros(2), cause[:-2]])  # effect lags by 2
    r = lead_lag_correlation(cause, effect, max_lag=4)
    assert r["best_lag"] == 2 and r["precedes"] is True


def test_did_recovers_a_planted_effect():
    rng = np.random.default_rng(9)
    ctrl_pre = rng.normal(4.2, 0.3, 400)
    ctrl_post = rng.normal(4.1, 0.3, 400)      # -0.1 secular drift
    treat_pre = rng.normal(4.2, 0.3, 400)
    treat_post = rng.normal(3.7, 0.3, 400)     # -0.5 total => -0.4 treatment
    res = difference_in_differences(treat_pre, treat_post, ctrl_pre, ctrl_post)
    assert res.did_estimate == pytest.approx(-0.4, abs=0.08)
    assert res.ci_high < 0


def test_cusum_finds_a_step_change():
    series = np.concatenate([np.full(30, 4.2), np.full(20, 3.6)])
    r = cusum(series, target=4.2, scale=0.15)
    assert r["detected"] and r["direction"] == "down" and 29 <= r["index"] <= 36


# ------------------------------------------------------------- scoring rules

def test_statistical_strength_is_bounded_and_monotone():
    weak = statistical_strength(0.01, -0.005, 0.025)
    strong = statistical_strength(0.50, 0.45, 0.55)
    assert 0.0 <= weak < strong <= 1.0


def test_composite_score_weights_sum_to_one():
    score, comps = composite_score(1.0, 1.0, 1.0, 1.0)
    assert score == pytest.approx(1.0)
    w = comps["weights"]
    assert sum(w.values()) == pytest.approx(1.0)


def test_composite_score_cannot_manufacture_evidence():
    """Zero measured evidence must produce zero score, whatever the weights."""
    score, _ = composite_score(0.0, 0.0, 0.0, 0.0)
    assert score == 0.0


# ------------------------------------------------------- end-to-end decisions

@pytest.fixture(scope="module")
def bundle():
    import json
    from engine import config as C
    path = C.ROOT / "docs" / "data.js"
    if not path.exists():
        pytest.skip("run `python build_demo.py` first")
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw[raw.index("{"):raw.rstrip().rstrip(";").rindex("}") + 1])


def test_engine_can_decline_as_well_as_confirm(bundle):
    """An engine that can only ever confirm is not an investigation."""
    states = {s["state"] for s in bundle["index"]}
    assert {"CONFIRMED", "INCONCLUSIVE", "NOISE"} <= states, (
        f"the engine must be able to abstain and to dismiss: {states}")


def test_artefact_is_reachable_and_gated_before_any_agent():
    """A broken feed must be caught whether or not the metric appears to move.

    Integrity used to run inside Scope 2, which only executes once triage
    reports a signal, so a feed that broke on a stable-looking metric was never
    checked. It now runs in Scope 0 against a recent-versus-prior window.
    """
    from pathlib import Path
    from engine.generic_run import investigate_upload
    f = Path(__file__).parent / "measurement_artefact.csv"
    if not f.exists():
        pytest.skip("fixture not generated; run tests/test_data_gen.py")
    r = investigate_upload(f.read_bytes(), metric_key="avg_value",
                           filename="measurement_artefact.csv")
    assert r["scope4_decision"]["state"] == "ARTEFACT"
    assert r["scope2_verdicts"] == [], "agents ran on data known to be broken"
    ig = r["governance"]["integrity"]
    assert ig["is_artefact"] and ig["n_structural"] >= 1


def test_volume_change_alone_never_triggers_artefact():
    """A business is allowed to grow. Only structural breakage counts."""
    from engine.integrity import check
    import pandas as pd, numpy as np
    rng = np.random.default_rng(4)
    weeks = pd.date_range("2024-01-01", periods=30, freq="W")
    rows = []
    for i, w in enumerate(weeks):
        n = 40 if i < 20 else 120            # volume triples, nothing else changes
        for _ in range(n):
            rows.append({"wk": str(w.date()), "value": float(rng.normal(10, 2)),
                         "seg": str(rng.choice(["a", "b", "c"]))})
    p = pd.DataFrame(rows)
    rep = check(p, "value", segment_columns=["seg"], value_columns=["value"])
    assert not rep.is_artefact, "a volume change alone was treated as a broken feed"


def test_short_series_abstains_instead_of_inventing_a_baseline():
    from engine.detect import triage
    import pandas as pd, numpy as np
    rng = np.random.default_rng(5)
    weeks = pd.date_range("2024-01-01", periods=6, freq="W")
    weekly = pd.DataFrame({"week": weeks, "orders": 100,
                           "m": rng.normal(10, 1, len(weeks))})
    panel = pd.DataFrame({"week": weeks, "wk": [str(w.date()) for w in weeks],
                          "delivered": True, "m": rng.normal(10, 1, len(weeks))})
    t = triage(panel, weekly, "m")
    assert t.verdict == "NO_BASELINE" and t.is_signal is False


def test_noise_path_stops_before_spawning_agents(bundle):
    noise = next(i for i in bundle["index"] if i["state"] == "NOISE")
    inv = bundle["investigations"][noise["key"]]
    assert inv["scope2_verdicts"] == [], "agents ran on a movement judged to be noise"
    assert inv["scope0_triage"]["is_signal"] is False


def test_inconclusive_withholds_a_recommendation(bundle):
    inc = next(i for i in bundle["index"] if i["state"] == "INCONCLUSIVE")
    dec = bundle["investigations"][inc["key"]]["scope4_decision"]
    assert dec["separability"]["separable"] is False
    assert "discriminating_checks" in dec["next_test"]["design"]


def test_confirmed_case_rejects_the_demand_hypothesis(bundle):
    """The engine must be able to reject a hypothesis, not only rank."""
    inv = bundle["investigations"]["national"]
    ext = next(v for v in inv["scope2_verdicts"] if v["cause_family"] == "demand")
    assert ext["supported"] is False


def test_confirmed_case_survives_every_adversarial_challenge(bundle):
    adv = bundle["investigations"]["national"]["scope3_adversary"]
    assert adv["survived"] is True and adv["n_passed"] == adv["n_total"]


def test_every_verdict_states_what_would_falsify_it(bundle):
    for key, inv in bundle["investigations"].items():
        for v in inv["scope2_verdicts"]:
            assert v["falsifiable_by"].strip(), f"{key}/{v['agent']} has no falsifier"


def test_every_evidence_item_carries_a_source(bundle):
    for key, inv in bundle["investigations"].items():
        for v in inv["scope2_verdicts"]:
            for e in v["evidence"]:
                assert e["source"].strip(), f"{key}/{v['agent']} evidence lacks a citation"


def test_corroborating_agents_do_not_compete(bundle):
    """Ops and VoC are two witnesses to one cause, so they share a family."""
    inv = bundle["investigations"]["national"]
    fams = {v["agent"]: v["cause_family"] for v in inv["scope2_verdicts"]}
    assert fams["Ops & Fulfilment"] == fams["Voice of Customer"] == "fulfilment"
    sep = inv["scope4_decision"]["separability"]
    assert "Voice of Customer" in sep["families"][0]["corroborated_by"]

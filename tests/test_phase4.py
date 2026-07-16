"""Phase 4 acceptance tests: endogeneity, behavior, substitution, interference."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace

import numpy as np
import pandas as pd

from demand_sim import (EndogenousConfig, ExperimentConfig, InterferenceConfig,
                        SubstitutionConfig, WaitingConfig, default_config,
                        demand_curve, demand_curve_ref, fit_2sls,
                        fit_control_function, fit_ols,
                        interference_design_bias, interference_probe,
                        run_interference_experiment, simulate_endogenous_market,
                        simulate_substitute_pair, simulate_with_waiting,
                        strategic_waiting_check)
from demand_sim.config import InventoryConfig
from demand_sim.interference import pool_capacity
from demand_sim.inventory import simulate_inventory
from demand_sim.panel import SeriesParams

CFG = default_config()


# ------------------------------------------------------------- endogeneity
def test_ols_biased_iv_cf_recover():
    df = simulate_endogenous_market(EndogenousConfig(gamma=0.5))
    theta = EndogenousConfig().theta
    assert fit_ols(df)["theta_hat"] - theta > 0.2          # attenuated
    iv = fit_2sls(df)
    assert abs(iv["theta_hat"] - theta) < 0.05
    assert iv["first_stage_F"] > 10 and not iv["weak_instrument"]
    assert abs(fit_control_function(df)["theta_hat"] - theta) < 0.05


def test_ols_fine_without_endogeneity():
    df = simulate_endogenous_market(EndogenousConfig(gamma=0.0))
    assert abs(fit_ols(df)["theta_hat"] - EndogenousConfig().theta) < 0.05


def test_weak_instrument_flagged():
    df = simulate_endogenous_market(EndogenousConfig(gamma=0.5, pi=0.005))
    assert fit_2sls(df)["weak_instrument"]


def test_cf_endogeneity_ttest_detects_gamma():
    hot = fit_control_function(
        simulate_endogenous_market(EndogenousConfig(gamma=0.8)))
    cold = fit_control_function(
        simulate_endogenous_market(EndogenousConfig(gamma=0.0)))
    assert abs(hot["endogeneity_t"]) > 5
    assert abs(cold["endogeneity_t"]) < 3


# --------------------------------------------------------- reference prices
def test_reference_price_kink():
    grid = np.array([25.0, 29.0, 33.0])
    cfg_ref = replace(CFG, segments=tuple(
        replace(s, ref_price_sensitivity=0.08) for s in CFG.segments))
    plain = demand_curve(CFG, grid)["P_aggregate"].to_numpy()
    kinked = demand_curve_ref(cfg_ref, grid, ref_price=29.0)["P_aggregate"].to_numpy()
    np.testing.assert_allclose(plain[:2], kinked[:2])       # identical <= anchor
    assert kinked[2] < plain[2]                             # harsher above


def test_reference_price_zero_sensitivity_reduces_to_phase1():
    grid = np.linspace(10, 60, 11)
    plain = demand_curve(CFG, grid)["P_aggregate"].to_numpy()
    ref = demand_curve_ref(CFG, grid, ref_price=29.0)["P_aggregate"].to_numpy()
    np.testing.assert_allclose(plain, ref)   # default sensitivity is 0


# ----------------------------------------------------------- demand shifting
def test_waiting_detector_fires_only_when_waiting_exists():
    hot = simulate_with_waiting(CFG, WaitingConfig(wait_prob=0.5,
                                                   n_consumers=10_000, seed=2))
    cold = simulate_with_waiting(CFG, WaitingConfig(wait_prob=0.0,
                                                    n_consumers=10_000, seed=2))
    assert strategic_waiting_check(hot)["waiting_alarm"]
    assert not strategic_waiting_check(cold)["waiting_alarm"]


# --------------------------------------------------- substitution / backorder
def _series(pid: str, base_rate: float = 8.0) -> SeriesParams:
    return SeriesParams(pid, "S01", "beverages", 10.0, -1.5, base_rate,
                        0.0, 0.1, 0.1, 0.0, 2.0, 0.2, 20, 60, 3)


def test_stockout_spill_reaches_substitute():
    dates = pd.date_range("2025-01-01", periods=365)
    promo = np.zeros(365, bool)
    flat = np.full(365, 10.0)
    pair = simulate_substitute_pair(_series("A"), _series("B"), dates,
                                    flat, flat, promo, promo,
                                    SubstitutionConfig(spill_rate=0.5, seed=1))
    assert (pair["units_sold"] <= pair["units_demanded"]).all()
    assert pair["spill_in"].sum() > 0                       # spill happened
    a = pair[pair["product_id"] == "A"]
    # demand on spill days exceeds own demand (the substitute's overflow)
    spill_days = a[a["spill_in"] > 0]
    assert (spill_days["units_demanded"]
            > spill_days["own_demand"]).all()


def test_cross_price_effect():
    dates = pd.date_range("2025-01-01", periods=365)
    promo = np.zeros(365, bool)
    flat = np.full(365, 10.0)
    dear_b = np.full(365, 14.0)   # rival 40% more expensive
    cheap = simulate_substitute_pair(_series("A"), _series("B"), dates,
                                     flat, flat, promo, promo,
                                     SubstitutionConfig(seed=3))
    dear = simulate_substitute_pair(_series("A"), _series("B"), dates,
                                    flat, dear_b, promo, promo,
                                    SubstitutionConfig(seed=3))
    own_a = lambda df: df.loc[df["product_id"] == "A", "own_demand"].sum()
    assert own_a(dear) > own_a(cheap)   # rival price up => my demand up


def test_backorder_shifts_rather_than_destroys_sales():
    inv = InventoryConfig(reorder_point=30, order_up_to=60, lead_time_days=3)
    demand = np.random.default_rng(0).poisson(8.0, 200)
    lost = simulate_inventory(demand, inv, np.random.default_rng(0))
    bo = simulate_inventory(demand,
                            replace(inv, stockout_behavior="backorder"),
                            np.random.default_rng(0))
    assert bo.units_sold.sum() >= lost.units_sold.sum()
    assert bo.units_sold.sum() <= demand.sum()              # conservation


# --------------------------------------------------------------- interference
BASE = ExperimentConfig(arm_prices=(29.0, 39.0), horizon_days=5,
                        sessions_per_day=2_000, window_hours=2,
                        n_markets=10, seed=5)
ICFG = InterferenceConfig(capacity_slack=0.85)


def test_interference_biases_split_designs_not_clustered_ones():
    tab = interference_design_bias(CFG, BASE, ICFG,
                                   designs=("session", "switchback_window"),
                                   n_reps=8).set_index("randomization_unit")
    assert abs(tab.loc["session", "bias"]) \
        > 3 * abs(tab.loc["switchback_window", "bias"])
    assert abs(tab.loc["switchback_window", "bias"]) < 0.01


def test_interference_probe_detects_leakage():
    # tighter capacity + wider arm gap => strong, unambiguous leakage
    exp = replace(BASE, randomization_unit="session",
                  arm_prices=(29.0, 49.0), horizon_days=7,
                  sessions_per_day=4_000)
    icfg = InterferenceConfig(capacity_slack=0.6)
    cap = pool_capacity(CFG, exp, icfg)
    out = run_interference_experiment(CFG, exp, icfg, capacity=cap)
    pure = run_interference_experiment(
        CFG, replace(exp, arm_prices=(29.0, 29.0), seed=99), icfg,
        capacity=cap)
    probe = interference_probe(out["sessions"], pure["sessions"])
    assert probe["interference_alarm"]
    # treatment at a HIGHER price frees capacity: in-experiment control
    # converts better than a pure-control market — leakage has a sign
    assert probe["diff"] > 0


def test_deterministic_under_seed():
    a = simulate_endogenous_market(EndogenousConfig())["log_q"].sum()
    b = simulate_endogenous_market(EndogenousConfig())["log_q"].sum()
    assert a == b
    w1 = simulate_with_waiting(CFG, WaitingConfig(n_consumers=2_000))
    w2 = simulate_with_waiting(CFG, WaitingConfig(n_consumers=2_000))
    assert w1["purchased"].sum() == w2["purchased"].sum()
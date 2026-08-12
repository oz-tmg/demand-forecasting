"""Phase 6 acceptance tests: recovering a demand system through a clearing market.

Encodes the docs/10 acceptance criteria:
  - price is an equilibrium outcome, so naive Berry-inversion OLS is biased;
  - 2SLS with the promo/rival instruments recovers alpha and lambda near truth,
    with a strong first stage and 95% CI that covers the truth;
  - the recovered diversion matrix matches the oracle;
  - the observable/oracle boundary holds (estimators never read xi_true);
  - generation is deterministic under a fixed seed.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace

import numpy as np

from demand_sim import (MarketConfig, simulate_market, fit_berry_ols,
                        fit_berry_2sls, score_diversion, market_scorecard,
                        nested_logit_shares, clear_market)
from demand_sim.market import ORACLE_COLS, OBSERVABLE_COLS

CFG = MarketConfig()
DF = simulate_market(CFG)


# --------------------------------------------------------------- demand system
def test_shares_are_a_probability_simplex():
    _, x, base = CFG.n_goods, None, None
    from demand_sim.market import _attributes
    nest_id, x, base_stock = _attributes(CFG)
    delta = CFG.beta_x * x
    s_j, s0, s_jg = nested_logit_shares(delta, nest_id, CFG.lam)
    assert abs(s_j.sum() + s0 - 1.0) < 1e-9          # inside + outside good = 1
    assert (s_j > 0).all() and 0 < s0 < 1
    # within-nest shares sum to 1 in every nest
    for g in np.unique(nest_id):
        assert abs(s_jg[nest_id == g].sum() - 1.0) < 1e-9


def test_market_clears_supply_equals_demand():
    from demand_sim.market import _attributes
    nest_id, x, base_stock = _attributes(CFG)
    mean_util = CFG.beta_x * x
    logp, q, s_j, s0, s_jg = clear_market(
        mean_util, CFG.alpha, base_stock, CFG.supply_elasticity,
        np.zeros(CFG.n_goods), nest_id, CFG.lam, CFG.population)
    supply = np.exp(base_stock + CFG.supply_elasticity * logp)   # omega = 0
    # demand == supply at the clearing price (tatonnement converged)
    assert np.max(np.abs(np.log(q) - np.log(supply))) < 1e-2


# ------------------------------------------------------- the headline result
def test_ols_biased_2sls_recovers_alpha_and_lambda():
    ols = fit_berry_ols(DF)
    iv = fit_berry_2sls(DF)
    # OLS is badly biased on price sensitivity (equilibrium simultaneity)
    assert abs(ols["alpha"] - CFG.alpha) > 0.3
    # 2SLS recovers both structural parameters
    assert abs(iv["alpha"] - CFG.alpha) < 0.08
    assert abs(iv["lam"] - CFG.lam) < 0.08


def test_first_stage_strong():
    iv = fit_berry_2sls(DF)
    assert iv["first_stage_F"] > 10 and not iv["weak_instrument"]


def test_2sls_ci_covers_truth():
    iv = fit_berry_2sls(DF)
    lo = iv["alpha"] - 1.96 * iv["alpha_se"]
    hi = iv["alpha"] + 1.96 * iv["alpha_se"]
    assert lo <= CFG.alpha <= hi


def test_diversion_matrix_matches_oracle():
    div = score_diversion(CFG, fit_berry_2sls(DF))
    assert div["diversion_mae"] < 0.02
    # substitution is fierce inside a nest, weak across — recovered structure agrees
    assert div["within_nest_share_true"] > 0.8
    assert abs(div["within_nest_share_hat"] - div["within_nest_share_true"]) < 0.03


# ------------------------------------------------- observable/oracle boundary
def test_estimators_ignore_the_oracle():
    """Permuting xi_true must not move any estimate — proof no estimator reads it."""
    poisoned = DF.copy()
    poisoned["xi_true"] = np.random.default_rng(0).permutation(poisoned["xi_true"].values)
    for fit in (fit_berry_ols, fit_berry_2sls):
        base, pois = fit(DF), fit(poisoned)
        assert abs(base["alpha"] - pois["alpha"]) < 1e-12
        assert abs(base["lam"] - pois["lam"]) < 1e-12


def test_oracle_column_is_not_observable():
    assert "xi_true" in ORACLE_COLS
    assert "xi_true" not in OBSERVABLE_COLS


# --------------------------------------------------------------- reproducibility
def test_deterministic_under_seed():
    a = simulate_market(CFG)
    b = simulate_market(CFG)
    assert np.array_equal(a["logp"].values, b["logp"].values)
    assert np.array_equal(a["q"].values, b["q"].values)


def test_no_endogeneity_ols_is_fine():
    """Kill the demand shock (xi_sd=0) and OLS attenuation should largely vanish."""
    clean = simulate_market(replace(CFG, xi_sd=0.0))
    ols = fit_berry_ols(clean)
    # with no unobserved demand shock the only endogeneity left is small
    assert abs(ols["alpha"] - CFG.alpha) < 0.15


# --------------------------------------------------------------- scorecard
def test_scorecard_ols_biased_2sls_unbiased_and_covers():
    small = replace(CFG, n_weeks=150)
    card = market_scorecard(small, n_reps=10).set_index("estimator")
    assert abs(card.loc["2sls", "mean_bias"]) < 0.05
    assert abs(card.loc["ols", "mean_bias"]) > 0.3
    assert card.loc["2sls", "coverage_95"] >= 0.7    # near-nominal (lenient for n_reps)

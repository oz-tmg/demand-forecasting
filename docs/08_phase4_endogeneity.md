# Phase 4 — Endogenous Pricing, Behavior, Substitution, Interference

**Status:** implemented (v0.4.0)
**Modules:** `endogenous.py`, `behavior.py`, `substitution.py`, `interference.py` (+ `inventory.py` backorder, `guardrails.py` detectors) · **Demo:** `examples/phase4_endogeneity.py`
**Spec:** §3.1 (endogenous rule, reference prices), §3.2 (demand shifting), §3.3 (substitute/backorder), §3.4 (interference) · **Design docs:** docs/05, docs/04 §3.4

Phase 4 adds every mechanism that makes real pricing data lie, each with a
strength dial and a known truth — so estimators and guardrails can be graded
on exactly how wrong they are and whether the alarms fire.

## 1. Endogenous pricing (`endogenous.py`) — the flagship

The docs/05 §2 DGP verbatim: the pricer partially observes the demand shock
and prices into it (`gamma` dial); a cost shifter `z` moves price but not
demand (`pi` = instrument-strength dial). Estimators: `fit_ols` (biased),
`fit_2sls` (with first-stage F and weak-instrument flag), and
`fit_control_function` (residual inclusion; the t on v̂ is the endogeneity
test). `endogeneity_scorecard` sweeps gamma and produces the docs/05 §8 bias
curves:

```
gamma      2sls    control_function    ols
0.00      0.002               0.002  0.003
0.25      0.002               0.002  0.237
0.50      0.002               0.002  0.401
0.75      0.002               0.002  0.480
1.00      0.002               0.002  0.499     (theta_true = -1.5)
```

OLS attenuates toward zero exactly as `plim θ̂ = θ + Cov(P̃,u)/Var(P̃)`
predicts; IV and CF stay flat while F is healthy. DML variants remain in the
optional `[estimators]` extra.

## 2. Reference prices & demand shifting (`behavior.py`)

**Reference prices** activate the `SegmentConfig.ref_price_sensitivity` hook:
utility takes an extra penalty only above the anchor, producing a kinked
demand curve (elasticity jumps at p_ref — why firms defend price points).
Zero sensitivity reduces exactly to Phase 1.

**Demand shifting**: `simulate_with_waiting` lets would-be buyers who see a
price above the anchor defer and revisit; organic (price-independent) return
traffic keeps the null world honest. `guardrails.strategic_waiting_check`
tests whether revisit sessions' *previous* prices skew above the served-price
distribution — fires at wait_prob 0.3+, silent at 0. (First implementation
compared cheap- vs dear-cell buyers' prior counts; that signal inverts under
WTP heterogeneity because deferrers are high-WTP consumers who buy anywhere.
The prev-price test is selection-proof.)

## 3. Substitution & backorder (`substitution.py`, `inventory.py`)

`simulate_substitute_pair` couples two series: cross-price demand
(`(p_rival/p0)^cross_elasticity`) plus same-day stockout spill
(`spill_rate` × rival's unmet demand). Oracle columns `own_demand` and
`spill_in` decompose observed demand. `InventoryConfig.stockout_behavior`
now supports `"backorder"` (unmet demand queues; sales shift rather than
vanish); `"substitute"` is handled at the portfolio level by design.

## 4. Interference (`interference.py`) — the Phase 3 payoff

Sessions want to purchase per the demand model; each (market, window) pool
fulfils at most `capacity` purchases (rationing at random). Capacity is a
**fixed physical resource** — `pool_capacity` computes it once from the
reference world and every counterfactual shares it. The benchmark is the
global-rollout contrast under the same mechanics, and the design-bias table
delivers the marketplace SUTVA story:

```
randomization_unit   true_diff   mean_estimate     bias
session                -0.0305         -0.0642  -0.0337   <- ~2x overstated
user                   -0.0305         -0.0619  -0.0315   <- ~2x overstated
switchback_window      -0.0305         -0.0314  -0.0010   <- clean
market                 -0.0305         -0.0292   0.0013   <- clean
```

Split designs promise more than a rollout delivers; switchback and geo
finally earn the variance cost Phase 3 documented.
`guardrails.interference_probe` (pure-control markets) detects the leakage,
with a signed diagnosis: a higher-priced treatment *frees* capacity, so
in-experiment control converts better than pure control.

## 5. Known gaps (beyond the spec's four phases)

- DML-PLR/PLIV scorecard rows (optional heavy deps: doubleml, lightgbm)
- CUPED / sequential-testing bounds (docs/04 §3.4-3.5)
- Endogenous pricing wired into the session/panel generators end-to-end
  (the DGP is standalone, matching docs/05's demonstration design)
- K>2 product substitution networks; backorder priority queues

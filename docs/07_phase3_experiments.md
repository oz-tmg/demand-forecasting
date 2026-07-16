# Phase 3 — Experiment Module: Designs, Guardrails, Power

**Status:** implemented (v0.3.0)
**Modules:** `experiments.py`, `guardrails.py`, `power.py`, `rd.py` · **Demo:** `examples/phase3_experiments.py`
**Spec:** docs/00_spec.md §2.2 (`fact_experiment`), §3.4 · **Design doc:** docs/04

Phase 3 ships what docs/04 §5 promised: assignment engines for all four
randomization units, automatic post-run guardrails, analytic + Monte Carlo power
tooling, surge-RD, and a design × estimator scorecard graded against oracle truth.

## 1. Assignment engines (`experiments.py`)

`ExperimentConfig(randomization_unit=...)` supports:

| unit | style | analysis unit | docs/04 |
|---|---|---|---|
| `user` | ZipRecruiter | consumer (user-level means) | §1.1 |
| `session` | Uber-adjacent | offer | §1.2 |
| `switchback_window` | Lyft/DoorDash | window (schedule generator + burn-in trim) | §1.3 |
| `market` | geo/cluster | market | §1.4 |

`run_experiment` returns observable/oracle session views plus `fact_experiment`
(one row per randomized unit, spec §2.2). Outcomes come from the Phase 1
ground-truth demand model, so every estimate has a known truth
(`true_contrast`). Sessions carry `segment_proxy` — the noisy observable
segment signal promised in spec §3.5 — and `market_id`.

`estimate_contrast` aggregates to the randomization unit before inference:
the same sessions yield ~42k units under session-split, ~26k users, ~170
windows, or 20 markets. That collapse in effective n **is** the cost of the
clustered designs.

## 2. Guardrails (`guardrails.py`)

Per docs/04 §3, run automatically via `run_guardrails`:

- `srm_check` — chi-square on realized vs planned unit counts; alarm at p < 0.001
- `covariate_balance` — SMDs on pre-assignment covariates (default `segment_proxy`); flag |SMD| > 0.1
- `dual_exposure` — users seen in >1 arm: structurally ~25% under session-split, must be 0 under user-split
- `aa_battery` — identical-arm replications; rejection rate must be ≈ α or the variance estimator is wrong for the design

## 3. Power (`power.py`)

`metrics.power_two_proportions` (Phase 1) is the analytic answer for iid
session-split; `monte_carlo_power` is the ground truth for every design — run
the full pipeline R times, count rejections, and also report bias, RMSE, and
CI coverage. `experiment_scorecard` sweeps all four designs at a fixed session
budget. Tests verify MC power matches the analytic formula for session-split.

## 4. Surge-RD (`rd.py`)

`run_surge_sessions` prices sessions with the `pricing.surge_price` rounding
rule (continuous latent multiplier → discrete surge levels). `rd_estimate`
fits a local-linear RD at a rounding cutpoint (HC1 SEs); `true_rd_jump` is the
oracle discontinuity. Reproduces the Cohen et al. (2016) identification idea
with a checkable answer key.

## 5. Sample output (`examples/phase3_experiments.py`)

```
randomization_unit  n_units  true_diff    bias   rmse  power  coverage  dual_exposed
              user    26511    -0.0506  0.0009 0.0056    1.0      0.90         0.000
           session    42000    -0.0506  0.0004 0.0046    1.0      0.97         0.246
 switchback_window      168    -0.0506 -0.0002 0.0049    1.0      0.87         0.240
            market       20    -0.0506  0.0001 0.0053    1.0      0.97         0.000

A/A battery: rejection_rate 0.075 (nominal 0.05) — pass
Surge-RD: jump -0.0169 (CI -0.0314, -0.0025) vs truth -0.0224 — covered
```

All designs are unbiased here **by construction**: consumers are exchangeable
across windows and markets, and there is no interference. Phase 4 adds the
pathologies (shared capacity, demand shifting, market heterogeneity) that make
the clustered designs earn their variance cost — and the guardrails detect them.

## 6. Known gaps vs spec (deferred)

- Interference / shared-capacity coupling between arms (spec §3.4, Phase 4) —
  without it, switchback and geo designs show no bias advantage
- Strategic-waiting detector (docs/04 §3.4) — needs Phase 4 demand shifting to
  generate the signature
- CUPED / pre-period adjustment and sequential-testing bounds (docs/04 §3.4–3.5)
- Market-level heterogeneity (markets are currently exchangeable, so geo SEs
  are optimistic relative to real geo experiments)

# demand-sim

A ground-truth **demand curve simulator** and **ideal-dataset generator** for demand
forecasting research. It simulates pricing experiments that trace out demand curves
(Uber/ZipRecruiter style), emits forecasting-ready panels (Amazon/Chronos style), and —
because it knows the truth it generated — **grades elasticity estimators on bias, RMSE, and
CI coverage**, something no real dataset can do.

Built as a study companion to the demand-estimation literature and the class of problems
Amazon's Demand Forecasting Science team works on (foundation time-series models, censored
demand, experiment-driven elasticity).

## Why a simulator?

Every method for learning a demand curve rests on assumptions you cannot verify in the field:
randomization integrity, no interference, unconfoundedness, instrument validity. A simulator
inverts the problem — *you* set the truth, then measure exactly how each method fails when its
assumptions break. The value is threefold:

1. **Methods testbed** — score RCT cell means, RD, DML, IV, and structural estimators against
   known elasticities, per population segment.
2. **Experiment design lab** — pick randomization units, window lengths, and sample sizes by
   Monte Carlo power before spending real traffic.
3. **Ideal forecasting dataset** — panels with known-future covariates (price, promo,
   holidays), controllable censoring (stockouts), and *uncensored true demand* as an oracle
   target for evaluating forecasting models honestly.

## Project layout

```
demand-sim/
├── README.md
├── pyproject.toml
├── docs/
│   ├── 00_spec.md                  # schema + full parameter registry (research-sourced)
│   ├── 01_research_foundations.md  # Uber RD, ZipRecruiter RCT, DML, structural, Amazon forecasting
│   ├── 02_censored_demand.md       # why inventory/stockouts are first-class
│   ├── 03_models_python_r.md       # every estimator, in Python AND R, with parity notes
│   ├── 04_experimentation.md       # randomization units, sample sizes, contamination guardrails
│   └── 05_endogenous_pricing.md    # OLS bias mechanics + IV/CF/DML-IV/RD fixes, both languages
├── src/demand_sim/
│   ├── config.py      # SegmentConfig / SimulationConfig (+ Phase 2-4 blocks)
│   ├── population.py  # heterogeneous consumer pool
│   ├── demand.py      # ORACLE: choice probs, true curves, elasticities, surplus
│   ├── pricing.py     # fixed / randomized cells (+ surge-RD stub)
│   ├── simulate.py    # session generator → observable + oracle views
│   └── metrics.py     # estimators + scoring vs truth, power calculator
├── examples/quickstart.py
└── tests/test_phase1.py            # acceptance criteria from the spec
```

## Quickstart

```bash
pip install numpy pandas scipy statsmodels
python examples/quickstart.py
python -m pytest tests/ -q
```

Sample output (default 3-segment market, 100k sessions, 7 randomized price cells):

```
=== Empirical demand curve vs truth ===
 quoted_price      n   conv   P_aggregate  covered
          9.0  14363  0.6235       0.6265     True
         19.0  14255  0.4614       0.4626     True
         ...
         99.0  14139  0.1012       0.0989     True

=== Arc elasticity vs truth ===
 price_mid  arc_elasticity  true_elasticity   bias
      14.0          -0.419           -0.430  0.012
      59.0          -1.050           -1.063  0.013

=== Pooled logit recovery ===
 b_hat = -0.0325 vs share-weighted truth -0.0693   # expected aggregation bias
                                                   # under segment mixtures — surfaced, not hidden

Consumer surplus per session at p=$29: $23.13      # the Uber integral
n/arm to detect 10.0% vs 8.5% conversion: 5,847    # the docs/04 worked example
```

Note what the run demonstrates: randomized cells recover the curve almost perfectly, while a
pooled (misspecified) logit is 53% off the weighted truth — heterogeneity is not a nuisance,
it is the phenomenon.

## Core concepts

- **Observable vs. oracle views.** Estimators only see what a real firm logs
  (`out["sessions"]`); truth (`sessions_oracle`, `wtp_true`, `purchase_prob_true`,
  `true_elasticity()`) exists solely for scoring.
- **Sessions first.** The unit of record is the *offer* (price shown + buy/no-buy), the
  crucial ingredient from Uber and ZipRecruiter that sales-only data lacks. Daily sales
  panels are aggregations of sessions.
- **Heterogeneity by construction.** The population is a finite mixture of segments with
  their own WTP distributions / price coefficients; segment-level and aggregate demand
  curves and elasticities come out of the same engine.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Static demand engine, randomized price cells, estimator scoring | ✅ this repo |
| 2 | NHPP arrivals, trend/seasonality/holidays/promos, inventory & censoring, daily panel + GluonTS export | spec'd (docs/00 §3.2–3.3, docs/02) |
| 3 | Experiment module: user/session/switchback/geo assignment, surge-RD, SRM & contamination guardrails, Monte Carlo power | spec'd (docs/04) |
| 4 | Endogenous pricing, substitution, reference prices, demand shifting, interference | spec'd (docs/05) |

## Key references

- Cohen, Hahn, Hall, Levitt & Metcalfe (2016), *Using Big Data to Estimate Consumer Surplus:
  The Case of Uber*, NBER WP 22627.
- Dubé & Misra (2023), *Personalized Pricing and Consumer Welfare*, JPE 131(1)
  (earlier: *Scalable Price Targeting*, NBER WP 23775).
- Chernozhukov et al. (2018), *Double/Debiased Machine Learning*, Econometrics Journal.
- Berry, Levinsohn & Pakes (1995), *Automobile Prices in Market Equilibrium*, Econometrica;
  Conlon & Gortmaker, *PyBLP*.
- Bojinov, Simchi-Levi & Zhao, *Design and Analysis of Switchback Experiments*; Lyft &
  DoorDash engineering posts on marketplace interference.
- Salinas et al. (2020), *DeepAR*; Amazon Science: MQTransformer, Chronos / Chronos-2.

# CLAUDE.md — Demand Curve Simulator (`demand-sim`)

_Last updated: 2026-07-21_

> AI operating rules for the **demand-sim** project. Project specifics — status,
> decisions, open questions, roadmap — live in `PROJECT_CONTEXT.md`. Research
> detail lives in `docs/`.

## Inheritance

This file extends `verticals/economics/CLAUDE.md`, which extends `core/CLAUDE.md`.
All global rules and all Economics-vertical econometric standards apply (real vs.
nominal discipline, standard errors + significance, stationarity checks, DiD/IV
causal hygiene, distinguishing correlation from causation). Rules here take
precedence where they conflict, and adapt those standards to a *simulated* setting
where the causal answer is known.

---

## Project Identity

**Project:** Demand Curve Simulator & Ideal Forecasting Dataset Generator (`demand-sim`)
**Vertical:** Economics → demand forecasting
**Purpose:** simulate pricing experiments that trace demand curves, emit an ideal
(oracle-backed) forecasting dataset, and grade elasticity/forecasting estimators
against known truth (bias / RMSE / CI coverage).
**Portfolio goal:** demonstrate identification and causal-inference fluency on the
exact problem class of Amazon-style demand science — heterogeneity, censored
demand, experiment-driven elasticity.
**Audience:** econometrics/DS reviewers, demand-science hiring managers, the
analytics community.

Act like a sharp applied econometrician already briefed on the project. Do not make
Alex restate context captured here, in `PROJECT_CONTEXT.md`, or in `docs/`.

---

## What this project is (and is not)

It **is** a synthetic, ground-truth simulator whose reason to exist is that it knows
the truth it generated and can therefore score methods honestly.

It **is not** a client engagement, a production service, or a claim that any
simulated elasticity describes a real market. If a task starts to look like "stand
up a real pipeline" or "productionize this," it does not belong here — note it in
`PROJECT_CONTEXT.md` and keep the project a methods/dataset lab.

---

## Standing Project Rules

These are the load-bearing invariants. Do not violate them without an explicit,
logged decision.

- **Ground truth is generative; estimation is downstream.** Demand comes from the
  random-utility population model. Every estimator (cell means, RD, DML, IV, logit
  MLE, structural) is scored against parameters we set — never treated as the truth.
- **Observable vs. oracle split is sacred.** Every table has two views. Estimators
  and forecasters may only read the observable view (`out["sessions"]`,
  `fact_sales_daily.units_sold`, known-future covariates). Truth (`sessions_oracle`,
  `wtp_true`, `purchase_prob_true`, `segment_id`, `units_demanded`, `lost_sales`,
  `true_elasticity()`, `ground_truth_curves`) is scoring-only. Never leak the oracle
  into an estimator's feature set.
- **Sessions are the unit of record.** The offer (price shown + buy/no-buy,
  *including non-buyers*) is primary; daily panels are aggregations of sessions,
  never the reverse.
- **Censoring is first-class.** `units_sold = min(units_demanded, on_hand)`. Keep
  `units_demanded` and `lost_sales` in the oracle. Never conflate sales with demand;
  when evaluating a forecaster, grade against uncensored true demand.
- **Heterogeneity is the phenomenon.** The population is a finite mixture of
  segments; aggregate demand is the share-weighted mixture and is deliberately *not*
  constant-elasticity. Surface aggregation bias in pooled estimators — do not hide
  or "fix" it.
- **Every mechanism has a strength dial and a known truth.** Endogeneity (`gamma`),
  instrument strength (`pi`), reference-price sensitivity, substitution, interference
  — each should let bias be plotted as a *function of* the mechanism strength.
- **Deterministic under a fixed seed.** Generation and analysis reproduce from a
  seed (default 42). No un-seeded randomness in load-bearing paths.

---

## Econometric Rules (adapting the vertical standards to a known-truth setting)

- Always report the **estimate against the truth**: bias, RMSE, and CI coverage, not
  just a point estimate. A coefficient without its error relative to ground truth is
  not a result here.
- State the **identification strategy** for every estimate — what makes price
  as-good-as-random (randomized cell, RD threshold, instrument, DML orthogonalization)
  and which assumption the simulator is currently satisfying or breaking.
- When showing OLS bias, show the **fix beside it** (IV/2SLS with first-stage F and a
  weak-instrument flag, control function, or RD) and the mechanism that causes the bias.
- **Cluster standard errors at the randomization unit** (user, session, window,
  market); one user's many sessions are not independent. Aggregate to the assignment
  unit before inference — the effective-n collapse *is* the cost of clustered designs.
- Run **guardrails** on any experiment output: SRM, covariate balance (|SMD| > 0.1),
  dual-exposure, and an A/A battery whose rejection rate must sit near α.
- Distinguish **statistical from practical** magnitude, and LATE (RD, valid only at
  thresholds) from ATE.

---

## Tools and Stack

- **Python primary** (`numpy`, `pandas`, `scipy`, `statsmodels`; `pyfixest`,
  `linearmodels`, `DoubleML`/`EconML`, `rdrobust`, `PyBLP`, GluonTS/Chronos/
  `statsforecast` where a phase reaches them). R only as reference snippets for
  parity (see `docs/03`); two pillars — BLP and deep forecasting — are Python-only.
- Keep the package structure in `src/demand_sim/` (config, population, demand,
  pricing, simulate, metrics, arrivals, inventory, panel, experiments, guardrails,
  power, rd, endogenous, behavior, substitution, interference). Estimators live in
  `metrics.py` and the phase modules; the oracle lives in `demand.py`.
- Examples in `examples/`, dataset CLI in `scripts/`, acceptance tests in `tests/`.
- Environment: current quickstart is `pip install`. Adopting the repo `uv` +
  `direnv` convention is an open question in `PROJECT_CONTEXT.md` — do not switch
  unilaterally.

---

## Code and Analysis Standards

- `lower_snake_case` for Python files, functions, and data columns.
- Type hints on public and non-trivial functions; Pydantic (or the existing
  `SegmentConfig` / `SimulationConfig` dataclasses) for structured config.
- Small, focused, mostly-pure transformation functions; keep the oracle strictly
  separated from the observable-view construction.
- Every new mechanism ships with (a) a config knob, (b) an oracle field if it
  creates truth, and (c) a test encoding its acceptance criterion from `docs/00_spec.md`.
- Comment the *why* (the assumption, the identification argument, the caveat), not
  the obvious *what*. Cite the source doc (`docs/NN`) for non-obvious modelling choices.
- Never hardcode secrets; there should be none — this project uses no real data.

---

## Vertical-Specific AI Behaviour Rules

### Always Do
- Start from this file, `PROJECT_CONTEXT.md`, and the relevant `docs/` doc.
- Keep the observable/oracle boundary intact in any new code or analysis.
- Grade estimators against the oracle and report bias/RMSE/coverage.
- Name the identification strategy and the assumption in play.
- Keep generation seed-reproducible; add or update a test for new behaviour.
- Make a recommendation when enough information exists; state the tradeoff, then pick.
- Flag a genuinely reusable asset as a possible generalization — separately, not silently.

### Never Do
- Never feed oracle fields (segment_id, wtp_true, units_demanded, true elasticity)
  into an estimator or forecaster.
- Never treat sales as demand, or a pooled estimate as the truth.
- Never present a demand estimate without its error relative to ground truth.
- Never claim a simulated elasticity describes a real market.
- Never introduce un-seeded randomness in a load-bearing path.
- Never let the project drift into a production pipeline or hosted service.
- Do not over-explain textbook econometrics or pad with generic disclaimers.

---

## Cross-Vertical Notes

- **Time-series decomposition, censoring/survival, and experiment design** developed
  here are shared methodology (see `core/analytical-methodology.md`): the censoring
  and cohort-forecasting concerns overlap directly with Game Analytics
  (`cohort-ltv-forecasting`), and the panel/FE and IV machinery is the same as the
  rest of the Economics vertical.
- The estimator-scoring harness (`metrics.py` + oracle) is the most likely candidate
  to generalize beyond this project — treat any such graduation as a deliberate decision.

---

## Default Response Style

Direct answer first, then the reasoning (identification + bias mechanics), then
concrete next steps. Short sections; tables only when they improve clarity. Strong
opinions, weakly held. Avoid long preambles, generic disclaimers, restating the
question, or listing twenty options when three suffice.

## How to Update This File

Update when a standing rule for this project changes, or when a repeated correction
from Alex should become a standing instruction. Keep project history, decisions, and
tasks in `PROJECT_CONTEXT.md`, not here.

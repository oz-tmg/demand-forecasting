# Research Foundations — How Demand Curves Have Been Estimated

This doc summarizes the five method families the simulator is built to replicate and stress-test.
Each section ends with **"What the simulator borrows."**

---

## 1. Uber — Natural experiment + regression discontinuity

**Paper:** Cohen, Hahn, Hall, Levitt & Metcalfe (2016), *Using Big Data to Estimate Consumer
Surplus: The Case of Uber*, NBER WP 22627.

**Setting.** ~50M UberX *sessions* (app opens with a quoted price) across Chicago, LA, NYC, SF
in 2015. ~21% of sessions had surge > 1.0x.

**Method.** The surge algorithm produces a continuous latent multiplier that is *rounded* to
discrete surge levels (1.2x, 1.3x, …). Two sessions with nearly identical latent values can
land on opposite sides of a rounding threshold and receive different prices — locally, price
assignment is as-good-as-random. A regression discontinuity design compares purchase
(ride-request) rates just left and right of each threshold, yielding a *local* elasticity at
each surge level. Stitching local elasticities across the surge distribution traces the demand
curve; integrating under it gives consumer surplus (~$2.9B in the four cities; ~$1.60 of
surplus per $1 spent).

**Data requirements.**
- Offer-level logs: quoted price + accept/decline for *every* session, including non-buyers.
- The raw (pre-rounding) assignment variable, to define distance-to-threshold.
- Enough traffic near each threshold (RD is data-hungry — this is why 50M sessions).

**Assumptions.**
- Continuity: potential outcomes are smooth through the threshold; only price jumps.
- No precise manipulation of the running variable by users (riders can't game rounding).
- No sorting: session composition doesn't change discontinuously at thresholds.

**Limitations.**
- Estimates are LATE — valid *at the thresholds*, extrapolation between them is assumption-laden.
- Short-run elasticity only; ignores intertemporal substitution (users waiting out the surge
  and re-opening the app later — which also threatens independence across sessions).
- Demand found to be surprisingly inelastic; generalizes poorly to markets with closer substitutes.

**What the simulator borrows.** Phase 3 implements a surge-like pricing rule (latent
multiplier + rounding) so RD estimators (`rdrobust`) can be tested against known truth; session
schema stores the latent multiplier in the oracle view.

---

## 2. ZipRecruiter — Randomized pricing (the gold standard)

**Paper:** Dubé & Misra (2023), *Personalized Pricing and Consumer Welfare*, JPE 131(1).
(Earlier version: *Scalable Price Targeting*, NBER WP 23775.)

**Setting.** New-customer signups at ZipRecruiter, standard price $99/month.

**Method.** Stage 1 (Sept 2015): new customers randomly assigned one of ~10 monthly price
cells from $19 to $399. Conversion rate per cell *is* the demand curve — no identification
assumptions beyond randomization. Findings: cleanly downward-sloping demand; far less elastic
than assumed ($99→$199 cut conversions only ~25%; the firm later moved to $249). Stage 2:
demand model with heterogeneous price effects trained on registration covariates
(industry, size, location) via regularized ML (lasso-family, weighted-likelihood bootstrap);
personalized prices predicted to raise profit 86% vs. status quo and 19% vs. the optimal
uniform price. A **second experiment validated the counterfactual predictions out of sample** —
predicted conversion/profit distributions closely matched realized ones.

**Data requirements.** Ability to randomize price at the unit level; covariates collected
*pre-assignment*; conversion outcome; enough units per cell.

**Assumptions.** Randomization integrity (no SRM); no interference across units (plausible
here — job-posting firms don't share a capacity pool); stable population during the test.

**Limitations.**
- New customers only; long-run effects (churn, LTV, reputation/fairness blowback) not captured.
- One-shot subscription purchase; harder in repeat-purchase retail where reference prices form.
- Legal/ethical constraints often cap price randomization ranges in practice.

**What the simulator borrows.** Phase 1's default mechanism is exactly this: session- or
user-level random price cells. The two-stage pattern (estimate → predict counterfactual →
*validate against a fresh experiment*) is the simulator's core evaluation loop, since we can
always run the "validation experiment" against ground truth.

---

## 3. Observational causal ML — Double Machine Learning (DML)

**Canon:** Chernozhukov et al. (2018), *Double/Debiased Machine Learning*; software: `DoubleML`
(Python + R), `EconML` (Python).

**Method.** Partially linear model `log Q = θ·log P + g(X) + u`, `log P = m(X) + v`. Fit
flexible ML (boosting/forests) for both nuisance functions with **cross-fitting**, then regress
outcome residuals on treatment residuals — Frisch–Waugh–Lovell with ML. θ is the elasticity;
Neyman-orthogonality makes it robust to slow ML convergence. Extends to heterogeneous
elasticity θ(X) via CATE learners (causal forests, DR-learner, `EconML`'s `LinearDML`).

**Data requirements.** Observational panel with rich confounders X (seasonality, promo,
product attributes, competitor context); price variation not fully explained by X.

**Assumptions.**
- **Unconfoundedness**: all common causes of price and quantity are in X. If price responds to
  *unobserved* demand shocks (the normal case for a profit-maximizing pricer), DML alone is
  biased — you need instruments (DML-IV) or the experiment. See `docs/05_endogenous_pricing.md`.
- Overlap: prices vary within every X stratum.
- Cross-fitting to kill regularization/overfitting bias.

**Limitations.** Zero-sales censoring (sales-only logs can't show what happened on no-sale
days); functional-form choice for the target (log-log imposes constant elasticity); inference
on CATEs is delicate.

**What the simulator borrows.** The oracle lets us generate data where unconfoundedness holds
(Phase 2) or fails (Phase 4 endogenous pricing) and measure exactly how wrong DML gets — the
kind of estimator due-diligence you can never do on real data.

---

## 4. Traditional structural models — BLP / random-coefficients logit & conjoint

**Canon:** Berry (1994); Berry, Levinsohn & Pakes (1995); Nevo (2000, 2001); software: `PyBLP`.

**Method.** Consumers have random utility `u_ij = x_j β_i − α_i p_j + ξ_j + ε_ij` with
*random coefficients* (β_i, α_i vary across consumers) — heterogeneity is the point. Market
shares are the integral of logit choice probabilities over the coefficient distribution.
Estimation inverts observed aggregate shares to recover mean utilities, then uses GMM with
**price instruments** to deal with the correlation between price and the unobserved product
quality ξ_j (the structural version of endogeneity). Output: full demand system with own- and
cross-price elasticities, welfare, merger/counterfactual simulation.

**Data requirements.** Aggregate market shares by product × market, prices, product
characteristics, and instruments (cost shifters, BLP instruments = rivals' characteristics,
Hausman instruments = same product's price in other markets).

**Assumptions.** Correct utility specification; instrument validity; equilibrium pricing
conduct (often Bertrand-Nash) if supply side is used.

**Limitations.** Heavy machinery; identification leans on functional form + instruments;
computationally finicky (though PyBLP largely tamed this). Conjoint analysis is the
stated-preference cousin — cheap, runs in surveys, estimated with mixed logit
(`mlogit`/R, `xlogit`/Python) — but measures stated, not revealed, WTP.

**What the simulator borrows.** The *generative* model. Our population IS a random-coefficients
discrete-choice model (finite mixture instead of continuous mixing). This means structural
estimators are, by construction, correctly specified in the base case — and we can break them
deliberately (misspecified mixing distribution, missing substitutes).

---

## 5. What Amazon is going after — forecasting at catalog scale

**Sources:** Amazon Science, "The history of Amazon's forecasting algorithm"; MQTransformer
(2020); Chronos / Chronos-2; the job posting itself.

**Trajectory.** Sparse local models → global deep models. DeepAR (2017-era): one RNN trained
across *all* series, probabilistic outputs (quantiles, not point forecasts) — the key insight
that cross-learning fixes cold-start and sparse series. MQ-CNN → MQTransformer: multi-horizon
quantile forecasts with attention, plus specialized components for seasonality and
**price-elasticity-driven demand spikes**. Chronos-2: a time-series *foundation model* —
zero-shot forecasting with support for known-future covariates (scheduled promotions, holiday
calendars, prices) and categorical covariates. The job posting adds the frontier: foundation
models that forecast products **with no sales history**, plus **synthetic data generation** to
augment training corpora.

**Data requirements (i.e., what our "ideal dataset" must contain).**
- Long panel of many related series (cross-learning is the whole trick).
- Known-future covariates: price, promo schedule + depth, holiday calendar.
- Static metadata: category, attributes (drives cold-start generalization).
- Honest demand signal or at least stock status — sales censored by OOS poison training
  (see `docs/02_censored_demand.md`).
- Probabilistic evaluation targets: quantile loss / CRPS, not just MAPE.

**Limitations of the paradigm.** Forecasters model *conditional correlation*, not causation:
a price coefficient learned from endogenous historical pricing does not answer "what if we
cut price 10%?" That question needs the experimental/causal half of this project — which is
exactly why a simulator that produces both (a forecastable panel AND a causally interpretable
experiment log) is more than the sum of its parts.

---

## The unifying table

| Family | Identification | Unit | Recovers | Fails when |
|---|---|---|---|---|
| Uber RD | algorithm discontinuity | session | local elasticities | far from thresholds; manipulation |
| ZipRecruiter RCT | randomization | user/session | whole curve on tested range | interference; range too narrow |
| DML | unconfoundedness | panel obs | θ, θ(X) | unobserved demand shocks drive price |
| BLP | instruments + structure | market shares | full demand system | bad instruments; wrong structure |
| Deep forecasting | none (predictive) | series | conditional forecasts | asked causal questions |

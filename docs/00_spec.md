# SPEC — Demand Curve Simulator & Ideal Forecasting Dataset Generator

**Status:** Phases 1–5 implemented (docs/06–09; Phase 5 = forecasting benchmark)
**Owner:** Alex
**Purpose:** A ground-truth demand simulator that (a) simulates pricing experiments capable of
tracing a demand curve, (b) emits an "ideal dataset" for demand forecasting research, and
(c) grades elasticity estimators against known truth (bias / RMSE / coverage).

---

## 1. Design principles

1. **Ground truth is generative, estimation is downstream.** Demand comes from a random-utility
   population model (the same object BLP-style structural models *assume*). Every estimator we
   test — cell means, RD, DML, logit MLE — is scored against parameters we set.
2. **Observable vs. oracle split.** Every output table exists in two views: the *observable*
   view (what a real firm logs) and the *oracle* view (true WTP, true demand, censoring amount).
   Estimators only ever see the observable view.
3. **Sessions, not just sales.** The single biggest lesson from Uber (Cohen et al. 2016) and
   ZipRecruiter (Dubé & Misra 2023): you need the *offer* level — quoted price + buy/no-buy —
   including the non-buyers. Aggregated sales panels are derived from sessions, never the
   other way around.
4. **Censoring is first-class.** Sales ≠ demand once inventory exists. The simulator always
   knows true demand; the observable view only shows sales (see `docs/02_censored_demand.md`).

---

## 2. Data schema

### 2.1 Dimension tables

**`dim_segment`** — the heterogeneity backbone (Dubé & Misra found WTP heterogeneity large
enough that personalized pricing beat optimal uniform pricing by 19%).

| column | type | notes |
|---|---|---|
| segment_id | str | PK |
| name | str | e.g. "price_hawks", "convenience", "whales" |
| share | float | population share, sums to 1 |
| wtp_mu, wtp_sigma | float | lognormal WTP params (threshold model) |
| base_util, price_coef | float | logit utility params (logit model) |
| promo_uplift | float | additive utility bump when promo shown |
| ref_price_sensitivity | float | Phase 4: loss aversion around reference price |

**`dim_product`** (Phase 2+)

| column | type | notes |
|---|---|---|
| product_id | str | PK |
| category | str | drives substitution sets |
| base_price | float | anchor price |
| launch_date, eol_date | date | lifecycle window |
| lifecycle_shape | str | "bass", "flat", "decay" |
| substitutes | list[str] | cross-elasticity links (Phase 4) |

### 2.2 Fact tables

**`fact_session`** — one row per shopping session (the Uber-style unit of analysis).

| column | type | view | notes |
|---|---|---|---|
| session_id | int | obs | PK |
| ts | datetime | obs | Phase 1: synthetic index; Phase 2: NHPP arrival |
| consumer_id | int | obs | stable across sessions → enables per-user randomization |
| segment_id | str | **oracle** | hidden from estimators; obs view may include noisy proxies |
| product_id | str | obs | Phase 1: single product |
| quoted_price | float | obs | price shown this session |
| price_mechanism | str | obs | "fixed" / "random_cell" / "surge" / "endogenous" |
| arm / cell | str | obs | experiment assignment label |
| surge_multiplier | float | obs | Phase 3: raw multiplier pre-rounding lives in oracle |
| promo_flag | bool | obs | |
| in_stock | bool | obs | Phase 2+; False ⇒ purchase impossible (censoring) |
| wtp_true | float | **oracle** | this consumer's realized WTP / utility draw |
| purchase_prob_true | float | **oracle** | model-implied P(buy) at quoted price |
| purchased | bool | obs | Bernoulli outcome |

**`fact_sales_daily`** (Phase 2+) — the forecasting dataset, Chronos/DeepAR-ready.

| column | type | view | notes |
|---|---|---|---|
| date, product_id | | obs | PK |
| units_sold | int | obs | censored by stock |
| units_demanded | int | **oracle** | true demand |
| lost_sales | int | **oracle** | demanded − sold |
| avg_price, promo_flag, promo_depth | | obs | known-future covariates |
| holiday, dow, week_of_year | | obs | calendar covariates |
| in_stock_hours | float | obs | exposure offset for censoring corrections |
| inventory_start, replenishment | | obs | |

**`fact_experiment`** (Phase 3)

| column | notes |
|---|---|
| experiment_id, arm | |
| randomization_unit | "user" / "session" / "switchback_window" / "market" |
| unit_id | consumer_id, session_id, or window_id |
| assigned_at, window_start, window_end | |
| price_multiplier or price_cell | |

**`ground_truth_curves`** — per segment and aggregate: price grid, D(p), elasticity(p),
consumer surplus. This is the answer key every estimator run is graded against.

---

## 3. Parameter registry

Everything the literature identified as a demand driver. "Source" = where the evidence comes
from; "Phase" = when the simulator implements it.

### 3.1 Price & pricing mechanism

| parameter | meaning | source | phase |
|---|---|---|---|
| price grid / cells | randomized price levels ($19–$399 style) | ZipRecruiter RCT | 1 |
| price_coef (per segment) | logit price sensitivity β_p | Dubé & Misra heterogeneous WTP | 1 |
| wtp_mu, wtp_sigma (per segment) | WTP distribution | same | 1 |
| surge thresholds & rounding | algorithmic discontinuities enabling RD | Uber (Cohen et al. 2016) | 3 |
| endogenous pricing rule | price responds to demand shocks → OLS bias | econometrics canon; docs/05 | 4 |
| reference price / loss aversion | demand dips harder above anchored price | pricing literature | 4 |

### 3.2 Time-series structure (the forecasting covariates Amazon's models consume)

| parameter | meaning | source | phase |
|---|---|---|---|
| base_arrival_rate λ | sessions/day (NHPP intensity) | — | 2 |
| trend | multiplicative growth/decay | M5, Amazon | 2 |
| weekly seasonality | dow multipliers | Chronos-2 covariates | 2 |
| annual seasonality | Fourier terms | Amazon seasonality add-on (MQ line) | 2 |
| holiday calendar + uplift | categorical covariates (Prime-Day-like events) | Chronos-2 | 2 |
| promo schedule, promo_depth, promo_uplift | known-future covariates; spike driver | Amazon elasticity-spike component | 2 (utility hook in 1) |
| lifecycle (launch/decay, cold-start) | new products with no history | Amazon job posting (cold-start focus) | 2 |
| demand shifting | consumers defer purchase when price high | Uber limitation (surge waiting) | 4 |

### 3.3 Supply / censoring

| parameter | meaning | source | phase |
|---|---|---|---|
| initial_inventory, reorder policy (s, S), lead_time | stockout process | Amazon OOS problem | 2 |
| stockout behavior | lost sale vs. substitute vs. backorder | censoring literature | 2/4 |
| oos_price_correlation | stockouts co-move with promos/price → biases naive elasticity | docs/02 | 2 |

### 3.4 Experimentation

| parameter | meaning | source | phase |
|---|---|---|---|
| randomization_unit | user / session / switchback window | Lyft, DoorDash | 3 |
| n_arms, allocation | price cells & weights | ZipRecruiter (10 cells) | 1 (basic), 3 |
| switchback window length | variance/carryover tradeoff | Bojinov et al. | 3 |
| interference strength | shared-resource coupling between arms | marketplace SUTVA literature | 4 |

### 3.5 Population

| parameter | meaning | source | phase |
|---|---|---|---|
| n_consumers, segment shares | mixture weights | — | 1 |
| choice_model | "wtp_threshold" or "logit" | structural canon | 1 |
| observable segment proxies + noise | features estimators may use for HTE | Dubé & Misra lasso targeting | 3 |

---

## 4. Ground-truth math (Phase 1)

**Logit model.** Consumer in segment *s* buys iff
`U = base_util_s + price_coef_s · p + ε > 0`, ε ~ Logistic(0,1), so
`P_s(p) = σ(base_util_s + price_coef_s · p)`.
Segment point elasticity: `ε_s(p) = price_coef_s · p · (1 − P_s(p))`.

**WTP-threshold model.** `P_s(p) = 1 − F_s(p)` with F lognormal(wtp_mu, wtp_sigma).

**Aggregate demand.** `D(p) = N · Σ_s share_s · P_s(p)`; aggregate elasticity is the
share-of-demand-weighted mixture — heterogeneity falls out for free, and the aggregate curve
is *not* constant-elasticity even when segments are simple. That is intentional: it lets us
test estimator misspecification.

**Consumer surplus.** `CS(p*) = N · Σ_s share_s · ∫_{p*}^{∞} P_s(p) dp` (numerical), enabling
Uber-style surplus replication.

---

## 5. Acceptance criteria (Phase 1)

1. Demand curve monotonically downward-sloping for all valid configs.
2. With session-level price randomization and N ≥ 100k sessions, cell-mean conversion rates
   recover true P(p) within CI coverage ≈ 95%.
3. Logit MLE on observable sessions recovers pooled price_coef within ±5% at N = 250k
   (single segment) and exhibits the *expected* aggregation bias under mixtures (documented,
   not hidden).
4. Arc-elasticity between adjacent cells matches true elasticity at cell midpoints within
   estimator noise; scoring report auto-generated by `metrics.py`.
5. Deterministic under fixed seed.

## 6. Roadmap

- **Phase 1 (this repo):** static demand engine, randomized price cells, estimator scoring.
- **Phase 2:** arrivals, seasonality/trend/holidays/promos, inventory & censoring, daily panel export.
- **Phase 3:** experiment module (user-split, session-split, surge-RD, switchback), power tooling, SRM/contamination guardrails.
- **Phase 4:** endogenous pricing, substitution, reference prices, demand shifting, interference.

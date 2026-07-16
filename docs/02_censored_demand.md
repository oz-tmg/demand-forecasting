# Why Inventory Belongs in the Simulator — Stockouts Censor Sales Below True Demand

**Claim:** adding inventory so stockouts censor sales is the single most important realism
feature in this simulator, and a core problem in Amazon's demand forecasting work. This doc
makes the case and specifies the implementation.

---

## 1. The core problem: sales are not demand

A forecaster is asked to predict **demand** — what customers *wanted* to buy — but is trained
on **sales** — what customers *were able* to buy. Whenever an item is out of stock (OOS),
observed sales are right-censored at available inventory:

```
units_sold(t) = min(units_demanded(t), inventory_available(t))
```

Every hour of stockout deletes demand signal from the training data. A model that treats sales
as demand will systematically under-forecast exactly the products that need forecasts most:
fast movers that keep selling out.

Amazon's own job descriptions for this team name the problem directly — scientists must
"model variations related with demand prediction, **out of stock**, seasonality, and different
lead times" — because at 190M+ products, some fraction of the catalog is always censored.

## 2. Three failure modes censoring creates

### 2.1 The doom loop (self-fulfilling under-forecast)
1. Item stocks out → sales truncated below demand.
2. Forecast trained on truncated sales → forecast too low.
3. Buying system orders to the low forecast → less inventory.
4. Item stocks out sooner → even more truncation.

The feedback loop compounds: forecast error becomes *policy*, and the system converges to
chronic under-stocking of best-sellers. Without an oracle you can't even measure how much
demand you're losing — which is precisely why a simulator with `units_demanded` in the ground
truth is the right tool for studying it.

### 2.2 Elasticity bias (censoring is not random)
Stockouts are *correlated with the covariates we care about*: they cluster during promotions,
price cuts, and holiday peaks — the exact moments of highest demand. So censoring shaves the
top off demand precisely where price is low, flattening the observed price–quantity
relationship and biasing elasticity estimates **toward zero**. Any experiment or observational
study that ignores stock status inherits this bias: a price cut that stocks out looks like a
price cut that "didn't do much."

### 2.3 Forecast evaluation corruption
If test-set targets are censored sales, a model that correctly predicts *true* demand is
penalized for "over-forecasting" during OOS periods. Teams then tune models toward the
censored target, institutionalizing the bias. Proper evaluation needs either uncensored
targets (only a simulator has them) or censoring-aware metrics.

## 3. What the literature does about it (and what we can test)

| Approach | Idea | Where implemented |
|---|---|---|
| Discard censored periods | drop OOS days | trivial; wastes data, biased if OOS ⊥̸ demand |
| Exposure offset | model demand rate with `in_stock_hours` as exposure in a Poisson/NB GLM | `statsmodels` GLM offset (Py), `glm(..., offset=)` (R) |
| Tobit / censored likelihood | likelihood contribution `P(D ≥ sold)` on censored days | `statsmodels`, `pymc`; `AER::tobit`, `survival` (R) |
| EM imputation | E-step imputes latent demand on OOS days, M-step refits | custom; classic Talluri–van Ryzin unconstraining |
| Substitute-aware demand transfer | OOS demand spills to substitutes (Phase 4) | choice-model based |

The simulator's job: generate (sold, demanded, in_stock_hours) triples so each correction can
be scored on how much true demand it recovers — an eval harness for uncensoring methods.

## 4. Implementation spec (Phase 2)

**State per product-day:** `inventory_start`, `replenishment` (policy-driven), `demand_t`
(from the demand engine), `sold_t = min(demand_t, stock)`, `lost_sales_t`, `in_stock_hours`
(fractional-day stockout via within-day arrival ordering).

**Replenishment policy:** (s, S) with `lead_time` days; parameters chosen to hit a target OOS
rate (e.g., 5–15% of product-days) so censoring is material but not degenerate.

**Correlation knob:** `oos_price_correlation` — when > 0, replenishment lags during
promos/price cuts, reproducing the elasticity-bias mechanism in §2.2 on demand.

**Stockout behavior modes:** `lost` (Phase 2 default), `substitute` (Phase 4, routes demand
to linked products), `backorder` (Phase 4, shifts demand forward — interacts with demand
shifting).

**Outputs:** observable view exposes `units_sold`, `in_stock_hours`, stock flags; oracle view
adds `units_demanded`, `lost_sales`. Acceptance test: naive elasticity on censored data is
attenuated vs. truth; exposure-offset GLM closes most of the gap.

## 5. Why this matters for the Amazon-style use case

The job posting emphasizes forecasting products "that have never been sold before" and
synthetic-data augmentation. Both live or die on the demand/sales distinction: cold-start
generalization transfers *demand structure* from analogous products, and if the source
products' histories are silently censored, the transferred structure is wrong. A simulator
that controls censoring severity is the only place you can quantify that transfer error
before betting inventory dollars on it.

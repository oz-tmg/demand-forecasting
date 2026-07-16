# Phase 2 — Forecasting Dataset: Arrivals & Inventory Censoring

**Status:** implemented (v0.2.0)
**Modules:** `arrivals.py`, `inventory.py`, `panel.py` · **CLI:** `scripts/generate_phase2_dataset.py`
**Spec:** docs/00_spec.md §2.2 (`fact_sales_daily`), §3.2 (time-series structure), §3.3 (censoring); docs/02

Phase 1 was a demand-curve lab: a static price → expected-demand mapping. Phase 2 turns it
into a forecasting dataset generator by adding an arrival process and inventory censoring.

## 1. Arrival process (`arrivals.py`)

Daily demand per product-store series is Poisson with NHPP intensity

```
lambda_t = base_rate × trend × weekly × annual × (p/p0)^elasticity × promo_lift
```

The price term is the Phase 1 constant-elasticity demand curve carried over as a
multiplier; swapping in the full random-utility population model (`demand.demand_curve`)
is a one-function change. Trend is multiplicative annual growth/decay; weekly seasonality
lifts Fri–Sun and dips Mon–Tue; annual seasonality is a phase-shifted sinusoid.

## 2. Inventory censoring (`inventory.py`)

Each series runs an (s, S) reorder policy with 3–4 day lead time, reordering on inventory
*position* (on hand + on order). Observed sales are censored at on-hand stock:

```
units_sold = min(units_demanded, on_hand)
```

`units_demanded` is retained in the oracle view so forecast models can be evaluated
against uncensored ground truth — the core experimental condition this project exists to
showcase. Stockout behavior is "lost"; substitute/backorder are Phase 4.

(s, S) is sized off *peak* demand (trend + seasonality) so stockouts land in a realistic
5–12%-of-days range and concentrate around promos — reproducing the promo-driven
censoring bias documented in docs/02 (`oos_price_correlation` emerges endogenously).

## 3. Outputs (`panel.py` → `data/`)

| File | Contents |
|---|---|
| `fact_sales_daily.csv` | one row per product × store × day (oracle view — full) |
| `dim_product_store.csv` | static params per series: category, base_price, elasticity, seasonality/trend/promo, (s,S) policy |
| `promo_calendar.csv` | promo windows (start, end, discount) — known-future covariates |

### Column dictionary — `fact_sales_daily`

| column | view | notes |
|---|---|---|
| date, product_id, store_id | obs | PK |
| dow, week_of_year | obs | calendar covariates |
| avg_price | obs | promo price on promo days, else base |
| promo_flag, promo_depth | obs | known-future covariates |
| inventory_start, replenishment | obs | post-receipt on-hand; units received |
| units_sold | obs | **censored** by stock |
| sold_out | obs | ended day at zero stock (observable stockout proxy) |
| units_demanded | **oracle** | true demand |
| lost_sales | **oracle** | demanded − sold |
| stockout | **oracle** | sold < demanded |
| lambda_true | **oracle** | NHPP intensity (oracle upper bound on accuracy) |

Use `demand_sim.observable_view(panel)` to strip oracle columns before handing the panel
to any forecast model.

## 4. Dataset properties (seed 42)

- 20 SKUs × 5 stores × 731 days (2024-07-01 → 2026-07-01), 73,100 rows
- Stockout-day rate 4.2% overall (per-series 0.8–17.2%, median 3.6%)
- 9.1% of true demand censored; censoring concentrates on promo days
  (32.8% stockout rate on promo vs 0.6% off) — the classic promo-driven bias
- Weekly (Fri–Sun) and annual seasonality; per-series trend −10%…+20%/yr
- Deterministic under seed; integrity invariants enforced in `tests/test_phase2.py`

## 5. Forecasting exercises this enables

Train on `units_sold`, evaluate against `units_demanded`: quantify censoring bias, compare
naive vs stockout-aware models, test demand-unconstraining methods, use `lambda_true` as
the oracle accuracy bound.

## 6. Known gaps vs spec (deferred)

- Holiday calendar + uplift (§3.2) — not yet in the intensity function
- Session-level `fact_session` with NHPP timestamps — Phase 2 generates the daily panel
  directly from a per-series demand curve rather than aggregating sessions
- Product lifecycle (launch/decay, cold-start) and GluonTS/Chronos export helpers

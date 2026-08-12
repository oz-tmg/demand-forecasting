# Phase 6 — Recovering a Demand System *Through a Market That Clears*

Everything through Phase 5 recovers a parameter from a **posted price the studio
set**: we randomize or instrument a price we control, and grade the estimate
against the oracle. This phase asks the harder question the transfer-market note
raised (`docs/transfer_market_demand_curve.md`): what if **price is not a
treatment at all**, but an equilibrium outcome — the number at which supply and
demand happen to cross? Recovering a parameter from a posted price is a solved
exercise. Recovering one *through a market that clears* is where the interesting
failure modes live, and it is the thing almost nobody demonstrates end-to-end on
data where the truth is known.

This doc states the identification problem, the DGP, the estimator, and the
graded result. The implementation is `src/demand_sim/market.py`; the acceptance
criteria are `tests/test_phase6.py`; a leaner standalone version is
`examples/phase6_market_clearing_sketch.py`.

> **Status.** This is Stage 1 of Phase 6: a *reduced-form* clearing market
> (upward-sloping supply cleared by tatonnement) with a nested-logit demand
> system. The full order book (asks, bids, expiries, matching, carryover) and the
> two estimators it unlocks — an order-book survival read on WTP, and a bunching
> estimator at price bands — are Stage 2/3, scoped in `PROJECT_CONTEXT.md`.

---

## 1. What changes once price is an outcome

In Phases 1–5 the unit of record is the **session**: an offer at a price we set,
and a buy/no-buy. That model breaks the moment the good trades between players:

- **Price is no longer assignable.** It is what a buyer and a seller agreed on.
  You cannot randomize it — you can only randomize things that *shift* it.
- **The unit is no longer the user.** Everyone trades against the same order book,
  so randomizing users inside one market violates SUTVA badly. The unit becomes
  the **market, the time block, or the item** — never the user.
- **The estimand is a matrix, not a scalar.** With many differentiated goods, the
  own-price elasticity of one card is close to meaningless — buyers choose *among
  substitutes*. The object of interest is a **diversion matrix**: of the demand a
  good loses when its price rises, where does it go.

This is why Phase 6 is a separate phase rather than an in-place extension: it
deliberately departs from the sessions-as-unit invariant. What it keeps — and in
fact strengthens — is the observable/oracle split: grading a diversion matrix
*requires* a known ground-truth demand system.

## 2. The identification problem: simultaneity

Plotting observed price against observed quantity does **not** trace a demand
curve. It traces the sequence of equilibria where supply and demand happened to
cross. This is the oldest identification problem in econometrics (Working, 1927),
and its only solution is a **shifter** — something that moves one curve while
leaving the other alone.

- A **supply shifter traces the demand curve.** Inject stock without changing how
  much anyone wants the good; price falls, quantity rises, and the locus of those
  points is demand.
- A **demand shifter traces the supply curve.** Raise willingness-to-pay without
  changing how many exist; price and quantity both rise.

The content calendar supplies both, as a side effect of its own operation:

| Lever | Economic role | Use |
|---|---|---|
| **Pack / promo** (injects cards) | supply shifter | instrument for demand |
| **SBC / objective** (consumes a card, raises its utility) | demand shifter | observed demand covariate / traces supply |

If those events are scheduled anyway, they are natural experiments already in the
data. If *which goods they touch* is randomized, they become instruments and the
demand system is identified properly rather than argued for.

## 3. The DGP (`market.py`)

**Demand — nested logit over differentiated goods.** Goods are grouped into nests
by (quality band × role). Mean utility of good *j* is

```
delta_j = beta_x * x_j  +  sbc_boost * SBC_j  +  xi_j  -  alpha * logp_j
```

where `x_j` is an observed attribute, `SBC_j` a (randomized, observed) demand
shifter, `xi_j` the **unobserved demand shock** (the source of endogeneity), and
`alpha` the true price sensitivity. With an outside good (u₀ = 0) and nesting
parameter `lam ∈ (0,1]`, shares are the standard nested-logit form
(`nested_logit_shares`). Smaller `lam` ⇒ tighter within-nest substitution.

**Supply — upward-sloping per good.** Sellers list more when price is high:

```
log q_j^supply = base_stock_j  +  promo_boost * PROMO_j  +  phi * logp_j  +  omega_j
```

`PROMO_j` is the (randomized, observed) supply shifter; `phi` the supply
elasticity; `omega_j` supply noise.

**Clearing.** For each of many weekly blocks, `clear_market` finds the log-price
vector where nested-logit demand equals supply for every good, by tatonnement
(raise a good's price when it is in excess demand). This is the reduced-form
stand-in for the order book; it couples goods through the shared nest/outside-good
denominators, exactly the interference that makes the unit the market, not the
user. Each week yields one equilibrium `(logp_j, q_j)` per good, plus the observed
shares. `xi_j` is retained **oracle-only**.

## 4. The estimator: Berry inversion, OLS vs 2SLS

Invert the nested-logit shares (Berry, 1994) into a linear equation. With `s_j`
the market share, `s_0` the outside share, and `s_{j|g}` the within-nest share:

```
ln s_j − ln s_0  =  b0 + beta*x_j + c*SBC_j  −  alpha*logp_j  +  (1 − lam)*ln s_{j|g}  +  xi_j
```

Two regressors are **endogenous**: `logp_j` (price is an equilibrium outcome,
correlated with `xi_j`) and `ln s_{j|g}` (a share, mechanically correlated with
`xi_j`). Recover `alpha = −coef(logp)` and `lam = 1 − coef(ln s_{j|g})`.

- **OLS** (`fit_berry_ols`) is biased: `xi_j` moves both the price and the shares,
  so the price coefficient is pulled toward zero (and, in the leaner
  `examples/` sketch that omits the SBC covariate, all the way to the wrong sign —
  the "we raised prices and sold more" illusion, now arising through the market).
- **2SLS** (`fit_berry_2sls`) instruments the two endogenous regressors with the
  **excluded shifters**: own PROMO (a pure supply shock, excluded from demand) and
  the BLP-style **rival** instruments — the count of same-nest rivals hit by a
  PROMO or an SBC that week (they affect good *j* only through equilibrium prices).
  A first-stage F guards against weak instruments (docs/05 convention: F < 10 ⇒
  weak).

Standard errors are **clustered by week**: one week's goods trade against a shared
denominator, so they are not independent observations (CLAUDE.md; docs/04). The
effective-n is the number of weeks, not good-weeks.

## 5. The graded result

Default config (3 nests × 4 goods, 400 weeks, 4,800 equilibria), `market.py`:

| | price sensitivity `alpha` | nest substitution `lam` |
|---|---|---|
| **truth** | 0.900 | 0.550 |
| OLS (biased) | 0.088 | 0.372 |
| 2SLS (promo/rival IV) | 0.921 | 0.560 |

First-stage F ≈ 558 (instruments strong); the 2SLS 95% CI `[0.84, 1.00]` covers
the truth. The recovered **diversion matrix** matches the oracle: off-diagonal MAE
≈ 0.001, and within-nest diversion share ≈ 0.94 (true) vs ≈ 0.94 (recovered) —
substitution is fierce inside a nest and weak across, and the estimator recovers
that structure, not just a scalar.

`market_scorecard` repeats this over seeds and reports OLS-vs-2SLS bias, RMSE, and
95% CI coverage — the Phase 6 analogue of `endogeneity_scorecard`, and the plot no
real dataset can produce: the estimate's error *as a function of* whether you used
the shifter.

## 6. What this stage does not yet do

The reduced-form clearing is a stand-in. Two mechanisms from the transfer-market
note need the real order book and are Stage 2/3:

- **Order book as a WTP instrument (§4a of the note).** Listing-level (price,
  time-to-sell) pairs give a survival curve — a direct read on demand intensity at
  each price — but with selection: sellers set asks using private demand
  information, biasing the naive curve toward looking inelastic.
- **Bunching at price bands (§4c).** If goods carry binding min/max prices, the
  mass piled at the cap identifies how far above it the market-clearing price would
  have been — a *second* censoring channel, distinct from the inventory censoring
  of Phases 2 and 5.

Both are demonstrations the simulator can support that obtainable real data cannot
(see the FUT case-study feasibility memo), which is itself part of the point.

## 7. References

- Working, E. J. (1927), *What Do Statistical "Demand Curves" Show?*, QJE.
- Berry, S. (1994), *Estimating Discrete-Choice Models of Product
  Differentiation*, RAND J. Econ. — the nested-logit inversion used here.
- Berry, Levinsohn & Pakes (1995), *Automobile Prices in Market Equilibrium*,
  Econometrica — the rival-characteristics instruments; PyBLP is the Stage-2
  estimation path.
- Cardell, N. S. (1997), *Variance Components Structures for the Extreme-Value and
  Logistic Distributions* — the nested-logit share algebra.
- Bojinov, Simchi-Levi & Zhao, *Design and Analysis of Switchback Experiments* —
  the time-block unit and its carryover cost.

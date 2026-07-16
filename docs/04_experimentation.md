# Experimentation Design — Randomization Units, Sample Sizes, and Guardrails

How to run a pricing experiment that actually identifies a demand curve, and how the simulator
stress-tests each design choice.

---

## 1. Choosing the randomization unit

### 1.1 Per-user randomization (ZipRecruiter style)
Each consumer is assigned a price cell once and sees it consistently.

- **Pros:** captures the full decision process including return visits; no within-user price
  flicker (a user seeing $19 then $99 behaves strangely and may arbitrage); the unit of
  business value (a customer) matches the unit of analysis; enables downstream LTV outcomes.
- **Cons:** slower to accumulate units than sessions; requires stable identity (logins,
  device IDs); price discrimination across users is visible if users compare notes —
  reputational/legal exposure (Amazon's own 2000 DVD-pricing test became a PR incident).
- **Analysis note:** the experiment unit is the user, so cluster standard errors at the user
  level; one user's many sessions are not independent observations.

### 1.2 Per-session randomization (Uber-adjacent)
Each session/offer independently draws a price.

- **Pros:** maximum statistical power per calendar day; maps exactly to offer-level demand
  curves; fine for anonymous or low-repeat traffic.
- **Cons:** repeat visitors see different prices → strategic waiting ("refresh until cheap"),
  which contaminates the estimate (the cheap-cell sample becomes enriched with patient
  bargain-hunters — a selection effect the simulator can reproduce via demand shifting,
  Phase 4); violates the independence assumption if the same user contributes many sessions
  (cluster by user anyway).
- **Use when:** purchase is one-shot per session and identity is weak.

### 1.3 Switchback / time-split (Lyft, DoorDash style)
The *whole market* toggles arms on a randomized schedule (e.g., 2-hour windows).

- **Why it exists:** in shared-resource marketplaces, user-split pricing tests violate SUTVA —
  a discount for treatment riders depletes driver supply for control riders, biasing the
  contrast. Switchbacks keep everyone in one arm at a time.
- **Costs:** the effective sample size is the number of *windows*, not users → high variance;
  carryover effects between adjacent windows (the market doesn't reset instantly) require
  burn-in trimming or window-length tuning; infeasible for persistent UI changes.
- **Analysis:** treat window as the unit; use Horvitz–Thompson / regression with window fixed
  effects; power depends on window count and autocorrelation.

### 1.4 Cluster / geo randomization
Randomize markets or regions. Kills interference across users within a market, costs power
(few clusters); analyze with cluster-robust or randomization inference. The practical choice
when identity and switchbacks both fail.

**Decision rule:** interference risk drives the choice. No shared inventory/capacity between
arms → per-user (default) or per-session (anonymous, one-shot). Shared capacity → switchback
or geo.

## 2. How big must samples be?

The estimand is a *difference in conversion rates between price cells* (that difference IS the
demand-curve slope), so standard two-proportion power math applies — the catch is that
realistic price effects are small in absolute conversion points.

Worked example (defaults in the simulator's power helper):
- Baseline conversion at the anchor price: **p₁ = 10%**
- Price cell +20% with elasticity −0.75 → conversion drops ~15% relative → **p₂ = 8.5%**
- α = 0.05 (two-sided), power = 0.80 → effect size h = 0.0512 → **n ≈ 6,000 per cell**
- 10 cells (ZipRecruiter-style curve) → ~60k sessions minimum; realistically 2–3× that
  because (a) outer cells sit at lower conversion, (b) you want per-segment curves.

Rules of thumb the simulator makes concrete:
1. **Sample size scales with 1/h² ≈ 1/(Δconversion)²**: halving the detectable price effect
   quadruples n. Elasticities near −0.5 in a 10%-conversion business need six-figure session
   counts per contrast — this is why Uber needed 50M sessions for a full curve.
2. **Curve estimation ≠ one contrast.** For K cells you're powering K−1 adjacent contrasts;
   allocate more mass to the anchor and the cells where curvature matters (optimal design
   favors extreme + anchor cells if you'll fit a parametric curve).
3. **Heterogeneity multiplies n.** Per-segment curves need the same math *within* each
   segment; a 20%-share segment needs 5× the traffic for equal precision. (Dubé & Misra
   solved this by pooling with regularized ML rather than splitting.)
4. **Clustering deflates effective n.** With per-user assignment and ~3 sessions/user,
   design effect ≈ 1 + (m̄−1)·ICC; budget sessions accordingly.
5. **Switchbacks:** power comes from window count; simulate before running (the simulator's
   Phase 3 harness does exactly this — generate the schedule, estimate, repeat, read off
   empirical power).

`metrics.power_two_proportions()` (Phase 1) implements the analytic version; the simulator's
Monte Carlo power (run the whole pipeline R times, count rejections) is the ground truth the
analytic formula is checked against.

## 3. Analytical guardrails

### 3.1 Sample-ratio mismatch (SRM)
If the realized arm counts deviate from planned allocation beyond chance, *stop* — assignment
or logging is broken and every downstream number is suspect.
```python
from scipy.stats import chisquare
chisquare(f_obs=[n_a, n_b], f_exp=[N*0.5, N*0.5])   # p < 0.001 ⇒ SRM alarm
```

### 3.2 A/A tests
Run the full pipeline with identical arms. Effect estimates should be null at the nominal
false-positive rate; if not, your variance estimator (clustering!) is wrong. Cheap in the
simulator — Phase 3 runs A/A batteries automatically.

### 3.3 Covariate balance
Pre-assignment covariates should be balanced across arms (standardized mean differences
< 0.1). Imbalance ⇒ randomization bug or unit-definition leak.

### 3.4 Cross-contamination detection (the pricing-specific ones)
- **Multi-arm exposure audit:** count users observed in >1 cell (identity resets, multi-device).
  Even a few % of dual-exposed heavy users can drag estimates; report and sensitivity-test by
  excluding them.
- **Strategic-waiting signature:** under per-session randomization, compare
  sessions-per-purchase and inter-session gaps across cells. Cheap-cell buyers with elevated
  prior-session counts at higher prices = refresh arbitrage. (The simulator's Phase 4 demand-
  shifting knob generates this signature on demand so the detector can be validated.)
- **Interference probes:** hold out "pure control" markets untouched by the experiment; if
  within-market control drifts away from pure control during the test, treatment is leaking
  through shared capacity → switch designs.
- **Pre-period regression / CUPED:** use pre-experiment outcomes as covariates — detects
  residual imbalance and cuts variance 30–50% for repeat-purchase metrics.

### 3.5 Temporal validity
- **Novelty/learning effects:** estimate effects by exposure week; a decaying effect means the
  short test overstates the long-run elasticity (reference prices haven't re-anchored yet).
- **Seasonal representativeness:** an elasticity measured in December may not hold in March;
  the simulator can generate season-varying elasticity to quantify how wrong extrapolation is.
- **Peeking / sequential testing:** fixed-horizon p-values are invalid under continuous
  monitoring; use group-sequential bounds or always-valid inference if you must peek.

### 3.6 Multiple comparisons
K cells and S segments = K·S contrasts. Pre-register the primary contrast; control FDR
(Benjamini–Hochberg) on the rest; prefer fitting one parametric curve over testing every pair.

## 4. Ethics & practicality notes
Price experiments discriminate by construction. Keep ranges defensible (test discounts off a
public anchor rather than surcharges), honor quoted prices, cap exposure duration, and check
regional price-discrimination law before user-level targeting. These constraints bound the
feasible price grid — which is itself a parameter the simulator should respect (`price_grid`
bounds in `SimulationConfig`).

## 5. What Phase 3 will ship
Assignment engines (user / session / switchback / geo), schedule generator, SRM + balance +
dual-exposure audits as automatic post-run checks, analytic and Monte Carlo power tools, and
an experiment scorecard: bias, RMSE, CI coverage, and guardrail outcomes per design × estimator.

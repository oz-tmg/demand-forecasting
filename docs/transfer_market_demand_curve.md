# Mapping a Demand Curve from a Player-Driven Transfer Market

_A working note, written 2026-08-03, recording a design discussion about EA Sports FC /
FIFA Ultimate Team's transfer market and what it would take to measure demand there._

This is **not** a practitioner reference like the other pages in `docs/`. It is a record
of a line of reasoning: where the idea came from, what is standard economics versus what
is inference about a specific game, and what would have to be true before any of it is
run. Read the provenance section first — several claims here are about a live product I
cannot verify from inside this repository.

---

## 0. Provenance — where this came from and how much to trust it

The question that started it: *when does a soft-currency price ladder stop resembling
the FUT transfer market, where there are thousands of differentiated player cards and
players bid on them?*

Three different kinds of claim are mixed together below, and they deserve different
levels of trust:

**(a) Standard economics, high confidence.** That a price set by trading is an
equilibrium outcome rather than an assignable treatment; that observed price–quantity
pairs trace neither supply nor demand without a shifter (the classic simultaneity
problem); that differentiated goods need a demand *system* rather than a scalar
elasticity; that binding price controls produce excess demand and bunching. None of this
is specific to games — it is textbook industrial organization and demand estimation, and
I am applying it, not reporting it.

**(b) Standard marketplace experimentation practice, high confidence.** Switchback
designs — randomizing time blocks rather than users — are how two-sided marketplaces
routinely test policies that move a shared price. That lineage is ride-hailing and
delivery marketplaces, not games. Item-level randomization and synthetic control on a
panel of goods are the other two conventional escapes from a small-cluster problem.

**(c) Inference about FUT's specific mechanics, medium confidence — verify before
using.** That the transfer market is platform-segmented; that there is a sale tax; that
cards carry minimum and maximum price bands; that rewards can be untradeable; that
squad-building challenges consume cards and act as demand sinks. These are widely known
features of the mode, but I have not verified them against the current release from
inside this session, and EA has changed several of them over the years. **Treat every
specific in this note as a hypothesis to confirm against the live economy**, including
whether markets are still separated by platform at all.

I did not read a source that says "here is how to map FUT's demand curve." The mapping
below is derived: take what the mode does mechanically, apply standard demand
estimation, and see what falls out. Where that produces a testable claim I have said so;
where it produces a design, I have said what would break it.

---

## 1. The dividing line is tradability

A price ladder — randomize the posted price across arms, measure quantity — is valid when
the **studio is the seller** and the good cannot be resold. It stops being valid the
moment the good trades between players, and it fails in two distinct ways at once:

1. **Price is no longer assignable.** In a transfer market the price is what two players
   agreed on. You cannot randomize it; you can only randomize things that shift it.
2. **Per-user price randomization becomes an exploit.** If you post different prices to
   different players for a tradable good, you have created an arbitrage: the cheap arm
   buys and resells into the expensive arm. That is worse than a biased experiment. You
   have shipped an economy bug and paid users to find it.

Point 2 is the one worth carrying around. In an untradeable economy a badly designed
price test produces a wrong number; in a tradable one it produces a wrong number *and* a
gold-farming strategy.

**FUT sits on both sides of this line simultaneously.** Packs, store prices and
untradeable rewards are studio-posted goods you can ladder normally. The transfer market
is the other thing entirely, and the same title needs two different measurement
programmes.

---

## 2. What changes once price is an outcome

| | Posted-price sink | Player-driven market |
|---|---|---|
| **Treatment** | the price itself | a *policy* that shifts supply or demand |
| **Outcome** | units bought | price index, liquidity, time-to-sell, fill rate |
| **Unit** | the user | the market, the time block, or the item — never the user |
| **Estimand** | a scalar elasticity | a substitution matrix over item attributes |

### The levers you can actually randomize

Since price is off the table, the treatment moves up a level to the rules that generate
it:

- **Sale tax** (the cut taken on a completed trade) — moves the supply side directly.
- **Price bands** (per-card minimum and maximum) — a literal price control; widening or
  narrowing them is a clean policy treatment.
- **Pack odds and drop rates** — a supply shock, injecting cards into the market.
- **Tradeable versus untradeable rewards** — changes how much of the reward flow reaches
  the market at all.
- **Listing duration, listing caps, market availability windows** — friction levers that
  move liquidity without touching price.
- **Objectives and squad-building challenges that consume specific cards** — a demand
  shock, and a targeted one.

That last pair matters more than it looks. A pack promo is close to a **pure supply
shifter** and an SBC requirement is close to a **pure demand shifter**, which is exactly
the asymmetry needed for identification. Section 4.

---

## 3. The unit problem, and three ways out

Prices are a shared object. Randomizing users inside one market violates SUTVA badly —
everyone trades against the same order book — so the unit has to be the market. If
markets are platform-segmented you have a handful of clusters, and a handful is not an
experiment. This repository already carries that failure mode as a worked example:
`exp_platform_store_layout` returns "inconclusive — too few clusters."

**1. Switchbacks.** Randomize *time blocks* within one market: alternate the tax rate or
band width by day or by block over many blocks. You trade cross-sectional units for
temporal ones. The cost is carryover — inventory and player stock persist across blocks,
so the washout period has to be reasoned about rather than assumed, and the analysis
needs HAC or block-bootstrap standard errors.

**2. Item-level randomization.** There are thousands of cards, so randomize the *good*.
Apply the policy to a random subset of cards and cluster on the substitution graph — by
rating band × position, say — so spillover stays inside treated groups. This is the same
logic this case study applies to alliances in the reengagement fixture, and it is the
design I would reach for first, because it converts a 3-cluster problem into a
several-hundred-cluster one.

**3. Synthetic control on a card panel.** For a one-shot policy change you cannot
randomize, build a donor pool of untreated cards and construct the counterfactual price
path. Weaker evidence, but often the only option after the fact.

---

## 4. How you would actually map the demand curve

This is the part the question was really about. Three sources of identification, in
increasing order of strength.

### 4a. The order book is a willingness-to-pay instrument

An auction market hands you something a posted-price store never does: **the
distribution of what buyers were willing to pay**. Bids are WTP draws. Buy-it-now
listings are posted prices with an observable outcome — sold or not, and how fast.

For a single card, the raw material is a set of (price, time-to-sell) pairs. The
survival curve of a listing at price *p* — the probability it sells within *t* — is a
direct read on demand intensity at that price. Sweep across the listings and you have an
empirical demand curve for that card, without running an experiment at all.

The catch is selection: sellers choose their asking prices, and they choose them using
information about demand. High asks appear when sellers believe demand is strong. So the
naive curve through observed listings is biased toward looking inelastic.

### 4b. The simultaneity problem, and why shifters are the answer

Plotting observed price against observed quantity does **not** trace a demand curve. It
traces the sequence of equilibria where supply and demand happened to cross. This is the
oldest identification problem in econometrics, and its solution is to find something
that moves one curve while leaving the other alone.

The transfer market supplies both:

- **A supply shifter traces the demand curve.** A pack promo injects cards without
  changing how much anyone wants them. Price falls, quantity rises, and the locus of
  those points is demand.
- **A demand shifter traces the supply curve.** An SBC that consumes a specific card
  raises demand for it without changing how many exist. Price rises, quantity rises.

If those events are scheduled by live-ops anyway, they are natural experiments already
in the data. If a few of them can be *randomized* — which cards an SBC requires, which
cards a promo pack contains — they become instruments, and the demand system is
identified properly rather than argued for.

That, concretely, is the answer to "can you map a demand curve from transfer-market
data." You do not get it from prices and quantities alone. You get it from prices and
quantities *plus* a shifter you can defend, and the mode generates the shifters as a
side effect of its own content calendar.

### 4c. Price bands give you a bunching estimator

If cards carry minimum and maximum prices, those are binding price controls, and a
binding control is informative. When a ceiling binds you observe excess demand directly:
listings clearing instantly, cards effectively unobtainable at the posted maximum, a
mass of transactions piled exactly at the cap. The size of that pile — the bunching mass
— identifies how far above the ceiling the market-clearing price would have been.

The economics here is the same as the cap-binding mechanic already simulated in this
repository's `exp_specialty_pack_caps`, transposed from one player's wallet to a whole
market: a cap only teaches you something when a meaningful share of demand sits above
it.

### 4d. The estimand is a matrix, not a number

With thousands of differentiated cards, the own-price elasticity of one card is close to
meaningless — nobody is deciding whether to buy a specific striker in isolation. They are
choosing among substitutes at a rating band, a position, a league, a chemistry link.

So the model is a **demand system over attributes**: a nested logit or a random-
coefficients discrete-choice model where the nest is defined by role and quality.
Substitution is fierce inside a nest and weak across nests, and the object you report is
a **diversion matrix** — of the demand a card loses, where does it go — rather than a
scalar.

This is exactly the structure built into `data/simulation/units.py` for unit investment,
at much smaller scale: a nested logit over role → item, where the nesting parameter *is*
the substitution structure. The transfer-market version is the same idea with thousands
of goods, an order book, and prices that clear rather than being posted.

---

## 5. What would have to be built to demonstrate this

The current simulator cannot express any of it. It has posted prices and studio-set
costs; it has no market, no order book, no clearing, and no player-to-player transfer.
The demand layer measures how quantity responds to a price the studio chose — which is
the right model for the sinks it covers, and the wrong one for a market.

A transfer-market fixture would need:

1. **Heterogeneous goods with attributes** — rating, position, league, chemistry — and a
   nest structure over them.
2. **An order book**: listings with asks, bids, expiries, and a matching rule.
3. **Faucets and sinks**: packs injecting cards, SBCs and objectives consuming them, a
   tax removing currency.
4. **A known ground-truth demand system**, so the recovered diversion matrix can be
   scored the way `verify_demand_recovery.py` scores the elasticities today.
5. **Policy treatments**: tax rate and band width randomized across time blocks or card
   nests, with the analysis having to recover the truth through the equilibrium.

That last point is what would make it worth building. Recovering a parameter from a
posted price is a solved exercise; recovering one *through a market that clears* is the
part almost nobody demonstrates, and it is where the interesting failure modes live —
selection in the order book, carryover across switchback blocks, censoring at the price
bands.

It is a market simulator, not an effect multiplier, and it is a substantially larger
build than the demand layer was.

---

## 6. Summary judgements

- **Keep the soft-currency price ladder for untradeable, studio-sold sinks.** It is
  valid there, cheap, and free of platform exposure.
- **Never randomize a tradable good's price per user.** Not because the estimate is
  biased — because it is an exploit.
- **In a market, randomize the policy and measure the price.** Tax, bands, pack odds,
  reward tradability, SBC requirements.
- **Randomize the item, not the user**, and cluster on the substitution graph. It is the
  only escape from a three-cluster problem that does not cost you the cross-section.
- **Use the content calendar as instruments.** Supply shocks trace demand; demand shocks
  trace supply. Randomize which cards they touch and the system is identified.
- **Report a diversion matrix, not an elasticity.**
- **Verify every FUT-specific mechanic in section 0(c) before designing anything.**

---

_Related: `docs/demand_measurement_experiment_design.md` (the posted-price programme,
which is built and verified), `data/simulation/economy.py` and
`data/simulation/units.py` (the engines), `data/simulation/verify_demand_recovery.py`
(the recovery scoreboard)._

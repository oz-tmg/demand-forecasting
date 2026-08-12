# FUT Transfer Market as a Savepoint Case Study — Feasibility Memo

_Written 2026-08-06. Companion to `docs/transfer_market_demand_curve.md` (the
identification note) and the Phase 6 scope in `PROJECT_CONTEXT.md`. Question: can
the transfer-market idea become a **standalone Savepoint case study on real FUT
data**, with `demand_sim` as the ground-truth validation engine?_

## Bottom line

Yes as a **method-validation case study**, no as a **microstructure study**. The
honest, defensible version pairs `demand_sim` (where truth is known) with a
**reduced-form event study on public aggregated price series** (where truth is
not). It is publishable, differentiated, and legally clean. What is *not*
feasible is the order-book layer of the note (§4a survival-curve WTP, §4c
bunching): nobody exposes the microdata it needs, and getting it would mean
violating EA's terms. So build the case study around the parts that survive
contact with the real data — the shifter/instrument identification (§4b) and the
demand system across cards (§4d) — and use the simulator to prove the estimator
is unbiased *before* pointing it at data where you can't check.

## What real data actually exists, and at what granularity

Three tiers, decreasing in legitimacy and increasing in depth:

| Source | What you get | Granularity | Access |
|---|---|---|---|
| **Third-party aggregators** (FUTBIN, FUT.GG, FUTWIZ, FutDatabase) | Per-card price index (min-BIN), coarse "sales/volume" proxies, sometimes a price *history* line | Snapshot: 30 min–24 h, rarer cards more often | FutDatabase has a documented keyed REST API; FUTBIN is bot-protected and scraped only via paid wrappers (Apify/Parse.bot) |
| **EA official** (Ultimate Team) | — | — | **No public API.** EA is *discussing* extending API access post-Pro-Clubs; not available today |
| **Live scrape of the in-game / web / companion market** | The order book itself (asks, some depth) | Real-time | **Prohibited** — see legal note |

The decisive limitation: **aggregators publish a summary price, not the order
book.** No bids, no listing-level time-to-sell, no fill rates, no per-listing
survival. That is exactly the raw material the note's strongest section (§4a, the
order book as a WTP instrument) is built on. It does not exist in any obtainable
feed.

Two secondary notes. First, the *shifter* side is fully public: promo/pack
release dates and SBC requirements are published on every content site, so the
instruments are observable as an event calendar even though the microdata is not.
Second, recent EA FC titles moved toward a **unified cross-platform market**,
largely removing the console/PC segmentation the note flags in §0(c) — which
*worsens* the small-cluster problem (fewer independent markets) and argues for the
item-level / event-study designs over switchbacks. (Confirm against the current
release before publishing; this is exactly the kind of live-product claim the note
warns to verify.)

> Caution: "FUT transfer market data" is easy to confuse with **Transfermarkt**
> (real-world player transfer *fees*). The abundant Kaggle datasets are
> Transfermarkt, a different thing entirely — real football economics, not the
> in-game card auction. They are not a substitute for FUT card prices.

## Legal / ToS reality — read before sourcing anything

EA's rules **prohibit bots, automation, scraping, and text/data mining** of EA
Services, and making transactions outside the game/web/companion app. Enforcement
is automated (transfer-market bans, account bans) and backed by kernel-level
anti-cheat. **Do not scrape EA directly, and do not automate a market account** —
that is both a ToS violation and a real ban/anti-cheat risk, and it is the one
path that touches the note's forbidden territory.

Third-party aggregators carry the ToS risk themselves and expose derived prices.
Even so, two cautions remain: (1) their own terms restrict bulk scraping (use a
provider that offers an actual API under terms, e.g. FutDatabase, rather than
scraping FUTBIN); and (2) card prices/images sit on EA IP, so a public case study
should **publish derived analysis, not republish bulk raw price data**, and cite
the provider. None of this blocks a portfolio piece — it just sets the sourcing
path.

## Feasibility of the note's four strategies on *real* data

| Note § | Strategy | On real aggregated data? |
|---|---|---|
| 4b | Content-calendar shifters (promo = supply shock, SBC = demand shock) as instruments / event studies | **Feasible.** Public event calendar + price panel = reduced-form event study or IV on aggregated series |
| 4d | Demand *system* across cards (nested logit by rating × position), diversion matrix | **Partly feasible.** Prices + a volume proxy + card attributes support a reduced-form / share-based system; a full BLP needs a quantity measure the feeds only approximate |
| 4a | Order book as a WTP instrument (survival curve of listings) | **Not feasible** — needs listing-level microdata no one exposes without violating ToS |
| 4c | Bunching at binding price bands | **Weakly feasible** — you can *see* pile-ups at min/max BIN in the price index, but without listing-level mass the bunching estimator is qualitative at best |

## Recommended case-study design — "validate on truth, then deploy"

This is the framing that makes it a *Savepoint* case study rather than a hobby
scrape, and it is the natural graduation of the estimator-scoring harness the repo
already flags as its most reusable asset:

1. **Simulator arm (`demand_sim` / Phase 6).** Generate a market with the same
   structure — differentiated cards, promo/SBC shocks, a clearing price — where
   the elasticities and diversion matrix are known. Show the event-study / IV
   estimator recovers truth and quantify its bias when assumptions break
   (carryover across weeks, unobserved concurrent promos, thin volume). This is
   the spike in `examples/phase6_market_clearing_sketch.py`, hardened.
2. **Real-data arm.** Pull an aggregated **price panel** for a card universe plus
   the **public promo/SBC calendar**. Run the *same* estimator: reduced-form price
   response to supply shocks (packs) and demand shocks (SBCs), and a
   share/attribute demand system yielding a diversion matrix (which cards absorb
   demand when a rival is SBC'd).
3. **The honest handshake.** The real elasticities are unknown — that is the whole
   problem. The simulator is what licenses the estimate: "this estimator is
   unbiased on ground truth with this structure; here is what it says on the live
   market, with these caveats." That is a genuinely strong, differentiated
   analytics story and it sidesteps every order-book/ToS problem.

## Sourcing options, ranked

1. **FutDatabase keyed REST API** — cleanest terms, real API, price + attributes.
   Start here for the price panel.
2. **Public content calendar** (promo/SBC dates) — hand-curated or lightly scraped
   from content sites; this is editorial data, low risk.
3. **A commercial wrapper** (Apify / Parse.bot over FUTBIN) — only if FutDatabase
   coverage is thin, and read the provider terms on redistribution.
4. **A historical community dataset** (e.g. GitHub FIFA-19 price time-series) — good
   for a static methods demo without any live pull at all; smallest legal surface.
5. **Never:** scraping EA's web/companion app or automating a market account.

## Standalone project vs. Phase 6 of `demand-sim`

Recommend **Phase 6 stays the generic methods engine inside `demand-sim`**
(product-agnostic, no live-product claims, safe for the portfolio's core), and the
**FUT case study is a thin standalone repo that imports `demand_sim`** for the
simulator arm and adds the real-data arm. That keeps the unverifiable-live-product
risk out of the methods lab (honouring the note's own §0 discipline) while giving
Savepoint a named, real-data showcase. It is also the concrete instance of the
"estimator-scoring harness graduates to a reusable asset" decision already
anticipated in `CLAUDE.md` and `PROJECT_CONTEXT.md`.

## Risks & caveats to state up front in any published version

- **No ground truth on the real arm** — the estimate is only as credible as the
  simulator validation and the instrument's defensibility.
- **Volume/quantity is a proxy** on aggregated feeds; a share-based demand system
  inherits that measurement error. Say so.
- **Concurrent-event confounding** — multiple promos/SBCs overlap; the clean
  supply-vs-demand shock separation the simulator enjoys is muddier live.
- **Live-product drift** — mechanics (platform segmentation, price bands, tax)
  change per release; verify each against the current title before asserting it.
- **IP / ToS hygiene** — derived analysis only, cite the data provider, no EA
  scraping, no market automation.

## Recommendation

Green-light the **method-validation framing**: build Phase 6 as the generic
market-clearing engine, then a thin FUT case-study repo that runs the identical
estimator on a FutDatabase price panel plus the public content calendar, with the
simulator as the truth harness. Scope it to §4b (shifter identification) and §4d
(demand system / diversion), and explicitly **descope the order book (§4a) and
bunching (§4c)** as not obtainable from real data — noting that the simulator can
still demonstrate them, which is itself a selling point ("here is the method the
live data can't yet support"). Drop any dependence on scraping EA or on
unverifiable FUT mechanics.

---

### Sources
- [EA — How to play EA SPORTS FC by the rules](https://help.ea.com/en/articles/ea-sports-fc/fc-rules/)
- [EA Positive Play Charter](https://www.ea.com/commitments/positive-play/charter)
- [EA Forums — request to extend API access to Ultimate Team](https://forums.ea.com/discussions/fc-26-ultimate-team-en/extend-api-access-to-ultimate-team-following-pro-clubs-integration/12798515)
- [FutDatabase — FUT database & price API](https://www.futdatabase.com/)
- [FUTBIN](https://www.futbin.com/) · [FUT.GG](https://www.fut.gg/)
- [Parse.bot — Futbin API wrapper](https://parse.bot/marketplace/1b6234f9-0dfb-4cca-99b4-2d6d37aec6a7/futbin-com-api) · [Apify — Futbin scraper](https://apify.com/getdataforme/futbin-player-details-scraper/api)
- [GitHub — PhilippeCodes/Web-Scraping-Futbin.com (FIFA-19 price time-series)](https://github.com/PhilippeCodes/Web-Scraping-Futbin.com)
- [Kaggle — Football Data from Transfermarkt (NB: real-world fees, *not* FUT)](https://www.kaggle.com/datasets/davidcariboo/player-scores/data)

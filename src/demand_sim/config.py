"""Configuration objects for the demand simulator.

Every parameter that the demand literature identifies as a driver appears here,
annotated with the phase in which it becomes active. Phase 1 uses: segments,
choice model, price mechanism (fixed / random cells), promo utility hook, n_sessions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ChoiceModel = Literal["logit", "wtp_threshold"]
PriceMechanism = Literal["fixed", "random_cell"]  # Phase 3 adds: "surge", "switchback"; Phase 4: "endogenous"


@dataclass(frozen=True)
class SegmentConfig:
    """One consumer segment. Heterogeneity backbone (Dubé & Misra 2023).

    Logit model:      U = base_util + price_coef * price + promo_uplift * promo + eps,
                      eps ~ Logistic(0, 1); buy iff U > 0.
    WTP threshold:    WTP ~ LogNormal(wtp_mu, wtp_sigma); buy iff WTP >= price.
    """

    name: str
    share: float                      # population share; shares must sum to 1
    # --- logit parameters ---
    base_util: float = 1.0
    price_coef: float = -0.05         # must be negative (law of demand)
    promo_uplift: float = 0.0         # additive utility when promo shown (Phase 2 schedules)
    # --- wtp_threshold parameters ---
    wtp_mu: float = 3.0               # log-dollars
    wtp_sigma: float = 0.6
    # --- Phase 4 hooks ---
    ref_price_sensitivity: float = 0.0  # loss aversion around reference price

    def __post_init__(self) -> None:
        if self.price_coef >= 0:
            raise ValueError(f"segment {self.name}: price_coef must be < 0")
        if not 0 < self.share <= 1:
            raise ValueError(f"segment {self.name}: share must be in (0, 1]")


@dataclass(frozen=True)
class ArrivalConfig:
    """Phase 2: non-homogeneous Poisson arrival process for sessions.

    Sources: M5 stylized facts; Chronos-2 known-future covariates (promos, holidays).
    Phase 1 ignores this and draws `n_sessions` directly.
    """

    base_rate_per_day: float = 2_000.0
    trend_daily_growth: float = 0.0          # multiplicative, e.g. 0.0005
    dow_multipliers: tuple[float, ...] = (1.0,) * 7
    annual_fourier_amp: float = 0.0          # yearly seasonality amplitude
    holiday_uplift: float = 0.0              # Prime-Day-like spikes


@dataclass(frozen=True)
class InventoryConfig:
    """Phase 2: (s, S) replenishment; stockouts censor sales (docs/02)."""

    enabled: bool = False
    initial_stock: int = 10_000
    reorder_point: int = 500                 # s
    order_up_to: int = 5_000                 # S
    lead_time_days: int = 3
    oos_price_correlation: float = 0.0       # >0: replenishment lags during promos
    stockout_behavior: Literal["lost", "substitute", "backorder"] = "lost"


@dataclass(frozen=True)
class SimulationConfig:
    """Top-level Phase 1 configuration."""

    segments: tuple[SegmentConfig, ...]
    n_sessions: int = 100_000
    choice_model: ChoiceModel = "logit"
    price_mechanism: PriceMechanism = "random_cell"
    fixed_price: float = 29.0
    price_cells: tuple[float, ...] = (9.0, 19.0, 29.0, 39.0, 49.0, 69.0, 99.0)
    cell_weights: tuple[float, ...] | None = None   # None = uniform allocation
    promo_prob: float = 0.0                  # Phase 1: iid promo flag; Phase 2: schedule
    sessions_per_consumer_mean: float = 1.0  # >1 enables per-user randomization realism
    seed: int = 42
    # Phase 2+ blocks ride along so configs are forward-compatible:
    arrivals: ArrivalConfig = field(default_factory=ArrivalConfig)
    inventory: InventoryConfig = field(default_factory=InventoryConfig)

    def __post_init__(self) -> None:
        total = sum(s.share for s in self.segments)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"segment shares sum to {total}, expected 1.0")
        if self.cell_weights is not None and len(self.cell_weights) != len(self.price_cells):
            raise ValueError("cell_weights length must match price_cells")


def default_config() -> SimulationConfig:
    """Three-segment market loosely inspired by the ZipRecruiter finding that
    demand is less elastic than intuition suggests, with a WTP-heterogeneous mix."""
    return SimulationConfig(
        segments=(
            SegmentConfig("price_hawks", share=0.45, base_util=1.2, price_coef=-0.11,
                          wtp_mu=2.6, wtp_sigma=0.5, promo_uplift=0.6),
            SegmentConfig("mainstream", share=0.40, base_util=1.0, price_coef=-0.045,
                          wtp_mu=3.3, wtp_sigma=0.5, promo_uplift=0.3),
            SegmentConfig("whales", share=0.15, base_util=1.5, price_coef=-0.012,
                          wtp_mu=4.2, wtp_sigma=0.6, promo_uplift=0.1),
        ),
    )

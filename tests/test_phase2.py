"""Phase 2 acceptance tests: arrivals, censoring, dataset integrity."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from demand_sim import PanelConfig, generate_panel, observable_view
from demand_sim.arrivals import build_intensity, price_multiplier
from demand_sim.config import InventoryConfig
from demand_sim.inventory import simulate_inventory
from demand_sim.panel import ORACLE_COLS

SMALL = PanelConfig(n_products=4, n_stores=2, n_days=365, seed=11)


@pytest.fixture(scope="module")
def panel():
    return generate_panel(SMALL)["panel_oracle"]


def test_censoring_invariants(panel):
    assert (panel.units_sold <= panel.units_demanded).all()
    assert (panel.units_sold <= panel.inventory_start).all()
    assert (panel.inventory_start >= 0).all()
    assert (panel.lost_sales == panel.units_demanded - panel.units_sold).all()
    assert (panel.stockout == (panel.units_sold < panel.units_demanded)).all()


def test_stockouts_exist_but_bounded(panel):
    rate = panel.stockout.mean()
    assert 0.005 < rate < 0.20


def test_censoring_concentrates_on_promos(panel):
    promo_rate = panel[panel.promo_flag].stockout.mean()
    base_rate = panel[~panel.promo_flag].stockout.mean()
    assert promo_rate > base_rate


def test_observable_view_hides_oracle(panel):
    obs = observable_view(panel)
    assert not set(ORACLE_COLS) & set(obs.columns)
    assert "units_sold" in obs.columns


def test_price_multiplier_law_of_demand():
    assert price_multiplier(np.array([20.0]), 10.0, -1.5)[0] < 1.0
    assert price_multiplier(np.array([10.0]), 10.0, -1.5)[0] == 1.0
    with pytest.raises(ValueError):
        price_multiplier(np.array([10.0]), 10.0, 0.5)


def test_intensity_components():
    dates = pd.date_range("2024-01-01", periods=730, freq="D")
    price = np.full(730, 10.0)
    promo = np.zeros(730, dtype=bool)
    lam = build_intensity(dates, base_rate=10.0, trend_pct_yr=0.2,
                          weekly_amp=0.3, annual_amp=0.0, annual_phase=0.0,
                          price=price, base_price=10.0, elasticity=-1.5,
                          on_promo=promo, promo_lift=2.0)
    # trend: year 2 mean > year 1 mean
    assert lam[365:].mean() > lam[:365].mean()
    # weekly: Fri-Sun > Mon-Tue
    dow = dates.dayofweek.values
    assert lam[np.isin(dow, [4, 5, 6])].mean() > lam[np.isin(dow, [0, 1])].mean()


def test_inventory_conservation():
    rng = np.random.default_rng(0)
    demand = rng.poisson(8.0, size=200)
    cfg = InventoryConfig(enabled=True, reorder_point=30, order_up_to=60,
                          lead_time_days=3)
    res = simulate_inventory(demand, cfg, rng)
    # stock flow conserves: S + total receipts - total sold = final on-hand
    final = res.inventory_start[-1] - res.units_sold[-1]
    assert cfg.order_up_to + res.replenishment.sum() - res.units_sold.sum() == final


def test_deterministic_under_seed():
    a = generate_panel(SMALL)["panel_oracle"].units_sold.sum()
    b = generate_panel(SMALL)["panel_oracle"].units_sold.sum()
    assert a == b

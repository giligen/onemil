"""Unit tests for per-tier MACD zone multipliers (S2-max ship, 2026-04-18).

Verifies:
  - A-tier (intraday ≥ 20%) uses A-tier defaults (1.8×/1.0×)
  - Extras-tier (10% ≤ intraday < 20%) uses extras_tier block (2.0×/0.0×)
  - edge-tier (intraday < 10%) falls back to A-tier defaults
  - Dead zone unchanged regardless of tier
  - Tier boundaries inclusive/exclusive correctly

Uses monkeypatch on `macd_histogram` to deterministically drive the zone
classifier, since synthetic bar series can't reliably produce target
histogram % values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import BacktestRunner
from trading.two_tier_filter import classify_tier, TIER_A, TIER_EXTRAS, TIER_EDGE


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _dummy_bars(n: int = 40):
    """40 bars of constant close, enough for MACD warmup (≥ 35)."""
    closes = [10.0] * n
    return pd.DataFrame({
        'open': closes, 'high': closes, 'low': closes, 'close': closes,
        'volume': [1000] * n,
    })


def _mock_hist(macd_pct: float, entry_price: float = 10.0):
    """Return a pandas Series whose last value produces `macd_pct` when
    divided by entry_price and multiplied by 100."""
    # hist_val / entry_price * 100 = macd_pct → hist_val = macd_pct * entry_price / 100
    hist_val = macd_pct * entry_price / 100
    return pd.Series([0.0] * 39 + [hist_val])


# -------------------------------------------------------------------------
# BT tier routing
# -------------------------------------------------------------------------

class TestBTPerTierRouting:
    """BacktestRunner._get_macd_zone_multiplier applies per-tier multipliers."""

    @pytest.fixture
    def r(self):
        runner = BacktestRunner()
        runner._prev_day_bars = None
        return runner

    @pytest.fixture
    def bars(self):
        return _dummy_bars()

    # --- A-tier (intraday ≥ 20%) → A-tier defaults ---

    def test_a_tier_strong_pos(self, r, bars, monkeypatch):
        """A-tier + macd > +0.5% → 1.8× (config default)."""
        monkeypatch.setattr('trading.indicators.macd_histogram',
                            lambda _: _mock_hist(+1.0))
        mult = r._get_macd_zone_multiplier(
            'X', bars, len(bars) - 1, 10.0, intraday_change_pct=25.0)
        assert mult == pytest.approx(r.macd_strong_pos_multiplier)
        assert mult == pytest.approx(1.8)

    def test_a_tier_strong_neg(self, r, bars, monkeypatch):
        """A-tier + macd < -0.5% → 1.8×."""
        monkeypatch.setattr('trading.indicators.macd_histogram',
                            lambda _: _mock_hist(-1.0))
        mult = r._get_macd_zone_multiplier(
            'X', bars, len(bars) - 1, 10.0, intraday_change_pct=25.0)
        assert mult == pytest.approx(r.macd_strong_neg_multiplier)

    def test_a_tier_normal(self, r, bars, monkeypatch):
        """A-tier + macd in [+0.1%, +0.5%] (out of dead AND strong) → 1.0×."""
        monkeypatch.setattr('trading.indicators.macd_histogram',
                            lambda _: _mock_hist(+0.3))
        mult = r._get_macd_zone_multiplier(
            'X', bars, len(bars) - 1, 10.0, intraday_change_pct=25.0)
        assert mult == pytest.approx(r.macd_normal_multiplier)

    def test_a_tier_dead_zone(self, r, bars, monkeypatch):
        """A-tier + macd in [-0.2%, +0.1%] → 0.0× (skip)."""
        monkeypatch.setattr('trading.indicators.macd_histogram',
                            lambda _: _mock_hist(+0.0))
        mult = r._get_macd_zone_multiplier(
            'X', bars, len(bars) - 1, 10.0, intraday_change_pct=25.0)
        assert mult == 0.0

    # --- Extras-tier (10% ≤ intraday < 20%) ---

    def test_extras_tier_strong_pos(self, r, bars, monkeypatch):
        """Extras + macd > +0.5% → 2.0× (extras_tier override)."""
        monkeypatch.setattr('trading.indicators.macd_histogram',
                            lambda _: _mock_hist(+1.0))
        mult = r._get_macd_zone_multiplier(
            'X', bars, len(bars) - 1, 10.0, intraday_change_pct=15.0)
        assert mult == pytest.approx(r.macd_extras_strong_pos_multiplier)
        assert mult == pytest.approx(2.0)

    def test_extras_tier_strong_neg(self, r, bars, monkeypatch):
        monkeypatch.setattr('trading.indicators.macd_histogram',
                            lambda _: _mock_hist(-1.0))
        mult = r._get_macd_zone_multiplier(
            'X', bars, len(bars) - 1, 10.0, intraday_change_pct=15.0)
        assert mult == pytest.approx(r.macd_extras_strong_neg_multiplier)
        assert mult == pytest.approx(2.0)

    def test_extras_tier_normal_is_zero(self, r, bars, monkeypatch):
        """Extras + macd normal zone → 0.0× (SKIP the landmine bucket)."""
        monkeypatch.setattr('trading.indicators.macd_histogram',
                            lambda _: _mock_hist(+0.3))
        mult = r._get_macd_zone_multiplier(
            'X', bars, len(bars) - 1, 10.0, intraday_change_pct=15.0)
        assert mult == pytest.approx(r.macd_extras_normal_multiplier)
        assert mult == 0.0

    def test_extras_tier_dead_zone(self, r, bars, monkeypatch):
        """Extras + dead zone → 0.0× (same as baseline)."""
        monkeypatch.setattr('trading.indicators.macd_histogram',
                            lambda _: _mock_hist(+0.0))
        mult = r._get_macd_zone_multiplier(
            'X', bars, len(bars) - 1, 10.0, intraday_change_pct=15.0)
        assert mult == 0.0

    # --- edge-tier (intraday < 10%) falls back to A-tier ---

    def test_edge_tier_falls_back_to_a_tier(self, r, bars, monkeypatch):
        monkeypatch.setattr('trading.indicators.macd_histogram',
                            lambda _: _mock_hist(+1.0))
        mult = r._get_macd_zone_multiplier(
            'X', bars, len(bars) - 1, 10.0, intraday_change_pct=5.0)
        assert mult == pytest.approx(r.macd_strong_pos_multiplier)
        assert mult == pytest.approx(1.8)

    # --- tier boundaries ---

    def test_boundary_intraday_20_is_a_tier(self, r, bars, monkeypatch):
        """intraday = 20.0 → A-tier (inclusive lower)."""
        assert classify_tier(20.0) == TIER_A
        monkeypatch.setattr('trading.indicators.macd_histogram',
                            lambda _: _mock_hist(+1.0))
        mult = r._get_macd_zone_multiplier(
            'X', bars, len(bars) - 1, 10.0, intraday_change_pct=20.0)
        assert mult == pytest.approx(r.macd_strong_pos_multiplier)

    def test_boundary_intraday_10_is_extras(self, r, bars, monkeypatch):
        """intraday = 10.0 → Extras (inclusive lower)."""
        assert classify_tier(10.0) == TIER_EXTRAS
        monkeypatch.setattr('trading.indicators.macd_histogram',
                            lambda _: _mock_hist(+1.0))
        mult = r._get_macd_zone_multiplier(
            'X', bars, len(bars) - 1, 10.0, intraday_change_pct=10.0)
        assert mult == pytest.approx(r.macd_extras_strong_pos_multiplier)

    def test_boundary_just_below_10_is_edge(self):
        assert classify_tier(9.99) == TIER_EDGE

    def test_default_intraday_0_is_edge_fallback(self, r, bars, monkeypatch):
        """intraday defaults to 0.0 (unknown) → edge → A-tier multipliers."""
        monkeypatch.setattr('trading.indicators.macd_histogram',
                            lambda _: _mock_hist(+1.0))
        mult = r._get_macd_zone_multiplier('X', bars, len(bars) - 1, 10.0)
        assert mult == pytest.approx(r.macd_strong_pos_multiplier)


# -------------------------------------------------------------------------
# Config loading
# -------------------------------------------------------------------------

class TestConfigLoading:
    def test_bt_loads_extras_tier_from_config(self):
        r = BacktestRunner()
        assert r.macd_extras_strong_pos_multiplier == 2.0
        assert r.macd_extras_strong_neg_multiplier == 2.0
        assert r.macd_extras_normal_multiplier == 0.0

    def test_bt_loads_a_tier_defaults_from_config(self):
        r = BacktestRunner()
        assert r.macd_strong_pos_multiplier == 1.8
        assert r.macd_strong_neg_multiplier == 1.8
        assert r.macd_normal_multiplier == 1.0

    def test_bt_loads_v_rev_bonus_1_0(self):
        r = BacktestRunner()
        assert r.v_reversal_bonus == 1.0
        assert r.v_reversal_enabled is True


# -------------------------------------------------------------------------
# Legacy sweet-spot + audit-fix removal verification
# -------------------------------------------------------------------------

class TestCleanupVerification:
    def test_sweet_spot_attrs_removed(self):
        r = BacktestRunner()
        assert not hasattr(r, 'macd_sweet_spot_min')
        assert not hasattr(r, 'macd_sweet_spot_max')
        assert not hasattr(r, 'macd_sweet_spot_multiplier')

    def test_audit_fix_attrs_removed(self):
        r = BacktestRunner()
        assert not hasattr(r, 'conviction_audit_fix_enabled')

    def test_conviction_function_no_audit_fix_kwarg(self):
        import inspect
        from backtest import BacktestRunner
        sig = inspect.signature(BacktestRunner._compute_conviction_score_setup)
        assert 'audit_fix_enabled' not in sig.parameters

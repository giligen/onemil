"""Phase 1d / 2a / 2b unit tests for macd_wave_backtest.

Covers:
  - apply_entry_slippage / apply_exit_slippage helpers (Phase 2a).
  - filter_signals's per-row slippage recompute from raw_*_price (Phase 2b).
  - Conviction-sizing default behavior + cache-not-double-scaled invariant
    (Phase 1d).

These pin the parity contract: the cache stores raw outcomes; runtime
filters and slippage are applied at filter time. That lets the same cache
serve conviction-on, conviction-off, and a slippage sweep without rebuild.
"""
from __future__ import annotations

import pytest

from macd_wave_backtest import (
    DEFAULT_ENTRY_MIN_CENTS,
    DEFAULT_EXIT_MIN_CENTS,
    apply_entry_slippage,
    apply_exit_slippage,
    filter_signals,
)


# =========================================================================
# Helpers
# =========================================================================

class TestApplyEntrySlippage:

    def test_pct_dominates_on_expensive_stock(self):
        # $50 stock, 30bps = $0.15 → dominates over $0.03 floor
        out = apply_entry_slippage(50.0, 0.003, min_cents=0.03)
        assert out == pytest.approx(50.15)

    def test_floor_dominates_on_cheap_stock(self):
        # $5 stock, 30bps = $0.015 → floor $0.03 dominates
        out = apply_entry_slippage(5.0, 0.003, min_cents=0.03)
        assert out == pytest.approx(5.03)

    def test_zero_slippage_with_zero_floor(self):
        out = apply_entry_slippage(10.0, 0.0, min_cents=0.0)
        assert out == 10.0

    def test_zero_pct_still_floor(self):
        out = apply_entry_slippage(10.0, 0.0, min_cents=0.05)
        assert out == pytest.approx(10.05)

    def test_returns_raw_on_zero_price(self):
        assert apply_entry_slippage(0.0, 0.003, 0.03) == 0.0

    def test_returns_raw_on_negative_price(self):
        # Degenerate input must not amplify into a more-degenerate output.
        assert apply_entry_slippage(-1.0, 0.003, 0.03) == -1.0

    def test_default_min_cents(self):
        out = apply_entry_slippage(5.0, 0.003)
        assert out == pytest.approx(5.0 + DEFAULT_ENTRY_MIN_CENTS)


class TestApplyExitSlippage:

    def test_pct_dominates_on_expensive_stock(self):
        out = apply_exit_slippage(50.0, 0.003, min_cents=0.03)
        assert out == pytest.approx(49.85)

    def test_floor_dominates_on_cheap_stock(self):
        out = apply_exit_slippage(5.0, 0.003, min_cents=0.03)
        assert out == pytest.approx(4.97)

    def test_never_goes_negative(self):
        # Pathological: slippage > price. Floor at $0.01.
        out = apply_exit_slippage(0.02, 0.5, min_cents=10.0)
        assert out == 0.01

    def test_returns_raw_on_zero_price(self):
        assert apply_exit_slippage(0.0, 0.003, 0.03) == 0.0

    def test_default_min_cents(self):
        out = apply_exit_slippage(5.0, 0.003)
        assert out == pytest.approx(5.0 - DEFAULT_EXIT_MIN_CENTS)


# =========================================================================
# filter_signals slippage recompute
# =========================================================================

def _sig(**overrides):
    """Minimal signal row good enough to pass filter_signals's filters."""
    base = {
        'symbol': 'TEST', 'date': '2026-04-01',
        'wave': 1, 'paper': False,
        'entry_price': 10.0, 'exit_price': 10.5,
        'raw_entry_price': 10.0, 'raw_exit_price': 10.5,
        'shares': 500, 'pnl_pct': 5.0, 'pnl_dollar': 250.0,
        'entry_time': '2026-04-01 09:35', 'exit_time': '2026-04-01 10:30',
        'exit_reason': 'trail', 'cross_time_min': 1,
        'vol_at_cross': 100_000, 'macd_hist_pct': 1.0,
        'w1_pnl': 0.0, 'conv_mult': 1.0,
    }
    base.update(overrides)
    return base


class TestSlippageRecompute:

    def _filters(self, **kw):
        f = {
            'cross_time_max_min': 0,
            'min_vol_at_cross': 0,
            'max_vol_at_cross': 0,
            'min_macd_hist_pct': 0.0,
            'max_price_at_entry': 0,
            'last_entry_minutes_after_open': 0,
            'recompute_slippage': True,
            'entry_pct': 0.003,
            'exit_pct': 0.003,
            'entry_min_cents': 0.03,
            'exit_min_cents': 0.03,
            'conviction_sizing': False,
        }
        f.update(kw)
        return f

    def test_recompute_applies_current_slippage_to_raw_prices(self):
        s = _sig(raw_entry_price=10.0, raw_exit_price=10.5,
                 entry_price=99.9, exit_price=99.9,  # poisoned baked values
                 shares=500, pnl_dollar=0.0)
        out = filter_signals([s], self._filters())
        assert len(out) == 1
        r = out[0]
        # 10.0 + max(0.003*10, 0.03) = 10.0 + 0.03 = 10.03
        assert r['entry_price'] == pytest.approx(10.03)
        # 10.5 - max(0.003*10.5, 0.03) = 10.5 - 0.0315 = 10.4685
        assert r['exit_price'] == pytest.approx(10.4685)
        # pnl_dollar recomputed from corrected fills
        assert r['pnl_dollar'] == pytest.approx((10.4685 - 10.03) * 500)

    def test_falls_back_to_baked_prices_when_raw_missing(self):
        # Legacy cache rows: no raw_* fields.
        s = _sig()
        s.pop('raw_entry_price')
        s.pop('raw_exit_price')
        out = filter_signals([s], self._filters())
        r = out[0]
        # Untouched
        assert r['entry_price'] == 10.0
        assert r['exit_price'] == 10.5
        assert r['pnl_dollar'] == 250.0

    def test_does_not_mutate_input_dict(self):
        # Critical: cache rows are reused across runs in batch mode. Mutating
        # them would corrupt subsequent calls.
        s = _sig(raw_entry_price=10.0, raw_exit_price=10.5,
                 entry_price=10.0, exit_price=10.5, pnl_dollar=250.0)
        before_entry = s['entry_price']
        before_pnl = s['pnl_dollar']
        _ = filter_signals([s], self._filters())
        assert s['entry_price'] == before_entry
        assert s['pnl_dollar'] == before_pnl

    def test_recompute_disabled_by_flag(self):
        s = _sig(raw_entry_price=10.0, raw_exit_price=10.5,
                 entry_price=99.0, exit_price=88.0, pnl_dollar=42.0)
        out = filter_signals([s], self._filters(recompute_slippage=False))
        r = out[0]
        # Baked values pass through unchanged
        assert r['entry_price'] == 99.0
        assert r['exit_price'] == 88.0
        assert r['pnl_dollar'] == 42.0

    def test_zero_slippage_round_trip_equals_raw(self):
        s = _sig(raw_entry_price=10.0, raw_exit_price=10.5,
                 shares=500, pnl_dollar=999.0)
        out = filter_signals(
            [s], self._filters(entry_pct=0.0, exit_pct=0.0,
                                entry_min_cents=0.0, exit_min_cents=0.0),
        )
        r = out[0]
        assert r['entry_price'] == 10.0
        assert r['exit_price'] == 10.5
        assert r['pnl_dollar'] == pytest.approx(0.5 * 500)


# =========================================================================
# Conviction sizing
# =========================================================================

class TestConvictionSizing:

    def _filters(self, conviction_sizing):
        return {
            'cross_time_max_min': 0,
            'min_vol_at_cross': 0,
            'max_vol_at_cross': 0,
            'min_macd_hist_pct': 0.0,
            'max_price_at_entry': 0,
            'last_entry_minutes_after_open': 0,
            'recompute_slippage': False,
            'conviction_sizing': conviction_sizing,
        }

    def test_off_leaves_shares_unchanged(self):
        s = _sig(shares=500, pnl_dollar=250.0, conv_mult=1.8)
        out = filter_signals([s], self._filters(False))
        assert out[0]['shares'] == 500
        assert out[0]['pnl_dollar'] == 250.0

    def test_on_scales_shares_and_pnl(self):
        s = _sig(shares=500, pnl_dollar=250.0, conv_mult=1.8)
        out = filter_signals([s], self._filters(True))
        assert out[0]['shares'] == int(500 * 1.8)
        assert out[0]['pnl_dollar'] == pytest.approx(250.0 * 1.8)

    def test_conv_mult_one_is_noop(self):
        s = _sig(shares=500, pnl_dollar=250.0, conv_mult=1.0)
        out = filter_signals([s], self._filters(True))
        assert out[0]['shares'] == 500
        assert out[0]['pnl_dollar'] == 250.0

    def test_string_conv_mult_parsed(self):
        # Cache rows arrive from CSV: conv_mult is a string.
        s = _sig(shares=500, pnl_dollar=250.0, conv_mult='1.8')
        out = filter_signals([s], self._filters(True))
        assert out[0]['shares'] == int(500 * 1.8)

    def test_missing_conv_mult_treated_as_one(self):
        s = _sig(shares=500, pnl_dollar=250.0)
        s.pop('conv_mult')
        out = filter_signals([s], self._filters(True))
        assert out[0]['shares'] == 500

    def test_does_not_mutate_input(self):
        s = _sig(shares=500, pnl_dollar=250.0, conv_mult=1.8)
        original_shares = s['shares']
        _ = filter_signals([s], self._filters(True))
        assert s['shares'] == original_shares

    def test_pnl_pct_unchanged_by_sizing(self):
        # pnl_pct is a ratio — sizing scales shares + dollars but not pct.
        s = _sig(shares=500, pnl_dollar=250.0, pnl_pct=5.0, conv_mult=1.8)
        out = filter_signals([s], self._filters(True))
        assert out[0]['pnl_pct'] == 5.0


# =========================================================================
# End-to-end: slippage + conviction together
# =========================================================================

class TestRecomputeAndConvictionStack:

    def _filters(self, **kw):
        f = {
            'cross_time_max_min': 0,
            'min_vol_at_cross': 0,
            'max_vol_at_cross': 0,
            'min_macd_hist_pct': 0.0,
            'max_price_at_entry': 0,
            'last_entry_minutes_after_open': 0,
            'recompute_slippage': True,
            'entry_pct': 0.003,
            'exit_pct': 0.003,
            'entry_min_cents': 0.03,
            'exit_min_cents': 0.03,
            'conviction_sizing': True,
        }
        f.update(kw)
        return f

    def test_stack_applies_in_correct_order(self):
        # raw_entry=10, raw_exit=10.5, shares=500, conv_mult=1.8
        # After slippage recompute: entry=10.03, exit=10.4685, shares=500
        #   pnl_dollar = (10.4685 - 10.03) * 500 = 219.25
        # After conviction scale: shares=900, pnl_dollar=219.25*1.8=394.65
        s = _sig(raw_entry_price=10.0, raw_exit_price=10.5,
                 shares=500, conv_mult=1.8)
        out = filter_signals([s], self._filters())
        r = out[0]
        assert r['entry_price'] == pytest.approx(10.03)
        assert r['exit_price'] == pytest.approx(10.4685)
        assert r['shares'] == 900
        assert r['pnl_dollar'] == pytest.approx((10.4685 - 10.03) * 500 * 1.8)

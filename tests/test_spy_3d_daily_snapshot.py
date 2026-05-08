"""Tests for the SPY 3d daily-snapshot fix (IREZ post-mortem 2026-05-08).

Pinned regression: `_get_spy_3d_range_live` MUST use bars STRICTLY BEFORE
today (T-1, T-2, T-3), matching BT's `_get_spy_3d_range` (`WHERE bar_date <
trade_date`). Today's intraday-updating partial daily bar must be excluded
so the value is stable across the entire trading day — which is the parity
guarantee that prevents another IREZ-style boundary-noise kill.

Before the fix:
  conviction time @ 14:26:00 — _get_spy_3d_range_live() returned 0.80
  post-fill check @ 14:26:10 — _get_spy_3d_range_live() returned 0.77
  Same 3-day metric, 9 seconds apart → kill switch fired.

After the fix:
  Both calls return the same T-1/T-2/T-3 mean regardless of how today's
  partial bar drifts.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from trading.market_regime import MarketRegimeFilter


def _make_bars():
    """Five days of SPY bars: T-3, T-2, T-1, T (today partial), T+1 unused.

    Today's bar has an unusually large range (1.5%) that would inflate
    a "drifting" 3-day mean to ~1.0%. T-1/T-2/T-3 are all ~0.5%, so the
    correct snapshot value is ~0.5%.
    """
    return [
        {'date': date(2026, 5, 5), 'open': 500.0, 'high': 502.5, 'low': 500.0, 'close': 501.0, 'volume': 1_000_000},  # T-3, range 0.5%
        {'date': date(2026, 5, 6), 'open': 501.0, 'high': 503.5, 'low': 501.0, 'close': 502.0, 'volume': 1_000_000},  # T-2, range 0.499%
        {'date': date(2026, 5, 7), 'open': 502.0, 'high': 504.5, 'low': 502.0, 'close': 503.0, 'volume': 1_000_000},  # T-1, range 0.498%
        {'date': date(2026, 5, 8), 'open': 503.0, 'high': 510.5, 'low': 503.0, 'close': 510.0, 'volume': 1_000_000},  # T (today), range 1.491%
    ]


def _build_engine(bars):
    """Build a TradingEngine with just enough plumbing to call _get_spy_3d_range_live.

    Bypasses __init__ to avoid the full config + alpaca + DB stack.
    """
    from trading.trading_engine import TradingEngine
    engine = TradingEngine.__new__(TradingEngine)
    engine.market_regime = MarketRegimeFilter(vol_threshold=5.0, sma_period=50)
    engine.market_regime.load_spy_bars(bars)
    engine._spy_bars_cache = None  # disables fallback path 2
    return engine


class TestExcludesToday:
    """The fix: today's partial bar must NOT contribute to the 3-day mean."""

    def test_today_excluded_when_today_in_data(self, monkeypatch):
        bars = _make_bars()
        engine = _build_engine(bars)

        # Pin "today" to 2026-05-08 — the day matching the largest bar in fixture.
        # If the function uses today's bar, mean would include 1.49% and pull up.
        # If it correctly excludes today, mean is over T-3/T-2/T-1 (~0.5%).
        import trading.trading_engine as te
        monkeypatch.setattr(te, 'date', _FrozenDate(date(2026, 5, 8)))

        result = engine._get_spy_3d_range_live()
        assert result is not None
        # Expected: mean of three ~0.5% ranges. If function leaks today's bar
        # (1.49%), result would be ~0.83% — fails this assertion clearly.
        assert 0.45 < result < 0.55, (
            f"Today's bar leaked into 3-day mean (got {result:.3f}, "
            f"expected ~0.5). The fix uses before_date=today to clamp."
        )

    def test_stable_when_today_drifts(self, monkeypatch):
        """Simulating IREZ: today's bar updates intraday — 3d mean must NOT change."""
        bars = _make_bars()
        engine = _build_engine(bars)

        import trading.trading_engine as te
        monkeypatch.setattr(te, 'date', _FrozenDate(date(2026, 5, 8)))

        # Snapshot 1: today's bar has range 1.49%
        first = engine._get_spy_3d_range_live()

        # Mutate today's bar to a much larger range (simulating intraday expansion).
        engine.market_regime._bars_by_date[date(2026, 5, 8)]['high'] = 525.0
        engine.market_regime._bars_by_date[date(2026, 5, 8)]['low'] = 495.0

        # Snapshot 2: today's bar now has range 6%
        second = engine._get_spy_3d_range_live()

        assert first == pytest.approx(second), (
            f"3-day mean drifted intraday: first={first:.4f}, second={second:.4f}. "
            f"This was the IREZ root cause — the fix excludes today's bar entirely."
        )


class TestPathStability:
    def test_returns_none_when_no_prior_bars(self, monkeypatch):
        """If only today's bar exists, no T-1/T-2/T-3 → None (regime unknown)."""
        bars = [{
            'date': date(2026, 5, 8), 'open': 500.0, 'high': 510.0,
            'low': 500.0, 'close': 510.0, 'volume': 1_000_000,
        }]
        engine = _build_engine(bars)

        import trading.trading_engine as te
        monkeypatch.setattr(te, 'date', _FrozenDate(date(2026, 5, 8)))

        result = engine._get_spy_3d_range_live()
        assert result is None, (
            "With only today's bar, function MUST return None so callers "
            "treat it as 'regime unknown' (-0.5 conviction penalty), not "
            "fall back to today's intraday range (which the EAF post-mortem "
            "ruled out)."
        )

    def test_uses_t1_t2_t3_when_today_missing(self, monkeypatch):
        """When today's bar isn't loaded, function still uses last 3 prior bars."""
        bars = _make_bars()[:-1]  # drop today
        engine = _build_engine(bars)

        import trading.trading_engine as te
        monkeypatch.setattr(te, 'date', _FrozenDate(date(2026, 5, 8)))

        result = engine._get_spy_3d_range_live()
        assert result is not None
        assert 0.45 < result < 0.55


class TestBtParityConstruction:
    """Live's snapshot must match BT's WHERE bar_date < trade_date filter."""

    def test_live_matches_bt_compute_helper(self, monkeypatch):
        """Live and BT both call compute_spy_3d_range on the same bars."""
        from trading.spy_regime import compute_spy_3d_range
        bars = _make_bars()
        engine = _build_engine(bars)

        import trading.trading_engine as te
        monkeypatch.setattr(te, 'date', _FrozenDate(date(2026, 5, 8)))

        live_result = engine._get_spy_3d_range_live()

        # BT-equivalent: take the 3 bars strictly before today
        prior_bars = [b for b in bars if b['date'] < date(2026, 5, 8)]
        bt_result = compute_spy_3d_range(prior_bars[-3:])

        assert live_result == pytest.approx(bt_result), (
            f"Live ({live_result}) must equal BT-helper ({bt_result}) "
            f"on identical bar inputs — this is the parity contract."
        )


# ---------------------------------------------------------------------------
# Helper to monkey-patch `date.today()` predictably without touching real time.
# ---------------------------------------------------------------------------


class _FrozenDate:
    """Stand-in for `datetime.date` that returns a fixed `today()`.

    Used to pin "today" in tests so we can control which bar the
    function-under-test treats as the partial intraday bar.
    """

    def __init__(self, today: date):
        self._today = today

    def today(self):
        return self._today

    # Pass-through other date constructors — lets the rest of the module's
    # date(...) calls work normally.
    def __call__(self, *args, **kwargs):
        return date(*args, **kwargs)


# ---------------------------------------------------------------------------
# Post-fill gate config thresholds (V1 ship)
# ---------------------------------------------------------------------------


class TestPostFillGateConfig:
    """Thresholds and enable flag come from config — defaults are 0.5/0.5/True."""

    def test_default_thresholds_are_v1(self):
        """Without explicit config, post_fill_gate_cfg returns 0.5/0.5/enabled."""
        from config import Config
        cfg = Config().post_fill_gate_cfg
        assert cfg['enabled'] is True, "Gate should default to enabled"
        assert cfg['spy_3d_threshold'] == 0.5, (
            f"Default SPY threshold should be 0.5 (V1), got {cfg['spy_3d_threshold']}"
        )
        assert cfg['bk_ratio_threshold'] == 0.5, (
            f"Default bk_ratio threshold should be 0.5 (V1), got {cfg['bk_ratio_threshold']}"
        )

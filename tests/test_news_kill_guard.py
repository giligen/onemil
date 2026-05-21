"""
Tests for the shared news-kill decision (trading/news_kill_guard.py) and its
integration into BacktestRunner._check_news_kill.

news_kill_decision is the single source of truth for the news-kill segment
gate — used by both the backtest and the live engine, so they cannot drift.
"""
import types

import pandas as pd
import pytest

from trading.news_kill_guard import news_kill_decision

# Default segment thresholds (config news_kill_rules.* defaults).
MAXV, MINP, MAXF = 3_000_000.0, 3.0, 30_000_000.0


def decide(has_catalyst=False, catalyst_exemption=False, avg_vol=500_000,
           entry_price=8.0, float_shares=5_000_000, pole_gain=5.0):
    """news_kill_decision with in-bounds (passing) defaults; override per test."""
    return news_kill_decision(
        has_catalyst=has_catalyst, catalyst_exemption=catalyst_exemption,
        avg_vol=avg_vol, entry_price=entry_price, float_shares=float_shares,
        pole_gain=pole_gain, max_avg_vol=MAXV, min_price=MINP, max_float=MAXF)


# ============================================================================
# Catalyst exemption ON (legacy behavior — rollback path)
# ============================================================================

class TestExemptionOn:
    def test_catalyst_exempts_from_segment_rules(self):
        """has_catalyst + exemption on → trades even in a bad segment."""
        ok, reason = decide(has_catalyst=True, catalyst_exemption=True,
                            avg_vol=10_000_000)
        assert ok is True and reason == "has_catalyst"

    def test_no_catalyst_still_killed(self):
        ok, reason = decide(has_catalyst=False, catalyst_exemption=True,
                            avg_vol=10_000_000)
        assert ok is False and "avg_vol" in reason

    def test_catalyst_irrelevant_in_good_segment(self):
        ok, reason = decide(has_catalyst=False, catalyst_exemption=True)
        assert ok is True and reason == "good_segment"


# ============================================================================
# Catalyst exemption OFF (shipped default 2026-05-21)
# ============================================================================

class TestExemptionOff:
    """Segment rules apply to EVERY trade — a catalyst no longer exempts."""

    def test_catalyst_does_not_exempt_bad_segment(self):
        """KEY new behavior: a real catalyst no longer saves a bad-segment trade."""
        ok, reason = decide(has_catalyst=True, catalyst_exemption=False,
                            avg_vol=10_000_000)
        assert ok is False and "avg_vol" in reason

    def test_catalyst_good_segment_still_trades(self):
        ok, reason = decide(has_catalyst=True, catalyst_exemption=False)
        assert ok is True and reason == "good_segment"

    def test_no_catalyst_bad_segment_killed(self):
        ok, reason = decide(has_catalyst=False, catalyst_exemption=False,
                            entry_price=2.0)
        assert ok is False and "price" in reason


# ============================================================================
# Segment rules
# ============================================================================

class TestSegmentRules:
    def test_rule1_high_volume_killed(self):
        ok, reason = decide(avg_vol=3_000_000)        # >= threshold
        assert ok is False and "avg_vol" in reason

    def test_rule1_just_under_threshold_passes(self):
        ok, _ = decide(avg_vol=2_999_999)
        assert ok is True

    def test_rule2_low_price_killed(self):
        ok, reason = decide(entry_price=2.99)
        assert ok is False and "price" in reason

    def test_rule2_exactly_min_price_passes(self):
        """price < min is strict — exactly $3.00 is not killed by R2."""
        ok, _ = decide(entry_price=3.0)
        assert ok is True

    def test_rule3_high_float_killed(self):
        ok, reason = decide(float_shares=30_000_000)
        assert ok is False and "float" in reason

    def test_rule3_just_under_threshold_passes(self):
        ok, _ = decide(float_shares=29_999_999)
        assert ok is True

    def test_rule4_overextended_midcap_killed(self):
        ok, reason = decide(entry_price=8.0, pole_gain=10.0)
        assert ok is False and "pole" in reason

    def test_rule4_pole_too_low_passes(self):
        ok, _ = decide(entry_price=8.0, pole_gain=7.9)
        assert ok is True

    def test_rule4_price_above_band_passes(self):
        ok, _ = decide(entry_price=12.0, pole_gain=10.0)
        assert ok is True

    def test_good_segment_passes(self):
        ok, reason = decide()
        assert ok is True and reason == "good_segment"

    def test_rule_precedence_volume_before_price(self):
        """A trade matching R1 and R2 is attributed to R1 (checked first)."""
        ok, reason = decide(avg_vol=5_000_000, entry_price=1.0)
        assert ok is False and "avg_vol" in reason


# ============================================================================
# BacktestRunner._check_news_kill integration
# ============================================================================

class TestBacktestNewsKillIntegration:
    @staticmethod
    def _bars():
        return pd.DataFrame([{'timestamp': '2026-01-15 09:35:00'}])

    @staticmethod
    def _runner():
        from backtest import BacktestRunner
        return BacktestRunner()

    def test_disabled_always_trades(self):
        r = self._runner()
        r.news_kill_enabled = False
        ok, _ = r._check_news_kill(
            'X', self._bars(), types.SimpleNamespace(pole_gain_pct=10.0),
            types.SimpleNamespace(entry_price=8.0), avg_daily_volume=10_000_000)
        assert ok is True

    def test_exemption_off_kills_bad_segment(self):
        r = self._runner()
        r.news_kill_enabled = True
        r.news_kill_catalyst_exemption = False
        ok, reason = r._check_news_kill(
            'X', self._bars(), types.SimpleNamespace(pole_gain_pct=5.0),
            types.SimpleNamespace(entry_price=8.0), avg_daily_volume=10_000_000)
        assert ok is False and "avg_vol" in reason

    def test_exemption_off_skips_catalyst_lookup(self, monkeypatch):
        """With the exemption off, the news classifier must NOT be consulted."""
        r = self._runner()
        r.news_kill_enabled = True
        r.news_kill_catalyst_exemption = False
        called = []
        monkeypatch.setattr(r, '_has_real_catalyst',
                            lambda *a, **k: called.append(1) or True)
        r._check_news_kill(
            'X', self._bars(), types.SimpleNamespace(pole_gain_pct=5.0),
            types.SimpleNamespace(entry_price=8.0), avg_daily_volume=500_000)
        assert called == []   # short-circuited — no news lookup

    def test_exemption_on_catalyst_saves_trade(self, monkeypatch):
        r = self._runner()
        r.news_kill_enabled = True
        r.news_kill_catalyst_exemption = True
        monkeypatch.setattr(r, '_has_real_catalyst', lambda *a, **k: True)
        ok, reason = r._check_news_kill(
            'X', self._bars(), types.SimpleNamespace(pole_gain_pct=5.0),
            types.SimpleNamespace(entry_price=8.0), avg_daily_volume=10_000_000)
        assert ok is True and reason == "has_catalyst"

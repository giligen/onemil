"""
Unit tests for trading.exhaustion_signals shared module.

Tests each signal detector in isolation and the check_exhaustion dispatcher.
"""

import pandas as pd
import pytest

from trading.exhaustion_signals import (
    check_exhaustion,
    sig_climax_candle,
    sig_shooting_star,
    sig_shrinking_bodies,
    sig_volume_divergence,
)


def _make_bars(rows):
    """Helper to build a bars DataFrame from list of dicts."""
    return pd.DataFrame(rows)


class TestSigClimaxCandle:
    """Test climax candle detection."""

    def test_fires_when_body_and_volume_exceed_threshold(self):
        """Climax: body and volume both >2× average of prior 5 bars."""
        bars = _make_bars([
            {'open': 5.0, 'close': 5.1, 'high': 5.2, 'low': 4.9, 'volume': 1000},
            {'open': 5.1, 'close': 5.2, 'high': 5.3, 'low': 5.0, 'volume': 1000},
            {'open': 5.2, 'close': 5.3, 'high': 5.4, 'low': 5.1, 'volume': 1000},
            {'open': 5.3, 'close': 5.4, 'high': 5.5, 'low': 5.2, 'volume': 1000},
            {'open': 5.4, 'close': 5.5, 'high': 5.6, 'low': 5.3, 'volume': 1000},
            # Climax: body 0.5 (5× avg 0.1), volume 3000 (3× avg 1000)
            {'open': 5.5, 'close': 6.0, 'high': 6.1, 'low': 5.4, 'volume': 3000},
        ])
        assert sig_climax_candle(bars, 5) == True

    def test_no_fire_normal_bar(self):
        """Normal bar doesn't trigger."""
        bars = _make_bars([
            {'open': 5.0, 'close': 5.1, 'high': 5.2, 'low': 4.9, 'volume': 1000},
            {'open': 5.1, 'close': 5.2, 'high': 5.3, 'low': 5.0, 'volume': 1000},
            {'open': 5.2, 'close': 5.3, 'high': 5.4, 'low': 5.1, 'volume': 1000},
            {'open': 5.3, 'close': 5.4, 'high': 5.5, 'low': 5.2, 'volume': 1000},
            {'open': 5.4, 'close': 5.5, 'high': 5.6, 'low': 5.3, 'volume': 1000},
            {'open': 5.5, 'close': 5.6, 'high': 5.7, 'low': 5.4, 'volume': 1000},
        ])
        assert sig_climax_candle(bars, 5) == False

    def test_insufficient_lookback(self):
        """Not enough bars → False."""
        bars = _make_bars([
            {'open': 5.0, 'close': 6.0, 'high': 6.1, 'low': 4.9, 'volume': 5000},
        ])
        assert sig_climax_candle(bars, 0) == False


class TestSigShootingStar:
    """Test shooting star detection."""

    def test_fires_on_shooting_star(self):
        """Long upper wick, close near low → shooting star."""
        bars = _make_bars([
            # open=5.0, close=5.05 (body=0.05), high=5.20 (wick=0.15, 3× body)
            # low=5.0, close_pos = (5.05-5.0)/(5.20-5.0) = 0.25 < 0.4
            {'open': 5.0, 'close': 5.05, 'high': 5.20, 'low': 5.0, 'volume': 1000},
        ])
        assert sig_shooting_star(bars, 0) == True

    def test_no_fire_bullish_candle(self):
        """Strong bullish candle → no shooting star."""
        bars = _make_bars([
            {'open': 5.0, 'close': 5.5, 'high': 5.55, 'low': 4.95, 'volume': 1000},
        ])
        assert sig_shooting_star(bars, 0) == False

    def test_no_fire_tiny_body(self):
        """Doji (body ≤ 0.001) → False."""
        bars = _make_bars([
            {'open': 5.0, 'close': 5.0005, 'high': 5.5, 'low': 4.9, 'volume': 1000},
        ])
        assert sig_shooting_star(bars, 0) == False


class TestSigVolumeDivergence:
    """Test volume divergence detection."""

    def test_fires_on_declining_volume_rising_highs(self):
        """3 bars of declining volume with higher highs."""
        bars = _make_bars([
            {'open': 5.0, 'close': 5.1, 'high': 5.2, 'low': 4.9, 'volume': 4000},
            {'open': 5.1, 'close': 5.2, 'high': 5.3, 'low': 5.0, 'volume': 3000},
            {'open': 5.2, 'close': 5.3, 'high': 5.4, 'low': 5.1, 'volume': 2000},
            {'open': 5.3, 'close': 5.4, 'high': 5.5, 'low': 5.2, 'volume': 1000},
        ])
        assert sig_volume_divergence(bars, 3) == True

    def test_no_fire_volume_increasing(self):
        """Volume increasing → no divergence."""
        bars = _make_bars([
            {'open': 5.0, 'close': 5.1, 'high': 5.2, 'low': 4.9, 'volume': 1000},
            {'open': 5.1, 'close': 5.2, 'high': 5.3, 'low': 5.0, 'volume': 2000},
            {'open': 5.2, 'close': 5.3, 'high': 5.4, 'low': 5.1, 'volume': 3000},
            {'open': 5.3, 'close': 5.4, 'high': 5.5, 'low': 5.2, 'volume': 4000},
        ])
        assert sig_volume_divergence(bars, 3) == False


class TestSigShrinkingBodies:
    """Test shrinking bodies detection."""

    def test_fires_on_shrinking_body_near_highs(self):
        """Current body < 50% of body 3 bars ago, price still near highs."""
        bars = _make_bars([
            {'open': 5.0, 'close': 5.5, 'high': 5.6, 'low': 4.9, 'volume': 1000},  # body=0.5
            {'open': 5.5, 'close': 5.6, 'high': 5.7, 'low': 5.4, 'volume': 1000},
            {'open': 5.6, 'close': 5.65, 'high': 5.7, 'low': 5.5, 'volume': 1000},
            # body=0.05 < 0.5 * 0.5 = 0.25, close 5.55 > 5.5 (3 bars ago close)
            {'open': 5.6, 'close': 5.55, 'high': 5.65, 'low': 5.5, 'volume': 1000},
        ])
        assert sig_shrinking_bodies(bars, 3) == True

    def test_no_fire_body_not_shrunk(self):
        """Body not shrunk enough → False."""
        bars = _make_bars([
            {'open': 5.0, 'close': 5.1, 'high': 5.2, 'low': 4.9, 'volume': 1000},
            {'open': 5.1, 'close': 5.2, 'high': 5.3, 'low': 5.0, 'volume': 1000},
            {'open': 5.2, 'close': 5.3, 'high': 5.4, 'low': 5.1, 'volume': 1000},
            {'open': 5.3, 'close': 5.4, 'high': 5.5, 'low': 5.2, 'volume': 1000},
        ])
        assert sig_shrinking_bodies(bars, 3) == False


class TestCheckExhaustion:
    """Test the dispatcher function."""

    def _climax_bars(self):
        """Build bars where the last bar is a climax candle."""
        return _make_bars([
            {'open': 5.0, 'close': 5.1, 'high': 5.2, 'low': 4.9, 'volume': 1000},
            {'open': 5.1, 'close': 5.2, 'high': 5.3, 'low': 5.0, 'volume': 1000},
            {'open': 5.2, 'close': 5.3, 'high': 5.4, 'low': 5.1, 'volume': 1000},
            {'open': 5.3, 'close': 5.4, 'high': 5.5, 'low': 5.2, 'volume': 1000},
            {'open': 5.4, 'close': 5.5, 'high': 5.6, 'low': 5.3, 'volume': 1000},
            {'open': 5.5, 'close': 6.0, 'high': 6.1, 'low': 5.4, 'volume': 3000},
        ])

    def test_dispatcher_fires_enabled_signal(self):
        """Enabled signal fires → True."""
        bars = self._climax_bars()
        signals = {'climax_candle': True, 'shooting_star': False,
                   'volume_divergence': False, 'shrinking_bodies': False}
        assert check_exhaustion(bars, 5, signals) == True

    def test_dispatcher_skips_disabled_signal(self):
        """All signals disabled → False even if pattern matches."""
        bars = self._climax_bars()
        signals = {'climax_candle': False, 'shooting_star': False,
                   'volume_divergence': False, 'shrinking_bodies': False}
        assert check_exhaustion(bars, 5, signals) == False

    def test_dispatcher_empty_signals_dict(self):
        """Empty signals dict → False."""
        bars = self._climax_bars()
        assert check_exhaustion(bars, 5, {}) == False

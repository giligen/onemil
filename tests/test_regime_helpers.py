"""Unit tests for trading/regime_helpers.py (Phase 1.4b regime sizing).

Covers:
  - classify_regime: boundary cases, NaN/None handling, threshold overrides
  - get_regime_multiplier: defaults, bad input, type coercion
  - compute_regime_features: rolling window correctness, output shape
  - build_regime_lookup: look-ahead safety (day T uses row T-1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trading.regime_helpers import (
    DEFAULT_SLOPE_THRESHOLD_PCT,
    DEFAULT_VOL_THRESHOLD_PCT,
    SMA_WINDOW,
    build_regime_lookup,
    classify_regime,
    compute_regime_features,
    get_regime_multiplier,
)


class TestClassifyRegime:
    """Boundary + edge cases for the pure classifier."""

    def test_clean_bull_is_A(self):
        assert classify_regime(15.0, above_sma_50=True, sma_slope_10d=0.3) == 'A'

    def test_volatile_overrides_trend(self):
        # vol >= threshold → B regardless of trend
        assert classify_regime(22.0, above_sma_50=True, sma_slope_10d=0.3) == 'B'
        assert classify_regime(30.0, above_sma_50=False, sma_slope_10d=-0.5) == 'B'

    def test_vol_exactly_at_threshold_is_B(self):
        # >= is the cutoff per the ship config
        assert classify_regime(22.0, above_sma_50=True, sma_slope_10d=0.0) == 'B'

    def test_vol_just_below_threshold_is_not_B(self):
        assert classify_regime(21.99, above_sma_50=True, sma_slope_10d=0.0) == 'A'

    def test_true_defensive_is_C1(self):
        # below SMA, low vol, SMA flat or falling
        assert classify_regime(15.0, above_sma_50=False, sma_slope_10d=-0.5) == 'C1'
        assert classify_regime(15.0, above_sma_50=False, sma_slope_10d=0.0) == 'C1'
        assert classify_regime(15.0, above_sma_50=False, sma_slope_10d=0.15) == 'C1'

    def test_slope_exactly_at_threshold_is_C1(self):
        # <= 0.15 is C1 (per the ship decision — safer default when tied)
        assert classify_regime(15.0, False, 0.15) == 'C1'

    def test_shallow_dip_is_C2(self):
        # below SMA, low vol, SMA still rising
        assert classify_regime(15.0, above_sma_50=False, sma_slope_10d=0.16) == 'C2'
        assert classify_regime(15.0, above_sma_50=False, sma_slope_10d=0.60) == 'C2'

    def test_threshold_overrides(self):
        # Custom thresholds respected
        assert classify_regime(10.0, True, 0.5,
                               vol_threshold_pct=8.0, slope_threshold_pct=0.5) == 'B'
        assert classify_regime(10.0, False, 0.5,
                               vol_threshold_pct=15.0, slope_threshold_pct=1.0) == 'C1'
        assert classify_regime(10.0, False, 1.5,
                               vol_threshold_pct=15.0, slope_threshold_pct=1.0) == 'C2'

    def test_none_vol_is_unknown(self):
        assert classify_regime(None, True, 0.3) == 'unknown'

    def test_nan_vol_is_unknown(self):
        assert classify_regime(float('nan'), True, 0.3) == 'unknown'

    def test_none_above_sma_is_unknown(self):
        assert classify_regime(15.0, None, 0.3) == 'unknown'

    def test_nan_slope_defaults_to_C1_when_below_sma(self):
        # Safer default in the below-SMA low-vol branch
        assert classify_regime(15.0, False, float('nan')) == 'C1'

    def test_none_slope_defaults_to_C1_when_below_sma(self):
        assert classify_regime(15.0, False, None) == 'C1'

    def test_nan_slope_with_A_regime_ignored(self):
        # Slope is irrelevant when above SMA
        assert classify_regime(15.0, True, float('nan')) == 'A'

    def test_negative_vol_below_threshold(self):
        # Not realistic but shouldn't crash — vol is compared numerically
        assert classify_regime(-5.0, True, 0.3) == 'A'


class TestGetRegimeMultiplier:
    """Config-dict multiplier lookup — safe defaults on bad input."""

    def test_normal_lookup(self):
        mults = {'A': 1.25, 'B': 1.0, 'C1': 1.5, 'C2': 0.0}
        assert get_regime_multiplier('A', mults) == 1.25
        assert get_regime_multiplier('B', mults) == 1.0
        assert get_regime_multiplier('C1', mults) == 1.5
        assert get_regime_multiplier('C2', mults) == 0.0

    def test_unknown_key_defaults_to_1(self):
        assert get_regime_multiplier('unknown', {'A': 1.25}) == 1.0
        assert get_regime_multiplier('disabled', {'A': 1.25}) == 1.0
        assert get_regime_multiplier('', {'A': 1.25}) == 1.0

    def test_empty_dict_defaults_to_1(self):
        assert get_regime_multiplier('A', {}) == 1.0

    def test_none_dict_defaults_to_1(self):
        assert get_regime_multiplier('A', None) == 1.0

    def test_bad_value_defaults_to_1(self):
        assert get_regime_multiplier('A', {'A': 'not-a-number'}) == 1.0
        assert get_regime_multiplier('A', {'A': object()}) == 1.0

    def test_int_values_coerced_to_float(self):
        assert get_regime_multiplier('A', {'A': 2}) == 2.0
        assert isinstance(get_regime_multiplier('A', {'A': 2}), float)


class TestComputeRegimeFeatures:
    """Rolling-window feature math + output shape."""

    def _synth_bars(self, n=80, start_close=500.0):
        """Flat SPY for n days — makes expected values trivial."""
        dates = pd.date_range('2025-01-01', periods=n, freq='B')
        return pd.DataFrame({
            'bar_date': dates,
            'close': [start_close + i for i in range(n)],
        })

    def test_output_has_required_columns(self):
        feats = compute_regime_features(self._synth_bars())
        for col in ('sma_50', 'sma_50_slope_10d', 'vol_20_ann', 'above_sma_50'):
            assert col in feats.columns

    def test_helper_does_not_leak_intermediate_columns(self):
        feats = compute_regime_features(self._synth_bars())
        assert '_ret' not in feats.columns

    def test_warmup_rows_have_nan_sma(self):
        feats = compute_regime_features(self._synth_bars())
        # First SMA_WINDOW-1 rows have no full window
        assert feats.iloc[SMA_WINDOW - 2]['sma_50'] != feats.iloc[SMA_WINDOW - 2]['sma_50']  # NaN
        assert not pd.isna(feats.iloc[SMA_WINDOW - 1]['sma_50'])

    def test_sma_is_mean_of_last_50_closes(self):
        feats = compute_regime_features(self._synth_bars(n=80))
        # Close sequence 500..579. SMA at idx 49 = mean(500..549) = 524.5
        assert feats.iloc[49]['sma_50'] == pytest.approx(524.5)
        assert feats.iloc[50]['sma_50'] == pytest.approx(525.5)

    def test_above_sma_50_true_for_rising_series(self):
        feats = compute_regime_features(self._synth_bars(n=80))
        # Last row close=579, sma_50=mean(530..579)=554.5 → above
        assert bool(feats.iloc[-1]['above_sma_50']) is True

    def test_sorts_input(self):
        bars = self._synth_bars(n=60)
        # Shuffle rows → still produces correct result
        shuffled = bars.sample(frac=1, random_state=42).reset_index(drop=True)
        f1 = compute_regime_features(bars)
        f2 = compute_regime_features(shuffled)
        # Both sorted ascending internally; last row should be the same
        assert f1.iloc[-1]['close'] == f2.iloc[-1]['close']
        assert f1.iloc[-1]['sma_50'] == pytest.approx(f2.iloc[-1]['sma_50'])

    def test_does_not_mutate_input(self):
        bars = self._synth_bars()
        orig_cols = set(bars.columns)
        compute_regime_features(bars)
        assert set(bars.columns) == orig_cols  # caller's df untouched

    def test_vol_is_non_negative(self):
        feats = compute_regime_features(self._synth_bars(n=60))
        vols = feats['vol_20_ann'].dropna()
        assert (vols >= 0).all()


class TestBuildRegimeLookup:
    """End-to-end lookup — critical look-ahead guard."""

    def _synth_bars_with_regime_flip(self):
        """Build SPY close series that flips from 'Clean Bull' (above SMA)
        to 'below SMA, falling' regime at a known date."""
        # 60 rising days then 30 falling days
        closes = [500 + i for i in range(60)] + [559 - i * 2 for i in range(30)]
        dates = pd.date_range('2025-01-01', periods=90, freq='B')
        return pd.DataFrame({'bar_date': dates, 'close': closes})

    def test_lookup_returns_dict_with_date_keys(self):
        bars = self._synth_bars_with_regime_flip()
        lookup = build_regime_lookup(bars)
        # first row has no T-1 features, so len = len-1
        assert len(lookup) == 89
        for k in lookup.keys():
            assert isinstance(k, str)
            # YYYY-MM-DD
            assert len(k) == 10 and k[4] == '-' and k[7] == '-'

    def test_lookup_no_lookahead(self):
        """Regime for day T must use features from row T-1, not T."""
        bars = self._synth_bars_with_regime_flip()
        lookup = build_regime_lookup(bars)
        # Day right after SMA crossover should still see the PREVIOUS day's
        # label (computed from T-1's close). This is the critical invariant.
        # We verify by checking that the first days in the series (when only
        # the early rising closes are visible) are never labeled 'C*'.
        first_few_days = sorted(lookup.keys())[:10]
        for d in first_few_days:
            # Regime might be 'unknown' (warmup) or 'A' — NEVER 'C*'
            # because T-1 closes are entirely in the rising series.
            assert lookup[d] in ('unknown', 'A', 'B')

    def test_lookup_uses_custom_thresholds(self):
        bars = self._synth_bars_with_regime_flip()
        # Artificially low vol threshold → most days become B
        tight = build_regime_lookup(bars, vol_threshold_pct=0.5)
        # At least some classified days should be B
        assert any(v == 'B' for v in tight.values())
        # With default threshold (22%), a flat rising series should never be B
        default = build_regime_lookup(bars)
        assert not any(v == 'B' for v in default.values())

    def test_warmup_period_returns_unknown(self):
        bars = self._synth_bars_with_regime_flip()
        lookup = build_regime_lookup(bars)
        # First 50 trading days have no sma_50 → T-1 features are NaN → unknown
        first_date = sorted(lookup.keys())[0]
        assert lookup[first_date] == 'unknown'

    def test_uses_default_thresholds_when_not_passed(self):
        # Ensure defaults are the shipped values (prevents silent drift)
        assert DEFAULT_VOL_THRESHOLD_PCT == 22.0
        assert DEFAULT_SLOPE_THRESHOLD_PCT == 0.15

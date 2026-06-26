"""Unit tests for trading/trail_vol_guard.py.

Covers: core decision boundary, None handling, zero/negative edge cases,
ratio sensitivity, non-numeric inputs.
"""
from __future__ import annotations

import pytest

from trading.trail_vol_guard import should_skip_trail_exit_on_low_vol


class TestCoreDecision:
    """Standard case: baseline 100k, ratio 1.0 → skip when bar_vol < 100k."""

    def test_skip_when_below_ratio(self):
        # bar_vol 50k < 100k baseline × 1.0 → skip
        assert should_skip_trail_exit_on_low_vol(50_000, 100_000, 1.0) is True

    def test_no_skip_when_at_ratio(self):
        # bar_vol 100k == 100k × 1.0 → strict less-than, so fire exit
        assert should_skip_trail_exit_on_low_vol(100_000, 100_000, 1.0) is False

    def test_no_skip_when_above_ratio(self):
        assert should_skip_trail_exit_on_low_vol(150_000, 100_000, 1.0) is False

    def test_ratio_0_5_threshold(self):
        # baseline 100k × 0.5 = 50k. 40k < 50k → skip. 60k > 50k → no skip.
        assert should_skip_trail_exit_on_low_vol(40_000, 100_000, 0.5) is True
        assert should_skip_trail_exit_on_low_vol(60_000, 100_000, 0.5) is False

    def test_ratio_1_5_threshold(self):
        # baseline 100k × 1.5 = 150k. 140k < 150k → skip. 160k > 150k → no skip.
        assert should_skip_trail_exit_on_low_vol(140_000, 100_000, 1.5) is True
        assert should_skip_trail_exit_on_low_vol(160_000, 100_000, 1.5) is False


class TestSafeDefaults:
    """When inputs are missing or invalid, never skip (conservative: fire exit)."""

    def test_flag_avg_none_no_skip(self):
        assert should_skip_trail_exit_on_low_vol(50_000, None, 1.0) is False

    def test_flag_avg_zero_no_skip(self):
        assert should_skip_trail_exit_on_low_vol(50_000, 0, 1.0) is False

    def test_flag_avg_negative_no_skip(self):
        assert should_skip_trail_exit_on_low_vol(50_000, -1, 1.0) is False

    def test_flag_avg_non_numeric_no_skip(self):
        assert should_skip_trail_exit_on_low_vol(50_000, "bad", 1.0) is False
        assert should_skip_trail_exit_on_low_vol(50_000, None, 1.0) is False

    def test_ratio_zero_no_skip(self):
        # ratio=0 effectively disables the check — any vol passes.
        assert should_skip_trail_exit_on_low_vol(50_000, 100_000, 0.0) is False

    def test_ratio_negative_no_skip(self):
        assert should_skip_trail_exit_on_low_vol(50_000, 100_000, -0.5) is False

    def test_ratio_non_numeric_no_skip(self):
        assert should_skip_trail_exit_on_low_vol(50_000, 100_000, "bad") is False


class TestBarVolumeEdgeCases:
    """bar_volume=None, 0, non-numeric."""

    def test_bar_volume_none_does_not_skip(self):
        """GLXG 2026-06-11 regression: None means "no bar observed yet" — NOT
        "this bar had zero trading". The pre-fix semantic (None → skip) held
        off the trail for ~20s on a fresh watch while the stock peaked.
        New semantic: None → fail open, fire the stop."""
        assert should_skip_trail_exit_on_low_vol(None, 100_000, 1.0) is False

    def test_bar_volume_none_still_no_skip_with_high_ratio(self):
        """Even with a very high ratio, None must NOT cause skip — the
        absence of bar data should never suppress downside protection."""
        assert should_skip_trail_exit_on_low_vol(None, 100_000, 10.0) is False

    def test_bar_volume_zero_skips(self):
        """0 (we observed a bar with zero trading) → skip if min_ratio > 0.
        Distinct from None — 0 is a real signal that the minute was dead."""
        assert should_skip_trail_exit_on_low_vol(0, 100_000, 1.0) is True

    def test_bar_volume_float_coerced(self):
        # int-expected but float passes through int() cast
        assert should_skip_trail_exit_on_low_vol(50_000.5, 100_000, 1.0) is True

    def test_bar_volume_non_numeric_treated_as_zero(self):
        # Defensive: string gets coerced to 0 → will skip if threshold > 0
        assert should_skip_trail_exit_on_low_vol("bad", 100_000, 1.0) is True


class TestWatchEntryDefault:
    """Pin the WatchEntry default. The pre-fix `last_bar_volume: int = 0`
    was the root of the GLXG incident — a fresh watch was indistinguishable
    from a real zero-volume bar. Anyone reverting the default to `0` (or any
    int) without re-thinking the guard semantics will trip this test."""

    def test_last_bar_volume_default_is_none(self):
        from trading.stop_monitor import WatchEntry
        # Construct with only the required fields; rely on default.
        e = WatchEntry(
            symbol='X', stop_price=10.0, shares=100,
            tp_leg_id='tp', sl_leg_id='sl',
        )
        assert e.last_bar_volume is None, (
            "WatchEntry.last_bar_volume default must be None — see GLXG "
            "2026-06-11 incident. Reverting to int=0 re-introduces the "
            "'no bar yet' vs 'dead minute' ambiguity that suppressed the "
            "trail exit for ~20s while the stock peaked."
        )


class TestGLXGScenarioIncidentReplay:
    """2026-06-11 GLXG incident: a watch added mid-session had
    `last_bar_volume=0` (its WatchEntry default at the time) before the
    first 1-min bar event arrived. The vol-conf guard saw bar_vol=0 and
    skipped the trail exit for ~20 seconds while the stock peaked at $3.75.
    The fix flips `last_bar_volume` default to `None` and the guard treats
    None as "no signal" instead of "low signal"."""

    def test_recently_armed_watch_no_bar_yet_still_fires(self):
        """The exact incident: flag_avg=589,668 (real GLXG number), ratio=1.0,
        bar_volume=None (no bar event seen since arm). Pre-fix: skip (bug).
        Post-fix: don't skip — trail fires."""
        assert should_skip_trail_exit_on_low_vol(None, 589_668, 1.0) is False

    def test_observed_zero_still_skips(self):
        """A LATER bar event arriving with volume=0 (truly dead minute) should
        still skip — the fix only changes the "no observation" semantics, not
        the "observed dead minute" semantics."""
        assert should_skip_trail_exit_on_low_vol(0, 589_668, 1.0) is True


class TestRealWorldScenarios:
    """Scenarios mirroring actual BT behavior."""

    def test_slow_burn_low_vol_drift(self):
        # CDNA-style: flag avg 300k, drift bar 50k (17% of baseline) → skip
        assert should_skip_trail_exit_on_low_vol(50_000, 300_000, 1.0) is True

    def test_fast_fade_high_vol_reversal(self):
        # LUNL-style: flag avg 300k, reversal bar 1.2M (4× baseline) → fire
        assert should_skip_trail_exit_on_low_vol(1_200_000, 300_000, 1.0) is False

    def test_borderline_exactly_at_threshold(self):
        # bar_vol == baseline × ratio (exactly) → strict less-than, not skipped
        assert should_skip_trail_exit_on_low_vol(300_000, 300_000, 1.0) is False

    def test_tiny_baseline_big_volume(self):
        # Edge: pattern had very light trading, real exit needs modest vol
        assert should_skip_trail_exit_on_low_vol(50_000, 1_000, 1.0) is False

    def test_sweep_ratio_monotonicity(self):
        # For fixed bar_vol + baseline, increasing ratio eventually causes skip
        # bar=150k, baseline=100k: skip threshold at ratio > 1.5
        assert should_skip_trail_exit_on_low_vol(150_000, 100_000, 1.0) is False
        assert should_skip_trail_exit_on_low_vol(150_000, 100_000, 1.5) is False  # ==
        assert should_skip_trail_exit_on_low_vol(150_000, 100_000, 1.51) is True
        assert should_skip_trail_exit_on_low_vol(150_000, 100_000, 2.0) is True

"""Unit tests for trading/two_tier_filter.py.

Covers:
  - Tier classifier boundary cases
  - Composite z-score with known frozen params
  - Surgical drop on extras + MACD
  - should_keep combined gate (enabled/disabled)
  - max_intraday_change_pre_entry over synthetic bars
  - build_features_from_trade extraction
"""
from __future__ import annotations

import pytest

from trading.two_tier_filter import (
    TIER_A,
    TIER_EDGE,
    TIER_EXTRAS,
    build_features_from_trade,
    classify_tier,
    composite_score,
    max_intraday_change_pre_entry,
    should_keep,
)


# ---------------------------------------------------------------------------
# classify_tier
# ---------------------------------------------------------------------------

class TestClassifyTier:
    def test_a_tier_at_20(self):
        assert classify_tier(20.0) == TIER_A
        assert classify_tier(25.0) == TIER_A
        assert classify_tier(100.0) == TIER_A

    def test_extras_between_10_and_20(self):
        assert classify_tier(10.0) == TIER_EXTRAS
        assert classify_tier(15.0) == TIER_EXTRAS
        assert classify_tier(19.99) == TIER_EXTRAS

    def test_edge_below_10(self):
        assert classify_tier(9.99) == TIER_EDGE
        assert classify_tier(0.0) == TIER_EDGE
        assert classify_tier(-5.0) == TIER_EDGE

    def test_boundary_exactly_20(self):
        # 20.0 is A-tier (inclusive), 19.99 is Extras
        assert classify_tier(20.0) == TIER_A
        assert classify_tier(19.99) == TIER_EXTRAS

    def test_boundary_exactly_10(self):
        # 10.0 is Extras (inclusive), 9.99 is edge
        assert classify_tier(10.0) == TIER_EXTRAS
        assert classify_tier(9.99) == TIER_EDGE

    def test_custom_thresholds(self):
        assert classify_tier(15.0, a_tier_lower=15.0, extras_lower=5.0) == TIER_A
        assert classify_tier(7.0, a_tier_lower=15.0, extras_lower=5.0) == TIER_EXTRAS
        assert classify_tier(3.0, a_tier_lower=15.0, extras_lower=5.0) == TIER_EDGE

    def test_none_input_returns_edge(self):
        assert classify_tier(None) == TIER_EDGE


# ---------------------------------------------------------------------------
# composite_score
# ---------------------------------------------------------------------------

FROZEN_PARAMS = {
    "conviction_mult":  {"mean": 2.0, "std": 0.5, "sign": -1},
    "qf_vwap_dist_pct": {"mean": 4.0, "std": 2.0, "sign": -1},
}


class TestCompositeScore:
    def test_value_at_mean_scores_zero(self):
        f = {"conviction_mult": 2.0, "qf_vwap_dist_pct": 4.0}
        assert composite_score(f, FROZEN_PARAMS) == 0.0

    def test_one_sigma_below_mean_with_negative_sign_scores_positive(self):
        # value below mean, sign=-1 -> z negative, sign*z positive
        # conviction 1.5: z = (1.5-2.0)/0.5 = -1.0, sign*z = +1.0
        # qf_vwap 2.0: z = (2.0-4.0)/2.0 = -1.0, sign*z = +1.0
        f = {"conviction_mult": 1.5, "qf_vwap_dist_pct": 2.0}
        assert composite_score(f, FROZEN_PARAMS) == pytest.approx(1.0)

    def test_one_sigma_above_mean_with_negative_sign_scores_negative(self):
        f = {"conviction_mult": 2.5, "qf_vwap_dist_pct": 6.0}
        assert composite_score(f, FROZEN_PARAMS) == pytest.approx(-1.0)

    def test_mixed_features_averages(self):
        # conviction 1.5 -> +1.0, qf_vwap 6.0 -> -1.0. avg = 0.
        f = {"conviction_mult": 1.5, "qf_vwap_dist_pct": 6.0}
        assert composite_score(f, FROZEN_PARAMS) == pytest.approx(0.0)

    def test_missing_feature_is_ignored(self):
        f = {"conviction_mult": 1.5}  # no qf_vwap
        # Only conviction contributes: z=-1, sign=-1, result +1. Average over 1.
        assert composite_score(f, FROZEN_PARAMS) == pytest.approx(1.0)

    def test_all_features_none_returns_none(self):
        assert composite_score({}, FROZEN_PARAMS) is None
        assert composite_score({"foo": 1.0}, FROZEN_PARAMS) is None

    def test_positive_sign_feature(self):
        params = {"higher_better": {"mean": 10.0, "std": 2.0, "sign": +1}}
        # value 12: z = +1, sign*z = +1
        assert composite_score({"higher_better": 12.0}, params) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# should_keep
# ---------------------------------------------------------------------------

COMPOSITE_CFG = {
    "enabled": True,
    "drop_extras_macd_below": 1.25,
    "composite_threshold": -0.50,
    "composite_features": FROZEN_PARAMS,
}


class TestShouldKeep:
    def test_disabled_keeps_all(self):
        cfg = {**COMPOSITE_CFG, "enabled": False}
        # Even a "bad" extras trade passes because gate is off
        keep, reason = should_keep(TIER_EXTRAS, 1.0, {"conviction_mult": 5.0}, cfg)
        assert keep is True
        assert reason == ""

    def test_a_tier_always_kept(self):
        keep, reason = should_keep(TIER_A, 1.0, {}, COMPOSITE_CFG)
        assert keep is True

    def test_edge_always_kept(self):
        keep, reason = should_keep(TIER_EDGE, 1.0, {}, COMPOSITE_CFG)
        assert keep is True

    def test_extras_macd_low_surgical_drop(self):
        # macd_zone_mult=1.0 < 1.25 threshold -> reject regardless of composite
        good_features = {"conviction_mult": 1.0, "qf_vwap_dist_pct": 1.0}  # very good score
        keep, reason = should_keep(TIER_EXTRAS, 1.0, good_features, COMPOSITE_CFG)
        assert keep is False
        assert reason == "extras_macd_surgical_drop"

    def test_extras_macd_high_passes_surgical(self):
        # macd_zone_mult=1.5 > 1.25 -> past surgical. Then composite gate.
        good_features = {"conviction_mult": 1.0, "qf_vwap_dist_pct": 1.0}
        keep, reason = should_keep(TIER_EXTRAS, 1.5, good_features, COMPOSITE_CFG)
        assert keep is True

    def test_extras_composite_below_threshold_rejected(self):
        # macd ok, composite too low. Bad features -> score < -0.5
        # conviction 3.0: z=+2, sign*z=-2; qf_vwap 8.0: z=+2, sign*z=-2. avg=-2.
        bad_features = {"conviction_mult": 3.0, "qf_vwap_dist_pct": 8.0}
        keep, reason = should_keep(TIER_EXTRAS, 2.0, bad_features, COMPOSITE_CFG)
        assert keep is False
        assert reason == "extras_composite_below_threshold"

    def test_extras_composite_at_threshold_kept(self):
        # Score == -0.5 exactly. "< threshold" rejects, so == threshold passes.
        # z1=+0.5 (sign=-1 -> -0.5), z2=+0.5 (sign=-1 -> -0.5). avg = -0.5.
        features = {"conviction_mult": 2.25, "qf_vwap_dist_pct": 5.0}
        keep, reason = should_keep(TIER_EXTRAS, 2.0, features, COMPOSITE_CFG)
        assert keep is True

    def test_extras_none_macd_skips_surgical_check(self):
        # macd_zone_mult=None = signal unavailable (e.g. macd_zones disabled in
        # live config). Should SKIP the surgical check and evaluate composite
        # only. With good composite features, trade is kept.
        good_features = {"conviction_mult": 1.5, "qf_vwap_dist_pct": 2.0}
        keep, reason = should_keep(TIER_EXTRAS, None, good_features, COMPOSITE_CFG)
        assert keep is True
        assert reason == ""

    def test_extras_none_macd_still_evaluates_composite(self):
        # macd None but composite below threshold -> rejected on composite
        bad_features = {"conviction_mult": 3.0, "qf_vwap_dist_pct": 8.0}
        keep, reason = should_keep(TIER_EXTRAS, None, bad_features, COMPOSITE_CFG)
        assert keep is False
        assert reason == "extras_composite_below_threshold"

    def test_extras_explicit_zero_macd_still_triggers_surgical(self):
        # 0.0 (e.g. MACD dead zone) IS a valid signal and DOES trigger drop
        keep, reason = should_keep(TIER_EXTRAS, 0.0, {"conviction_mult": 1.5}, COMPOSITE_CFG)
        assert keep is False
        assert reason == "extras_macd_surgical_drop"

    def test_extras_no_features_rejected(self):
        # macd fine but no features -> composite returns None -> reject
        cfg = {**COMPOSITE_CFG, "drop_extras_macd_below": 0.0}
        keep, reason = should_keep(TIER_EXTRAS, 1.5, {}, cfg)
        assert keep is False
        assert reason == "extras_composite_no_features"


# ---------------------------------------------------------------------------
# max_intraday_change_pre_entry
# ---------------------------------------------------------------------------

class TestMaxIntradayChangePreEntry:
    def test_returns_none_for_empty_bars(self):
        assert max_intraday_change_pre_entry([], 100.0, "2025-01-02T14:40:00+00:00") is None

    def test_all_bars_at_or_after_entry_returns_none(self):
        bars = [
            ("2025-01-02T14:40:00+00:00", 10, 11, 9, 10.5),
            ("2025-01-02T14:41:00+00:00", 10.5, 11, 10, 10.8),
        ]
        # entry exactly at first bar -> no pre-entry bars
        assert max_intraday_change_pre_entry(bars, 10.0, "2025-01-02T14:40:00+00:00") is None

    def test_gap_pct_computed_from_close_vs_prev_close(self):
        bars = [
            ("2025-01-02T14:30:00+00:00", 10.0, 10.0, 10.0, 10.0),
            ("2025-01-02T14:31:00+00:00", 10.0, 10.5, 9.8, 10.5),  # gap +5%
        ]
        r = max_intraday_change_pre_entry(bars, 10.0, "2025-01-02T14:40:00+00:00")
        # range_pct over these bars = (10.5 - 9.8)/9.8*100 = 7.14%
        # gap_pct at last pre-entry bar = (10.5-10.0)/10.0*100 = 5%
        # max(gap, range) = 7.14 at bar 2
        assert r == pytest.approx(7.14, rel=0.01)

    def test_range_pct_dominates_when_wider(self):
        bars = [
            ("2025-01-02T14:30:00+00:00", 10.0, 12.0, 10.0, 11.0),  # high 12
            ("2025-01-02T14:31:00+00:00", 11.0, 11.0, 9.0, 10.0),   # low 9
        ]
        r = max_intraday_change_pre_entry(bars, 10.0, "2025-01-02T14:40:00+00:00")
        # range_pct at bar 2: (12 - 9)/9 * 100 = 33.3%
        assert r == pytest.approx(33.33, rel=0.01)

    def test_stops_at_entry_timestamp(self):
        bars = [
            ("2025-01-02T14:30:00+00:00", 10, 11, 10, 10.5),     # range 10%, gap 5%
            ("2025-01-02T14:40:00+00:00", 10.5, 20, 10.5, 19),   # would be 100% gap if included
        ]
        r = max_intraday_change_pre_entry(bars, 10.0, "2025-01-02T14:40:00+00:00")
        # Second bar is AT entry time, excluded. First bar only:
        # range_pct = (11-10)/10 * 100 = 10%, gap_pct at close = (10.5-10)/10 = 5%.
        # max = 10%.
        assert r == pytest.approx(10.0, rel=0.01)

    def test_none_high_or_low_skipped_safely(self):
        # Bar with None high -> should not crash, just skip that bar's high update
        bars = [
            ("2025-01-02T14:30:00+00:00", 10, 11, 10, 10.5),
            ("2025-01-02T14:31:00+00:00", 10.5, None, 10.5, 10.8),   # bad high
            ("2025-01-02T14:32:00+00:00", 10.8, 12, 10.0, 11.5),     # real extremes
        ]
        r = max_intraday_change_pre_entry(bars, 10.0, "2025-01-02T14:40:00+00:00")
        # Final bar: day_high=12, day_low=10. range=(12-10)/10*100=20%
        assert r == pytest.approx(20.0, rel=0.01)

    def test_tracks_max_not_last(self):
        bars = [
            ("2025-01-02T14:30:00+00:00", 10, 15, 10, 15),       # +50%
            ("2025-01-02T14:31:00+00:00", 15, 15, 12, 12),       # back to 20%
            ("2025-01-02T14:32:00+00:00", 12, 12, 12, 12),       # 20% still
        ]
        r = max_intraday_change_pre_entry(bars, 10.0, "2025-01-02T14:40:00+00:00")
        # At bar 1: gap=50%, range=50%. At bar 2: range=(15-10)/10=50% (unchanged), gap=20%
        # max overall should be 50%
        assert r == pytest.approx(50.0, rel=0.01)

    def test_prev_close_none_uses_range_only(self):
        bars = [
            ("2025-01-02T14:30:00+00:00", 10, 11, 9, 10.5),
        ]
        r = max_intraday_change_pre_entry(bars, None, "2025-01-02T14:40:00+00:00")
        # range_pct = (11-9)/9 * 100 = 22.2%
        assert r == pytest.approx(22.22, rel=0.01)


# ---------------------------------------------------------------------------
# build_features_from_trade
# ---------------------------------------------------------------------------

class TestBuildFeaturesFromTrade:
    def test_extracts_4_features(self):
        trade = {
            "conviction_mult": "2.0",
            "qf_vwap_dist_pct": "4.5",
            "qf_fill_vwap_dist_pct": "4.8",
            "entry_time_et": "09:45:00",
        }
        f = build_features_from_trade(trade)
        assert f["conviction_mult"] == 2.0
        assert f["qf_vwap_dist_pct"] == 4.5
        assert f["qf_fill_vwap_dist_pct"] == 4.8
        assert f["entry_minute"] == pytest.approx(9 * 60 + 45)

    def test_none_for_missing_fields(self):
        trade = {"conviction_mult": "1.5"}
        f = build_features_from_trade(trade)
        assert f["conviction_mult"] == 1.5
        assert f["qf_vwap_dist_pct"] is None
        assert f["entry_minute"] is None

    def test_entry_minute_malformed_returns_none(self):
        trade = {"entry_time_et": "bogus"}
        f = build_features_from_trade(trade)
        assert f["entry_minute"] is None

    def test_empty_strings_treated_as_none(self):
        trade = {"conviction_mult": "", "qf_vwap_dist_pct": "None"}
        f = build_features_from_trade(trade)
        assert f["conviction_mult"] is None
        assert f["qf_vwap_dist_pct"] is None

    def test_fill_vwap_falls_back_to_setup_vwap(self):
        """Post-fill-exit BT rows lack fill VWAP; must fall back to setup VWAP
        so composite stays on 4-feature scale matching frozen-fit params."""
        trade = {"qf_vwap_dist_pct": "4.2", "qf_fill_vwap_dist_pct": ""}
        f = build_features_from_trade(trade)
        assert f["qf_vwap_dist_pct"] == 4.2
        assert f["qf_fill_vwap_dist_pct"] == 4.2

    def test_fill_vwap_stays_independent_when_both_populated(self):
        """When fill VWAP is present, keep it (don't overwrite with setup)."""
        trade = {"qf_vwap_dist_pct": "4.2", "qf_fill_vwap_dist_pct": "4.8"}
        f = build_features_from_trade(trade)
        assert f["qf_vwap_dist_pct"] == 4.2
        assert f["qf_fill_vwap_dist_pct"] == 4.8

    def test_fill_vwap_none_when_both_missing(self):
        """When neither is populated, fill VWAP is None (no fabrication)."""
        trade = {"entry_time_et": "09:45:00"}
        f = build_features_from_trade(trade)
        assert f["qf_vwap_dist_pct"] is None
        assert f["qf_fill_vwap_dist_pct"] is None

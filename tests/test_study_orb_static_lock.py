"""BT integration tests for study_orb_pipeline_static_lock.simulate_static_lock.

Verifies:
  - Rule M fires on weak breakout bar -> exit_reason='tag_bb', correct price.
  - Rule D fires on bar-1 revert -> exit_reason='tag_b1', correct price.
  - Existing 'stop' / 'lock' / 'eod' paths unchanged (regression guards).
  - Disabled via env var matches pre-change behavior.
"""
import importlib
import os
import sys
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest


def _make_bars(rows):
    """Build a bars DataFrame with UTC-tz timestamps.

    rows: list of (minute_offset_from_9_30_et, open, high, low, close, volume).
    """
    # 9:30 ET = 13:30 UTC (assume EST for test determinism)
    base = pd.Timestamp('2025-06-02 13:30:00', tz='UTC')
    records = []
    for offset, o, h, l, c, v in rows:
        records.append({
            'timestamp': base + pd.Timedelta(minutes=offset),
            'open': o, 'high': h, 'low': l, 'close': c, 'volume': v,
        })
    return pd.DataFrame(records)


@pytest.fixture(autouse=True)
def _clean_env():
    """Clear ORB_TOUCHGO_* env vars between tests."""
    keys = ['ORB_TOUCHGO_ENABLED', 'ORB_TOUCHGO_RULE_M_ENABLED',
            'ORB_TOUCHGO_RULE_M_THRESH', 'ORB_TOUCHGO_RULE_D_ENABLED',
            'ORB_TOUCHGO_RULE_D_R', 'ORB_TOUCHGO_RULE_D_EXIT_R']
    for k in keys:
        os.environ.pop(k, None)
    yield
    for k in keys:
        os.environ.pop(k, None)


@pytest.fixture
def sim_with_default_cfg():
    """Re-import the pipeline module to pick up cleaned env vars."""
    if 'study_orb_pipeline_static_lock' in sys.modules:
        del sys.modules['study_orb_pipeline_static_lock']
    import study_orb_pipeline_static_lock as m
    return m


@pytest.fixture
def sim_with_touchgo_disabled():
    """Re-import with master disable env var."""
    os.environ['ORB_TOUCHGO_ENABLED'] = '0'
    if 'study_orb_pipeline_static_lock' in sys.modules:
        del sys.modules['study_orb_pipeline_static_lock']
    import study_orb_pipeline_static_lock as m
    return m


# =====================================================================
# Rule M tests
# =====================================================================
class TestRuleMTriggers:
    def test_rule_m_fires_on_weak_breakout(self, sim_with_default_cfg):
        """Entry bar closes in bottom 20% of its range -> tag_bb exit."""
        m = sim_with_default_cfg
        # 5-min range: bars at offsets 0..4. range_high=10.0, range_low=9.0.
        # Entry bar at offset 5: opens 9.95, breaks above 10.0 (high=10.1),
        # then collapses to close 9.7 (close_pos = (9.7-9.6)/(10.1-9.6)=0.2)
        bars = _make_bars([
            (0, 9.5, 10.0, 9.0, 9.8, 1000),
            (1, 9.8, 9.9, 9.5, 9.7, 1000),
            (2, 9.7, 9.95, 9.6, 9.85, 1000),
            (3, 9.85, 9.95, 9.7, 9.8, 1000),
            (4, 9.8, 9.95, 9.65, 9.7, 1000),
            (5, 9.95, 10.1, 9.6, 9.7, 5000),  # entry bar — weak close
            (6, 9.7, 9.75, 9.5, 9.55, 1000),
        ])
        entry_price = 10.03  # range_high * 1.003
        entry_time = bars.iloc[5]['timestamp']
        exit_p, reason = m.simulate_static_lock(
            bars, entry_price, 10.0, 9.0, entry_time,
        )
        assert reason == 'tag_bb'
        # Exit price = entry_bar.close * (1 - 10bps slip) = 9.7 * 0.999 = 9.6903
        expected = 9.7 * (1 - m.EXIT_SLIP_BPS / 10000)
        assert exit_p == pytest.approx(expected, abs=0.001)

    def test_rule_m_does_not_fire_on_strong_breakout(self, sim_with_default_cfg):
        """Entry bar closes in top 80% of its range -> NOT tag_bb."""
        m = sim_with_default_cfg
        bars = _make_bars([
            (0, 9.5, 10.0, 9.0, 9.8, 1000),
            (1, 9.8, 9.9, 9.5, 9.7, 1000),
            (2, 9.7, 9.95, 9.6, 9.85, 1000),
            (3, 9.85, 9.95, 9.7, 9.8, 1000),
            (4, 9.8, 9.95, 9.65, 9.7, 1000),
            # Entry bar: opens 9.95, breaks 10.0, closes 10.08 (top of bar)
            (5, 9.95, 10.1, 9.9, 10.08, 5000),
            # Subsequent bars don't trigger lock or stop within window
            (6, 10.08, 10.15, 10.0, 10.1, 1000),
            (7, 10.1, 10.2, 10.05, 10.15, 1000),
            (8, 10.15, 10.18, 10.1, 10.12, 1000),
        ])
        entry_price = 10.03
        entry_time = bars.iloc[5]['timestamp']
        exit_p, reason = m.simulate_static_lock(
            bars, entry_price, 10.0, 9.0, entry_time,
        )
        assert reason != 'tag_bb'

    def test_rule_m_disabled_via_env_var(self, sim_with_touchgo_disabled):
        """When ORB_TOUCHGO_ENABLED=0, weak breakout does NOT exit via tag_bb."""
        m = sim_with_touchgo_disabled
        # Same weak-breakout setup as test_rule_m_fires_on_weak_breakout
        # but with rule disabled, falls through to static-lock loop.
        # Entry bar closes 9.7, range_low=9.0. Next bar low=9.5 (above stop).
        # Then we hold to EOD.
        bars = _make_bars([
            (0, 9.5, 10.0, 9.0, 9.8, 1000),
            (1, 9.8, 9.9, 9.5, 9.7, 1000),
            (2, 9.7, 9.95, 9.6, 9.85, 1000),
            (3, 9.85, 9.95, 9.7, 9.8, 1000),
            (4, 9.8, 9.95, 9.65, 9.7, 1000),
            (5, 9.95, 10.1, 9.6, 9.7, 5000),  # weak breakout
            (6, 9.7, 9.8, 9.55, 9.65, 1000),
            (7, 9.65, 9.7, 9.6, 9.65, 1000),  # holds above stop, no lock trigger
        ])
        entry_price = 10.03
        entry_time = bars.iloc[5]['timestamp']
        exit_p, reason = m.simulate_static_lock(
            bars, entry_price, 10.0, 9.0, entry_time,
        )
        # NOT tag_bb. Must be one of stop/lock/eod.
        assert reason in ('stop', 'lock', 'eod')


# =====================================================================
# Rule D tests
# =====================================================================
class TestRuleDTriggers:
    def test_rule_d_fires_on_bar1_deep_revert(self, sim_with_default_cfg):
        """Bar 1 revert >= 0.75R -> tag_b1 exit at entry - 0.5R."""
        m = sim_with_default_cfg
        # range_high=10.0, range_low=9.0, range_size=1.0
        # Entry bar (offset 5): closes strong (top half — does NOT trigger Rule M).
        # Bar 1 (offset 6): low = 9.2 -> revert = (10.03 - 9.2) / 1.0 = 0.83R >= 0.75
        bars = _make_bars([
            (0, 9.5, 10.0, 9.0, 9.8, 1000),
            (1, 9.8, 9.9, 9.5, 9.7, 1000),
            (2, 9.7, 9.95, 9.6, 9.85, 1000),
            (3, 9.85, 9.95, 9.7, 9.8, 1000),
            (4, 9.8, 9.95, 9.65, 9.7, 1000),
            (5, 9.95, 10.1, 9.9, 10.08, 5000),  # strong breakout
            (6, 10.05, 10.07, 9.2, 9.25, 8000),  # deep revert in bar 1
            (7, 9.25, 9.3, 9.1, 9.15, 1000),
        ])
        entry_price = 10.03
        entry_time = bars.iloc[5]['timestamp']
        exit_p, reason = m.simulate_static_lock(
            bars, entry_price, 10.0, 9.0, entry_time,
        )
        assert reason == 'tag_b1'
        # exit = (entry + (-0.5) * 1.0) * (1 - 10bps) = 9.53 * 0.999 = 9.521...
        expected = (10.03 - 0.5) * (1 - m.EXIT_SLIP_BPS / 10000)
        assert exit_p == pytest.approx(expected, abs=0.001)

    def test_rule_d_does_not_fire_on_shallow_revert(self, sim_with_default_cfg):
        """Bar 1 revert < 0.75R -> NOT tag_b1."""
        m = sim_with_default_cfg
        # bar 1 low = 9.7 -> revert = (10.03 - 9.7) / 1.0 = 0.33R < 0.75
        bars = _make_bars([
            (0, 9.5, 10.0, 9.0, 9.8, 1000),
            (1, 9.8, 9.9, 9.5, 9.7, 1000),
            (2, 9.7, 9.95, 9.6, 9.85, 1000),
            (3, 9.85, 9.95, 9.7, 9.8, 1000),
            (4, 9.8, 9.95, 9.65, 9.7, 1000),
            (5, 9.95, 10.1, 9.9, 10.08, 5000),
            (6, 10.05, 10.12, 9.7, 10.0, 5000),  # shallow revert
            (7, 10.0, 10.15, 9.95, 10.1, 1000),
            (8, 10.1, 10.15, 10.05, 10.1, 1000),
        ])
        entry_price = 10.03
        entry_time = bars.iloc[5]['timestamp']
        exit_p, reason = m.simulate_static_lock(
            bars, entry_price, 10.0, 9.0, entry_time,
        )
        assert reason != 'tag_b1'

    def test_rule_d_disabled_via_env_var(self, sim_with_touchgo_disabled):
        m = sim_with_touchgo_disabled
        # Same deep-revert setup as test_rule_d_fires_on_bar1_deep_revert
        bars = _make_bars([
            (0, 9.5, 10.0, 9.0, 9.8, 1000),
            (1, 9.8, 9.9, 9.5, 9.7, 1000),
            (2, 9.7, 9.95, 9.6, 9.85, 1000),
            (3, 9.85, 9.95, 9.7, 9.8, 1000),
            (4, 9.8, 9.95, 9.65, 9.7, 1000),
            (5, 9.95, 10.1, 9.9, 10.08, 5000),
            (6, 10.05, 10.07, 9.2, 9.25, 8000),  # deep revert
            (7, 9.25, 9.3, 9.0, 9.05, 1000),    # hits stop at range_low
        ])
        entry_price = 10.03
        entry_time = bars.iloc[5]['timestamp']
        exit_p, reason = m.simulate_static_lock(
            bars, entry_price, 10.0, 9.0, entry_time,
        )
        # Falls through to static-lock loop. Bar 7 low=9.0 hits stop_price=9.0.
        assert reason == 'stop'


# =====================================================================
# Regression guards: existing exit paths unchanged
# =====================================================================
class TestExistingPathsUnchanged:
    """Sanity-check that 'stop', 'lock', and 'eod' paths still work when
    neither touchgo rule fires.
    """

    def test_stop_path_unchanged(self, sim_with_default_cfg):
        """Bars hit stop without firing M or D."""
        m = sim_with_default_cfg
        bars = _make_bars([
            (0, 9.5, 10.0, 9.0, 9.8, 1000),
            (1, 9.8, 9.9, 9.5, 9.7, 1000),
            (2, 9.7, 9.95, 9.6, 9.85, 1000),
            (3, 9.85, 9.95, 9.7, 9.8, 1000),
            (4, 9.8, 9.95, 9.65, 9.7, 1000),
            # Strong entry-bar close (no Rule M), small bar-1 revert (no Rule D)
            (5, 9.95, 10.1, 9.9, 10.08, 5000),
            (6, 10.05, 10.12, 9.85, 9.9, 5000),  # revert 0.13R, no Rule D
            (7, 9.9, 9.95, 9.7, 9.75, 1000),
            (8, 9.75, 9.8, 9.5, 9.55, 1000),
            (9, 9.55, 9.6, 9.0, 9.05, 1000),  # hits range_low -> stop
        ])
        entry_price = 10.03
        entry_time = bars.iloc[5]['timestamp']
        exit_p, reason = m.simulate_static_lock(
            bars, entry_price, 10.0, 9.0, entry_time,
        )
        assert reason == 'stop'
        # exit = 9.0 * (1 - 10bps) = 8.991
        expected = 9.0 * (1 - m.EXIT_SLIP_BPS / 10000)
        assert exit_p == pytest.approx(expected, abs=0.001)

    def test_lock_path_unchanged(self, sim_with_default_cfg):
        """Lock arms then triggers."""
        m = sim_with_default_cfg
        # range_size = 1.0; lock_trigger = entry + 1.75 = 11.78
        # lock_stop after arming = entry + 0.5 = 10.53
        bars = _make_bars([
            (0, 9.5, 10.0, 9.0, 9.8, 1000),
            (1, 9.8, 9.9, 9.5, 9.7, 1000),
            (2, 9.7, 9.95, 9.6, 9.85, 1000),
            (3, 9.85, 9.95, 9.7, 9.8, 1000),
            (4, 9.8, 9.95, 9.65, 9.7, 1000),
            (5, 9.95, 10.1, 9.95, 10.08, 5000),  # entry bar, strong
            (6, 10.05, 10.5, 10.0, 10.4, 5000),  # bar 1, no revert
            (7, 10.4, 11.85, 10.35, 11.7, 5000),  # arms lock at 11.78
            (8, 11.7, 11.8, 10.5, 10.5, 5000),   # hits lock_stop 10.53
        ])
        entry_price = 10.03
        entry_time = bars.iloc[5]['timestamp']
        exit_p, reason = m.simulate_static_lock(
            bars, entry_price, 10.0, 9.0, entry_time,
        )
        assert reason == 'lock'

    def test_eod_path_unchanged(self, sim_with_default_cfg):
        """No exits triggered; close at last bar."""
        m = sim_with_default_cfg
        bars = _make_bars([
            (0, 9.5, 10.0, 9.0, 9.8, 1000),
            (1, 9.8, 9.9, 9.5, 9.7, 1000),
            (2, 9.7, 9.95, 9.6, 9.85, 1000),
            (3, 9.85, 9.95, 9.7, 9.8, 1000),
            (4, 9.8, 9.95, 9.65, 9.7, 1000),
            (5, 9.95, 10.1, 9.95, 10.08, 5000),
            (6, 10.05, 10.2, 10.0, 10.15, 5000),
            (7, 10.15, 10.4, 10.1, 10.3, 5000),
            (8, 10.3, 10.45, 10.2, 10.35, 5000),
        ])
        entry_price = 10.03
        entry_time = bars.iloc[5]['timestamp']
        exit_p, reason = m.simulate_static_lock(
            bars, entry_price, 10.0, 9.0, entry_time,
        )
        assert reason == 'eod'
        expected = 10.35 * (1 - m.EXIT_SLIP_BPS / 10000)
        assert exit_p == pytest.approx(expected, abs=0.001)


# =====================================================================
# Rule precedence
# =====================================================================
class TestRulePrecedence:
    def test_rule_m_fires_before_rule_d(self, sim_with_default_cfg):
        """When both M and D would fire, M wins (evaluated first)."""
        m = sim_with_default_cfg
        # Weak breakout (M fires) AND bar 1 reverts deeply (D would fire)
        bars = _make_bars([
            (0, 9.5, 10.0, 9.0, 9.8, 1000),
            (1, 9.8, 9.9, 9.5, 9.7, 1000),
            (2, 9.7, 9.95, 9.6, 9.85, 1000),
            (3, 9.85, 9.95, 9.7, 9.8, 1000),
            (4, 9.8, 9.95, 9.65, 9.7, 1000),
            (5, 9.95, 10.1, 9.6, 9.7, 5000),  # weak: M fires
            (6, 9.7, 9.75, 9.2, 9.25, 5000),  # would fire D too
        ])
        entry_price = 10.03
        entry_time = bars.iloc[5]['timestamp']
        _, reason = m.simulate_static_lock(
            bars, entry_price, 10.0, 9.0, entry_time,
        )
        assert reason == 'tag_bb'  # M wins

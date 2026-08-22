"""BT-side winner-stack tests: simulate_winner_stack frozen semantics +
load_bt_config flag plumbing (study_orb_pipeline_static_lock.py).

Golden cases from docs/orb_winner_stack_review_aug2026.md:
  * both-hit bar => scale FILLS, runner lock-stops same bar (P0-1, the
    NCNA 2025-08-21 class — the real-bars NCNA golden lives in
    tests/test_orb_winner_stack_goldens.py against cache.db)
  * touchgo prefire => whole position, no scale
  * tiny qty => all-runner
  * flags off => identical to simulate_static_lock
"""
import os

import pandas as pd
import pytest

import study_orb_pipeline_static_lock as bt
from study_orb_pipeline_static_lock import (
    EXIT_SLIP_BPS, load_bt_config, simulate_static_lock, simulate_winner_stack,
)

SLIP = 1 - EXIT_SLIP_BPS / 10000


def _bars(specs, date_str='2026-04-20'):
    """specs: list of (hh, mm, o, h, l, c). UTC timestamps (EDT: 13:30=9:30ET)."""
    rows = []
    for (h, m, o, hi, lo, c) in specs:
        rows.append({'timestamp': pd.Timestamp(f'{date_str} {h:02d}:{m:02d}:00',
                                               tz='UTC'),
                     'open': o, 'high': hi, 'low': lo, 'close': c,
                     'volume': 1000})
    return pd.DataFrame(rows)


ENTRY_TS = pd.Timestamp('2026-04-20 13:35:00', tz='UTC')
RH, RL = 10.0, 9.0          # R = 1.0
ENTRY = 10.0


def _base_specs():
    """Entry bar strong (no touchgo), then drift."""
    return [(13, 35, 10.0, 10.4, 9.95, 10.35),   # entry bar: closes top half
            (13, 36, 10.35, 10.6, 10.0, 10.5),   # bar 1: no deep revert
            (13, 37, 10.5, 10.8, 10.2, 10.6)]


class TestFlagsOffEquivalence:
    def test_matches_static_lock_stop(self):
        specs = _base_specs() + [(13, 38, 10.6, 10.7, 8.9, 8.95)]
        bars = _bars(specs)
        a = simulate_static_lock(bars, ENTRY, RH, RL, ENTRY_TS)
        b = simulate_winner_stack(bars, ENTRY, RH, RL, ENTRY_TS, shares=1000)
        assert a == b

    def test_matches_static_lock_eod(self):
        bars = _bars(_base_specs())
        a = simulate_static_lock(bars, ENTRY, RH, RL, ENTRY_TS)
        b = simulate_winner_stack(bars, ENTRY, RH, RL, ENTRY_TS, shares=1000)
        assert a == b

    def test_matches_static_lock_lock_exit(self):
        specs = _base_specs() + [
            (13, 38, 10.6, 11.8, 10.5, 11.7),    # arms lock (trig 11.75)
            (13, 39, 11.7, 11.7, 10.4, 10.45),   # hits lock stop 10.5
        ]
        bars = _bars(specs)
        a = simulate_static_lock(bars, ENTRY, RH, RL, ENTRY_TS)
        b = simulate_winner_stack(bars, ENTRY, RH, RL, ENTRY_TS, shares=1000)
        assert a == b
        assert a[1] == 'lock'

    def test_matches_static_lock_touchgo(self):
        specs = [(13, 35, 10.0, 10.4, 9.8, 9.9)] + _base_specs()[1:]
        bars = _bars(specs)
        a = simulate_static_lock(bars, ENTRY, RH, RL, ENTRY_TS)
        b = simulate_winner_stack(bars, ENTRY, RH, RL, ENTRY_TS, shares=1000)
        assert a == b
        assert a[1] == 'tag_bb'


class TestAtrFloor:
    def test_floor_binds_and_stops(self):
        """Floor 9.5 (atr=2, k=0.25) stops a dip to 9.4 that range_low 9.0
        would have survived."""
        specs = _base_specs() + [(13, 38, 10.6, 10.7, 9.4, 9.6),
                                 (13, 39, 9.6, 9.8, 9.55, 9.7)]
        bars = _bars(specs)
        base = simulate_winner_stack(bars, ENTRY, RH, RL, ENTRY_TS,
                                     shares=1000)
        floored = simulate_winner_stack(
            bars, ENTRY, RH, RL, ENTRY_TS, shares=1000,
            atr14=2.0, atr_floor_enabled=True, atr_floor_k=0.25)
        assert base[1] == 'eod'
        assert floored[1] == 'stop'
        assert floored[0] == pytest.approx(9.5 * SLIP)

    def test_no_atr_fail_open(self):
        specs = _base_specs() + [(13, 38, 10.6, 10.7, 9.4, 9.6)]
        bars = _bars(specs)
        base = simulate_winner_stack(bars, ENTRY, RH, RL, ENTRY_TS,
                                     shares=1000)
        floored = simulate_winner_stack(
            bars, ENTRY, RH, RL, ENTRY_TS, shares=1000,
            atr14=None, atr_floor_enabled=True)
        assert base == floored

    def test_sizing_untouched_by_floor(self):
        """The floor never changes share count — it is exit-side only (the
        pipeline sizes from range_size_pct before the walk). Sanity: the
        function does not even receive sizing inputs beyond `shares`."""
        import inspect
        params = inspect.signature(simulate_winner_stack).parameters
        assert 'risk' not in params and 'position' not in params


class TestScaleOut:
    def test_scale_fills_runner_locks(self):
        """Clean scale at +3R (13.0) then runner exits at the lock."""
        specs = _base_specs() + [
            (13, 38, 10.6, 13.1, 10.5, 12.9),    # hits scale 13.0 (and trig)
            (13, 39, 12.9, 12.9, 10.4, 10.45),   # runner lock-stops at 10.5
        ]
        bars = _bars(specs)
        px, reason = simulate_winner_stack(
            bars, ENTRY, RH, RL, ENTRY_TS, shares=1000,
            scale_enabled=True, scale_frac=0.40, scale_level_r=3.0)
        assert reason == 'scale_lock'
        exp_ret = 0.4 * (13.0 * SLIP / 10 - 1) + 0.6 * (10.5 * SLIP / 10 - 1)
        assert px == pytest.approx(10 * (1 + exp_ret))

    def test_both_hit_bar_scale_fills_first(self):
        """P0-1 golden (NCNA class): a bar touching BOTH the stop and the
        scale level FILLS the scale (stop check gated high<scale_px); the
        same bar arms the lock, so the runner exits at +0.5R — NOT the
        whole position at the stop."""
        specs = _base_specs() + [
            (13, 38, 10.6, 13.5, 8.5, 9.0),      # both-hit bar
        ]
        bars = _bars(specs)
        px, reason = simulate_winner_stack(
            bars, ENTRY, RH, RL, ENTRY_TS, shares=1000,
            scale_enabled=True, scale_frac=0.40, scale_level_r=3.0)
        assert reason == 'scale_lock'
        exp_ret = 0.4 * (13.0 * SLIP / 10 - 1) + 0.6 * (10.5 * SLIP / 10 - 1)
        assert px == pytest.approx(10 * (1 + exp_ret))
        # Contrast: without the scale the same path stops at the lock only
        # (arm-before-stop convention) — materially lower P&L.
        base_px, base_reason = simulate_winner_stack(
            bars, ENTRY, RH, RL, ENTRY_TS, shares=1000)
        assert base_reason == 'lock'
        assert px > base_px

    def test_touchgo_prefire_no_scale(self):
        """A tag_bb trade never scales even when later bars hit +3R."""
        specs = [(13, 35, 10.0, 10.4, 9.8, 9.9),   # Rule M fires (bottom half)
                 (13, 36, 9.9, 13.5, 9.8, 13.4)]   # would have scaled
        bars = _bars(specs)
        px, reason = simulate_winner_stack(
            bars, ENTRY, RH, RL, ENTRY_TS, shares=1000,
            scale_enabled=True)
        assert reason == 'tag_bb'
        assert px == pytest.approx(9.9 * SLIP)

    def test_tiny_qty_all_runner(self):
        """floor(0.4*2) < 1 => no scale — identical to the no-scale walk."""
        specs = _base_specs() + [
            (13, 38, 10.6, 13.1, 10.5, 12.9),
            (13, 39, 12.9, 12.9, 10.4, 10.45),
        ]
        bars = _bars(specs)
        scaled = simulate_winner_stack(bars, ENTRY, RH, RL, ENTRY_TS,
                                       shares=2, scale_enabled=True)
        plain = simulate_winner_stack(bars, ENTRY, RH, RL, ENTRY_TS,
                                      shares=2)
        assert scaled == plain

    def test_runner_rides_to_eod(self):
        specs = _base_specs() + [
            (13, 38, 10.6, 13.1, 10.6, 13.0),    # scale fills
            (13, 39, 13.0, 13.4, 12.8, 13.2),    # runner holds
        ]
        bars = _bars(specs)
        px, reason = simulate_winner_stack(
            bars, ENTRY, RH, RL, ENTRY_TS, shares=1000, scale_enabled=True)
        assert reason == 'scale_eod'
        exp_ret = 0.4 * (13.0 * SLIP / 10 - 1) + 0.6 * (13.2 * SLIP / 10 - 1)
        assert px == pytest.approx(10 * (1 + exp_ret))

    def test_stop_before_scale_whole_position(self):
        """Stop hit on a bar BELOW the scale level exits everything."""
        specs = _base_specs() + [
            (13, 38, 10.6, 10.7, 8.9, 9.0),      # stop, high < scale
            (13, 39, 9.0, 13.5, 8.9, 13.4),      # too late
        ]
        bars = _bars(specs)
        px, reason = simulate_winner_stack(
            bars, ENTRY, RH, RL, ENTRY_TS, shares=1000, scale_enabled=True)
        assert reason == 'stop'
        assert px == pytest.approx(9.0 * SLIP)

    def test_composed_floor_plus_scale(self):
        """C-point composition: runner keeps the FLOORED initial stop."""
        specs = _base_specs() + [
            (13, 38, 10.6, 13.1, 10.6, 13.0),    # scale fills (arms lock too)
            (13, 39, 13.0, 13.0, 10.4, 10.45),   # runner lock-stops (10.5)
        ]
        bars = _bars(specs)
        px, reason = simulate_winner_stack(
            bars, ENTRY, RH, RL, ENTRY_TS, shares=1000,
            atr14=2.0, atr_floor_enabled=True, scale_enabled=True)
        assert reason == 'scale_lock'
        # floored stop 9.5 never mattered here (lock 10.5 above it), but the
        # composition path executed — now a variant where the floor IS the
        # runner exit:
        specs2 = _base_specs() + [
            (13, 38, 10.6, 13.1, 10.6, 13.0),
            (13, 39, 13.0, 13.0, 10.55, 10.6),   # above lock stop, no exit
        ]
        # remove arming? trig=11.75 was hit by the scale bar, so lock stop
        # 10.5 rules; floor visibility needs a no-arm path: scale at lower R
        px2, reason2 = simulate_winner_stack(
            _bars(_base_specs() + [
                (13, 38, 10.6, 11.6, 10.6, 11.5),    # scale @ +1.5R=11.5, no arm
                (13, 39, 11.5, 11.5, 9.4, 9.45),     # runner hits FLOOR 9.5
            ]), ENTRY, RH, RL, ENTRY_TS, shares=1000,
            atr14=2.0, atr_floor_enabled=True,
            scale_enabled=True, scale_level_r=1.5)
        assert reason2 == 'scale_stop'
        exp_ret = (0.4 * (11.5 * SLIP / 10 - 1)
                   + 0.6 * (9.5 * SLIP / 10 - 1))
        assert px2 == pytest.approx(10 * (1 + exp_ret))


class TestLoadBtConfigFlags:
    def _write_yaml(self, tmp_path, exit_block):
        p = tmp_path / 'orb.yaml'
        p.write_text("sizing:\n  account_budget_usd: 10000\n"
                     "  max_concurrent: 3\n  risk_per_trade_usd: 375\n"
                     "filter:\n  threshold: 0.5\n" + exit_block)
        return str(p)

    def test_absent_keys_default_off(self, tmp_path, monkeypatch):
        monkeypatch.delenv('ORB_ATR_FLOOR', raising=False)
        monkeypatch.delenv('ORB_SCALE_OUT', raising=False)
        cfg = load_bt_config(self._write_yaml(tmp_path, ""))
        assert cfg['atr_floor_enabled'] is False
        assert cfg['scale_enabled'] is False
        assert cfg['atr_floor_k'] == 0.25
        assert cfg['scale_frac'] == 0.40
        assert cfg['scale_level_r'] == 3.0

    def test_yaml_flags_on(self, tmp_path, monkeypatch):
        monkeypatch.delenv('ORB_ATR_FLOOR', raising=False)
        monkeypatch.delenv('ORB_SCALE_OUT', raising=False)
        cfg = load_bt_config(self._write_yaml(
            tmp_path,
            "exit:\n  atr_stop_floor:\n    enabled: true\n    k: 0.3\n"
            "  scale_out:\n    enabled: true\n    frac: 0.5\n"
            "    level_r: 2.5\n"))
        assert cfg['atr_floor_enabled'] is True
        assert cfg['atr_floor_k'] == 0.3
        assert cfg['scale_enabled'] is True
        assert cfg['scale_frac'] == 0.5
        assert cfg['scale_level_r'] == 2.5

    def test_env_kill_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv('ORB_ATR_FLOOR', '0')
        monkeypatch.setenv('ORB_SCALE_OUT', '0')
        cfg = load_bt_config(self._write_yaml(
            tmp_path,
            "exit:\n  atr_stop_floor:\n    enabled: true\n"
            "  scale_out:\n    enabled: true\n"))
        assert cfg['atr_floor_enabled'] is False
        assert cfg['scale_enabled'] is False

    def test_env_force_on(self, tmp_path, monkeypatch):
        monkeypatch.setenv('ORB_ATR_FLOOR', '1')
        monkeypatch.setenv('ORB_SCALE_OUT', '1')
        cfg = load_bt_config(self._write_yaml(tmp_path, ""))
        assert cfg['atr_floor_enabled'] is True
        assert cfg['scale_enabled'] is True

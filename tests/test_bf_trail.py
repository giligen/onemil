"""BF trail unification (2026-09-05) — shared math + BT/LIVE parity.

Three layers:
  1. Unit: trading/bf_trail.py pure functions.
  2. Parity: the SAME 1-min bar tape drives backtest.TradeSimulator and
     the live StopMonitor (closed-bar path + tick exit) → identical stop
     path, identical exit bar, identical exit reason.
  3. Golden: CWVX 2026-08-03 (the trade that exposed the split — live
     +$313 at 9:58 vs the fill-R cache +$2,381 at 13:32). Under the
     unified spec BOTH sides exit on the 9:57 bar at the plan-R trail.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from trading.bf_trail import (
    R_BASIS_FILL, R_BASIS_PLAN, arm_and_ratchet, entry_bar_excluded,
    normalize_r_basis, r_baseline_and_unit,
)


# ---------------------------------------------------------------------------
# 1. Unit
# ---------------------------------------------------------------------------

class TestRBasis:
    def test_normalize_defaults_to_plan(self):
        assert normalize_r_basis(None) == R_BASIS_PLAN
        assert normalize_r_basis('') == R_BASIS_PLAN

    @pytest.mark.parametrize('v', ['plan', 'PLAN', ' fill ', 'Fill'])
    def test_normalize_accepts_case_and_space(self, v):
        assert normalize_r_basis(v) in (R_BASIS_PLAN, R_BASIS_FILL)

    def test_normalize_rejects_garbage(self):
        with pytest.raises(ValueError):
            normalize_r_basis('planned')

    def test_plan_basis_uses_setup_numbers(self):
        base, unit = r_baseline_and_unit(13.9795, 13.77, 14.175, 13.77, 'plan')
        assert base == pytest.approx(13.9795)
        assert unit == pytest.approx(0.2095)

    def test_fill_basis_uses_fill_numbers(self):
        base, unit = r_baseline_and_unit(13.9795, 13.77, 14.175, 13.77, 'fill')
        assert base == pytest.approx(14.175)
        assert unit == pytest.approx(0.405)

    def test_plan_basis_falls_back_to_fill_when_planned_unusable(self):
        assert r_baseline_and_unit(0.0, 0.0, 10.0, 9.5, 'plan') == (10.0, pytest.approx(0.5))
        assert r_baseline_and_unit(9.0, 9.5, 10.0, 9.5, 'plan') == (10.0, pytest.approx(0.5))


class TestArmAndRatchet:
    def test_no_trail_when_trail_r_zero(self):
        s = arm_and_ratchet(5.0, 4.0, 3.9, False, 4.0, 0.1, 2.0, 0.0)
        assert (s.highest, s.stop, s.trailing_active) == (5.0, 3.9, False)

    def test_below_arm_threshold_no_change(self):
        s = arm_and_ratchet(4.15, 4.0, 3.9, False, 4.0, 0.1, 2.0, 1.0)
        assert s.trailing_active is False and s.stop == 3.9 and s.highest == 4.15

    def test_arms_and_ratchets_at_threshold(self):
        s = arm_and_ratchet(4.20, 4.0, 3.9, False, 4.0, 0.1, 2.0, 1.0)
        assert s.armed_now and s.trailing_active and s.ratcheted
        assert s.stop == pytest.approx(4.10)
        assert s.r_gain == pytest.approx(2.0)

    def test_monotone_never_lowers(self):
        s = arm_and_ratchet(4.30, 4.30, 4.20, True, 4.0, 0.1, 2.0, 1.0)
        s2 = arm_and_ratchet(4.25, s.highest, s.stop, s.trailing_active, 4.0, 0.1, 2.0, 1.0)
        assert s2.highest == 4.30 and s2.stop == pytest.approx(4.20) and not s2.ratcheted

    def test_idempotent(self):
        s = arm_and_ratchet(4.30, 4.0, 3.9, False, 4.0, 0.1, 2.0, 1.0)
        s2 = arm_and_ratchet(4.30, s.highest, s.stop, s.trailing_active, 4.0, 0.1, 2.0, 1.0)
        assert (s2.highest, s2.stop, s2.trailing_active) == (s.highest, s.stop, s.trailing_active)
        assert not s2.armed_now and not s2.ratcheted


class TestEntryBarExcluded:
    def test_no_skip_when_unset(self):
        assert entry_bar_excluded(1000.0, 0.0) is False

    def test_entry_bar_excluded_next_bar_counts(self):
        end_of_fill_min = 1_000_060.0
        assert entry_bar_excluded(1_000_000.0, end_of_fill_min) is True
        assert entry_bar_excluded(1_000_060.0, end_of_fill_min) is False


# ---------------------------------------------------------------------------
# 2 + 3. Parity harness
# ---------------------------------------------------------------------------

# CWVX 2026-08-03, 1-min bars 09:50-10:05 ET (UTC-4), from data/cache.db.
# Live: planned breakout 13.9795, stop 13.77, filled 14.175 at 09:54.
CWVX_BARS = [
    ('2026-08-03T13:50:00+00:00', 13.70, 13.93, 13.70, 13.9029, 48301),
    ('2026-08-03T13:51:00+00:00', 13.95, 13.9795, 13.835, 13.8572, 10229),
    ('2026-08-03T13:52:00+00:00', 13.86, 13.86, 13.78, 13.78, 7649),
    ('2026-08-03T13:53:00+00:00', 13.885, 14.18, 13.885, 14.16, 81253),
    ('2026-08-03T13:54:00+00:00', 14.1569, 14.41, 14.14, 14.3995, 45095),   # entry bar
    ('2026-08-03T13:55:00+00:00', 14.39, 14.79, 14.39, 14.79, 40005),
    ('2026-08-03T13:56:00+00:00', 14.80, 14.96, 14.72, 14.9401, 45747),
    ('2026-08-03T13:57:00+00:00', 14.92, 14.95, 14.68, 14.72, 21390),
    ('2026-08-03T13:58:00+00:00', 14.765, 14.86, 14.746, 14.8499, 30148),
    ('2026-08-03T13:59:00+00:00', 14.80, 15.01, 14.77, 14.9893, 21095),
    ('2026-08-03T14:00:00+00:00', 15.00, 15.035, 14.80, 14.80, 56610),
    ('2026-08-03T14:01:00+00:00', 14.842, 14.842, 14.53, 14.69, 38908),
]
CWVX_PLANNED_ENTRY = 13.9795
CWVX_STOP = 13.77
CWVX_FILL = 14.175
CWVX_ENTRY_IDX = 4


def _bars_df(rows):
    df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    return df


def _plan(planned_entry, stop, shares=536):
    from trading.pattern_detector import BullFlagPattern
    from trading.trade_planner import TradePlan
    pattern = BullFlagPattern(
        symbol='CWVX', pole_start_idx=0, pole_end_idx=0, flag_start_idx=0,
        flag_end_idx=0, pole_low=stop, pole_high=planned_entry,
        pole_height=planned_entry - stop, pole_gain_pct=0.0, flag_low=stop,
        flag_high=planned_entry, retracement_pct=0.0, pullback_candle_count=1,
        avg_pole_volume=8938, avg_flag_volume=8938.5, breakout_level=planned_entry,
    )
    rps = planned_entry - stop
    return TradePlan(
        symbol='CWVX', entry_price=planned_entry, stop_loss_price=stop,
        take_profit_price=planned_entry + 2.5 * rps, risk_per_share=rps,
        reward_per_share=2.5 * rps, risk_reward_ratio=2.5, shares=shares,
        total_risk=rps * shares, pattern=pattern,
    )


def _bt_stop_path(bars, plan, fill, r_basis, vol_conf=False):
    """Run TradeSimulator and return (exit_bar_ts, exit_reason)."""
    from backtest import TradeSimulator
    sim = TradeSimulator(
        trailing_stop_r=1.0, trailing_activate_at_r=2.0, r_basis=r_basis,
        vol_confirmed_trail_enabled=vol_conf, vol_confirmed_trail_min_ratio=1.0,
        force_close_time_et=None,
    )
    t = sim.simulate(plan, bars, CWVX_ENTRY_IDX, entry_price_override=fill)
    return t.exit_time, t.exit_reason


async def _live_stop_path(bars, plan, fill, r_basis, vol_conf=False):
    """Drive StopMonitor with the same tape: for each bar AFTER entry, first
    replay its low as a tick (BT's `bar_low <= stop` check happens against
    the stop from prior bars), then close the bar. Returns
    (exit_bar_ts, exit_reason, stop_path)."""
    from trading.stop_monitor import StopMonitor
    alpaca = MagicMock()
    mon = StopMonitor(api_key='k', api_secret='s', alpaca_client=alpaca,
                      marketable_limit_offset=0.03, marketable_limit_offset_pct=0.005)
    events = []

    async def capture(symbol, price, watch, **kw):
        events.append((symbol, price, kw.get('exit_reason')))
    mon._execute_stop_exit = capture

    entry_ts = bars.iloc[CWVX_ENTRY_IDX]['timestamp']
    skip_until = entry_ts.timestamp() + 60
    mon.add_watch(
        'CWVX', stop_price=CWVX_STOP, shares=plan.shares, tp_leg_id='', sl_leg_id='',
        entry_price=fill, risk_per_share=fill - CWVX_STOP, trail_r=1.0, activate_at_r=2.0,
        avg_flag_volume=plan.pattern.avg_flag_volume,
        vol_confirmed_trail_enabled=vol_conf, vol_confirmed_trail_min_ratio=1.0,
        planned_entry_price=plan.entry_price, planned_risk_per_share=plan.risk_per_share,
        r_basis=r_basis, skip_exits_until_ts=skip_until,
    )
    mon._bar_symbols.add('CWVX')
    stop_path = []
    # Entry bar closes (must be ignored for trail state)
    eb = bars.iloc[CWVX_ENTRY_IDX]
    mon._last_data_ts = eb['timestamp'].timestamp()
    await mon._on_bar(MagicMock(symbol='CWVX', timestamp=eb['timestamp'], open=eb['open'],
                                high=eb['high'], low=eb['low'], close=eb['close'],
                                volume=int(eb['volume'])))
    for i in range(CWVX_ENTRY_IDX + 1, len(bars)):
        b = bars.iloc[i]
        # ticks inside bar i: the low (BT checks bar low against prior stop)
        mon._last_data_ts = b['timestamp'].timestamp() + 30
        tick = MagicMock(); tick.symbol = 'CWVX'; tick.price = float(b['low'])
        await mon._on_trade(tick)
        if events:
            return b['timestamp'], events[0][2], stop_path
        await mon._on_bar(MagicMock(symbol='CWVX', timestamp=b['timestamp'], open=b['open'],
                                    high=b['high'], low=b['low'], close=b['close'],
                                    volume=int(b['volume'])))
        with mon._watch_lock:
            stop_path.append(round(mon._watches['CWVX'].stop_price, 4))
    return None, None, stop_path


class TestBtLiveParity:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('r_basis', ['plan', 'fill'])
    async def test_cwvx_same_exit_bar_and_reason(self, r_basis):
        bars = _bars_df(CWVX_BARS)
        plan = _plan(CWVX_PLANNED_ENTRY, CWVX_STOP)
        bt_ts, bt_reason = _bt_stop_path(bars, plan, CWVX_FILL, r_basis)
        live_ts, live_reason, _ = await _live_stop_path(bars, plan, CWVX_FILL, r_basis)
        assert live_ts == bt_ts, f"{r_basis}: live exit bar {live_ts} != BT {bt_ts}"
        assert live_reason == bt_reason

    @pytest.mark.asyncio
    async def test_cwvx_golden_plan_r_exits_0957(self):
        """The unified spec on the real tape: plan-R R=$0.2095, 9:56 high
        14.96 → stop 14.7505; 9:57 low 14.68 trips it. BOTH sides."""
        bars = _bars_df(CWVX_BARS)
        plan = _plan(CWVX_PLANNED_ENTRY, CWVX_STOP)
        bt_ts, bt_reason = _bt_stop_path(bars, plan, CWVX_FILL, 'plan')
        live_ts, live_reason, path = await _live_stop_path(bars, plan, CWVX_FILL, 'plan')
        assert bt_reason == 'trail_stop' and live_reason == 'trail_stop'
        assert bt_ts == pd.Timestamp('2026-08-03T13:57:00+00:00') == live_ts
        # stop after 9:55 close: armed (14.79 >= 13.9795+2×0.2095=14.3985) → 14.5805
        # stop after 9:56 close: 14.96-0.2095 = 14.7505
        assert path[:2] == [14.5805, 14.7505]

    @pytest.mark.asyncio
    async def test_cwvx_fill_r_rides_past_1001(self):
        """Fill-R (the retired cache's basis): R=$0.405, arm 14.985 → the
        9:59 high 15.01 arms, stop 14.605; 10:01 low 14.53 trips. Both sides."""
        bars = _bars_df(CWVX_BARS)
        plan = _plan(CWVX_PLANNED_ENTRY, CWVX_STOP)
        bt_ts, _ = _bt_stop_path(bars, plan, CWVX_FILL, 'fill')
        live_ts, _, _ = await _live_stop_path(bars, plan, CWVX_FILL, 'fill')
        assert bt_ts == pd.Timestamp('2026-08-03T14:01:00+00:00') == live_ts

    @pytest.mark.asyncio
    async def test_entry_bar_high_does_not_arm_either_side(self):
        """A monster entry bar (high +3R) must not arm on either side."""
        rows = list(CWVX_BARS)
        rows[CWVX_ENTRY_IDX] = ('2026-08-03T13:54:00+00:00', 14.1569, 15.50, 14.14, 14.3995, 45095)
        # following bars flat & below any trail: no exits expected from an entry-bar arm
        rows[5] = ('2026-08-03T13:55:00+00:00', 14.39, 14.40, 14.30, 14.35, 40005)
        bars = _bars_df(rows[:6])
        plan = _plan(CWVX_PLANNED_ENTRY, CWVX_STOP)
        bt_ts, bt_reason = _bt_stop_path(bars, plan, CWVX_FILL, 'plan')
        live_ts, live_reason, path = await _live_stop_path(bars, plan, CWVX_FILL, 'plan')
        # Had the entry bar counted, stop would be 15.50-0.2095=15.2905 and
        # the 9:55 low 14.30 would have stopped both sides out. Instead the
        # 9:55 bar itself (high 14.40 >= arm 14.3985) arms → stop 14.1905,
        # and nothing trips: BT reaches EOD, live never exits.
        assert bt_reason == 'eod' and live_ts is None
        assert path == [pytest.approx(14.1905, abs=1e-4)]


class TestVolGuardParity:
    @pytest.mark.asyncio
    async def test_prev_bar_volume_gates_both_sides(self):
        """Vol guard reads the PREVIOUS closed bar on both sides. Make the
        9:56 bar (the one before the 9:57 trip) low-volume → both hold
        through 9:57; the 9:58 bar (prev = 9:57 normal volume) then trips
        at low 14.746 < 14.7505 on both."""
        rows = list(CWVX_BARS)
        rows[6] = ('2026-08-03T13:56:00+00:00', 14.80, 14.96, 14.72, 14.9401, 100)  # low vol
        bars = _bars_df(rows)
        plan = _plan(CWVX_PLANNED_ENTRY, CWVX_STOP)
        bt_ts, bt_reason = _bt_stop_path(bars, plan, CWVX_FILL, 'plan', vol_conf=True)
        live_ts, live_reason, _ = await _live_stop_path(bars, plan, CWVX_FILL, 'plan', vol_conf=True)
        assert bt_ts == pd.Timestamp('2026-08-03T13:58:00+00:00') == live_ts
        assert bt_reason == live_reason == 'trail_stop'

"""Profit partial — ONE spec for BT and live (trading/bf_profit_partial.py).

Pins:
  * shared functions (level, trigger, shares, breakeven);
  * TradeSimulator: flags off = identical to the legacy walk; on = partial
    at the trigger bar's close, stop to the fill, remainder keeps trailing,
    P&L = partial + remainder, reasons prefixed 'pp+';
  * StopMonitor: arming computes the level from the SAME plan-R baseline,
    pending_profit_partials fires on the closed-bar high, never on a tick;
  * parity: the same tape fires on the same bar on both sides.
"""
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest

from backtest import TradeSimulator
from trading.bf_profit_partial import (ProfitPartialConfig, breakeven_stop, load_profit_partial_config,
                                       partial_level, partial_shares, profit_partial_fires)
from trading.pattern_detector import BullFlagPattern
from trading.trade_planner import TradePlan


# ---------------------------------------------------------------------------
# shared functions
# ---------------------------------------------------------------------------

def test_shared_functions():
    assert partial_level(10.0, 0.5, 2.0) == 11.0
    assert profit_partial_fires(11.0, 11.0) and not profit_partial_fires(10.99, 11.0)
    assert not profit_partial_fires(99.0, 0.0)          # level 0 = disarmed
    assert partial_shares(100, 0.5) == 50 and partial_shares(1, 0.5) == 0
    assert partial_shares(2, 0.67) == 1 and partial_shares(3, 0.67) == 2 and partial_shares(2, 0.5) == 1
    assert breakeven_stop(9.0, 10.03) == 10.03 and breakeven_stop(10.5, 10.03) == 10.5


def test_config_loader_and_validation():
    assert not load_profit_partial_config({}).enabled
    c = load_profit_partial_config({'profit_partial': {'enabled': True, 'r_multiple': 2.0, 'fraction': 0.67}})
    assert c.enabled and c.r_multiple == 2.0 and c.fraction == 0.67 and c.move_to_breakeven
    with pytest.raises(ValueError):
        load_profit_partial_config({'profit_partial': {'enabled': True, 'fraction': 1.0}})


# ---------------------------------------------------------------------------
# BT simulator
# ---------------------------------------------------------------------------

def _tape(rows, start='2026-03-02 14:35'):
    ts = pd.date_range(start, periods=len(rows), freq='1min', tz='UTC')
    df = pd.DataFrame(rows, columns=['open', 'high', 'low', 'close'])
    df['volume'] = 5000
    df['timestamp'] = ts
    return df


def _plan():
    pat = BullFlagPattern(symbol='T', pole_start_idx=0, pole_end_idx=2, flag_start_idx=3, flag_end_idx=4,
                          pole_low=9.0, pole_high=10.0, pole_height=1.0, pole_gain_pct=11.0,
                          flag_low=9.5, flag_high=10.0, retracement_pct=50.0, pullback_candle_count=2,
                          avg_pole_volume=8000, avg_flag_volume=4000, breakout_level=10.0)
    return TradePlan(symbol='T', entry_price=10.0, stop_loss_price=9.5, take_profit_price=11.25,
                     risk_per_share=0.5, reward_per_share=1.25, risk_reward_ratio=2.5, shares=1000,
                     total_risk=500.0, pattern=pat)


# plan-R: baseline 10.0, R 0.5 -> +2R level = 11.0. Fill 10.03 on bar 0.
ROWS = [
    (10.02, 10.10, 10.00, 10.05),   # 0 entry bar
    (10.05, 10.40, 10.02, 10.35),   # 1
    (10.35, 10.80, 10.30, 10.70),   # 2
    (10.70, 11.05, 10.65, 10.95),   # 3 high 11.05 >= 11.0 -> partial at close 10.95
    (10.95, 11.10, 10.60, 10.62),   # 4
    (10.62, 10.70, 10.00, 10.01),   # 5 low 10.00 < breakeven 10.03 -> remainder stops out
    (10.01, 10.20, 9.90, 10.10),    # 6
]


def _sim(pp=None):
    return TradeSimulator(exit_slippage_pct=0.0, trailing_stop_r=1.0, trailing_activate_at_r=2.0,
                          profit_partial=pp, r_basis='plan')


def test_flags_off_matches_legacy_walk():
    a = _sim(None).simulate(_plan(), _tape(ROWS), 0, entry_price_override=10.03)
    b = _sim(ProfitPartialConfig(enabled=False)).simulate(_plan(), _tape(ROWS), 0, entry_price_override=10.03)
    assert (a.exit_reason, a.exit_price, a.pnl) == (b.exit_reason, b.exit_price, b.pnl)
    assert not a.partial_exit_taken


def test_partial_fires_on_closed_bar_high_then_breakeven_stop():
    t = _sim(ProfitPartialConfig(enabled=True, r_multiple=2.0, fraction=0.5)).simulate(
        _plan(), _tape(ROWS), 0, entry_price_override=10.03)
    assert t.partial_exit_taken
    assert t.partial_shares == 500
    assert t.partial_exit_price == pytest.approx(10.95)          # trigger bar's CLOSE, not the level
    assert t.partial_exit_time == _tape(ROWS)['timestamp'].iloc[3]
    assert t.partial_pnl == pytest.approx((10.95 - 10.03) * 500)
    # remainder: the trail armed at +2R (bar 3) and ratcheted to 11.05-0.5=10.55 -> bar 4 low 10.60 holds,
    # bar 5 low 10.00 hits the trail stop (max of breakeven 10.03 and the ratcheted stop)
    assert t.exit_reason.startswith('pp+')
    assert t.exit_price >= 10.03                                  # never below breakeven
    assert t.pnl == pytest.approx(t.partial_pnl + (t.exit_price - 10.03) * 500)


def test_no_partial_when_level_never_reached():
    rows = [(10.02, 10.10, 10.00, 10.05), (10.05, 10.60, 10.02, 10.50), (10.50, 10.55, 9.40, 9.45)]
    t = _sim(ProfitPartialConfig(enabled=True, r_multiple=2.0, fraction=0.5)).simulate(
        _plan(), _tape(rows), 0, entry_price_override=10.03)
    assert not t.partial_exit_taken and t.exit_reason == 'stop'


# ---------------------------------------------------------------------------
# StopMonitor (live twin) — arming, closed-bar trigger, no tick trigger
# ---------------------------------------------------------------------------

@pytest.fixture
def monitor():
    from trading.stop_monitor import StopMonitor
    from data_sources.alpaca_client import AlpacaClient
    m = StopMonitor.__new__(StopMonitor)
    # minimal state used by add_watch / arming / pending
    import threading
    m._watches = {}
    m._watch_lock = threading.RLock()
    m._exit_lock = threading.Lock()
    m._exit_in_progress = {}
    m._exit_started_at = {}
    m._alpaca = MagicMock(spec=AlpacaClient)
    return m


def test_live_arms_same_level_and_fires_on_closed_bar_only(monitor):
    from trading.stop_monitor import WatchEntry
    w = WatchEntry(symbol='T', stop_price=9.5, shares=1000, tp_leg_id='', sl_leg_id='',
                   entry_price=10.03, risk_per_share=0.53, trail_r=1.0, activate_at_r=2.0,
                   highest_since_entry=10.03, planned_entry_price=10.0, planned_risk_per_share=0.5,
                   r_basis='plan', pp_fraction=0.5)
    monitor._watches['T'] = w
    level = monitor.arm_profit_partial('T', 2.0)
    assert level == pytest.approx(partial_level(10.0, 0.5, 2.0))     # same plan-R math as the BT
    assert monitor.pending_profit_partials('bull_flag') == []
    w.highest_since_entry = 10.99                                     # tick above? no: highest advances on closed bars only
    assert monitor.pending_profit_partials('bull_flag') == []
    w.highest_since_entry = 11.05                                     # the closed bar 3 high from the BT tape
    assert monitor.pending_profit_partials('bull_flag') == ['T']
    w.pp_taken = True
    assert monitor.pending_profit_partials('bull_flag') == []


def test_parity_same_tape_same_bar(monitor):
    """BT fires on bar 3 (high 11.05); live's highest_since_entry after bar 3 closes is 11.05."""
    from trading.stop_monitor import WatchEntry
    bt = _sim(ProfitPartialConfig(enabled=True, r_multiple=2.0, fraction=0.5)).simulate(
        _plan(), _tape(ROWS), 0, entry_price_override=10.03)
    w = WatchEntry(symbol='T', stop_price=9.5, shares=1000, tp_leg_id='', sl_leg_id='',
                   entry_price=10.03, risk_per_share=0.53, planned_entry_price=10.0,
                   planned_risk_per_share=0.5, r_basis='plan', pp_fraction=0.5, highest_since_entry=10.03)
    monitor._watches['T'] = w
    monitor.arm_profit_partial('T', 2.0)
    fired_bar = None
    for i, (o, h, l, c) in enumerate(ROWS[1:], start=1):   # closed bars after the entry bar
        w.highest_since_entry = max(w.highest_since_entry, h)
        if monitor.pending_profit_partials('bull_flag') and fired_bar is None:
            fired_bar = i
    assert fired_bar == 3
    assert bt.partial_exit_time == _tape(ROWS)['timestamp'].iloc[fired_bar]


def test_shadow_flag_loads_and_marks_without_selling():
    """shadow=true: config parses; StopMonitor marks the partial taken (once)
    without touching the position — the live 10-session shadow window."""
    from trading.bf_profit_partial import load_profit_partial_config
    cfg = load_profit_partial_config({'profit_partial': {
        'enabled': True, 'r_multiple': 2.0, 'fraction': 0.5, 'shadow': True}})
    assert cfg.shadow is True and cfg.enabled is True
    from trading.stop_monitor import StopMonitor, WatchEntry
    sm = StopMonitor.__new__(StopMonitor)
    import threading
    sm._watch_lock = threading.RLock()
    sm._watches = {'ABC': WatchEntry(symbol='ABC', stop_price=9.5, shares=100, tp_leg_id='',
                                     sl_leg_id='', entry_price=10.0, strategy='bull_flag')}
    sm.mark_profit_partial_shadow('ABC')
    assert sm._watches['ABC'].pp_taken is True
    assert sm._watches['ABC'].shares == 100          # nothing sold
    assert sm._watches['ABC'].stop_price == 9.5      # stop untouched
    sm.mark_profit_partial_shadow('ZZZ')             # unknown symbol: no-op


class TestEngineLivePartialPath:
    """P1 goes live 2026-09-07 with NO shadow: the engine's real path must
    call the StopMonitor's partial executor with reason='profit_partial'."""

    def _engine(self, shadow=False, enabled=True):
        from trading.trading_engine import TradingEngine
        from trading.stop_monitor import StopMonitor
        from trading.bf_profit_partial import ProfitPartialConfig
        eng = TradingEngine.__new__(TradingEngine)
        eng.profit_partial = ProfitPartialConfig(enabled=enabled, r_multiple=2.0, fraction=0.5,
                                                 move_to_breakeven=True, shadow=shadow)
        sm = MagicMock(spec=StopMonitor)
        sm.pending_profit_partials.return_value = ['ABCD']
        sm.get_watch_snapshot.return_value = {'highest_since_entry': 5.40, 'trail_r': 1.0}
        sm.execute_partial_exit.return_value = None
        eng.stop_monitor = sm
        eng._process_exhaustion_partial_event = MagicMock()
        return eng, sm

    def test_live_path_sells_half_with_profit_partial_reason(self):
        eng, sm = self._engine()
        eng._check_profit_partials()
        sm.pending_profit_partials.assert_called_once_with('bull_flag')
        sm.execute_partial_exit.assert_called_once_with(
            'ABCD', fraction=0.5, tighter_trail_r=1.0, reason='profit_partial')
        sm.mark_profit_partial_shadow.assert_not_called()

    def test_event_is_processed(self):
        eng, sm = self._engine()
        sm.execute_partial_exit.return_value = {'symbol': 'ABCD', 'exit_reason': 'profit_partial'}
        eng._check_profit_partials()
        eng._process_exhaustion_partial_event.assert_called_once()

    def test_shadow_marks_and_never_sells(self):
        eng, sm = self._engine(shadow=True)
        eng._check_profit_partials()
        sm.execute_partial_exit.assert_not_called()
        sm.mark_profit_partial_shadow.assert_called_once_with('ABCD')

    def test_disabled_does_nothing(self):
        eng, sm = self._engine(enabled=False)
        eng._check_profit_partials()
        sm.pending_profit_partials.assert_not_called()


class TestLiveParityGuards:
    """Found in the 9/6 pre-launch review: (A) the exhaustion rule must not
    take a second partial after the +2R partial (BT skips it once pp_taken);
    (B) the partial must not fire inside the entry bar (tick path raises
    highest_since_entry during the fill minute; BT walks from entry+1)."""

    def _sm(self):
        import threading
        from trading.stop_monitor import StopMonitor, WatchEntry
        sm = StopMonitor.__new__(StopMonitor)
        sm._watch_lock = threading.RLock()
        sm._last_data_ts = 0.0
        w = WatchEntry(symbol='ABCD', stop_price=9.5, shares=200, tp_leg_id='', sl_leg_id='',
                       entry_price=10.0, risk_per_share=0.5, strategy='bull_flag')
        w.pp_level = 11.0; w.pp_taken = False; w.highest_since_entry = 11.2
        sm._watches = {'ABCD': w}
        return sm, w

    def test_snapshot_carries_pp_taken_and_exhaustion_skips(self):
        sm, w = self._sm()
        from trading.stop_monitor import StopMonitor
        w.pp_taken = True
        snap = StopMonitor.get_watch_snapshot(sm, 'ABCD')
        assert snap['pp_taken'] is True
        import inspect
        from trading.trading_engine import TradingEngine
        src = inspect.getsource(TradingEngine._check_exhaustion_exits)
        assert "snapshot.get('pp_taken')" in src

    def test_partial_waits_for_entry_bar_to_close(self):
        sm, w = self._sm()
        from trading.stop_monitor import StopMonitor
        w.skip_exits_until_ts = 1000.0
        sm._last_data_ts = 999.0                       # still inside the fill minute
        assert StopMonitor.pending_profit_partials(sm, 'bull_flag') == []
        sm._last_data_ts = 1000.0                      # entry bar closed
        assert StopMonitor.pending_profit_partials(sm, 'bull_flag') == ['ABCD']
        w.pp_taken = True
        assert StopMonitor.pending_profit_partials(sm, 'bull_flag') == []

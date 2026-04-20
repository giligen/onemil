"""Unit tests for MACDWaveEngine._sync_intraday_state — mid-day restart recovery.

Covers:
- No-op outside the 9:30-15:45 ET trading window
- universe_opens populated from snapshots
- Reverse-split detection routes to `invalidated`
- crossed_stocks reconstructed when a cross happened within cross_time_max_min
- crossed_stocks NOT added when cross is outside window (missed entry)
- Symbols already in open_positions are skipped
- Symbols at max_waves cap are skipped
- Volume filter (max) routes to invalidated; (min) silently skipped
- Closed trades replay into daily_pnl + trades_today
- Missing bars for a candidate don't fail the whole recovery
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone, time as dtime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import pytz

from trading.macd_wave_engine import (
    MACDWaveEngine, CrossedStock, OpenPosition,
)


ET = pytz.timezone('US/Eastern')


def _make_engine(
    universe=None,
    open_positions=None,
    cross_time_max_min=10,
    max_vol_at_cross=300000,
    min_vol_at_cross=0,
    min_intraday_pct=10.0,
    max_waves=1,
):
    cfg = {
        'universe': {},
        'entry': {
            'cross_time_max_min': cross_time_max_min,
            'max_vol_at_cross': max_vol_at_cross,
            'min_vol_at_cross': min_vol_at_cross,
        },
        'macd': {},
        'sizing': {'position_size': 50000, 'max_concurrent': 3},
        'risk': {'daily_loss_limit': -5000},
        'slippage': {},
        'waves': {'max_waves': max_waves},
    }
    # Inject min_intraday_pct via universe section (MACDWaveEngine reads it from there)
    cfg['universe']['min_intraday_pct'] = min_intraday_pct
    alpaca = MagicMock()
    db = MagicMock()
    db.get_trades_by_date.return_value = []
    db.get_open_trades.return_value = []
    e = MACDWaveEngine(
        alpaca_client=alpaca, db=db, config=cfg,
        stop_monitor=None,
    )
    if universe is not None:
        e.universe = list(universe)
    if open_positions is not None:
        e.open_positions = dict(open_positions)
    return e


def _mk_bars(open_price, highs, start_ts_utc, vol=10000):
    """Build a 1-min DataFrame. `highs` is a list — one entry per minute.
    First bar at start_ts_utc, each subsequent bar +1min.
    """
    rows = []
    for i, h in enumerate(highs):
        ts = start_ts_utc + timedelta(minutes=i)
        rows.append({
            'timestamp': ts,
            'open': open_price if i == 0 else highs[i - 1],
            'high': h,
            'low': open_price,
            'close': h,
            'volume': vol,
        })
    return pd.DataFrame(rows)


def _fake_et_now(hh, mm, date_str='2026-04-20'):
    """Return a fake `datetime.now(ET)` pinned to the given ET time on a weekday."""
    return ET.localize(datetime(2026, 4, 20, hh, mm, 0))


@pytest.fixture
def frozen_10_15_et():
    """ET = 10:15 (within trading window, 45 min past cross_time_max_min=10)."""
    target = _fake_et_now(10, 15)

    # We patch `datetime.now(ET)` usage inside macd_wave_engine. Since the
    # module imports ET and datetime separately, patch datetime.now at module level.
    class _FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return target.astimezone(tz)
            return target.replace(tzinfo=None)
    with patch('trading.macd_wave_engine.datetime', _FakeDT):
        yield target


@pytest.fixture
def frozen_9_00_et():
    target = _fake_et_now(9, 0)
    class _FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return target.astimezone(tz)
            return target.replace(tzinfo=None)
    with patch('trading.macd_wave_engine.datetime', _FakeDT):
        yield target


@pytest.fixture
def frozen_15_50_et():
    target = _fake_et_now(15, 50)
    class _FakeDT(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None:
                return target.astimezone(tz)
            return target.replace(tzinfo=None)
    with patch('trading.macd_wave_engine.datetime', _FakeDT):
        yield target


class TestTimeWindowGuards:
    def test_noop_before_market_open(self, frozen_9_00_et):
        e = _make_engine(universe=['AAA'])
        e.alpaca.get_snapshots = MagicMock()
        e._sync_intraday_state()
        e.alpaca.get_snapshots.assert_not_called()
        assert e.universe_opens == {}
        assert e.crossed_stocks == {}

    def test_noop_after_close(self, frozen_15_50_et):
        e = _make_engine(universe=['AAA'])
        e.alpaca.get_snapshots = MagicMock()
        e._sync_intraday_state()
        e.alpaca.get_snapshots.assert_not_called()


class TestUniverseOpensPopulation:
    def test_populates_universe_opens(self, frozen_10_15_et):
        e = _make_engine(universe=['AAA', 'BBB', 'CCC'])
        e.alpaca.get_snapshots = MagicMock(return_value={
            'AAA': {'open': 10.0, 'prev_close': 9.5, 'high': 10.2},
            'BBB': {'open': 15.0, 'prev_close': 14.8, 'high': 15.3},
            'CCC': {'open': 20.0, 'prev_close': 19.0, 'high': 21.0},
        })
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        e._sync_intraday_state()
        assert e.universe_opens == {'AAA': 10.0, 'BBB': 15.0, 'CCC': 20.0}

    def test_skips_reverse_splits(self, frozen_10_15_et):
        e = _make_engine(universe=['SPLIT', 'NORMAL'])
        e.alpaca.get_snapshots = MagicMock(return_value={
            # 150% overnight jump → reverse-split or corporate action
            'SPLIT': {'open': 25.0, 'prev_close': 10.0, 'high': 26.0},
            'NORMAL': {'open': 10.0, 'prev_close': 9.9, 'high': 10.3},
        })
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        e._sync_intraday_state()
        assert 'SPLIT' in e.invalidated
        assert 'SPLIT' not in e.universe_opens
        assert e.universe_opens.get('NORMAL') == 10.0


class TestCrossReconstruction:
    def test_reconstructs_cross_within_window(self, frozen_10_15_et):
        """Open $10, bar #3 high $11.50 (+15%) → crossed_stocks with cross_time_min=3."""
        e = _make_engine(universe=['MOVER'], cross_time_max_min=10, min_intraday_pct=10.0)
        e.alpaca.get_snapshots = MagicMock(return_value={
            'MOVER': {'open': 10.0, 'prev_close': 9.9, 'high': 12.0},  # high > +10%
        })
        # Market open (9:30 ET) in UTC
        open_utc = ET.localize(datetime(2026, 4, 20, 9, 30)).astimezone(timezone.utc)
        bars = _mk_bars(
            open_price=10.0,
            highs=[10.2, 10.5, 11.5, 11.8, 12.0],  # bar #2 (index 2) is first >=11.0
            start_ts_utc=open_utc,
            vol=5000,
        )
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={'MOVER': bars})
        e._sync_intraday_state()
        assert 'MOVER' in e.crossed_stocks
        cs = e.crossed_stocks['MOVER']
        assert cs.cross_time_min == 2
        assert cs.open_price == 10.0
        assert cs.vol_at_cross == 15000  # cumulative 3 bars × 5000

    def test_skips_cross_outside_window(self, frozen_10_15_et):
        """Cross at bar #12 (minute 12) vs max=10 → NOT added."""
        e = _make_engine(universe=['LATE'], cross_time_max_min=10, min_intraday_pct=10.0)
        e.alpaca.get_snapshots = MagicMock(return_value={
            'LATE': {'open': 10.0, 'prev_close': 9.9, 'high': 11.5},
        })
        open_utc = ET.localize(datetime(2026, 4, 20, 9, 30)).astimezone(timezone.utc)
        # 15 bars, cross at bar #12
        highs = [10.05] * 12 + [11.2, 11.3, 11.4]
        bars = _mk_bars(10.0, highs, open_utc, vol=5000)
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={'LATE': bars})
        e._sync_intraday_state()
        assert 'LATE' not in e.crossed_stocks

    def test_skips_symbol_already_in_open_positions(self, frozen_10_15_et):
        """open_positions takes precedence — don't re-add to crossed_stocks."""
        existing_pos = OpenPosition(
            symbol='HELD', entry_price=10.5, shares=100,
            hard_stop=10.0, trade_id=1, order_id='',
            entry_time=datetime.now(timezone.utc),
        )
        e = _make_engine(
            universe=['HELD'],
            open_positions={'HELD': existing_pos},
        )
        e.alpaca.get_snapshots = MagicMock(return_value={
            'HELD': {'open': 10.0, 'prev_close': 9.9, 'high': 12.0},
        })
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        e._sync_intraday_state()
        assert 'HELD' not in e.crossed_stocks

    def test_skips_symbol_at_max_waves(self, frozen_10_15_et):
        """DB has a filled MACD wave trade for SYM + max_waves=1 → skip."""
        e = _make_engine(universe=['DONE'], max_waves=1)
        e.db.get_trades_by_date = MagicMock(return_value=[
            {'symbol': 'DONE', 'strategy': 'macd_wave',
             'exit_price': 10.5, 'pnl': 150.0, 'exited_at': '2026-04-20T10:00:00Z'},
        ])
        e.alpaca.get_snapshots = MagicMock(return_value={
            'DONE': {'open': 10.0, 'prev_close': 9.9, 'high': 12.0},
        })
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        e._sync_intraday_state()
        assert 'DONE' not in e.crossed_stocks

    def test_vol_max_filter_routes_to_invalidated(self, frozen_10_15_et):
        """vol_at_cross > max → invalidated, not crossed_stocks."""
        e = _make_engine(
            universe=['HIVOL'],
            cross_time_max_min=10,
            max_vol_at_cross=10000,  # low cap
        )
        e.alpaca.get_snapshots = MagicMock(return_value={
            'HIVOL': {'open': 10.0, 'prev_close': 9.9, 'high': 12.0},
        })
        open_utc = ET.localize(datetime(2026, 4, 20, 9, 30)).astimezone(timezone.utc)
        # 1 bar at +15%, vol 50000 > 10000 cap
        bars = _mk_bars(10.0, [11.5], open_utc, vol=50000)
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={'HIVOL': bars})
        e._sync_intraday_state()
        assert 'HIVOL' in e.invalidated
        assert 'HIVOL' not in e.crossed_stocks

    def test_vol_min_filter_silently_skips(self, frozen_10_15_et):
        e = _make_engine(
            universe=['LOVOL'],
            cross_time_max_min=10,
            min_vol_at_cross=100000,
        )
        e.alpaca.get_snapshots = MagicMock(return_value={
            'LOVOL': {'open': 10.0, 'prev_close': 9.9, 'high': 12.0},
        })
        open_utc = ET.localize(datetime(2026, 4, 20, 9, 30)).astimezone(timezone.utc)
        bars = _mk_bars(10.0, [11.5], open_utc, vol=5000)  # <100K min
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={'LOVOL': bars})
        e._sync_intraday_state()
        assert 'LOVOL' not in e.crossed_stocks
        assert 'LOVOL' not in e.invalidated


class TestTradeReplay:
    def test_replays_closed_trades_into_daily_pnl(self, frozen_10_15_et):
        """DB has 3 closed MACD wave trades → daily_pnl + trades_today updated."""
        e = _make_engine(universe=[])
        e.alpaca.get_snapshots = MagicMock(return_value={})
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        e.db.get_trades_by_date = MagicMock(return_value=[
            {'symbol': 'A', 'strategy': 'macd_wave', 'exit_price': 11,
             'pnl': 100.0, 'exited_at': '2026-04-20T10:00:00Z'},
            {'symbol': 'B', 'strategy': 'macd_wave', 'exit_price': 9,
             'pnl': -50.0, 'exited_at': '2026-04-20T10:05:00Z'},
            {'symbol': 'C', 'strategy': 'macd_wave', 'exit_price': 8,
             'pnl': -75.0, 'exited_at': '2026-04-20T10:10:00Z'},
        ])
        # Force universe non-empty so the method doesn't bail early.
        e.universe = ['X']
        e._sync_intraday_state()
        assert e.daily_pnl == pytest.approx(-25.0)
        assert e.trades_today == 3

    def test_ignores_other_strategy_trades(self, frozen_10_15_et):
        e = _make_engine(universe=['X'])
        e.alpaca.get_snapshots = MagicMock(return_value={})
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        e.db.get_trades_by_date = MagicMock(return_value=[
            {'symbol': 'A', 'strategy': 'bull_flag', 'exit_price': 11,
             'pnl': 500.0, 'exited_at': '2026-04-20T10:00:00Z'},
            {'symbol': 'B', 'strategy': 'orb', 'exit_price': 9,
             'pnl': -200.0, 'exited_at': '2026-04-20T10:05:00Z'},
        ])
        e._sync_intraday_state()
        assert e.daily_pnl == 0.0
        assert e.trades_today == 0

    def test_skips_unfilled_trades(self, frozen_10_15_et):
        """exit_price=None means the trade is still open — don't count it."""
        e = _make_engine(universe=['X'])
        e.alpaca.get_snapshots = MagicMock(return_value={})
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        e.db.get_trades_by_date = MagicMock(return_value=[
            {'symbol': 'A', 'strategy': 'macd_wave', 'exit_price': None,
             'pnl': None, 'exited_at': None},
        ])
        e._sync_intraday_state()
        assert e.daily_pnl == 0.0
        assert e.trades_today == 0


class TestRobustness:
    def test_handles_missing_bars_gracefully(self, frozen_10_15_et):
        """One symbol has bars, another returns empty → good one still recovered."""
        e = _make_engine(universe=['GOOD', 'BAD'], cross_time_max_min=10)
        e.alpaca.get_snapshots = MagicMock(return_value={
            'GOOD': {'open': 10.0, 'prev_close': 9.9, 'high': 12.0},
            'BAD': {'open': 5.0, 'prev_close': 4.9, 'high': 6.0},
        })
        open_utc = ET.localize(datetime(2026, 4, 20, 9, 30)).astimezone(timezone.utc)
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={
            'GOOD': _mk_bars(10.0, [11.2], open_utc, vol=5000),
            'BAD': pd.DataFrame(),  # empty — simulate missing data
        })
        e._sync_intraday_state()
        assert 'GOOD' in e.crossed_stocks
        assert 'BAD' not in e.crossed_stocks
        assert 'BAD' not in e.invalidated  # just missing, not bad

    def test_snapshot_fetch_exception_doesnt_kill_sync(self, frozen_10_15_et):
        """If snapshot chunk fails, method logs warning + continues."""
        e = _make_engine(universe=['X'])
        e.alpaca.get_snapshots = MagicMock(side_effect=RuntimeError("API down"))
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        e.db.get_trades_by_date = MagicMock(return_value=[])
        # Should not raise
        e._sync_intraday_state()
        assert e.universe_opens == {}
        assert e.crossed_stocks == {}

    def test_telegram_notification_fires_on_recovery(self, frozen_10_15_et):
        notifier = MagicMock()
        e = _make_engine(universe=['MOVER'], cross_time_max_min=10)
        e.notifier = notifier
        e.alpaca.get_snapshots = MagicMock(return_value={
            'MOVER': {'open': 10.0, 'prev_close': 9.9, 'high': 12.0},
        })
        open_utc = ET.localize(datetime(2026, 4, 20, 9, 30)).astimezone(timezone.utc)
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={
            'MOVER': _mk_bars(10.0, [11.2], open_utc, vol=5000),
        })
        e._sync_intraday_state()
        notifier.send_message_sync.assert_called_once()
        msg = notifier.send_message_sync.call_args[0][0]
        assert '[MACD Wave]' in msg
        assert 'Restart recovery' in msg

    def test_no_notification_when_nothing_recovered(self, frozen_10_15_et):
        notifier = MagicMock()
        e = _make_engine(universe=['NOTHING'])
        e.notifier = notifier
        e.alpaca.get_snapshots = MagicMock(return_value={
            'NOTHING': {'open': 10.0, 'prev_close': 9.9, 'high': 10.2},  # <+10%
        })
        e.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        e.db.get_trades_by_date = MagicMock(return_value=[])
        e._sync_intraday_state()
        notifier.send_message_sync.assert_not_called()

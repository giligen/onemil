"""Regression for 2026-04-21 slow-entry bug.

Bars for 10 of 11 ORB candidates existed on Alpaca's historical API from
9:30-9:34 ET, but our engine only built range_data for 2-4 of them within
15 minutes of range close. Root cause: `_backfill_range_if_needed` was
called only from `_subscribe_bars` and had an early-return if et_time < 9:35.
Since universe build fires at ~9:31 ET, backfill returned immediately,
and we relied on the WS bar stream — which only delivers bars from
subscribe-time forward and misses the 9:30/9:31 bars that already closed.

Fix: `_ensure_ranges_post_open` sweeps all candidates missing range_data
after 9:35 ET, batch-fetches 1-min bars, ingests them. One-shot via
`_post_open_range_sweep_done` flag; resets on `reset_daily`.
"""
from __future__ import annotations

from datetime import datetime, timezone, time as dtime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine
from trading.stop_monitor import StopMonitor


@pytest.fixture
def orb_cfg():
    yaml_path = Path(__file__).parent.parent / 'orb.yaml'
    with open(yaml_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def engine(orb_cfg):
    orb_cfg['strategy']['enabled'] = True
    alpaca = MagicMock(spec=AlpacaClient)
    alpaca.get_open_positions.return_value = []
    alpaca.get_account_info.return_value = {'buying_power': 100_000.0}
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 1
    db.get_open_trades.return_value = []
    db.get_trades_by_date.return_value = []
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    return ORBEngine(
        alpaca_client=alpaca, db=db, stop_monitor=sm, config=orb_cfg,
    )


def _bars(open_, highs, lows=None, vol=10000, start_utc=None):
    """Build a 5-bar DataFrame for 9:30-9:34 ET (13:30-13:34 UTC on EDT)."""
    if start_utc is None:
        start_utc = datetime(2026, 4, 21, 13, 30, tzinfo=timezone.utc)
    rows = []
    for i, h in enumerate(highs):
        ts = start_utc.replace(minute=30 + i)
        lo = lows[i] if lows else open_
        rows.append({
            'timestamp': ts,
            'open': open_ if i == 0 else highs[i - 1],
            'high': h, 'low': lo, 'close': h, 'volume': vol,
        })
    return pd.DataFrame(rows)


class TestPostOpenSweep:
    def test_backfills_candidates_missing_range(self, engine):
        """Universe has 3 candidates, none have range_data (bars lost via WS).
        Sweep should batch-fetch bars and populate range_data for all."""
        engine.build_universe(source_loader=lambda: ['AAA', 'BBB', 'CCC'])
        # All three have no range_data
        for sym in ['AAA', 'BBB', 'CCC']:
            assert engine.candidates[sym].range_data is None

        # Mock alpaca bar-multi to return complete 5-bar ranges for all three
        engine.alpaca.get_1min_bars_multi = MagicMock(return_value={
            'AAA': _bars(10.0, [10.1, 10.15, 10.2, 10.25, 10.3]),
            'BBB': _bars(20.0, [20.5, 21.0, 20.8, 20.9, 21.1]),
            'CCC': _bars(5.0,  [5.05, 5.1, 5.15, 5.12, 5.18]),
        })

        # Mock et_now to be 9:36 ET (past the 9:35 gate)
        fake_et = datetime(2026, 4, 21, 9, 36, tzinfo=timezone.utc).astimezone()
        with patch('trading.orb_engine.datetime') as MockDT:
            # Our code calls datetime.now(timezone.utc) and then .astimezone(ZoneInfo(...))
            MockDT.now = lambda tz=None: datetime(2026, 4, 21, 13, 36, tzinfo=timezone.utc)
            MockDT.combine = datetime.combine
            MockDT.side_effect = lambda *args, **kw: datetime(*args, **kw)
            engine._ensure_ranges_post_open()

        assert engine.candidates['AAA'].range_data is not None
        assert engine.candidates['BBB'].range_data is not None
        assert engine.candidates['CCC'].range_data is not None
        assert engine._post_open_range_sweep_done is True

    def test_no_op_before_935_et(self, engine):
        """Pre-9:35 ET (e.g., 9:31 ET during universe build) — don't sweep."""
        engine.build_universe(source_loader=lambda: ['AAA'])
        engine.alpaca.get_1min_bars_multi = MagicMock()
        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 21, 13, 31, tzinfo=timezone.utc)
        # Call it — should not invoke bar-multi fetch
            engine._ensure_ranges_post_open()
        engine.alpaca.get_1min_bars_multi.assert_not_called()
        assert engine._post_open_range_sweep_done is False

    def test_no_op_after_1100_et(self, engine):
        """After 11:00 ET — stale; no sweep (mirrors _backfill window)."""
        engine.build_universe(source_loader=lambda: ['AAA'])
        engine.alpaca.get_1min_bars_multi = MagicMock()
        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 21, 15, 30, tzinfo=timezone.utc)
            engine._ensure_ranges_post_open()
        engine.alpaca.get_1min_bars_multi.assert_not_called()

    def test_idempotent_within_day(self, engine):
        """Second call same day is a no-op even past 9:35.

        Since the 2026-07-03 selection-race fix, the FIRST invocation may
        fetch twice (initial + one consolidation-lag retry when candidates
        stay rangeless) — so the idempotency contract is: exactly one
        initial fetch + at most one retry per DAY, regardless of how many
        times the sweep is invoked."""
        engine.build_universe(source_loader=lambda: ['AAA'])
        engine.alpaca.get_1min_bars_multi = MagicMock(return_value={})
        engine.sweep_retry_delay_s = 0.01  # keep the retry, skip the 4s sleep
        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 21, 13, 36, tzinfo=timezone.utc)
            engine._ensure_ranges_post_open()
            engine._ensure_ranges_post_open()
            engine._ensure_ranges_post_open()
        # initial + 1 retry on first invocation; invocations 2-3 no-op
        assert engine.alpaca.get_1min_bars_multi.call_count == 2
        # Even if the grace gate re-arms the sweep, the retry stays
        # once-per-day (no repeated sleep storms on the drain thread).
        engine._post_open_range_sweep_done = False
        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 21, 13, 36, tzinfo=timezone.utc)
            engine._ensure_ranges_post_open()
        assert engine.alpaca.get_1min_bars_multi.call_count == 3  # re-sweep fetch, NO extra retry

    def test_skips_candidates_that_already_have_range(self, engine):
        """Symbols with range_data should be excluded from batch fetch."""
        from trading.orb_engine import RangeData
        engine.build_universe(source_loader=lambda: ['DONE', 'TODO'])
        engine.candidates['DONE'].range_data = RangeData(
            symbol='DONE', range_high=10, range_low=9.5, range_volume=1000,
            range_avg_bar_range_pct=1.0, range_close=9.9,
            range_start_ts=pd.Timestamp.utcnow(),
        )
        captured = {}
        def capture(syms, **kw):
            captured['syms'] = list(syms)
            return {'TODO': _bars(10.0, [10.1, 10.15, 10.2, 10.25, 10.3])}
        engine.alpaca.get_1min_bars_multi = MagicMock(side_effect=capture)
        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 21, 13, 36, tzinfo=timezone.utc)
            engine._ensure_ranges_post_open()
        assert captured.get('syms') == ['TODO']  # DONE excluded

    def test_reset_daily_clears_done_flag(self, engine):
        """reset_daily (called at day boundary) must re-arm the sweep."""
        engine._post_open_range_sweep_done = True
        engine.reset_daily()
        assert engine._post_open_range_sweep_done is False

    def test_empty_bars_doesnt_crash(self, engine):
        """If bar fetch returns empty for some symbols, others still succeed."""
        engine.build_universe(source_loader=lambda: ['GOOD', 'EMPTY'])
        engine.alpaca.get_1min_bars_multi = MagicMock(return_value={
            'GOOD': _bars(10.0, [10.1, 10.15, 10.2, 10.25, 10.3]),
            'EMPTY': pd.DataFrame(),
        })
        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 21, 13, 36, tzinfo=timezone.utc)
            engine._ensure_ranges_post_open()
        assert engine.candidates['GOOD'].range_data is not None
        assert engine.candidates['EMPTY'].range_data is None
        # Flag still set — we don't retry on empty; next WS bar event can still fill
        assert engine._post_open_range_sweep_done is True

    def test_fetch_exception_doesnt_crash_check_entries(self, engine):
        """Sweep failure during check_entries shouldn't abort entry flow."""
        engine.build_universe(source_loader=lambda: ['A'])
        engine.alpaca.get_1min_bars_multi = MagicMock(side_effect=RuntimeError("API down"))
        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 21, 13, 36, tzinfo=timezone.utc)
            # Should NOT raise
            engine._ensure_ranges_post_open()

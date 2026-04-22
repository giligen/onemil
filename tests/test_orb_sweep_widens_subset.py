"""Regression for 2026-04-22 subset-only check_entries bug.

Bug: `check_entries(symbols=subset)` — called from scanner's WS drain path
with only symbols whose bars just arrived via WebSocket — evaluated ONLY
the subset, ignoring candidates that `_ensure_ranges_post_open` had just
filled via REST sweep. Result: Q4 winners (ETHT/RDTL/NBIG/VNCE) skipped,
4 slots filled by Q5 WS subset (CRMX/BKKT/RGTX/BITU) — inverted the
orb.yaml `ranking.order=[Q4, Q5, ...]` policy.

Fix: `_ensure_ranges_post_open` returns the set of sweep-filled symbols;
`check_entries` unions them into `symbols` when running subset-scoped, so
Q4 candidates that arrived only via the REST sweep are still considered.

Also: one INFO log per scored candidate with full feature dump, so live
composite drift (Bug #2 from 4/22) is traceable next session.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, time as dtime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, CandidateState, RangeData
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
    alpaca.get_account_info.return_value = {'buying_power': 500_000.0}
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 1
    db.get_open_trades.return_value = []
    db.get_trades_by_date.return_value = []
    db.get_daily_bars_cached.return_value = {}  # force alpaca fallback / empty context
    sm = MagicMock(spec=StopMonitor)
    sm.polling_mode = False
    sm.drain_exit_events.return_value = []
    return ORBEngine(
        alpaca_client=alpaca, db=db, stop_monitor=sm, config=orb_cfg,
    )


def _bars_5min(open_p: float, highs, lows=None, start_utc=None, vol=10_000):
    """Synthetic 5-bar DataFrame 9:30–9:34 ET."""
    if start_utc is None:
        start_utc = datetime(2026, 4, 22, 13, 30, tzinfo=timezone.utc)
    rows = []
    for i, h in enumerate(highs):
        ts = start_utc.replace(minute=30 + i)
        lo = lows[i] if lows else open_p
        rows.append({
            'timestamp': ts,
            'open': open_p if i == 0 else highs[i - 1],
            'high': h, 'low': lo, 'close': h, 'volume': vol,
        })
    return pd.DataFrame(rows)


class TestSweepReturnsFilledSymbols:
    """_ensure_ranges_post_open now returns the set of sweep-filled syms."""

    def test_returns_filled_symbols_when_sweep_runs(self, engine):
        engine.build_universe(source_loader=lambda: ['AAA', 'BBB', 'CCC'])
        engine.alpaca.get_1min_bars_multi = MagicMock(return_value={
            'AAA': _bars_5min(10.0, [10.1, 10.15, 10.2, 10.25, 10.3]),
            'BBB': _bars_5min(20.0, [20.5, 21.0, 20.8, 20.9, 21.1]),
            'CCC': _bars_5min(5.0,  [5.05, 5.1, 5.15, 5.12, 5.18]),
        })
        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 22, 13, 36, tzinfo=timezone.utc)
            filled = engine._ensure_ranges_post_open()
        assert filled == {'AAA', 'BBB', 'CCC'}

    def test_returns_empty_set_before_935_et(self, engine):
        engine.build_universe(source_loader=lambda: ['AAA'])
        engine.alpaca.get_1min_bars_multi = MagicMock()
        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 22, 13, 31, tzinfo=timezone.utc)
            filled = engine._ensure_ranges_post_open()
        assert filled == set()

    def test_returns_empty_set_after_1100_et(self, engine):
        engine.build_universe(source_loader=lambda: ['AAA'])
        engine.alpaca.get_1min_bars_multi = MagicMock()
        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 22, 15, 30, tzinfo=timezone.utc)
            filled = engine._ensure_ranges_post_open()
        assert filled == set()

    def test_returns_empty_set_when_already_done(self, engine):
        engine.build_universe(source_loader=lambda: ['AAA'])
        engine._post_open_range_sweep_done = True
        engine.alpaca.get_1min_bars_multi = MagicMock()
        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 22, 13, 36, tzinfo=timezone.utc)
            filled = engine._ensure_ranges_post_open()
        assert filled == set()
        engine.alpaca.get_1min_bars_multi.assert_not_called()

    def test_returns_empty_set_when_no_missing(self, engine):
        """All candidates already have range_data — no sweep, empty return."""
        engine.build_universe(source_loader=lambda: ['X'])
        engine.candidates['X'].range_data = RangeData(
            symbol='X', range_high=10, range_low=9.5, range_volume=1000,
            range_avg_bar_range_pct=1.0, range_close=9.9,
            range_start_ts=pd.Timestamp('2026-04-22 13:30:00+00:00'),
        )
        engine.alpaca.get_1min_bars_multi = MagicMock()
        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 22, 13, 36, tzinfo=timezone.utc)
            filled = engine._ensure_ranges_post_open()
        assert filled == set()
        assert engine._post_open_range_sweep_done is True


class TestCheckEntriesWidensSymbolsAfterSweep:
    """The 2026-04-22 bug proper: subset-scoped check_entries must widen to
    include sweep-filled candidates so Q4 winners aren't silently dropped."""

    def test_sweep_filled_candidates_included_when_called_with_subset(self, engine):
        """Call check_entries(symbols={'WSHP'}) simulating WS drain.
        Sweep fills ranges for {'ETHT','WSHP'} — both must be in eligible set."""
        engine.build_universe(source_loader=lambda: ['WSHP', 'ETHT'])
        # Both candidates missing range_data initially (pre-sweep state)
        assert engine.candidates['WSHP'].range_data is None
        assert engine.candidates['ETHT'].range_data is None

        engine.alpaca.get_1min_bars_multi = MagicMock(return_value={
            'WSHP': _bars_5min(11.0, [11.2, 11.3, 11.25, 11.4, 11.5]),
            'ETHT': _bars_5min(19.4, [19.5, 19.53, 19.45, 19.5, 19.48]),
        })

        captured_syms = []
        orig_compute = engine._compute_features

        def _capture_compute(cand, *a, **kw):
            captured_syms.append(cand.symbol)
            return orig_compute(cand, *a, **kw)

        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 22, 13, 36, tzinfo=timezone.utc)
            with patch.object(engine, '_compute_features', side_effect=_capture_compute):
                # Simulate scanner's WS drain path: only WSHP was touched by WS
                engine.check_entries(symbols={'WSHP'})
        # BUG FIX: both WSHP (from subset) AND ETHT (from sweep) must score.
        # Before fix, only WSHP would have been in captured_syms.
        assert 'WSHP' in captured_syms, "WSHP (subset) should be scored"
        assert 'ETHT' in captured_syms, (
            "ETHT (sweep-filled, not in WS subset) MUST be scored — "
            "this is the 2026-04-22 regression"
        )

    def test_no_widening_when_called_with_none_symbols(self, engine):
        """check_entries(symbols=None) already iterates all candidates —
        no widening needed or done; semantics unchanged."""
        engine.build_universe(source_loader=lambda: ['AAA'])
        engine.alpaca.get_1min_bars_multi = MagicMock(return_value={
            'AAA': _bars_5min(10.0, [10.1, 10.15, 10.2, 10.25, 10.3]),
        })
        captured_syms = []

        def _capture_compute(cand, *a, **kw):
            captured_syms.append(cand.symbol)
            return {}

        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 22, 13, 36, tzinfo=timezone.utc)
            with patch.object(engine, '_compute_features', side_effect=_capture_compute):
                engine.check_entries(symbols=None)  # full scan — no subset
        assert 'AAA' in captured_syms


class TestOrbScoredTelemetry:
    """Per-scored-symbol INFO log — used to diff live vs BT composites."""

    def test_emits_orb_scored_log_line(self, engine, caplog):
        """Every candidate that survives to quintile assignment must emit
        one 'ORB SCORED' line with composite + quintile + 7 features."""
        engine.build_universe(source_loader=lambda: ['WSHP'])
        cand = engine.candidates['WSHP']
        cand.range_data = RangeData(
            symbol='WSHP', range_high=11.5, range_low=11.0, range_volume=50_000,
            range_avg_bar_range_pct=2.0, range_close=11.4,
            range_start_ts=pd.Timestamp('2026-04-22 13:30:00+00:00'),
            range_open=11.1,
        )
        providers = {'WSHP': {
            'prev_day_bar': {'open': 10.5, 'high': 10.8, 'low': 10.0,
                             'close': 10.2, 'volume': 100_000},
            'daily_stats_20d': {'high_20d': 12.0, 'volume_20d': 80_000},
        }}
        with patch('trading.orb_engine.datetime') as MockDT:
            MockDT.now = lambda tz=None: datetime(2026, 4, 22, 13, 36, tzinfo=timezone.utc)
            with caplog.at_level(logging.INFO, logger='trading.orb_engine'):
                engine.check_entries(symbols={'WSHP'}, feature_providers=providers)
        scored_lines = [r.message for r in caplog.records
                        if 'ORB SCORED' in r.message]
        assert len(scored_lines) == 1, f"expected 1 scored line, got {scored_lines}"
        msg = scored_lines[0]
        # Must carry composite, quintile, and all 7 feature tokens
        assert 'WSHP' in msg
        assert 'comp=' in msg
        assert 'gap=' in msg
        assert 'rtv=' in msg        # range_total_volume
        assert 'rabr=' in msg       # range_avg_bar_range_pct
        assert 'rs=' in msg         # range_size_pct
        assert 'p20h=' in msg       # price_vs_20d_high_pct
        assert 'pdcp=' in msg       # prev_day_close_position
        assert 'rcp=' in msg        # range_close_position
        assert 'prev_close=' in msg
        assert 'range_open=' in msg

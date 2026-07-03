"""Selection-race fix (2026-07-03) — CRCD/AVEX/FABC/RGNX missed-winner class.

Incident: the 9:34 bar consolidates at the vendor with 2-10s lag, so the
first post-open sweep left ~20% of candidates rangeless (7/2: 21/27). The
9:35:01 check_entries then ranked the READY subset and burned all
max_concurrent daily slots within 3 seconds; late-consolidating names —
including BT-selected winners (CRCD +$15.8K model on 6/30) — were locked
out for the day by the daily cap. BT ranks the full field (full-day bars),
so this was a pure live-vs-BT selection divergence.

Fix layers under test:
  A. `_ensure_ranges_post_open` retries still-rangeless candidates once
     after `entry.sweep_retry_delay_s`.
  B. check_entries first-rank grace gate: before the day's FIRST placement,
     if pool candidates are rangeless and we're within
     `entry.first_rank_grace_s` of range end, defer ranking (return []).
     After grace expiry or once anything has been placed, proceed.
  C. `_audit_selection` persists every placement-burst's ranked field to
     logs/orb_selection_audit.jsonl.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.orb_engine import ORBEngine, RangeData, CandidateState
from trading.stop_monitor import StopMonitor


@pytest.fixture
def orb_cfg():
    with open(Path(__file__).parent.parent / 'orb.yaml') as f:
        cfg = yaml.safe_load(f)
    cfg['strategy']['enabled'] = True
    return cfg


@pytest.fixture
def mock_alpaca():
    c = MagicMock(spec=AlpacaClient)
    c.get_open_positions.return_value = []
    c.get_account_info.return_value = {'buying_power': 100_000.0}
    c.get_latest_quote.return_value = {'bid_price': 9.98, 'ask_price': 10.00}
    c.submit_stop_bracket_order.return_value = {'id': 'o-1', 'status': 'accepted'}
    c.get_daily_bars.return_value = {}
    return c


@pytest.fixture
def engine(orb_cfg, mock_alpaca):
    db = MagicMock(spec=Database)
    db.save_trade.return_value = 1
    db.get_open_trades.return_value = []
    sm = MagicMock(spec=StopMonitor)
    sm.drain_exit_events.return_value = []
    eng = ORBEngine(alpaca_client=mock_alpaca, db=db, stop_monitor=sm, config=orb_cfg)
    # no DB-recorded entries today
    eng._symbols_entered_today_db = MagicMock(return_value=set())
    return eng


def _rng(sym):
    return RangeData(symbol=sym, range_high=10.0, range_low=9.5, range_volume=50_000,
                     range_avg_bar_range_pct=1.0, range_close=9.9,
                     range_start_ts=pd.Timestamp.utcnow(), range_open=9.6)


class TestConfigKnobs:
    def test_knobs_loaded_from_yaml(self, engine):
        assert engine.sweep_retry_delay_s == 4.0
        assert engine.first_rank_grace_s == 25.0


class TestFirstRankGraceGate:
    def _seed(self, engine, ranged, rangeless):
        engine.build_universe(source_loader=lambda: ranged + rangeless)
        for s in ranged:
            engine.candidates[s].range_data = _rng(s)
        # rangeless candidates keep range_data=None

    def _at(self, et_hh, et_mm, et_ss):
        """UTC datetime mapping to the given ET wall time (July -> EDT)."""
        return datetime(2026, 7, 6, et_hh + 4, et_mm, et_ss, tzinfo=timezone.utc)

    def _gate(self, engine, pool, at):
        # `pool` kept in the helper signature to document each scenario's
        # calling context, but the gate inspects engine.candidates directly
        # (2026-07-04 fix: subset-scoping made it blind on the WS-drain path).
        with patch('trading.orb_engine.datetime') as mdt:
            mdt.now.return_value = at
            mdt.combine = datetime.combine
            return engine._should_defer_first_rank()

    def test_defers_within_grace_when_field_incomplete(self, engine, mock_alpaca):
        """9:35:05, CRCD still rangeless -> defer (the incident scenario)."""
        self._seed(engine, ranged=['AAA', 'BBB', 'CCC'], rangeless=['CRCD'])
        assert self._gate(engine, ['AAA', 'BBB', 'CCC', 'CRCD'],
                          self._at(9, 35, 5)) is True

    def test_proceeds_when_field_complete(self, engine, mock_alpaca):
        self._seed(engine, ranged=['AAA', 'BBB', 'CCC', 'CRCD'], rangeless=[])
        assert self._gate(engine, ['AAA', 'BBB', 'CCC', 'CRCD'],
                          self._at(9, 35, 5)) is False

    def test_proceeds_after_grace_expiry(self, engine, mock_alpaca):
        """9:35:40 (grace 25s expired): never wait forever — fills matter."""
        self._seed(engine, ranged=['AAA', 'BBB'], rangeless=['CRCD'])
        assert self._gate(engine, ['AAA', 'BBB', 'CRCD'],
                          self._at(9, 35, 40)) is False

    def test_gate_off_when_grace_zero(self, engine, mock_alpaca):
        engine.first_rank_grace_s = 0.0
        self._seed(engine, ranged=['AAA'], rangeless=['CRCD'])
        assert self._gate(engine, ['AAA', 'CRCD'], self._at(9, 35, 5)) is False

    def test_gate_inactive_before_range_end(self, engine, mock_alpaca):
        """9:33 (range not even complete): gate must not claim the defer —
        pre-range ticks are governed by range detection, not the gate."""
        self._seed(engine, ranged=[], rangeless=['CRCD'])
        assert self._gate(engine, ['CRCD'], self._at(9, 33, 0)) is False

    def test_check_entries_wires_gate_before_ranking(self, engine, mock_alpaca):
        """End-to-end: within grace + incomplete field + nothing placed today
        -> check_entries returns [] and submits nothing."""
        self._seed(engine, ranged=['AAA', 'BBB', 'CCC'], rangeless=['CRCD'])
        engine._audit_selection = MagicMock()
        with patch('trading.orb_engine.datetime') as mdt:
            mdt.now.return_value = self._at(9, 35, 5)
            mdt.combine = datetime.combine
            mdt.strptime = datetime.strptime
            out = engine.check_entries()
        assert out == []
        engine._audit_selection.assert_not_called()
        mock_alpaca.submit_stop_bracket_order.assert_not_called()

    def test_gate_skipped_after_first_placement(self, engine, mock_alpaca):
        """check_entries only consults the gate when nothing has been placed
        today (the day's selection is committed after the first burst)."""
        engine._symbols_entered_today_db = MagicMock(return_value={'AAA'})
        self._seed(engine, ranged=['BBB'], rangeless=['CRCD'])
        engine._should_defer_first_rank = MagicMock(return_value=True)
        with patch('trading.orb_engine.datetime') as mdt:
            mdt.now.return_value = self._at(9, 35, 5)
            mdt.combine = datetime.combine
            mdt.strptime = datetime.strptime
            engine.check_entries()
        engine._should_defer_first_rank.assert_not_called()


class TestSweepRetry:
    def test_retry_fetches_unfilled_once(self, engine, mock_alpaca):
        """First multi-fetch misses one candidate's bars; the retry supplies
        them; sweep reports it filled."""
        engine.build_universe(source_loader=lambda: ['AAA', 'LATE'])
        # 5 complete 9:30-9:34 bars (13:30-13:34 UTC in July/EDT)
        def bars(sym):
            return pd.DataFrame([
                {'timestamp': pd.Timestamp(f'2026-07-06 13:{30+i}:00', tz='UTC'),
                 'open': 9.6, 'high': 10.0, 'low': 9.5, 'close': 9.9,
                 'volume': 10_000}
                for i in range(5)
            ])
        calls = {'n': 0}
        def multi(symbols, lookback_minutes=60):
            calls['n'] += 1
            if calls['n'] == 1:
                return {'AAA': bars('AAA')}          # LATE missing (lag)
            return {'LATE': bars('LATE')}            # retry delivers
        mock_alpaca.get_1min_bars_multi = MagicMock(side_effect=multi)
        engine.sweep_retry_delay_s = 0.01            # fast test
        with patch('trading.orb_engine._first_session_open_ts_utc',
                   wraps=None) as _:
            pass
        # Freeze ET gate inside sweep (9:36 ET)
        with patch('trading.orb_engine.datetime') as mdt:
            mdt.now.return_value = datetime(2026, 7, 6, 13, 36, 0, tzinfo=timezone.utc)
            mdt.combine = datetime.combine
            filled = engine._ensure_ranges_post_open()
        assert calls['n'] == 2, "retry fetch must run for unfilled candidates"
        assert 'AAA' in filled and 'LATE' in filled
        assert engine.candidates['LATE'].range_data is not None

    def test_no_retry_when_disabled(self, engine, mock_alpaca):
        engine.build_universe(source_loader=lambda: ['LATE'])
        mock_alpaca.get_1min_bars_multi = MagicMock(return_value={})
        engine.sweep_retry_delay_s = 0.0
        with patch('trading.orb_engine.datetime') as mdt:
            mdt.now.return_value = datetime(2026, 7, 6, 13, 36, 0, tzinfo=timezone.utc)
            mdt.combine = datetime.combine
            engine._ensure_ranges_post_open()
        assert mock_alpaca.get_1min_bars_multi.call_count == 1


class TestSelectionAudit:
    def test_audit_written_direct(self, engine, tmp_path):
        """_audit_selection appends a JSONL record with ranked field, picks,
        and the rangeless pool."""
        engine.build_universe(source_loader=lambda: ['AAA', 'CRCD'])
        engine.candidates['AAA'].range_data = _rng('AAA')
        engine.candidates['AAA'].composite = 0.42
        engine.candidates['AAA'].quintile = 'Q4'
        scored = [engine.candidates['AAA']]
        with patch('trading.orb_engine.__file__',
                   str(tmp_path / 'trading' / 'orb_engine.py')):
            engine._audit_selection(['AAA'], ['AAA'], scored)
        p = tmp_path / 'logs' / 'orb_selection_audit.jsonl'
        assert p.exists()
        rec = json.loads(p.read_text().strip().splitlines()[-1])
        assert rec['picks'] == ['AAA']
        assert rec['ranked'][0]['sym'] == 'AAA'
        assert rec['ranked'][0]['q'] == 'Q4'
        assert 'CRCD' in rec['rangeless_pool']

    def test_audit_never_raises(self, engine):
        """Unwritable path must not break the trade path."""
        with patch('trading.orb_engine.__file__', '/dev/null/x/orb_engine.py'):
            engine._audit_selection([], [], [])   # must not raise


class TestGateFullPoolScope:
    """2026-07-04 review fix: the gate must inspect the FULL candidate pool,
    not the caller's subset. The day's first burst arrives via the WS drain
    as check_entries(symbols=<ready subset>) — every member has a range by
    construction, so a subset-scoped rangeless check never fired on exactly
    the racing path (the original CRCD/AVEX/FABC/RGNX failure survived the
    'fix')."""

    def _at(self, et_hh, et_mm, et_ss):
        return datetime(2026, 7, 6, et_hh + 4, et_mm, et_ss, tzinfo=timezone.utc)

    def test_gate_fires_even_when_caller_scope_is_ready_subset(self, engine):
        """WS-drain scenario: CRCD rangeless in the pool; the drain calls
        with only the ready names. Gate must still defer."""
        engine.build_universe(source_loader=lambda: ['AAA', 'BBB', 'CRCD'])
        engine.candidates['AAA'].range_data = _rng('AAA')
        engine.candidates['BBB'].range_data = _rng('BBB')
        # CRCD rangeless — WS drain would call check_entries(['AAA','BBB'])
        engine._audit_selection = MagicMock()
        with patch('trading.orb_engine.datetime') as mdt:
            mdt.now.return_value = self._at(9, 35, 5)
            mdt.combine = datetime.combine
            mdt.strptime = datetime.strptime
            out = engine.check_entries(symbols=['AAA', 'BBB'])
        assert out == []
        engine._audit_selection.assert_not_called()

    def test_defer_rearms_post_open_sweep(self, engine):
        """Deferral must re-arm the sweep so stragglers get actively
        re-fetched (WS alone lacks the 9:30 anchor bar — without re-arm the
        defer is pure entry delay and the field can never complete)."""
        engine.build_universe(source_loader=lambda: ['AAA', 'CRCD'])
        engine.candidates['AAA'].range_data = _rng('AAA')
        engine._post_open_range_sweep_done = True
        with patch('trading.orb_engine.datetime') as mdt:
            mdt.now.return_value = self._at(9, 35, 5)
            mdt.combine = datetime.combine
            mdt.strptime = datetime.strptime
            engine.check_entries(symbols=['AAA'])
        assert engine._post_open_range_sweep_done is False


class TestEtOffsetDstAccuracy:
    """2026-07-04 review fix: month-granularity DST offsets zeroed ORB
    entries for the Mar-1→2nd-Sunday and Nov-1→1st-Sunday windows (the 9:30
    session-open mask matched nothing). ZoneInfo is now authoritative."""

    def test_transition_weeks(self):
        from trading.orb_engine import _et_offset_hours
        cases = [
            # 2027: DST Mar 14 → Nov 7
            (datetime(2027, 3, 10, 14, 30, tzinfo=timezone.utc), 5),  # pre-spring: EST
            (datetime(2027, 3, 15, 13, 30, tzinfo=timezone.utc), 4),  # post-spring: EDT
            (datetime(2027, 11, 3, 13, 30, tzinfo=timezone.utc), 4),  # pre-fall: still EDT
            (datetime(2027, 11, 8, 14, 30, tzinfo=timezone.utc), 5),  # post-fall: EST
        ]
        for dt, expected in cases:
            assert _et_offset_hours(dt) == expected, dt

    def test_naive_datetime_treated_as_utc(self):
        from trading.orb_engine import _et_offset_hours
        assert _et_offset_hours(datetime(2027, 7, 6, 14, 0)) == 4

"""Forward-instrument counterfactuals for the HOD-break dry run (`hod_break.log_counterfactuals`, default OFF,
module docstring of `trading.hod_break_engine`). Three columns collected on the resting-entry dry run, never
touching a gate/size/order: (1) `scanner_qualified_at_arm`, (2) `cf_floor_stop_px`, (3) `cf_floor_stop_hit` /
`cf_stoplimit_exit_px` (the latter two in a separate file, `cf_ledger_path`, via `CFWatch`).

Reuses the resting-entry fixtures/tape from tests/test_hod_resting_entry.py (same synthetic day: level 11.0,
stop 10.6, mocked ask 11.015) so a dry fill is driven exactly as the production bar-fallback path drives one.
"""
import csv
from datetime import timedelta

import pytest

from trading.hod_break_engine import HodBreakEngine
from tests.test_hod_break_engine import cfg, bars_df, admit
from tests.test_hod_resting_entry import mock_alpaca, mock_db, mock_sm, read_ledger, tape, trades_db


def cf_resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path, is_qualified=None, **over):
    """Same setup as tests/test_hod_resting_entry.py::resting_engine, plus the `is_qualified` constructor kwarg
    (item 1) that helper doesn't forward (it only builds the cfg dict, not __init__ kwargs)."""
    over.setdefault('cf_ledger_path', str(tmp_path / 'hod_dry_counterfactuals.csv'))
    c = cfg(dry_run=True, entry_mode='resting_stop_limit', **over)
    c['dry_ledger_path'] = str(tmp_path / 'hod_dry_entry_ledger.csv')
    e = HodBreakEngine(mock_alpaca, mock_db, mock_sm, cfg=c, is_qualified=is_qualified)
    e._roll_session()
    e.live_since = e._bar_close_et(0)
    return e


class TestFlagOff:
    def test_columns_present_but_blank_and_no_behaviour_change(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = cf_resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path)   # log_counterfactuals defaults False
        assert e.log_counterfactuals is False
        admit(e); e._ingest_bars('ABC', bars_df(tape()))
        rows = read_ledger(e.dry_ledger_path)
        assert rows[0]['filled'] == '1'                                  # unchanged behaviour
        assert rows[0]['scanner_qualified_at_arm'] == '' and rows[0]['cf_floor_stop_px'] == ''
        assert e.candidates['ABC'].symbol == 'ABC'                       # sanity: fill happened normally
        assert not __import__('os').path.exists(e.cf_ledger_path)        # no watch ever opened -> no file


class TestIsQualifiedWiring:
    def test_none_callable_warns_once_at_boot(self, mock_alpaca, mock_db, mock_sm, tmp_path, caplog):
        with caplog.at_level('WARNING'):
            e = cf_resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path, log_counterfactuals=True)
        assert e.is_qualified is None
        assert any('is_qualified' in r.message for r in caplog.records)

    def test_scanner_qualified_at_arm_evaluated_at_arm_time(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        seen = []
        e = cf_resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path, log_counterfactuals=True,
                               is_qualified=lambda s: seen.append(s) or (s == 'ABC'))
        admit(e); e._ingest_bars('ABC', bars_df(tape()))
        assert seen and seen[0] == 'ABC'
        rows = read_ledger(e.dry_ledger_path)
        assert rows[0]['scanner_qualified_at_arm'] == '1'


class TestCfFloorStopPx:
    def test_computed_at_fill_as_min_of_stop_and_fill_times_0975(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = cf_resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path, log_counterfactuals=True, is_qualified=lambda s: True)
        admit(e); e._ingest_bars('ABC', bars_df(tape()))
        cand = e.candidates['ABC']
        w = e._cf_watches['ABC']
        fill_px = 11.015                                                   # the mocked ask (<= limit) — production fill price
        expected = min(w.actual_stop, fill_px * 0.975)                      # the arm active AT the fill, re-armed several times by the consolidation walk
        assert cand.cf_floor_stop_px == pytest.approx(expected)
        rows = read_ledger(e.dry_ledger_path)
        assert float(rows[0]['cf_floor_stop_px']) == pytest.approx(expected, abs=1e-3)


class TestHeaderBackwardCompatibility:
    def test_existing_old_header_file_is_never_rewritten(self, mock_alpaca, mock_db, mock_sm, tmp_path, caplog):
        path = tmp_path / 'hod_dry_entry_ledger.csv'
        old_header = ['date', 'symbol', 'arm_ts', 'cross_ts', 'level', 'trigger', 'limit', 'ask',
                      'filled', 'fill_px', 'stop', 'target', 'tape_accurate', 'live']
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(old_header); w.writerow(['2026-01-01', 'ZZZ'] + ['x'] * 12)
        e = cf_resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path, log_counterfactuals=True, is_qualified=lambda s: True)
        e.dry_ledger_path = str(path)
        with caplog.at_level('WARNING'):
            admit(e); e._ingest_bars('ABC', bars_df(tape()))
        with open(path, newline='') as fh:
            all_rows = list(csv.reader(fh))
        assert all_rows[0] == old_header                                 # header untouched
        assert len(all_rows[1]) == len(old_header)                       # pre-existing row untouched
        assert len(all_rows[2]) == len(old_header)                       # NEW row keeps the OLD shape too
        assert any('predates the counterfactual columns' in r.message for r in caplog.records)


class TestCfWatchViaPrints:
    """Drives a dry fill (bar-fallback) then feeds synthetic trade prints through the real `_on_trade_print`
    handler — the same entry point the WS thread calls in production — to an exit, and checks the separate
    counterfactual file (item 3)."""

    def _armed_engine(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = cf_resting_engine(mock_alpaca, mock_db, mock_sm, tmp_path, log_counterfactuals=True, is_qualified=lambda s: True)
        admit(e); e._ingest_bars('ABC', bars_df(tape()))
        assert e.candidates['ABC'].resting_filled is True
        assert 'ABC' in e._cf_watches                                    # watch opened on the dry fill
        return e

    def test_target_exit_without_a_floor_touch_resolves_immediately(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = self._armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        w = e._cf_watches['ABC']
        assert w.floor_stop_px == pytest.approx(min(w.actual_stop, w.fill_px * 0.975))
        # runs straight to the target, never dipping to the floor/actual stop -> no stop-limit leg to wait on
        e._on_trade_print('ABC', w.actual_target + 0.01, 100, 0.0)
        assert 'ABC' not in e._cf_watches                                # resolved and dropped
        rows = read_ledger(e.cf_ledger_path)
        assert len(rows) == 1
        assert rows[0]['exit_reason'] == 'target' and rows[0]['cf_floor_stop_hit'] == '0'
        assert rows[0]['cf_stoplimit_px'] == '' and rows[0]['cf_stoplimit_exit_px'] == ''
        assert rows[0]['scanner_qualified_at_arm'] == '1'                 # item 1, carried onto the self-sufficient row
        assert rows[0]['exit_ts'] != '' and rows[0]['arm_level'] != '' and rows[0]['arm_trigger'] != ''
        mock_sm.unsubscribe.assert_any_call(['ABC'])

    def test_eod_flat_with_no_touch_writes_one_row(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        """A fill that never reaches its target or its actual stop all day must still get exactly one row
        (item (1) of the review: the target/stop paths write, but so must EOD/flat)."""
        e = self._armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        now = [e._et_now().replace(hour=15, minute=56)]
        e._et_now = lambda: now[0]
        assert e.is_force_close_time() is True
        e._sweep_cf_watch_timeouts()
        assert 'ABC' not in e._cf_watches
        rows = read_ledger(e.cf_ledger_path)
        assert len(rows) == 1
        assert rows[0]['exit_reason'] == 'eod'
        assert float(rows[0]['exit_px']) == pytest.approx(float(rows[0]['fill_px']))   # no print ever seen -> falls back to the fill price
        assert rows[0]['scanner_qualified_at_arm'] == '1'
        mock_sm.unsubscribe.assert_any_call(['ABC'])

    def test_eod_flat_while_stoplimit_leg_still_pending_closes_both(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        """Session ends mid-60s-window (actual stop touched, counterfactual stop-limit still resting) — EOD must
        close the whole watch in one row, not leave it stuck waiting past the close."""
        e = self._armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        w = e._cf_watches['ABC']
        now = [e._et_now()]
        e._et_now = lambda: now[0]
        e._on_trade_print('ABC', w.actual_stop - 0.01, 100, 0.0)          # touches the actual stop -> arms the stop-limit + is the exit
        assert 'ABC' in e._cf_watches
        now[0] = now[0].replace(hour=15, minute=56)                      # session flat before the 60s deadline elapses
        e._sweep_cf_watch_timeouts()
        assert 'ABC' not in e._cf_watches
        rows = read_ledger(e.cf_ledger_path)
        assert len(rows) == 1 and rows[0]['exit_reason'] == 'stop'
        assert float(rows[0]['cf_stoplimit_exit_px']) == pytest.approx(w.actual_stop - 0.01)  # last print seen

    def test_floor_and_actual_stop_touch_sets_hit_and_arms_stoplimit(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        """floor_stop_px = min(stop, fill*0.975) == the actual stop itself in this fixture's numbers (fill*0.975
        > stop), so a print at the actual stop is simultaneously the floor touch and the recorded exit."""
        e = self._armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        w = e._cf_watches['ABC']
        e._on_trade_print('ABC', w.actual_stop - 0.001, 100, 0.0)
        rows_pending = e._cf_watches['ABC']                               # not yet resolved: waiting on the stop-limit leg
        assert rows_pending.floor_hit is True and rows_pending.exit_reason == 'stop'

    def test_stoplimit_counterfactual_fills_inside_60s(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = self._armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        w = e._cf_watches['ABC']
        now = [e._et_now()]
        e._et_now = lambda: now[0]
        e._on_trade_print('ABC', w.actual_stop - 0.01, 100, 0.0)          # touches the ACTUAL stop -> arms stop-limit + is the recorded exit
        assert w.stoplimit_px == pytest.approx(round(w.actual_stop * (1 - 0.0020), 4))
        assert 'ABC' in e._cf_watches                                     # still waiting on the stop-limit leg
        now[0] = now[0] + timedelta(seconds=10)
        e._on_trade_print('ABC', w.stoplimit_px + 0.005, 50, 0.0)         # fills the counterfactual stop-limit
        assert 'ABC' not in e._cf_watches
        rows = read_ledger(e.cf_ledger_path)
        assert rows[0]['exit_reason'] == 'stop'
        assert float(rows[0]['cf_stoplimit_exit_px']) == pytest.approx(w.stoplimit_px + 0.005, abs=1e-3)

    def test_stoplimit_no_fill_tail_resolves_at_60s_via_bar_sweep(self, mock_alpaca, mock_db, mock_sm, tmp_path):
        e = self._armed_engine(mock_alpaca, mock_db, mock_sm, tmp_path)
        w = e._cf_watches['ABC']
        now = [e._et_now()]
        e._et_now = lambda: now[0]
        e._on_trade_print('ABC', w.actual_stop - 0.01, 100, 0.0)          # touches actual stop, arms + exits
        last_seen = w.actual_stop - 0.5                                  # well below the stop-limit (0.20% of actual_stop) — never fills it
        e._on_trade_print('ABC', last_seen, 50, 0.0)
        now[0] = now[0] + timedelta(seconds=61)
        e._sweep_cf_watch_timeouts()                                     # called once/bar close in production
        assert 'ABC' not in e._cf_watches
        rows = read_ledger(e.cf_ledger_path)
        assert float(rows[0]['cf_stoplimit_exit_px']) == pytest.approx(last_seen, abs=1e-6)   # no-fill tail = last print seen

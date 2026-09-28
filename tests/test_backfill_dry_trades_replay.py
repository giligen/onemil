"""Unit + integration tests for scripts/backfill_dry_trades.py's capped-book replay (owner 9/28 ops
task: "why did 9/28 produce counterfactual watches only and no capped dry entries").

Diagnosis this fix encodes: the resting-fill simulator (trading/hod_break_engine.py's dry-fill path,
~890-935) has NO max_per_day/max_concurrent check — those caps only gate the separate LIVE-order path
(_arm_live_order, ~1267-1270/1556-1563, `if not self.dry_run`). With log_counterfactuals on (shipped
9/26; logs/hod_dry_counterfactuals.csv has no row before 9/28), every dry fill becomes part of an
UNCAPPED all-armed population (source='counterfactual_all', renamed from the old 'backfill_ledger'), not
the ~12/day 4-concurrent book. build_capped_replay_rows reconstructs the actual capped book by replaying
that population through trading.hod_break.run_book (the SAME cap-simulation helper the backtest and
scripts/hod_break_eod_check.py use), first-come by ARM time, using the ACTUAL exit leg — never the floor
counterfactual.
"""
import csv

import pytest

from persistence.database import Database

import scripts.backfill_dry_trades as backfill

ENTRY_HEADER = ['date', 'symbol', 'arm_ts', 'cross_ts', 'level', 'trigger', 'limit', 'ask',
                'filled', 'fill_px', 'stop', 'target', 'tape_accurate', 'live']
CF_HEADER = ['date', 'symbol', 'fill_ts', 'fill_px', 'scanner_qualified_at_arm', 'arm_level', 'arm_trigger',
             'actual_stop', 'actual_target', 'cf_floor_stop_px', 'cf_floor_stop_hit', 'cf_stoplimit_px',
             'cf_stoplimit_exit_px', 'exit_px', 'exit_reason', 'exit_ts']


def _entry_row(date, symbol, arm_ts, cross_ts, fill_px, stop, target=None, filled='1', extra_13col=False):
    """One 14-column (or, with extra_13col=True, legacy 13-column) entry-ledger row, positionally
    matching trading/hod_break_engine.py's _append_dry_ledger."""
    row = [date, symbol, arm_ts, cross_ts, '0', '0', '0', '0', filled,
           '' if fill_px is None else f"{fill_px:.4f}", f"{stop:.4f}",
           '' if target is None else f"{target:.4f}", '0']
    if not extra_13col:
        row.append('1')  # `live`
    return row


def _cf_row(date, symbol, fill_ts, exit_px, exit_reason, exit_ts,
            floor_stop_px=None, stoplimit_exit_px=None):
    """One counterfactual-ledger row. floor_stop_px/stoplimit_exit_px default to values that would
    yield an OPPOSITE-signed r_multiple if a caller mistakenly used them instead of exit_px/exit_reason
    — see TestActualVsFloorLeg."""
    return {
        'date': date, 'symbol': symbol, 'fill_ts': fill_ts, 'fill_px': '', 'scanner_qualified_at_arm': '0',
        'arm_level': '', 'arm_trigger': '', 'actual_stop': '', 'actual_target': '',
        'cf_floor_stop_px': '' if floor_stop_px is None else f"{floor_stop_px:.4f}",
        'cf_floor_stop_hit': '0', 'cf_stoplimit_px': '',
        'cf_stoplimit_exit_px': '' if stoplimit_exit_px is None else f"{stoplimit_exit_px:.4f}",
        'exit_px': f"{exit_px:.4f}", 'exit_reason': exit_reason, 'exit_ts': exit_ts,
    }


def _write_entry_csv(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(ENTRY_HEADER)
        for r in rows:
            w.writerow(r)


def _write_cf_csv(path, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=CF_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow(r)


ET_DATE = '2026-09-28'


# --------------------------------------------------------------------------------------------- unit: cap ordering
class TestCapOrdering:
    def test_first_come_by_arm_time_not_fill_time(self, tmp_path):
        """LATE arms first (09:30) but FILLS last (09:36); EARLY arms second (09:32) but fills first
        (09:33). max_concurrent=1 forces a choice. run_book must pick LATE (arm-time order) — a
        cross_ts/fill-time-ordered implementation would pick EARLY instead."""
        entry_csv = tmp_path / 'entry.csv'
        cf_csv = tmp_path / 'cf.csv'
        _write_entry_csv(entry_csv, [
            _entry_row(ET_DATE, 'LATE', f'{ET_DATE}T09:30:00-04:00', f'{ET_DATE}T09:36:00-04:00', 10.0, 9.5),
            _entry_row(ET_DATE, 'EARLY', f'{ET_DATE}T09:32:00-04:00', f'{ET_DATE}T09:33:00-04:00', 20.0, 19.5),
        ])
        _write_cf_csv(cf_csv, [
            _cf_row(ET_DATE, 'LATE', f'{ET_DATE}T09:36:00-04:00', 9.5, 'stop', f'{ET_DATE}T09:40:00-04:00'),
            _cf_row(ET_DATE, 'EARLY', f'{ET_DATE}T09:33:00-04:00', 21.0, 'target', f'{ET_DATE}T09:45:00-04:00'),
        ])
        rows = backfill.build_capped_replay_rows(str(entry_csv), str(cf_csv), max_per_day=10, max_concurrent=1)
        assert [r['symbol'] for r in rows] == ['LATE']

    def test_concurrency_limit_and_causal_freeing(self, tmp_path):
        """A(09:30) opens the only slot (max_concurrent=1) until it exits at 09:40. B arms 09:31 while
        A is still open -> rejected. C arms 09:41, strictly AFTER A's 09:40 exit -> the slot is free
        (causal: exit_m < entry_m, never <=) -> C taken."""
        entry_csv = tmp_path / 'entry.csv'
        cf_csv = tmp_path / 'cf.csv'
        _write_entry_csv(entry_csv, [
            _entry_row(ET_DATE, 'A', f'{ET_DATE}T09:30:00-04:00', f'{ET_DATE}T09:30:05-04:00', 10.0, 9.5),
            _entry_row(ET_DATE, 'B', f'{ET_DATE}T09:31:00-04:00', f'{ET_DATE}T09:31:05-04:00', 20.0, 19.5),
            _entry_row(ET_DATE, 'C', f'{ET_DATE}T09:41:00-04:00', f'{ET_DATE}T09:41:05-04:00', 30.0, 29.5),
        ])
        _write_cf_csv(cf_csv, [
            _cf_row(ET_DATE, 'A', f'{ET_DATE}T09:30:05-04:00', 9.5, 'stop', f'{ET_DATE}T09:40:00-04:00'),
            _cf_row(ET_DATE, 'B', f'{ET_DATE}T09:31:05-04:00', 21.0, 'target', f'{ET_DATE}T09:50:00-04:00'),
            _cf_row(ET_DATE, 'C', f'{ET_DATE}T09:41:05-04:00', 31.0, 'target', f'{ET_DATE}T09:55:00-04:00'),
        ])
        rows = backfill.build_capped_replay_rows(str(entry_csv), str(cf_csv), max_per_day=10, max_concurrent=1)
        assert [r['symbol'] for r in rows] == ['A', 'C']

    def test_per_day_limit(self, tmp_path):
        """Three non-overlapping fills, max_per_day=2: only the first two (arm-time order) survive,
        regardless of max_concurrent being generous."""
        entry_csv = tmp_path / 'entry.csv'
        cf_csv = tmp_path / 'cf.csv'
        _write_entry_csv(entry_csv, [
            _entry_row(ET_DATE, 'A', f'{ET_DATE}T09:30:00-04:00', f'{ET_DATE}T09:30:05-04:00', 10.0, 9.5),
            _entry_row(ET_DATE, 'B', f'{ET_DATE}T09:40:00-04:00', f'{ET_DATE}T09:40:05-04:00', 20.0, 19.5),
            _entry_row(ET_DATE, 'C', f'{ET_DATE}T09:50:00-04:00', f'{ET_DATE}T09:50:05-04:00', 30.0, 29.5),
        ])
        _write_cf_csv(cf_csv, [
            _cf_row(ET_DATE, 'A', f'{ET_DATE}T09:30:05-04:00', 10.5, 'target', f'{ET_DATE}T09:35:00-04:00'),
            _cf_row(ET_DATE, 'B', f'{ET_DATE}T09:40:05-04:00', 20.5, 'target', f'{ET_DATE}T09:45:00-04:00'),
            _cf_row(ET_DATE, 'C', f'{ET_DATE}T09:50:05-04:00', 30.5, 'target', f'{ET_DATE}T09:55:00-04:00'),
        ])
        rows = backfill.build_capped_replay_rows(str(entry_csv), str(cf_csv), max_per_day=2, max_concurrent=10)
        assert [r['symbol'] for r in rows] == ['A', 'B']


# --------------------------------------------------------------------------------------------- unit: actual vs floor leg
class TestActualVsFloorLeg:
    def test_replay_uses_the_actual_exit_not_the_floor_counterfactual(self, tmp_path):
        """entry=10.00 stop=9.50 (risk 0.50). ACTUAL leg exits at 11.00 (target) -> r=+2.0. The floor
        leg (cf_floor_stop_px=9.90, cf_stoplimit_exit_px=9.80) would give r=(9.80-10)/(10-9.5)=-0.40 if
        wrongly used — asserting +2.0 proves the floor columns are ignored."""
        entry_csv = tmp_path / 'entry.csv'
        cf_csv = tmp_path / 'cf.csv'
        _write_entry_csv(entry_csv, [
            _entry_row(ET_DATE, 'A', f'{ET_DATE}T09:30:00-04:00', f'{ET_DATE}T09:30:05-04:00', 10.0, 9.5),
        ])
        _write_cf_csv(cf_csv, [
            _cf_row(ET_DATE, 'A', f'{ET_DATE}T09:30:05-04:00', 11.0, 'target', f'{ET_DATE}T09:40:00-04:00',
                    floor_stop_px=9.90, stoplimit_exit_px=9.80),
        ])
        rows = backfill.build_capped_replay_rows(str(entry_csv), str(cf_csv))
        assert len(rows) == 1
        assert rows[0]['r_multiple'] == pytest.approx(2.0)
        assert rows[0]['exit_reason'] == 'target'
        assert rows[0]['pnl_usd'] == pytest.approx(2.0 * backfill.DEFAULT_RISK_USD)


# --------------------------------------------------------------------------------------------- unit: exclusions
class TestReplayExclusions:
    def test_13_column_rows_excluded_from_replay(self, tmp_path):
        entry_csv = tmp_path / 'entry.csv'
        cf_csv = tmp_path / 'cf.csv'
        _write_entry_csv(entry_csv, [
            _entry_row('2026-09-20', 'OLD', '2026-09-20T09:30:00-04:00', '2026-09-20T09:30:05-04:00',
                       10.0, 9.5, extra_13col=True),
            _entry_row(ET_DATE, 'NEW', f'{ET_DATE}T09:30:00-04:00', f'{ET_DATE}T09:30:05-04:00', 10.0, 9.5),
        ])
        _write_cf_csv(cf_csv, [
            _cf_row(ET_DATE, 'NEW', f'{ET_DATE}T09:30:05-04:00', 11.0, 'target', f'{ET_DATE}T09:40:00-04:00'),
        ])
        rows = backfill.build_capped_replay_rows(str(entry_csv), str(cf_csv))
        assert {r['symbol'] for r in rows} == {'NEW'}

    def test_unresolved_arm_excluded_from_replay(self, tmp_path):
        """A filled row with NO matching cf-ledger exit can't causally free a concurrency slot and must
        be excluded, not silently treated as always-open or always-free."""
        entry_csv = tmp_path / 'entry.csv'
        cf_csv = tmp_path / 'cf.csv'
        _write_entry_csv(entry_csv, [
            _entry_row(ET_DATE, 'NOEXIT', f'{ET_DATE}T09:30:00-04:00', f'{ET_DATE}T09:30:05-04:00', 10.0, 9.5),
        ])
        _write_cf_csv(cf_csv, [])
        rows = backfill.build_capped_replay_rows(str(entry_csv), str(cf_csv))
        assert rows == []


# --------------------------------------------------------------------------------------------- unit: Database.delete_dry_trades
class TestDeleteDryTrades:
    def test_deletes_only_matching_strategy_and_source(self, tmp_path):
        db = Database(db_path=str(tmp_path / 'trades.db'))
        db.insert_dry_entry({'strategy': 'hod_break', 'trade_date': ET_DATE, 'symbol': 'A', 'source': 'counterfactual_all'})
        db.insert_dry_entry({'strategy': 'hod_break', 'trade_date': ET_DATE, 'symbol': 'B', 'source': 'replay_capped'})
        db.insert_dry_entry({'strategy': 'orb', 'trade_date': ET_DATE, 'symbol': 'C', 'source': 'counterfactual_all'})

        deleted = db.delete_dry_trades('hod_break', 'counterfactual_all')

        assert deleted == 1
        remaining = {(r['strategy'], r['source']) for r in
                     db.get_dry_trades('hod_break') + db.get_dry_trades('orb')}
        assert remaining == {('hod_break', 'replay_capped'), ('orb', 'counterfactual_all')}

    def test_bounded_by_date_range(self, tmp_path):
        db = Database(db_path=str(tmp_path / 'trades.db'))
        db.insert_dry_entry({'strategy': 'hod_break', 'trade_date': '2026-09-01', 'symbol': 'A', 'source': 'replay_capped'})
        db.insert_dry_entry({'strategy': 'hod_break', 'trade_date': ET_DATE, 'symbol': 'B', 'source': 'replay_capped'})

        db.delete_dry_trades('hod_break', 'replay_capped', start=ET_DATE, end=ET_DATE)

        assert [r['symbol'] for r in db.get_dry_trades('hod_break')] == ['A']


# --------------------------------------------------------------------------------------------- integration
class TestCappedReplayIntegration:
    """Real CSV ledgers -> real backfill functions -> real sqlite Database -> real hod_dry_ledger split,
    including a second (idempotent) run."""

    def _build_fixture(self, tmp_path):
        entry_csv = tmp_path / 'entry.csv'
        cf_csv = tmp_path / 'cf.csv'
        _write_entry_csv(entry_csv, [
            _entry_row(ET_DATE, 'A', f'{ET_DATE}T09:30:00-04:00', f'{ET_DATE}T09:30:05-04:00', 10.0, 9.5),
            _entry_row(ET_DATE, 'B', f'{ET_DATE}T09:40:00-04:00', f'{ET_DATE}T09:40:05-04:00', 20.0, 19.5),
            _entry_row(ET_DATE, 'C', f'{ET_DATE}T09:50:00-04:00', f'{ET_DATE}T09:50:05-04:00', 30.0, 29.5),
        ])
        _write_cf_csv(cf_csv, [
            _cf_row(ET_DATE, 'A', f'{ET_DATE}T09:30:05-04:00', 10.5, 'target', f'{ET_DATE}T09:35:00-04:00'),
            _cf_row(ET_DATE, 'B', f'{ET_DATE}T09:40:05-04:00', 19.0, 'stop', f'{ET_DATE}T09:45:00-04:00'),
            _cf_row(ET_DATE, 'C', f'{ET_DATE}T09:50:05-04:00', 30.5, 'target', f'{ET_DATE}T09:55:00-04:00'),
        ])
        return str(entry_csv), str(cf_csv)

    def _run_once(self, db, entry_csv, cf_csv):
        cf_rows = backfill.build_ledger_rows(entry_csv, cf_csv)
        replay_rows = backfill.build_capped_replay_rows(entry_csv, cf_csv, max_per_day=2, max_concurrent=10)

        cf_lo, cf_hi = backfill._date_bounds(cf_rows)
        db.delete_dry_trades('hod_break', 'counterfactual_all', start=cf_lo, end=cf_hi)
        backfill.insert_fresh_rows(db, cf_rows, dry_run=False)

        rp_lo, rp_hi = backfill._date_bounds(replay_rows)
        db.delete_dry_trades('hod_break', 'replay_capped', start=rp_lo, end=rp_hi)
        backfill.insert_fresh_rows(db, replay_rows, dry_run=False)
        return cf_rows, replay_rows

    def test_end_to_end_populations_and_rerun_idempotency(self, tmp_path):
        entry_csv, cf_csv = self._build_fixture(tmp_path)
        db = Database(db_path=str(tmp_path / 'trades.db'))

        cf_rows, replay_rows = self._run_once(db, entry_csv, cf_csv)
        assert len(cf_rows) == 3          # all-armed: A, B, C
        assert len(replay_rows) == 2      # capped (max_per_day=2): A, B only — C excluded

        all_rows = db.get_dry_trades('hod_break')
        assert len(all_rows) == 5
        assert sum(1 for r in all_rows if r['source'] == 'counterfactual_all') == 3
        assert sum(1 for r in all_rows if r['source'] == 'replay_capped') == 2

        # re-run: delete-and-rebuild must not duplicate
        self._run_once(db, entry_csv, cf_csv)
        all_rows_2 = db.get_dry_trades('hod_break')
        assert len(all_rows_2) == 5

    def test_hod_dry_ledger_never_sums_the_two_populations(self, tmp_path):
        import scripts.hod_dry_ledger as hdl
        from datetime import datetime

        entry_csv, cf_csv = self._build_fixture(tmp_path)
        db = Database(db_path=str(tmp_path / 'trades.db'))
        self._run_once(db, entry_csv, cf_csv)

        end = datetime.strptime(ET_DATE, '%Y-%m-%d').date()
        ledger = hdl.ledger_from_db(ET_DATE, end, db=db)
        armed = hdl.armed_population_from_db(ET_DATE, end, db=db)

        ledger_trades = sum(t for _, t, _, _, _, _, _ in ledger)
        armed_trades = sum(t for _, t, _, _, _, _, _ in armed)
        assert ledger_trades == 2      # replay_capped only (A, B)
        assert armed_trades == 3       # counterfactual_all only (A, B, C)

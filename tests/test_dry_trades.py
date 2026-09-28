"""dry_trades persistence (owner 9/28: "hod dry-run is not in the DB???").

Three layers: (1) unit tests on persistence.database.Database's dry_trades methods with a real sqlite
in tmp_path, (2) integration tests that drive the REAL HOD-break and ORB engine code paths (resting-entry
fill -> CFWatch exit for HOD, the production dry-run WOULD BUY branch for ORB) against a real Database,
(3) an idempotency test for scripts/backfill_dry_trades.py's ledger-CSV path.
"""
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from data_sources.alpaca_client import AlpacaClient
from persistence.database import Database
from trading.stop_monitor import StopMonitor
from trading.hod_break_engine import HodBreakEngine
from tests.test_hod_break_engine import cfg, bars_df, admit
from tests.test_hod_resting_entry import tape

import scripts.backfill_dry_trades as backfill


# --------------------------------------------------------------------------------------------- unit: Database
@pytest.fixture
def db(tmp_path):
    return Database(db_path=str(tmp_path / 'trades.db'))


class TestDryTradesDbMethods:
    def test_insert_and_get(self, db):
        rid = db.insert_dry_entry({'strategy': 'hod_break', 'trade_date': '2026-09-28', 'symbol': 'ABC',
                                    'entry_px': 10.0, 'stop_px': 9.5, 'shares': 100, 'source': 'live_dry'})
        assert rid is not None
        rows = db.get_dry_trades('hod_break')
        assert len(rows) == 1 and rows[0]['symbol'] == 'ABC' and rows[0]['exit_px'] is None

    def test_insert_missing_required_field_returns_none_and_warns(self, db, caplog):
        assert db.insert_dry_entry({'strategy': 'hod_break', 'symbol': 'ABC', 'source': 'live_dry'}) is None
        assert db.get_dry_trades('hod_break') == []

    def test_close_dry_trade_sets_exit_fields(self, db):
        rid = db.insert_dry_entry({'strategy': 'orb', 'trade_date': '2026-09-28', 'symbol': 'XYZ', 'source': 'live_dry'})
        db.close_dry_trade(rid, exit_ts='2026-09-28T11:00:00', exit_px=12.0, exit_reason='target', r_multiple=1.5, pnl_usd=150.0)
        row = db.get_dry_trades('orb')[0]
        assert row['exit_px'] == 12.0 and row['exit_reason'] == 'target' and row['r_multiple'] == 1.5

    def test_close_dry_trade_none_id_is_a_noop(self, db):
        db.close_dry_trade(None, exit_px=1.0)  # must not raise

    def test_get_dry_trades_filters_by_date_range(self, db):
        db.insert_dry_entry({'strategy': 'hod_break', 'trade_date': '2026-09-01', 'symbol': 'A', 'source': 'live_dry'})
        db.insert_dry_entry({'strategy': 'hod_break', 'trade_date': '2026-09-28', 'symbol': 'B', 'source': 'live_dry'})
        rows = db.get_dry_trades('hod_break', start='2026-09-15', end='2026-09-30')
        assert len(rows) == 1 and rows[0]['symbol'] == 'B'

    def test_daily_summary(self, db):
        rid = db.insert_dry_entry({'strategy': 'hod_break', 'trade_date': '2026-09-28', 'symbol': 'A', 'source': 'live_dry'})
        db.close_dry_trade(rid, exit_px=11.0, exit_reason='target', r_multiple=2.0, pnl_usd=100.0)
        db.insert_dry_entry({'strategy': 'hod_break', 'trade_date': '2026-09-28', 'symbol': 'B', 'source': 'live_dry'})
        summary = db.get_dry_trades_daily_summary('hod_break')
        assert summary[0]['trades'] == 2 and summary[0]['r_total'] == 2.0 and summary[0]['is_green'] is True

    def test_index_and_table_created_idempotently(self, db):
        db2 = Database(db_path=db._trades_path)  # re-open same file — migration 16 must be a no-op, not an error
        assert db2.get_dry_trades('hod_break') == []


# --------------------------------------------------------------------------------------------- integration: HOD engine
@pytest.fixture
def mock_alpaca():
    a = MagicMock(spec=AlpacaClient)
    a.get_latest_quote.return_value = {'bid_price': 11.00, 'ask_price': 11.015, 'bid_size': 100, 'ask_size': 100}
    a.get_1min_bars_multi.return_value = {}
    a.get_open_positions.return_value = []
    return a


@pytest.fixture
def mock_sm():
    s = MagicMock(spec=StopMonitor); s.polling_mode = False; return s


@pytest.fixture
def real_db_engine(mock_alpaca, mock_sm, tmp_path):
    """A real HodBreakEngine wired to a REAL sqlite Database (not a mock) via resting_stop_limit +
    log_counterfactuals=True — the exact live config (config.yaml log_counterfactuals: true, 9/26)."""
    real_db = Database(db_path=str(tmp_path / 'trades.db'))
    real_db.get_active_universe = lambda: [{'symbol': 'ABC', 'avg_volume_daily': 1_000_000}]
    real_db.get_open_trades = lambda *a, **k: []
    real_db._cache_path = None  # force _load_adv_map's fallback to get_active_universe (no daily_bars fixture here)
    c = cfg(dry_run=True, entry_mode='resting_stop_limit', log_counterfactuals=True)
    c['dry_ledger_path'] = str(tmp_path / 'hod_dry_entry_ledger.csv')
    c['cf_ledger_path'] = str(tmp_path / 'hod_dry_counterfactuals.csv')
    e = HodBreakEngine(mock_alpaca, real_db, mock_sm, cfg=c)
    e._roll_session()
    e.live_since = e._bar_close_et(0)
    return e, real_db


class TestHodDryTradePersistence:
    def test_dry_fill_persists_a_dry_trades_row(self, real_db_engine):
        """Real engine code path: admit + _ingest_bars drives arm_state/resting_entry_fill to a fill,
        which calls _record_dry_fill -> db.insert_dry_entry (trading/hod_break_engine.py)."""
        e, real_db = real_db_engine
        admit(e); e._ingest_bars('ABC', bars_df(tape()))
        rows = real_db.get_dry_trades('hod_break')
        assert len(rows) == 1
        row = rows[0]
        assert row['symbol'] == 'ABC' and row['source'] == 'live_dry'
        assert row['entry_px'] == pytest.approx(11.015)
        assert row['exit_px'] is None                       # not yet resolved
        assert e.candidates['ABC'].dry_trade_id == row['id']

    def test_dry_fill_then_eod_close_persists_exit(self, real_db_engine):
        """Real engine code path: after the fill, force the session flat (is_force_close_time) and call
        _sweep_cf_watch_timeouts — the actual EOD path CFWatch uses — which calls _close_cf_watch ->
        _close_dry_trade_db -> db.close_dry_trade."""
        e, real_db = real_db_engine
        admit(e); e._ingest_bars('ABC', bars_df(tape()))
        assert 'ABC' in e._cf_watches
        e.is_force_close_time = lambda: True
        e._sweep_cf_watch_timeouts()
        assert 'ABC' not in e._cf_watches
        row = real_db.get_dry_trades('hod_break')[0]
        assert row['exit_reason'] == 'eod'
        assert row['exit_px'] is not None
        assert row['r_multiple'] is not None


# --------------------------------------------------------------------------------------------- integration: ORB engine
class _Plan:
    def __init__(self):
        self.entry_price = 5.10
        self.stop_price = 4.90
        self.shares = 50
        self.total_risk = 10.0


class TestOrbDryTradePersistence:
    def test_would_buy_persists_a_dry_trades_row_with_null_exit(self, tmp_path):
        from datetime import datetime, timezone
        from trading.orb_engine import ORBEngine, STRATEGY_NAME
        real_db = Database(db_path=str(tmp_path / 'trades.db'))
        e = object.__new__(ORBEngine)   # avoid the full ctor's live-config plumbing — only db + STRATEGY_NAME are used
        e.db = real_db
        e.STRATEGY_NAME = STRATEGY_NAME
        ts = datetime(2026, 9, 28, 14, 0, tzinfo=timezone.utc)
        e._record_orb_dry_entry('ORBX', ts, _Plan())
        rows = real_db.get_dry_trades('orb')
        assert len(rows) == 1
        row = rows[0]
        assert row['symbol'] == 'ORBX' and row['source'] == 'live_dry'
        assert row['entry_px'] == pytest.approx(5.10) and row['stop_px'] == pytest.approx(4.90)
        assert row['target_px'] is None and row['exit_px'] is None   # ORB dry has no exit simulation yet

    def test_db_none_is_a_noop(self):
        from datetime import datetime, timezone
        from trading.orb_engine import ORBEngine, STRATEGY_NAME
        e = object.__new__(ORBEngine)
        e.db = None
        e.STRATEGY_NAME = STRATEGY_NAME
        e._record_orb_dry_entry('X', datetime(2026, 9, 28, tzinfo=timezone.utc), _Plan())  # must not raise


# --------------------------------------------------------------------------------------------- backfill idempotency
class TestBackfillIdempotency:
    def _write_ledger(self, path, rows13=0, rows14=0):
        import csv
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(backfill._ENTRY_COLS[:13])   # legacy 13-col header
            for i in range(rows13):
                w.writerow(['2026-09-20', f'OLD{i}', '2026-09-20T09:30:00-04:00', '2026-09-20T09:31:00-04:00',
                            '10.0', '10.01', '10.02', '10.00', '1', '10.00', '9.5', '10.5', '1'])
            for i in range(rows14):
                w.writerow(['2026-09-27', f'NEW{i}', '2026-09-27T09:30:00-04:00', '2026-09-27T09:31:00-04:00',
                            '10.0', '10.01', '10.02', '10.00', '1', '10.00', '9.5', '10.5', '1', '1'])

    def test_parses_mixed_13_and_14_column_rows(self, tmp_path):
        """13-column rows predate log_counterfactuals entirely (no cf-ledger row can ever resolve
        them) and are EXCLUDED — see parse_entry_ledger's docstring. Only the 14-column row survives,
        tagged 'counterfactual_all' (the uncapped all-armed population, not the capped dry book)."""
        p = tmp_path / 'entry.csv'
        self._write_ledger(p, rows13=1, rows14=1)
        rows = backfill.build_ledger_rows(str(p), str(tmp_path / 'missing_cf.csv'))
        assert {r['symbol'] for r in rows} == {'NEW0'}
        assert all(r['source'] == 'counterfactual_all' for r in rows)

    def test_second_run_skips_already_present_rows(self, tmp_path):
        p = tmp_path / 'entry.csv'
        self._write_ledger(p, rows13=0, rows14=2)
        rows = backfill.build_ledger_rows(str(p), str(tmp_path / 'missing_cf.csv'))
        real_db = Database(db_path=str(tmp_path / 'trades.db'))

        inserted_1, updated_1, skipped_1, failed_1 = backfill.upsert_rows(real_db, rows, dry_run=False)
        assert inserted_1 == 2 and failed_1 == 0
        assert len(real_db.get_dry_trades('hod_break')) == 2

        inserted_2, updated_2, skipped_2, failed_2 = backfill.upsert_rows(real_db, rows, dry_run=False)
        assert inserted_2 == 0 and updated_2 == 0 and skipped_2 == 2   # idempotent: nothing new/changed
        assert len(real_db.get_dry_trades('hod_break')) == 2


# --------------------------------------------------------------------------------------------- cross-source dedup
class TestBackfillCrossSourceDedup:
    """9/28 defect: a ledger row and a journal row of the SAME real trade were inserted as two separate
    dry_trades rows (130 rows / -5.2R / $0 instead of 67 / +6.2R / +$891) because the old idempotency key
    (strategy+symbol+entry_ts) never matched across sources — the journal has no per-trade entry_ts.
    upsert_rows/_same_trade now match across sources within a 120s window (or a wildcard match when
    either side lacks a timestamp, always true for a journal row)."""

    def _ledger_row(self, symbol='ABC', entry_ts='2026-09-27T09:31:00-04:00'):
        return {'strategy': 'hod_break', 'trade_date': '2026-09-27', 'symbol': symbol,
                'entry_ts': entry_ts, 'entry_px': 10.0, 'shares': None,
                'stop_px': 9.5, 'target_px': 10.5,
                'exit_ts': None, 'exit_px': None, 'exit_reason': None,
                'r_multiple': None, 'pnl_usd': None, 'risk_usd': None,
                'source': 'backfill_ledger'}

    def _journal_row(self, symbol='ABC', entry_ts=None):
        return {'strategy': 'hod_break', 'trade_date': '2026-09-27', 'symbol': symbol,
                'entry_ts': entry_ts, 'entry_px': None, 'shares': None,
                'stop_px': None, 'target_px': None,
                'exit_ts': None, 'exit_px': None, 'exit_reason': 'eod',
                'r_multiple': 1.5, 'pnl_usd': 75.0, 'risk_usd': 50.0,
                'source': 'backfill_journal'}

    def test_ledger_and_journal_40s_apart_collapse_to_one_row_journal_wins(self, tmp_path):
        ledger_row = self._ledger_row(entry_ts='2026-09-27T09:31:00-04:00')
        journal_row = self._journal_row(entry_ts='2026-09-27T09:31:40-04:00')  # 40s apart, no timestamp in production
        real_db = Database(db_path=str(tmp_path / 'trades.db'))

        inserted, updated, skipped, failed = backfill.upsert_rows(real_db, [ledger_row, journal_row], dry_run=False)

        rows = real_db.get_dry_trades('hod_break')
        assert len(rows) == 1
        assert rows[0]['source'] == 'backfill_journal'
        assert rows[0]['r_multiple'] == 1.5 and rows[0]['pnl_usd'] == 75.0
        assert inserted == 1 and skipped == 1 and failed == 0

    def test_unmatched_ledger_row_is_inserted_open(self, tmp_path):
        ledger_row = self._ledger_row(symbol='XYZ')
        real_db = Database(db_path=str(tmp_path / 'trades.db'))

        inserted, updated, skipped, failed = backfill.upsert_rows(real_db, [ledger_row], dry_run=False)

        rows = real_db.get_dry_trades('hod_break')
        assert len(rows) == 1 and inserted == 1
        assert rows[0]['source'] == 'backfill_ledger'
        assert rows[0]['exit_ts'] is None and rows[0]['r_multiple'] is None   # OPEN

    def test_journal_backfill_populates_pnl_usd(self, tmp_path, monkeypatch):
        """Addendum: backfill_journal rows used to leave pnl_usd NULL (hod_dry_ledger.py printed $+0 for
        every day). build_journal_rows now derives it from the day's aggregate $ and signed R total."""
        import scripts.hod_dry_ledger as hdl

        def fake_analyze_day(day_str):
            if day_str == '2026-09-28':  # a Monday
                return (2, 1.0, 50.0, [('AAA', 1.5), ('BBB', -0.5)], True)
            return (0, 0.0, 0.0, [], False)

        monkeypatch.setattr(hdl, 'analyze_day', fake_analyze_day)
        rows = backfill.build_journal_rows('2026-09-28', '2026-09-28')

        by_symbol = {r['symbol']: r for r in rows}
        assert by_symbol['AAA']['pnl_usd'] == pytest.approx(75.0)     # risk_usd = 50/1.0 = 50; 1.5 * 50
        assert by_symbol['AAA']['risk_usd'] == pytest.approx(50.0)
        assert by_symbol['BBB']['pnl_usd'] == pytest.approx(-25.0)    # -0.5 * 50

    def test_rerun_updates_existing_journal_row_instead_of_duplicating(self, tmp_path):
        """The addendum's idempotency requirement: re-running after pnl_usd starts being populated must
        UPDATE a previously-inserted (pre-fix) NULL-pnl_usd row, not skip it or duplicate it."""
        real_db = Database(db_path=str(tmp_path / 'trades.db'))
        stale = self._journal_row()
        stale['pnl_usd'] = None
        stale['risk_usd'] = None
        backfill.upsert_rows(real_db, [stale], dry_run=False)
        assert real_db.get_dry_trades('hod_break')[0]['pnl_usd'] is None

        fresh = self._journal_row()  # same trade, now with pnl_usd/risk_usd populated
        inserted, updated, skipped, failed = backfill.upsert_rows(real_db, [fresh], dry_run=False)

        rows = real_db.get_dry_trades('hod_break')
        assert len(rows) == 1                    # healed in place, not duplicated
        assert rows[0]['pnl_usd'] == 75.0
        assert inserted == 0 and updated == 1 and skipped == 0


# --------------------------------------------------------------------------------------------- hod_dry_ledger.py summary
class TestHodDryLedgerSummary:
    """9/28 defect: scripts/hod_dry_ledger.py counted OPEN (unresolved) dry_trades rows toward trades/R/$,
    and the $ column summed pnl_usd which was NULL on every backfill_journal row (prints $+0). The DB read
    path now counts only CLOSED rows (exit_ts not null) and reports open rows on their own line."""

    def test_counts_only_closed_rows_and_reports_open_separately(self, tmp_path, capsys):
        """source='replay_capped' — one of the two DRY_BOOK_SOURCES (scripts/hod_dry_ledger.py) that
        make up the capped dry book. 'live_dry' (the engine's own real-time insert) is deliberately NOT
        used here: it comes from the same uncapped resting-fill simulator as 'counterfactual_all' (no
        max_per_day/max_concurrent check) and so belongs to the all-armed population, not this book —
        see armed_population_from_db / TestArmedPopulationSplit below."""
        import scripts.hod_dry_ledger as hdl
        db = Database(db_path=str(tmp_path / 'trades.db'))
        rid1 = db.insert_dry_entry({'strategy': 'hod_break', 'trade_date': '2026-09-28', 'symbol': 'AAA', 'source': 'replay_capped'})
        db.close_dry_trade(rid1, exit_ts='2026-09-28T15:00:00', exit_px=11.0, exit_reason='target', r_multiple=1.0, pnl_usd=50.0)
        rid2 = db.insert_dry_entry({'strategy': 'hod_break', 'trade_date': '2026-09-28', 'symbol': 'BBB', 'source': 'replay_capped'})
        db.close_dry_trade(rid2, exit_ts='2026-09-28T15:05:00', exit_px=9.5, exit_reason='stop', r_multiple=-0.5, pnl_usd=-25.0)
        db.insert_dry_entry({'strategy': 'hod_break', 'trade_date': '2026-09-28', 'symbol': 'CCC', 'source': 'replay_capped'})  # still open

        ledger = hdl.ledger_from_db('2026-09-28', datetime(2026, 9, 28).date(), db=db)
        assert len(ledger) == 1
        d, trades, dr, dusd, symbols, status, open_n = ledger[0]
        assert trades == 2
        assert dr == pytest.approx(0.5)
        assert dusd == pytest.approx(25.0)
        assert open_n == 1

        hdl._print_report(ledger, skipped=[], json_output=False)
        out = capsys.readouterr().out
        assert 'open: 1' in out
        assert 'TOTAL:' in out

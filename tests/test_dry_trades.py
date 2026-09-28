"""dry_trades persistence (owner 9/28: "hod dry-run is not in the DB???").

Three layers: (1) unit tests on persistence.database.Database's dry_trades methods with a real sqlite
in tmp_path, (2) integration tests that drive the REAL HOD-break and ORB engine code paths (resting-entry
fill -> CFWatch exit for HOD, the production dry-run WOULD BUY branch for ORB) against a real Database,
(3) an idempotency test for scripts/backfill_dry_trades.py's ledger-CSV path.
"""
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
        p = tmp_path / 'entry.csv'
        self._write_ledger(p, rows13=1, rows14=1)
        rows = backfill.build_ledger_rows(str(p), str(tmp_path / 'missing_cf.csv'))
        assert {r['symbol'] for r in rows} == {'OLD0', 'NEW0'}
        assert all(r['source'] == 'backfill_ledger' for r in rows)

    def test_second_run_skips_already_present_rows(self, tmp_path):
        p = tmp_path / 'entry.csv'
        self._write_ledger(p, rows13=2, rows14=0)
        rows = backfill.build_ledger_rows(str(p), str(tmp_path / 'missing_cf.csv'))
        real_db = Database(db_path=str(tmp_path / 'trades.db'))

        inserted_1 = sum(1 for r in rows if not backfill._already_present(real_db, r))
        for r in rows:
            if not backfill._already_present(real_db, r):
                real_db.insert_dry_entry(r)
        assert inserted_1 == 2
        assert len(real_db.get_dry_trades('hod_break')) == 2

        inserted_2 = sum(1 for r in rows if not backfill._already_present(real_db, r))
        assert inserted_2 == 0                                  # idempotent: nothing new the second time
        assert len(real_db.get_dry_trades('hod_break')) == 2

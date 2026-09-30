"""Tests for scripts/ops_fix_trades_20260930.py, the 2026-09-30 HOD-break fill-persistence backfill.

trading/hod_break_engine.py:_on_live_fill booked every resting-order fill without ever writing
fill_price/filled_at/filled_qty to the trades row (fixed in the same change as this script). This
script backfills those three fields plus pnl/pnl_pct from the broker-verified fills the owner read
directly off the account, for the 11 rows the defect produced on 2026-09-30.

Reuses the 2026-09-29 incident test's real-schema helpers (_insert/TRADE_COLUMNS) so this table is
never hand-copied and never drifts from production.
"""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persistence.database import Database  # noqa: E402
import scripts.ops_fix_trades_20260930 as ops_fix  # noqa: E402
from tests.test_ops_fix_trades_20260929 import _insert, TRADE_COLUMNS  # noqa: E402,F401

ROWS = ops_fix.FILLS_20260930   # [(symbol, row_id, filled_at_time, filled_qty, fill_price), ...]


@pytest.fixture
def seeded_db(tmp_path):
    """Real trades.db schema seeded with all 11 2026-09-30 rows exactly as the live defect left them:
    entry_price/stop_loss_price/shares/exit_price/exit_reason/pnl set, fill_price/filled_at/filled_qty
    NULL (order_status is 'closed' — these are already-exited trades, same as the real rows)."""
    path = tmp_path / "trades.db"
    db = Database(db_path=":memory:", cache_path=":memory:", trades_path=str(path))
    conn = db._trades_conn
    for symbol, row_id, filled_at_time, filled_qty, fill_price in ROWS:
        entry_price = round(fill_price * 1.0003, 6)     # close to fill_price, like the real rows
        exit_price = round(entry_price * 1.01, 4)
        old_pnl = round((exit_price - entry_price) * filled_qty, 4)
        _insert(conn, id=row_id, trade_date="2026-09-30", symbol=symbol, entry_price=entry_price,
                stop_loss_price=round(entry_price * 0.98, 4), take_profit_price=round(entry_price * 1.02, 4),
                shares=filled_qty, order_id=f"o-{symbol}", order_status="closed",
                fill_price=None, filled_at=None,
                exit_price=exit_price, exit_reason="target", exited_at="2026-09-30T20:00:00+00:00",
                pnl=old_pnl, pnl_pct=round(old_pnl / (entry_price * filled_qty) * 100, 6), account="paper")
    conn.commit()
    conn.close()
    return path


def _connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _get(conn, trade_id):
    row = conn.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
    return dict(row) if row else None


class TestBackfill20260930:
    def test_all_eleven_rows_get_fill_price_filled_at_filled_qty(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        for symbol, row_id, filled_at_time, filled_qty, fill_price in ROWS:
            row = _get(conn, row_id)
            assert row["fill_price"] == pytest.approx(fill_price)
            assert row["filled_at"] == f"2026-09-30T{filled_at_time}+00:00"
            assert row["filled_qty"] == filled_qty

    def test_pnl_recomputed_from_exit_minus_fill_times_shares(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        for symbol, row_id, filled_at_time, filled_qty, fill_price in ROWS:
            row = _get(conn, row_id)
            expected = round((row["exit_price"] - fill_price) * filled_qty, 4)
            assert row["pnl"] == pytest.approx(expected)

    def test_order_status_and_exit_fields_untouched(self, seeded_db):
        conn = _connect(seeded_db)
        before = {row_id: _get(conn, row_id) for _sym, row_id, *_rest in ROWS}
        ops_fix.apply_corrections(conn, dry_run=False)
        for symbol, row_id, *_rest in ROWS:
            row = _get(conn, row_id)
            assert row["order_status"] == before[row_id]["order_status"] == "closed"
            assert row["exit_price"] == before[row_id]["exit_price"]
            assert row["exit_reason"] == before[row_id]["exit_reason"]

    def test_idempotent_second_run_is_a_noop(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        before = [_get(conn, r[1]) for r in ROWS]
        second_changes = ops_fix.apply_corrections(conn, dry_run=False)
        after = [_get(conn, r[1]) for r in ROWS]
        assert second_changes == []
        assert before == after

    def test_dry_run_changes_nothing(self, seeded_db):
        conn = _connect(seeded_db)
        before = [_get(conn, r[1]) for r in ROWS]
        changes = ops_fix.apply_corrections(conn, dry_run=True)
        after = [_get(conn, r[1]) for r in ROWS]
        assert len(changes) == 11
        assert before == after

    def test_symbol_mismatch_is_refused(self, seeded_db):
        conn = _connect(seeded_db)
        target_row_id = ROWS[0][1]
        conn.execute("UPDATE trades SET symbol='WRONG' WHERE id=?", (target_row_id,))
        conn.commit()
        ops_fix.apply_corrections(conn, dry_run=False)
        row = _get(conn, target_row_id)
        assert row["fill_price"] is None


class TestRowSymbolMappingIndependentOfFillsTable:
    """2026-09-30 follow-up: the first FILLS_20260930 assumed row-id order == fill-timestamp order and
    put SWMR at 405 / HIMZ at 406; the real table has HIMZ at 405 / SWMR at 406 (29s apart, row insertion
    didn't match fill order). fix_fill_row's symbol check correctly SKIPPED both rather than cross-writing
    — caught only because the owner read the live table independently, since a seeded_db test built FROM
    FILLS_20260930 can never catch a row-id error in that same table. This seeds the two rows' symbols as
    literals (not derived from FILLS_20260930), so a future swap fails this test instead of passing it
    vacuously."""

    def test_himz_405_and_swmr_406_both_get_backfilled(self, tmp_path):
        path = tmp_path / "trades.db"
        db = Database(db_path=":memory:", cache_path=":memory:", trades_path=str(path))
        conn = db._trades_conn
        _insert(conn, id=405, trade_date="2026-09-30", symbol="HIMZ", entry_price=28.08, stop_loss_price=27.5,
                take_profit_price=29.0, shares=65, order_id="o-HIMZ", order_status="closed",
                fill_price=None, filled_at=None, exit_price=27.33, exit_reason="stop_loss",
                exited_at="2026-09-30T13:49:06+00:00", pnl=-48.86, pnl_pct=-2.6, account="paper")
        _insert(conn, id=406, trade_date="2026-09-30", symbol="SWMR", entry_price=17.53, stop_loss_price=17.2,
                take_profit_price=17.9, shares=75, order_id="o-SWMR", order_status="closed",
                fill_price=None, filled_at=None, exit_price=17.80, exit_reason="target",
                exited_at="2026-09-30T19:55:41+00:00", pnl=20.25, pnl_pct=1.5, account="paper")
        conn.commit()
        conn.close()
        conn = _connect(path)
        changes = ops_fix.apply_corrections(conn, dry_run=False)
        himz, swmr = _get(conn, 405), _get(conn, 406)
        assert himz["symbol"] == "HIMZ" and himz["fill_price"] == pytest.approx(28.081692) and himz["filled_qty"] == 65
        assert swmr["symbol"] == "SWMR" and swmr["fill_price"] == pytest.approx(17.53) and swmr["filled_qty"] == 75
        assert any("405" in c and "HIMZ" in c for c in changes)
        assert any("406" in c and "SWMR" in c for c in changes)

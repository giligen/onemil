"""Tests for scripts/ops_fix_trades_20260929.py, the 2026-09-25 -> 2026-09-29
HOD-break trades-ledger incident corrections.

The synthetic trades table is built with persistence.database.Database's own
CREATE TABLE statements (never a hand-copied schema, so this never drifts from
production) in a temp file, then seeded with rows shaped exactly like the real
381 / 393 / 394 / 395 incident rows (explicit ids, so the script's hardcoded row
lookups apply to this fixture the same way they apply to the real table).
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from persistence.database import Database  # noqa: E402
import scripts.ops_fix_trades_20260929 as ops_fix  # noqa: E402

TRADE_COLUMNS = [
    "id", "trade_date", "symbol", "side", "entry_price", "stop_loss_price", "take_profit_price",
    "shares", "risk_per_share", "total_risk", "risk_reward_ratio", "order_id", "order_status",
    "fill_price", "filled_at", "exit_price", "exit_reason", "exited_at", "pnl", "pnl_pct",
    "pattern_data", "strategy", "account", "created_at", "updated_at",
]


def _insert(conn: sqlite3.Connection, **kw) -> None:
    """Insert one synthetic trades row, explicit id included, unset columns NULL."""
    defaults = dict.fromkeys(TRADE_COLUMNS)
    defaults.update({
        "side": "buy", "risk_per_share": 0.0, "total_risk": 0.0, "risk_reward_ratio": 2.0,
        "order_status": "pending_new", "strategy": "hod_break", "account": "",
        "created_at": "2026-09-29T00:00:00+00:00", "updated_at": "2026-09-29T00:00:00+00:00",
    })
    defaults.update(kw)
    cols = [c for c in TRADE_COLUMNS]
    conn.execute(
        f"INSERT INTO trades ({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
        [defaults[c] for c in cols],
    )


@pytest.fixture
def seeded_db(tmp_path):
    """A real trades.db schema seeded with rows shaped like the actual incident
    rows 381 (CDNA, wrong 'eod' exit), 393 (TTAN, stuck exit_pending_verification),
    394 (CONI, boot-adopted 87sh) and 395 (CONI, real resting fill 25sh, the
    duplicate). Returns the db path; caller opens its own connection."""
    path = tmp_path / "trades.db"
    db = Database(db_path=":memory:", cache_path=":memory:", trades_path=str(path))
    conn = db._trades_conn

    _insert(conn, id=381, trade_date="2026-09-25", symbol="CDNA", entry_price=65.0468,
            stop_loss_price=64.19, take_profit_price=66.76, shares=57, order_id="f7cf81f9",
            order_status="closed", fill_price=65.0468, filled_at="2026-09-25T18:59:04+00:00",
            exit_price=63.5001, exit_reason="eod", exited_at="2026-09-25T19:56:01+00:00",
            pnl=-88.1619000000001, pnl_pct=-2.37782642651138, account="")

    _insert(conn, id=393, trade_date="2026-09-29", symbol="TTAN", entry_price=63.66,
            stop_loss_price=63.16, take_profit_price=64.66, shares=31, order_id="655340dd",
            order_status="exit_pending_verification", fill_price=63.66,
            filled_at="2026-09-29T15:14:29+00:00", account="paper")

    _insert(conn, id=394, trade_date="2026-09-29", symbol="CONI", entry_price=22.88,
            stop_loss_price=22.63, take_profit_price=23.38, shares=87, order_id="adopted-CONI",
            order_status="pending_new",
            pattern_data=json.dumps({"book": "hod_break", "entry_mode": "resting_stop_limit",
                                      "adopted_on_boot": True}),
            account="paper")

    _insert(conn, id=395, trade_date="2026-09-29", symbol="CONI", entry_price=22.89,
            stop_loss_price=22.63, take_profit_price=23.41, shares=25, order_id="3778637d",
            order_status="exit_pending_verification", fill_price=22.89,
            filled_at="2026-09-29T17:18:58+00:00",
            pattern_data=json.dumps({"book": "hod_break", "tp_leg_id": "x", "sl_leg_id": "y",
                                      "client_order_id": "hod-rest-CONI-09-29-1db707a0"}),
            account="paper")

    _insert(conn, id=396, trade_date="2026-09-29", symbol="CONI", entry_price=22.882231,
            stop_loss_price=22.63, take_profit_price=23.39, shares=112, order_id="adopted-CONI",
            order_status="pending_new",
            pattern_data=json.dumps({"book": "hod_break", "entry_mode": "resting_stop_limit",
                                      "adopted_on_boot": True}),
            account="paper")

    conn.commit()
    conn.close()
    return path


@pytest.fixture(autouse=True)
def no_broker_calls(monkeypatch):
    """Every test runs with the broker lookup patched to the fallback (None) —
    never depend on this box's ALPACA_API_KEY/SECRET, never touch the live
    account from a unit test."""
    monkeypatch.setattr(ops_fix, "_astn_entry_ts_from_broker", lambda: None)


def _connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _get(conn, trade_id):
    row = conn.execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
    return dict(row) if row else None


def _full_dump(conn):
    """Every row, every column, order-independent — for before/after equality
    checks (idempotency, dry-run)."""
    rows = conn.execute("SELECT * FROM trades ORDER BY id").fetchall()
    return [dict(r) for r in rows]


class TestRow381Cdna:
    def test_exit_corrected_to_real_stop_loss(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        row = _get(conn, 381)
        assert row["exit_reason"] == "stop_loss"
        assert row["exit_price"] == pytest.approx(64.1043)
        assert row["exited_at"] == "2026-09-25T18:55:05+00:00"
        assert row["pnl"] == pytest.approx(-53.72)
        conn.close()

    def test_entry_and_shares_untouched(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        row = _get(conn, 381)
        assert row["entry_price"] == pytest.approx(65.0468)
        assert row["shares"] == 57
        conn.close()


class TestCdnaOverExitShortInsert:
    def test_short_row_inserted_with_correct_fields(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        row = conn.execute(
            "SELECT * FROM trades WHERE symbol='CDNA' AND exit_reason='over_exit_short'"
        ).fetchone()
        assert row is not None
        row = dict(row)
        assert row["side"] == "sell"
        assert row["entry_price"] == pytest.approx(63.5001)
        assert row["exit_price"] == pytest.approx(65.1152)
        assert row["shares"] == 57
        assert row["pnl"] == pytest.approx(-92.06)
        assert row["account"] is None
        assert row["trade_date"] == "2026-09-25"
        conn.close()

    def test_pattern_data_marks_direction_short(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        row = dict(conn.execute(
            "SELECT * FROM trades WHERE symbol='CDNA' AND exit_reason='over_exit_short'"
        ).fetchone())
        pd = json.loads(row["pattern_data"])
        assert pd["direction"] == "short"
        conn.close()

    def test_does_not_touch_row_381(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        assert _get(conn, 381)["symbol"] == "CDNA"
        total_cdna = conn.execute("SELECT COUNT(*) c FROM trades WHERE symbol='CDNA'").fetchone()["c"]
        assert total_cdna == 2  # row 381 (corrected) + the new short row
        conn.close()


class TestOpsFlattenInserts:
    @pytest.mark.parametrize("symbol,entry_price,exit_price,shares,pnl", [
        ("PRIM", 77.7386, 76.0351, 25, -42.59),
        ("ASTN", 17.23, 17.3187, 83, 7.36),
        ("WRBY", 26.1905, 26.3759, 76, 14.09),
    ])
    def test_row_inserted(self, seeded_db, symbol, entry_price, exit_price, shares, pnl):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        row = conn.execute(
            "SELECT * FROM trades WHERE symbol=? AND exit_reason='ops_flatten'", (symbol,)
        ).fetchone()
        assert row is not None
        row = dict(row)
        assert row["side"] == "buy"
        assert row["entry_price"] == pytest.approx(entry_price)
        assert row["exit_price"] == pytest.approx(exit_price)
        assert row["shares"] == shares
        assert row["pnl"] == pytest.approx(pnl)
        assert row["account"] is None
        assert row["trade_date"] == "2026-09-29"
        conn.close()

    def test_astn_uses_fallback_timestamp_when_broker_unreachable(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        row = dict(conn.execute(
            "SELECT * FROM trades WHERE symbol='ASTN' AND exit_reason='ops_flatten'"
        ).fetchone())
        assert row["filled_at"] == "2026-09-29T13:52:00+00:00"
        conn.close()

    def test_astn_uses_broker_timestamp_when_reachable(self, seeded_db, monkeypatch):
        monkeypatch.setattr(ops_fix, "_astn_entry_ts_from_broker",
                             lambda: "2026-09-29T13:50:01+00:00")
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        row = dict(conn.execute(
            "SELECT * FROM trades WHERE symbol='ASTN' AND exit_reason='ops_flatten'"
        ).fetchone())
        assert row["filled_at"] == "2026-09-29T13:50:01+00:00"
        conn.close()


class TestTtan393:
    def test_closed_with_real_market_fallback_fill(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        row = _get(conn, 393)
        assert row["exit_reason"] == "stop_loss"
        assert row["exit_price"] == pytest.approx(63.12)
        assert row["exited_at"] == "2026-09-29T15:13:37+00:00"
        assert row["pnl"] == pytest.approx(-16.74)
        conn.close()


class TestConiTripleMerge:
    def test_row_396_becomes_the_position(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        row = _get(conn, 396)
        assert row["order_status"] == "filled"
        assert row["shares"] == 112
        assert row["filled_qty"] == 112
        assert row["fill_price"] == pytest.approx(22.882231)
        assert row["filled_at"] == "2026-09-29T15:17:20+00:00"
        assert row["stop_loss_price"] == pytest.approx(22.63)
        conn.close()

    def test_take_profit_computed_from_config_target_r_and_original_entry(self, seeded_db):
        conn = _connect(seeded_db)
        target_r = ops_fix._hod_target_r()
        expected_tp = round(22.88 + target_r * (22.88 - 22.63), 2)
        ops_fix.apply_corrections(conn, dry_run=False)
        row = _get(conn, 396)
        assert row["take_profit_price"] == pytest.approx(expected_tp)
        conn.close()

    def test_take_profit_uses_injected_target_r_not_config(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.merge_coni_triple(conn, dry_run=False, changes=[], target_r=1.0)
        row = _get(conn, 396)
        assert row["take_profit_price"] == pytest.approx(22.88 + 1.0 * (22.88 - 22.63))
        conn.close()

    def test_pattern_data_merged_from_and_empty_legs(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        pd = json.loads(_get(conn, 396)["pattern_data"])
        assert pd["merged_from"] == [394, 395]
        assert pd["tp_leg_id"] == ""
        assert pd["sl_leg_id"] == ""
        conn.close()

    def test_rows_394_395_canceled_with_null_exit_fields(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        for rid in (394, 395):
            row = _get(conn, rid)
            assert row["order_status"] == "canceled"
            assert row["exit_price"] is None
            assert row["exit_reason"] is None
            assert row["exited_at"] is None
            assert row["pnl"] is None
        conn.close()

    def test_rows_394_395_pattern_data_merged_into_396(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        for rid in (394, 395):
            pd = json.loads(_get(conn, rid)["pattern_data"])
            assert pd["merged_into"] == 396
            assert pd["note"]
        conn.close()

    def test_394_395_entry_data_preserved_for_audit_trail(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        assert _get(conn, 394)["shares"] == 87
        assert _get(conn, 395)["shares"] == 25
        assert _get(conn, 395)["order_id"] == "3778637d"
        conn.close()

    def test_idempotent_second_run_no_changes(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        before = _full_dump(conn)
        second_changes = ops_fix.apply_corrections(conn, dry_run=False)
        after = _full_dump(conn)
        assert before == after
        assert not any("396" in c for c in second_changes)
        conn.close()

    def test_dry_run_leaves_all_three_rows_untouched(self, seeded_db):
        conn = _connect(seeded_db)
        before = _full_dump(conn)
        changes = ops_fix.apply_corrections(conn, dry_run=True)
        after = _full_dump(conn)
        assert before == after
        assert any("396" in c for c in changes)
        conn.close()

    def test_missing_row_396_is_a_no_op(self, seeded_db):
        conn = _connect(seeded_db)
        conn.execute("DELETE FROM trades WHERE id=396")
        conn.commit()
        changes = []
        ops_fix.merge_coni_triple(conn, dry_run=False, changes=changes)
        assert changes == []
        assert _get(conn, 394)["order_status"] == "pending_new"  # untouched
        conn.close()


class TestConiPartialExit:
    def test_partial_exit_booked_on_row_396(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        row = _get(conn, 396)
        assert row["partial_exit_shares"] == 25
        assert row["partial_exit_price"] == pytest.approx(23.48)
        assert row["partial_exit_pnl"] == pytest.approx(round(25 * (23.48 - 22.882231), 2))
        assert row["partial_exit_reason"] == "target"
        assert row["partial_exited_at"] == "2026-09-29T17:38:15+00:00"
        conn.close()

    def test_pattern_data_closed_qty_and_notional(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        pd = json.loads(_get(conn, 396)["pattern_data"])
        assert pd["closed_qty"] == 25
        assert pd["closed_notional"] == pytest.approx(587.00)
        conn.close()

    def test_shares_and_order_status_unchanged_by_partial_exit(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        row = _get(conn, 396)
        assert row["shares"] == 112
        assert row["order_status"] == "filled"
        conn.close()

    def test_open_qty_matches_broker_holding(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        row = _get(conn, 396)
        pd = json.loads(row["pattern_data"])
        assert row["shares"] - pd["closed_qty"] == 87
        conn.close()

    def test_merge_fields_still_present_alongside_partial_exit(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        pd = json.loads(_get(conn, 396)["pattern_data"])
        assert pd["merged_from"] == [394, 395]
        assert pd["closed_qty"] == 25
        conn.close()

    def test_idempotent_second_run(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        before = _full_dump(conn)
        second = ops_fix.apply_corrections(conn, dry_run=False)
        after = _full_dump(conn)
        assert before == after
        assert not any("partial exit" in c for c in second)
        conn.close()

    def test_dry_run_writes_nothing(self, seeded_db):
        conn = _connect(seeded_db)
        before = _full_dump(conn)
        changes = ops_fix.apply_corrections(conn, dry_run=True)
        after = _full_dump(conn)
        assert before == after
        assert any("partial exit" in c for c in changes)
        conn.close()

    def test_missing_row_396_is_a_no_op(self, seeded_db):
        conn = _connect(seeded_db)
        conn.execute("DELETE FROM trades WHERE id=396")
        conn.commit()
        changes = []
        ops_fix.book_coni_partial_exit(conn, dry_run=False, changes=changes)
        assert changes == []
        conn.close()


def test_hod_target_r_reads_real_config():
    """The real config.yaml must parse to a positive target_r — the script
    reads this value, it never guesses or hardcodes it."""
    assert ops_fix._hod_target_r() > 0


class TestIdempotency:
    def test_second_run_produces_no_changes(self, seeded_db):
        conn = _connect(seeded_db)
        first = ops_fix.apply_corrections(conn, dry_run=False)
        assert len(first) > 0
        second = ops_fix.apply_corrections(conn, dry_run=False)
        assert second == []
        conn.close()

    def test_second_run_leaves_db_byte_identical_rows(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        before = _full_dump(conn)
        ops_fix.apply_corrections(conn, dry_run=False)
        after = _full_dump(conn)
        assert before == after
        conn.close()

    def test_no_duplicate_inserts_on_second_run(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=False)
        ops_fix.apply_corrections(conn, dry_run=False)
        n = conn.execute("SELECT COUNT(*) c FROM trades").fetchone()["c"]
        assert n == 5 + 4  # 5 seeded rows (incl. CONI 394/395/396) + 4 inserted (CDNA short, PRIM, ASTN, WRBY)
        conn.close()


class TestDryRun:
    def test_dry_run_writes_nothing(self, seeded_db):
        conn = _connect(seeded_db)
        before = _full_dump(conn)
        changes = ops_fix.apply_corrections(conn, dry_run=True)
        after = _full_dump(conn)
        assert before == after
        assert len(changes) > 0  # still reports what WOULD change
        conn.close()

    def test_dry_run_then_real_run_still_applies(self, seeded_db):
        conn = _connect(seeded_db)
        ops_fix.apply_corrections(conn, dry_run=True)
        row = _get(conn, 381)
        assert row["exit_reason"] == "eod"  # unchanged by the dry run
        ops_fix.apply_corrections(conn, dry_run=False)
        row = _get(conn, 381)
        assert row["exit_reason"] == "stop_loss"
        conn.close()


class TestBackup:
    def test_backup_creates_timestamped_copy(self, tmp_path):
        src = tmp_path / "trades.db"
        db = Database(db_path=":memory:", cache_path=":memory:", trades_path=str(src))
        db._trades_conn.close()
        backup_path = ops_fix.backup_db(src)
        assert backup_path.exists()
        assert backup_path.name.startswith("trades.db.bak_")
        assert backup_path.read_bytes() == src.read_bytes()

    def test_missing_db_errors_cleanly(self, tmp_path, capsys):
        missing = tmp_path / "does_not_exist.db"
        with pytest.raises(SystemExit):
            sys_argv_backup = sys.argv
            sys.argv = ["ops_fix_trades_20260929.py", "--db", str(missing)]
            try:
                ops_fix.main()
            finally:
                sys.argv = sys_argv_backup
        assert "does not exist" in capsys.readouterr().out

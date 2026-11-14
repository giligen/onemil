"""Tests for scripts/gap_short_parity.py: reason classifier, comparison, summary line, idempotent CSV, 0-pick day."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gap_short_parity as gpar  # noqa: E402
import gap_short_paper as gp  # noqa: E402

D = "2026-11-16"
CTX = {"pause_active": False, "universe": {"AAA", "BBB"}, "open_price": 10.0}


@pytest.mark.parametrize("row,ctx,want", [
    ({"entry_status": "not_submitted", "note": "opg deadline"}, {}, "opg_deadline"),
    ({"entry_status": "skipped", "note": "existing position/order"}, {}, "existing_position"),
    ({"entry_status": "rejected", "note": "asset cannot be sold short"}, {}, "rejected_htb"),
    ({"entry_status": "rejected", "note": "SSR restriction in effect"}, {}, "rejected_ssr"),
    ({"entry_status": "rejected", "note": "???"}, {}, "unexplained"),
    (None, {"pause_active": True}, "pause_flag"),
    (None, {"open_price": 400.0}, "qty_floor"),
    (None, {"universe": set()}, "universe_diff"),
    (None, {}, "unexplained"),
])
def test_classifier(row, ctx, want):
    got = gpar.classify_bt_only("AAA", row, dict(CTX, **ctx))
    assert got == want and got in gpar.REASONS


def test_compare_matched_bt_only_live_only():
    bt = [{"symbol": "AAA", "open": 10.0, "ask": 9.0, "net": 1000.0}, {"symbol": "BBB", "open": 10.0, "ask": 9.5, "net": 400.0}]
    led = [{"date": D, "symbol": "AAA", "entry_fill": "9.9", "cover_fill": "9.09", "net_bp": "900", "order_id": "x", "entry_status": "accepted"},
           {"date": D, "symbol": "ZZZ", "entry_fill": "5", "net_bp": "10", "order_id": "y"}]
    rows = gpar.compare(D, bt, led, CTX)
    by = {r["symbol"]: r for r in rows}
    assert by["AAA"]["side"] == "matched" and by["AAA"]["entry_slip_bp"] == 100.0 and by["AAA"]["exit_slip_bp"] == 100.0
    assert by["AAA"]["net_diff_bp"] == -100.0
    assert by["BBB"]["side"] == "bt_only" and by["BBB"]["reason"] == "unexplained"
    assert by["ZZZ"]["side"] == "live_only" and by["ZZZ"]["reason"] == "universe_diff"
    line = gpar.summary_line(D, rows)
    assert "BT picks 2 | live 2 | matched 1 | unexplained 1" in line and "ERROR" in line and line.startswith("[GAPSHORT PARITY]")


def test_zero_pick_day_line_is_info_and_has_no_error():
    rows = gpar.compare(D, [], [], CTX)
    line = gpar.summary_line(D, rows)
    assert "BT picks 0 | live 0 | matched 0 | unexplained 0 | BT net n/a bp vs live n/a bp" in line and "ERROR" not in line


def test_api_failure_is_info_unless_two_days_running():
    line = gpar.summary_line(D, [], "boom")
    assert "api error: boom" in line and "ERROR" not in line
    assert "ERROR" in gpar.summary_line(D, [], "boom", escalate=True)


def test_api_error_writes_one_row_and_no_universe_diff(tmp_path, monkeypatch):
    monkeypatch.setattr(gpar, "PARITY_CSV", tmp_path / "parity.csv")
    monkeypatch.setattr(gp, "LEDGER", tmp_path / "ledger.csv")
    monkeypatch.setattr(gp, "PAUSE_FLAG", tmp_path / "pause.flag")
    gp.write_ledger([{"date": D, "symbol": "ZZZ", "entry_fill": "10"}])
    monkeypatch.setattr(gp, "make_client", lambda: (_ for _ in ()).throw(RuntimeError("down")))
    from datetime import datetime
    now = datetime(2026, 11, 16, 16, 35, tzinfo=gp.R.ET)
    assert gpar.run_parity(False, now) == 0
    rows = gp.read_ledger(gpar.PARITY_CSV)
    assert len(rows) == 1 and rows[0]["reason"] == "api_error" and "api_error" in gpar.REASONS
    assert gpar.api_error_yesterday(gp.read_ledger(gpar.PARITY_CSV), "2026-11-17") is True
    assert gpar.api_error_yesterday(gp.read_ledger(gpar.PARITY_CSV), "2026-11-16") is False


def test_prev_close_fallback_logs_warning(caplog):
    from unittest.mock import MagicMock
    from data_sources.alpaca_client import AlpacaClient
    c = MagicMock(spec=AlpacaClient)
    c.get_snapshots.return_value = {"AAA": {"open": 15.0, "prev_close": None, "daily_bar_date": D}}
    c.get_historical_1min_bars.return_value = None
    gpar.bt_picks(D, c, {"AAA": 10.0}, {})
    assert any(r.levelname == "WARNING" and "prev_close" in r.message for r in caplog.records)


def test_write_rows_idempotent(tmp_path):
    p = tmp_path / "parity.csv"
    assert gpar.write_rows(D, [], p) is True
    assert gpar.write_rows(D, [{"date": D, "symbol": "AAA", "side": "matched"}], p) is False
    assert len(gp.read_ledger(p)) == 1

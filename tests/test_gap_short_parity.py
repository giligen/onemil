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
    ({"entry_status": "skipped", "note": "not shortable (asset.shortable=false)"}, {}, "not_shortable"),
    ({"entry_status": "rejected", "note": "SSR restriction in effect"}, {}, "rejected_ssr"),
    ({"entry_status": "rejected", "note": "???"}, {}, "entry_not_filled"),
    ({"entry_status": "expired", "note": ""}, {}, "entry_not_filled"),
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


def test_expired_entry_and_day_fill_vs_open_reasons():
    bt = [{"symbol": "AAA", "open": 10.0, "ask": 9.0, "net": 1.0}, {"symbol": "BBB", "open": 10.0, "ask": 9.0, "net": 1.0}]
    led = [{"date": D, "symbol": "AAA", "entry_status": "expired", "order_id": "x"},
           {"date": D, "symbol": "BBB", "entry_status": "filled", "entry_fill": "9.985", "order_id": "y", "entry_tif": "day"}]
    by = {r["symbol"]: r for r in gpar.compare(D, bt, led, CTX)}
    assert by["AAA"]["side"] == "bt_only" and by["AAA"]["reason"] == "entry_not_filled" and "expired" in by["AAA"]["detail"]
    assert by["BBB"]["side"] == "matched" and by["BBB"]["reason"] == "entry_day_vs_open" and by["BBB"]["entry_slip_bp"] == 15.0
    assert "15.0 bp" in by["BBB"]["detail"] and by["BBB"]["reason"] in gpar.REASONS


SPAI = {"symbol": "SPAI", "open": 3.01, "ask": 3.0, "net": -7.5, "prev_close": 2.47, "pm_last_0919": 2.97, "l1_0919": 17_552_563.0}


@pytest.mark.parametrize("cand,lg,want", [
    (SPAI, None, "price_gate_live"),                                                                  # 2026-11-17: $2.97 at 09:19 < $3 floor
    (dict(SPAI, pm_last_0919=10.2, prev_close=10.0), None, "gap_gate_live"),
    (dict(SPAI, pm_last_0919=None, l1_0919=None), None, "gap_gate_live"),                             # no pre-market print
    (dict(SPAI, pm_last_0919=11.0, prev_close=10.0, l1_0919=1e6), None, "l1_gate_live"),
    (dict(SPAI, pm_last_0919=11.0, prev_close=10.0), {"SPAI": {"picked": "0", "l1": "17552563", "gap_pct": "10"}}, "rank_displaced_live"),
    (dict(SPAI, pm_last_0919=11.0, prev_close=10.0), {"SPAI": {"picked": "0", "l1": "100"}}, "l1_gate_live"),
    (dict(SPAI, pm_last_0919=11.0, prev_close=10.0), None, "unexplained"),                            # passed every live gate, still absent
    ({"symbol": "SPAI", "open": 3.01}, {"SPAI": {"picked": "0", "l1": "100"}}, "l1_gate_live"),        # no 09:19 fields, live log says L1 short
    ({"symbol": "SPAI", "open": 3.01}, {"SPAI": {"picked": "1", "l1": "9000000"}}, "unexplained"),     # picked live but no ledger row = defect
    ({"symbol": "SPAI", "open": 3.01}, None, "unexplained"),
])
def test_live_gate_classifier(cand, lg, want):
    ctx = dict(CTX, universe={"SPAI"}, open_price=cand["open"], bt_cand=cand, live_gappers=lg)
    got = gpar.classify_bt_only("SPAI", None, ctx)
    assert got == want and got in gpar.REASONS


def test_spai_price_gate_day_is_explained_not_error():
    """2026-11-17 shape: BT picked SPAI off the 3.01 open, live saw 2.97 at 09:19 (below the $3 floor) -> no ledger row."""
    rows = gpar.compare(D, [SPAI], [], dict(CTX, universe={"SPAI"}, live_gappers={"COHX": {"picked": "0"}}))
    assert rows[0]["side"] == "bt_only" and rows[0]["reason"] == "price_gate_live"
    assert "2.97" in rows[0]["detail"] and "3.01" in rows[0]["detail"]
    line = gpar.summary_line(D, rows)
    assert "unexplained 0" in line and "ERROR" not in line


def test_rank_displaced_detail_and_unexplained_detail_blank():
    cand = dict(SPAI, pm_last_0919=11.0, prev_close=10.0)
    rows = gpar.compare(D, [cand], [], dict(CTX, universe={"SPAI"}, live_gappers={"SPAI": {"picked": "0", "l1": "17552563", "gap_pct": "10.0"}}))
    assert rows[0]["reason"] == "rank_displaced_live" and "top-10" in rows[0]["detail"]
    rows = gpar.compare(D, [cand], [], dict(CTX, universe={"SPAI"}, live_gappers=None))
    assert rows[0]["reason"] == "unexplained" and rows[0].get("detail", "") == ""


def test_bt_picks_carries_the_live_0919_view(monkeypatch):
    """bt_picks adds prev_close / pm_last_0919 / l1_0919 from the same 1-min bars (bars starting at/after 09:20 ET excluded)."""
    from unittest.mock import MagicMock
    import pandas as pd
    from data_sources.alpaca_client import AlpacaClient
    c = MagicMock(spec=AlpacaClient)
    c.get_snapshots.return_value = {"SPAI": {"open": 3.01, "prev_close": 2.47, "daily_bar_date": D}}
    # 09:17, 09:18, 09:19, 09:25 ET: L1 counts bars that END by 09:19 (09:17, 09:18); pm_last is the last bar starting before 09:20
    ts = pd.to_datetime(["2026-11-16 14:17", "2026-11-16 14:18", "2026-11-16 14:19", "2026-11-16 14:25"], utc=True)
    c.get_historical_1min_bars.return_value = pd.DataFrame({"timestamp": ts, "close": [2.95, 2.96, 2.97, 3.0],
                                                            "volume": [1_000_000, 2_000_000, 500, 5_000_000]})
    monkeypatch.setattr(gpar, "fetch_ask_1030", lambda headers, sym, day: 3.0)
    out = gpar.bt_picks(D, c, {"SPAI": 2.47}, {})
    assert len(out) == 1 and out[0]["symbol"] == "SPAI" and out[0]["prev_close"] == 2.47
    assert out[0]["pm_last_0919"] == 2.97 and out[0]["l1_0919"] == 2.95 * 1_000_000 + 2.96 * 2_000_000
    from datetime import datetime
    assert gpar.live_window_fields([], datetime(2026, 11, 16, tzinfo=gp.R.ET)) == {"pm_last_0919": None, "l1_0919": None}

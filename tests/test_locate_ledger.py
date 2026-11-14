"""Unit tests for scripts/locate_ledger.py (real response shapes, gap maths, idempotent CSV, no POST, ERROR path)."""
import csv
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import locate_ledger as ll  # noqa: E402

REAL = {
    "quotes": [
        {"symbol": "ATRA", "price": "0.008", "available_qty": "27300", "quoted_at": "2026-11-14T14:58:09.42871372Z"},
        {"symbol": "XPOF", "price": "0.0155", "available_qty": "1200", "quoted_at": "2026-11-14T14:58:09Z"},
    ],
    "errors": [
        {"code": "easy_to_borrow", "message": "etb", "symbol": "AAPL"},
        {"code": "threshold_security", "message": "thr", "symbol": "THR"},
        {"code": "symbol_not_found", "message": "nf", "symbol": "ZZZ"},
        {"code": "weird", "message": "?", "symbol": "WNW"},
    ],
}


def _resp(payload, status=200):
    """Build a requests.Response-like mock."""
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.json.return_value = payload
    r.text = str(payload)
    return r


def test_parse_locate_response_shapes():
    out = ll.parse_locate_response(REAL)
    assert out["ATRA"] == {"locate_status": "quoted", "fee_per_share": 0.008, "available_qty": 27300}
    assert out["AAPL"]["locate_status"] == "etb"
    assert out["THR"]["locate_status"] == "threshold"
    assert out["ZZZ"]["locate_status"] == "not_found"
    assert out["WNW"]["locate_status"] == "error"


def test_gap_pct_and_selection():
    assert ll.gap_pct(10.0, 11.0) == pytest.approx(10.0)
    assert ll.gap_pct(0, 5) is None
    prev = {"A": 10.0, "B": 10.0, "C": 10.0, "D": 0.5}
    snaps = {"A": {"latest_price": 11.0, "volume": 5}, "B": {"latest_price": 10.5},
             "C": {"latest_price": 0.9}, "D": {"latest_price": 0.8}}
    got = ll.select_gappers(prev, snaps)
    assert [g["symbol"] for g in got] == ["A"]
    assert got[0]["gap_pct"] == pytest.approx(10.0)


def test_fetch_locates_batches_and_never_posts():
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _resp(REAL)
    syms = [f"S{i}" for i in range(60)]
    ll.fetch_locates(session, syms)
    assert session.get.call_count == 3  # 25 + 25 + 10
    session.post.assert_not_called()
    assert all("/v1/locates/quotes" in c.args[0] for c in session.get.call_args_list)


def test_source_has_no_post_call():
    src = Path(ll.__file__).read_text()
    assert ".post(" not in src and "requests.post" not in src


def test_borrow_status_and_error_logged(caplog):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _resp({"borrow_status": "hard_to_borrow"})
    assert ll.fetch_borrow_status(session, "ATRA") == "hard_to_borrow"
    session.get.return_value = _resp({}, status=500)
    with caplog.at_level(logging.ERROR):
        assert ll.fetch_borrow_status(session, "ATRA") == "error"
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_locate_api_failure_logs_error(caplog):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _resp({}, status=404)
    with caplog.at_level(logging.ERROR):
        out = ll.fetch_locates(session, ["A"])
    assert out["A"]["locate_status"] == "error"
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_build_row_fee_bp():
    g = {"symbol": "ATRA", "prev_close": 7.0, "pm_price": 7.7, "gap_pct": 10.0, "pm_vol": 100}
    row = ll.build_row("2026-11-16", "13:25:00", g, "hard_to_borrow",
                       {"locate_status": "quoted", "fee_per_share": 0.008, "available_qty": 27300})
    assert row["fee_bp_of_price"] == pytest.approx(0.008 / 7.7 * 1e4)
    assert list(row) == ll.LEDGER_COLUMNS


def test_append_idempotent(tmp_path):
    p = tmp_path / "l.csv"
    row = dict.fromkeys(ll.LEDGER_COLUMNS, "")
    row.update(date="2026-11-16", symbol="A")
    assert ll.append_rows(p, [row]) == 1
    assert ll.append_rows(p, [row]) == 0
    row2 = dict(row, symbol="B")
    assert ll.append_rows(p, [row, row2]) == 1
    assert len(list(csv.DictReader(p.open()))) == 2


def test_score_and_weekly_net(tmp_path):
    rows = [
        dict(dict.fromkeys(ll.LEDGER_COLUMNS, ""), date="2026-11-16", symbol="A", locate_status="quoted",
             fee_bp_of_price="20", borrow_status="hard_to_borrow"),
        dict(dict.fromkeys(ll.LEDGER_COLUMNS, ""), date="2026-11-16", symbol="B", locate_status="etb",
             fee_bp_of_price="0", borrow_status="easy_to_borrow"),
    ]
    snaps = {"A": {"open": 10.0, "close": 9.0, "high": 10.5, "low": 8.9},
             "B": {"open": 10.0, "close": 10.0, "high": 10.0, "low": 10.0}}
    scored = ll.score_rows(rows, snaps)
    assert scored[0]["fade_bp"] == pytest.approx(1000.0)
    s = ll.weekly_summary(scored)
    assert s["n"] == 2 and s["etb_share"] == pytest.approx(0.5)
    assert s["mean_fade_bp_gross"] == pytest.approx(500.0)
    assert s["mean_fade_bp_net"] == pytest.approx(500.0 - 10.0 - ll.BORROW_BP_PER_DAY)


def test_quote_without_price_zero_qty():
    out = ll.parse_locate_response({"quotes": [{"symbol": "AZ", "available_qty": 0, "quoted_at": "x"}], "errors": []})
    assert out["AZ"] == {"locate_status": "quoted", "fee_per_share": "", "available_qty": 0}

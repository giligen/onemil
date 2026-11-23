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
    from datetime import datetime
    from trading import gap_short_rules as R
    ts = datetime(2026, 11, 16, 9, 10, tzinfo=R.ET)
    for v in snaps.values():
        v["latest_trade_ts"] = ts
    snaps["B"]["latest_trade_ts"] = datetime(2026, 11, 13, 15, 0, tzinfo=R.ET)       # yesterday's print: not a live gapper
    snaps["A"]["latest_trade_ts"] = ts
    got = ll.select_gappers(prev, snaps, "2026-11-16")
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
    with caplog.at_level(logging.WARNING):
        assert ll.fetch_borrow_status(session, "ATRA") == "error"
    assert not any(r.levelno == logging.ERROR for r in caplog.records)  # 11/23: aggregated by run_ledger, not per symbol
    assert any(r.levelno == logging.WARNING for r in caplog.records)


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


def test_make_session_reads_repo_relative_env_and_stale_bars_refuse(tmp_path, caplog):
    import inspect
    assert "/home/ec2-user" not in inspect.getsource(ll.make_session)
    import sqlite3
    db = tmp_path / "c.db"
    con = sqlite3.connect(db)
    con.execute("create table daily_bars(symbol text, bar_date text, close real, volume real)")
    con.execute("insert into daily_bars values('OK','2026-11-12',4.0,9e6)")
    con.commit()
    con.close()
    assert ll.load_universe(db, "2026-11-16") == {}
    assert any("daily_bars stale (2026-11-12) - refusing" in r.message for r in caplog.records if r.levelname == "ERROR")


# ---------------------------------------------------------------------------------------------- 2026-11-23: 429 spec
import time  # noqa: E402


@pytest.fixture(autouse=False)
def fast(monkeypatch):
    """No pacing and no real sleeping for the 429 tests."""
    monkeypatch.setattr(ll, "ASSET_GET_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(ll.time, "sleep", lambda s: None)


def _r(payload, status=200, retry_after=None):
    """Response mock with a headers dict."""
    r = _resp(payload, status)
    r.headers = {} if retry_after is None else {"Retry-After": str(retry_after)}
    return r


def _asset(sym, bs="easy_to_borrow", shortable=True):
    """One /v2/assets list element."""
    return {"symbol": sym, "borrow_status": bs, "shortable": shortable, "easy_to_borrow": bs == "easy_to_borrow"}


def test_asset_table_parse_and_hit_makes_no_get(fast):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _r([_asset("AAA"), _asset("BBB", "hard_to_borrow", False)])
    tab = ll.fetch_asset_table(session)
    assert session.get.call_count == 1
    url = session.get.call_args.args[0]
    assert url.endswith("/v2/assets") and session.get.call_args.kwargs["params"] == {"status": "active", "asset_class": "us_equity"}
    assert tab["BBB"] == {"borrow_status": "hard_to_borrow", "shortable": False, "easy_to_borrow": False}
    session.get.reset_mock()
    assert ll.fetch_borrow_status(session, "AAA", table=tab) == "easy_to_borrow"
    session.get.assert_not_called()


def test_asset_table_failure_warns_and_returns_empty(fast, caplog):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _r({}, 500)
    with caplog.at_level(logging.WARNING):
        assert ll.fetch_asset_table(session) == {}
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_table_miss_is_one_get(fast):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _r({"borrow_status": "hard_to_borrow"})
    assert ll.fetch_borrow_status(session, "ZZZ", table={"AAA": {"borrow_status": "x"}}) == "hard_to_borrow"
    assert session.get.call_count == 1 and session.get.call_args.args[0].endswith("/v2/assets/ZZZ")


def test_429_then_200_one_warning_no_error(fast, caplog):
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [_r({}, 429, retry_after=0), _r({"borrow_status": "easy_to_borrow"})]
    with caplog.at_level(logging.INFO):
        assert ll.fetch_borrow_status(session, "AAA") == "easy_to_borrow"
    assert session.get.call_count == 2
    assert sum(r.levelno == logging.WARNING for r in caplog.records) == 1
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_429_exhausted_returns_error_without_error_record(fast, caplog):
    session = MagicMock(spec=requests.Session)
    session.get.return_value = _r({}, 429, retry_after=0)
    with caplog.at_level(logging.INFO):
        assert ll.fetch_borrow_status(session, "AAA") == "error"
    assert session.get.call_count == ll.ASSET_GET_RETRIES + 1
    assert not any(r.levelno >= logging.ERROR for r in caplog.records)


def test_429_without_retry_after_sleeps_power_of_two(monkeypatch):
    monkeypatch.setattr(ll, "ASSET_GET_MIN_INTERVAL_S", 0.0)
    sleeps = []
    monkeypatch.setattr(ll.time, "sleep", sleeps.append)
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = [_r({}, 429), _r({}, 429), _r({"borrow_status": "x"})]
    assert ll.fetch_borrow_status(session, "AAA") == "x"
    assert sleeps == [2 ** 0, 2 ** 1] or sleeps == [2, 4]


def _patch_run(monkeypatch, tmp_path, statuses, table=None):
    """Patch run_ledger's collaborators: 3 gappers, per-symbol statuses via fake session."""
    monkeypatch.setattr(ll, "ASSET_GET_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(ll.time, "sleep", lambda s: None)
    monkeypatch.setattr(ll, "load_universe", lambda *a, **k: {"A": 5.0, "B": 5.0, "C": 5.0})
    gaps = [{"symbol": s, "prev_close": 5.0, "pm_price": 6.0, "gap_pct": 20.0, "pm_vol": 1} for s in "ABC"]
    monkeypatch.setattr(ll, "select_gappers", lambda *a, **k: gaps)
    monkeypatch.setattr(ll, "fetch_locates", lambda s, syms: {x: {"locate_status": "etb", "fee_per_share": "", "available_qty": ""} for x in syms})
    monkeypatch.setattr(ll, "fetch_asset_table", lambda s: table if table is not None else {})
    monkeypatch.setattr(ll, "LEDGER", tmp_path / "l.csv")
    session = MagicMock(spec=requests.Session)

    def get(url, **kw):
        sym = url.rsplit("/", 1)[1]
        st = statuses[sym]
        return _r({}, st) if isinstance(st, int) else _r({"borrow_status": st})
    session.get.side_effect = get
    client = MagicMock()
    client.get_snapshots.return_value = {"A": {}}
    return session, client


def test_run_ledger_one_aggregated_error(monkeypatch, tmp_path, caplog):
    session, client = _patch_run(monkeypatch, tmp_path, {"A": "easy_to_borrow", "B": 500, "C": 429})
    caplog.set_level(logging.INFO)
    assert ll.run_ledger(False, None, session=session, client=client) == 0
    errs = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errs) == 1 and "failed for 2 of 3" in errs[0].getMessage()
    assert "HTTP 429 x 1" in errs[0].getMessage() and "other x 1" in errs[0].getMessage()
    assert any("error 2" in r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
    by = {r["symbol"]: r["borrow_status"] for r in ll.read_csv(tmp_path / "l.csv")}
    assert by == {"A": "easy_to_borrow", "B": "error", "C": "error"}


def test_summary_line_error_count_only_when_positive():
    ok = dict.fromkeys(ll.LEDGER_COLUMNS, "")
    ok.update(locate_status="etb", borrow_status="easy_to_borrow")
    bad = dict(ok, borrow_status="error")
    assert "error" not in ll.summary_line([ok])
    assert "error 1" in ll.summary_line([ok, bad])


def test_pacing_spans_min_interval(monkeypatch):
    monkeypatch.setattr(ll, "ASSET_GET_MIN_INTERVAL_S", 0.05)
    session = MagicMock(spec=requests.Session)
    stamps = []

    def get(url, **kw):
        stamps.append(time.monotonic())
        return _r({"borrow_status": "x"})
    session.get.side_effect = get
    for s in ("A", "B", "C"):
        ll.fetch_borrow_status(session, s)
    assert stamps[-1] - stamps[0] >= 0.1 - 1e-3


def _ledger_rows():
    """4 rows: 2 same-day error, 1 same-day ok, 1 other-day error."""
    def row(d, s, bs):
        return dict(dict.fromkeys(ll.LEDGER_COLUMNS, ""), date=d, symbol=s, borrow_status=bs, locate_status="etb", gap_pct="12.5")
    return [row("2026-11-23", "A", "error"), row("2026-11-23", "B", "error"),
            row("2026-11-23", "C", "easy_to_borrow"), row("2026-11-20", "D", "error")]


def test_repair_borrow_rewrites_only_error_rows(fast, tmp_path, caplog):
    p = tmp_path / "l.csv"
    ll.append_rows(p, _ledger_rows())
    before = p.read_text().splitlines()
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = lambda url, **kw: _r({}, 500) if url.endswith("/B") else _r({"borrow_status": "hard_to_borrow"})
    with caplog.at_level(logging.INFO):
        out = ll.repair_borrow("2026-11-23", path=p, session=session, table={"A": {"borrow_status": "hard_to_borrow", "shortable": False, "easy_to_borrow": False}})
    after = p.read_text().splitlines()
    assert after[0] == before[0] and len(after) == len(before)
    assert after[3] == before[3] and after[4] == before[4]          # C and other-day D byte-identical
    assert ",hard_to_borrow," in after[1] and ",error," in after[2]
    assert out == {"repaired": 1, "total": 2, "still_error": 1}
    assert any("repaired 1 of 2 error rows for 2026-11-23; still error: 1" in r.getMessage() for r in caplog.records)
    assert not list(tmp_path.glob("*.tmp"))


def test_repair_borrow_main_integration(fast, monkeypatch, tmp_path):
    p = tmp_path / "l.csv"
    monkeypatch.setattr(ll, "LEDGER", p)
    assert ll.append_rows(p, _ledger_rows()) == 4
    monkeypatch.setattr(ll, "make_session", lambda: MagicMock(spec=requests.Session))
    monkeypatch.setattr(ll, "fetch_asset_table", lambda s: {"A": {"borrow_status": "easy_to_borrow", "shortable": True, "easy_to_borrow": True},
                                                            "B": {"borrow_status": "hard_to_borrow", "shortable": False, "easy_to_borrow": False}})
    import monitoring.cron_guard as cg
    monkeypatch.setattr(cg, "attach_telegram_error_handler", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["locate_ledger.py", "--repair-borrow", "2026-11-23"])
    assert ll.main() == 0
    got = {(r["date"], r["symbol"]): r["borrow_status"] for r in ll.read_csv(p)}
    assert got == {("2026-11-23", "A"): "easy_to_borrow", ("2026-11-23", "B"): "hard_to_borrow",
                   ("2026-11-23", "C"): "easy_to_borrow", ("2026-11-20", "D"): "error"}
    assert list(csv.DictReader(p.open()).fieldnames) == ll.LEDGER_COLUMNS

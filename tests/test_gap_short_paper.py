"""Unit tests for scripts/gap_short_paper.py (mocked broker/client with spec=, tmp ledger, no real orders)."""
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import gap_short_paper as gp  # noqa: E402
from data_sources.alpaca_client import AlpacaClient  # noqa: E402
from trading import gap_short_rules as R  # noqa: E402

NOW = datetime(2026, 11, 16, 9, 20, tzinfo=R.ET)
TS = datetime(2026, 11, 16, 9, 10, tzinfo=R.ET)
DAY = "2026-11-16"


@pytest.fixture
def paths(tmp_path, monkeypatch):
    """Redirect ledger/gappers/pause files to tmp."""
    monkeypatch.setattr(gp, "LEDGER", tmp_path / "ledger.csv")
    monkeypatch.setattr(gp, "GAPPERS", tmp_path / "gappers.csv")
    monkeypatch.setattr(gp, "PAUSE_FLAG", tmp_path / "pause.flag")
    return tmp_path


def _bars(n_dollars):
    """Pre-market bars df whose 04:00-09:25 dollar volume is n_dollars."""
    ts = pd.Timestamp("2026-11-16 05:00", tz="America/New_York")
    return pd.DataFrame([{"timestamp": ts, "close": 1.0, "volume": n_dollars}])


def _client(snaps, bars):
    c = MagicMock(spec=AlpacaClient)
    c.get_snapshots.return_value = snaps
    c.get_premarket_1min_bars_multi.return_value = bars
    return c


def _broker(positions=None, orders=None):
    b = MagicMock(spec=gp.PaperBroker)
    b.positions.return_value = positions or {}
    b.orders_today.return_value = orders or []
    b.short_opg.return_value = {"id": "oid1", "status": "accepted"}
    return b


def _session():
    s = MagicMock(spec=requests.Session)
    r = MagicMock(spec=requests.Response)
    r.status_code = 200
    r.json.return_value = {"quotes": [{"symbol": "AAA", "price": "0.01", "available_qty": "100"}],
                           "errors": [{"code": "easy_to_borrow", "symbol": "BBB", "message": "etb"}]}
    r.text = ""
    s.get.return_value = r
    return s


UNI = {"AAA": 10.0, "BBB": 10.0, "CCC": 10.0}
SNAPS = {"AAA": {"latest_price": 15.0, "latest_trade_ts": TS}, "BBB": {"latest_price": 13.0, "latest_trade_ts": TS},
         "CCC": {"latest_price": 10.3, "latest_trade_ts": TS}}


def test_pick_submits_opg_shorts_and_writes_ledger(paths):
    client = _client(SNAPS, {"AAA": _bars(9e6), "BBB": _bars(8e6)})
    broker = _broker()
    assert gp.run_pick(False, client, broker, _session(), UNI, NOW) == 0
    assert [c.args for c in broker.short_opg.call_args_list] == [("AAA", 20, "gapshort-2026-11-16-AAA"), ("BBB", 23, "gapshort-2026-11-16-BBB")]
    rows = gp.read_ledger()
    assert {r["symbol"] for r in rows} == {"AAA", "BBB"} and rows[0]["order_id"] == "oid1"
    assert rows[1]["borrow_status"] != "" and rows[1]["locate_status"] == "etb"
    assert gp.read_ledger(gp.GAPPERS) is not None


def test_pick_dry_run_places_no_order_and_writes_nothing(paths):
    client = _client(SNAPS, {"AAA": _bars(9e6)})
    broker = _broker()
    gp.run_pick(True, client, broker, _session(), UNI, NOW)
    broker.short_opg.assert_not_called()
    assert not gp.LEDGER.exists()


def test_pick_skips_existing_position_and_low_l1(paths):
    client = _client(SNAPS, {"AAA": _bars(9e6), "BBB": _bars(1e6)})
    broker = _broker(positions={"AAA": (-5.0, 15.0)})
    gp.run_pick(False, client, broker, _session(), UNI, NOW)
    broker.short_opg.assert_not_called()
    assert [r["note"] for r in gp.read_ledger()] == ["existing position/order"]


def test_pick_rejection_logged_error_no_retry(paths, caplog):
    client = _client(SNAPS, {"AAA": _bars(9e6)})
    broker = _broker()
    broker.short_opg.side_effect = RuntimeError("asset not shortable")
    gp.run_pick(False, client, broker, _session(), UNI, NOW)
    assert broker.short_opg.call_count == 1
    assert any("ALL 1 submitted picks" in r.message for r in caplog.records if r.levelname == "ERROR")
    assert gp.read_ledger()[0]["entry_status"] == "rejected"


def test_pick_refuses_after_moo_cutoff_and_when_paused(paths, caplog):
    client = _client(SNAPS, {})
    late = datetime(2026, 11, 16, 9, 27, 31, tzinfo=R.ET)
    assert gp.run_pick(False, client, _broker(), _session(), UNI, late) == 1
    gp.PAUSE_FLAG.write_text("x")
    assert gp.run_pick(False, client, _broker(), _session(), UNI, NOW) == 0
    client.get_snapshots.assert_not_called()


def test_cover_limit_at_ask_plus_10bp_then_market_when_unfilled(paths):
    client = MagicMock(spec=AlpacaClient)
    client.get_latest_quote.return_value = {"ask_price": 10.0, "bid_price": 9.9}
    broker = _broker(positions={"AAA": (-20.0, 15.0)}, orders=[{"client_order_id": f"gapshort-{DAY}-AAA", "symbol": "AAA"}])
    broker.cover_limit.return_value = {"id": "c1", "status": "accepted"}
    broker.order.return_value = {"status": "new", "filled_qty": 0.0, "filled_avg_price": None}
    gp.write_ledger([{"date": DAY, "symbol": "AAA"}])
    gp.run_cover(False, client, broker, NOW.replace(hour=10, minute=30), poll_s=0)
    assert broker.cover_limit.call_args.args[:3] == ("AAA", 20, 10.01)
    assert broker.cancel.call_count == 2          # cancel/replace at 60 s, cancel + market at 120 s
    assert broker.cover_market.call_args.args[:2] == ("AAA", 20)
    assert gp.read_ledger()[0]["ask_1030"] == "10.0"


def test_cover_dry_run_submits_nothing(paths):
    client = MagicMock(spec=AlpacaClient)
    client.get_latest_quote.return_value = {"ask_price": 10.0}
    broker = _broker(positions={"AAA": (-20.0, 15.0)}, orders=[{"client_order_id": f"gapshort-{DAY}-AAA", "symbol": "AAA"}])
    gp.run_cover(True, client, broker, NOW, poll_s=0)
    broker.cover_limit.assert_not_called()
    broker.cover_market.assert_not_called()


def test_flatten_covers_remaining_and_logs_error(paths, caplog):
    broker = _broker(positions={"AAA": (-20.0, 15.0), "OWN": (-5.0, 3.0)},
                     orders=[{"client_order_id": f"gapshort-{DAY}-AAA", "symbol": "AAA", "side": "sell", "status": "filled"}])
    gp.run_flatten(False, MagicMock(spec=AlpacaClient), broker, NOW)
    assert broker.cover_market.call_count == 1 and broker.cover_market.call_args.args[0] == "AAA"   # owner's OWN untouched
    assert any("FLATTEN" in r.message for r in caplog.records if r.levelname == "ERROR")


def test_score_row_fills_slip_net_and_pnl():
    row = {"date": DAY, "symbol": "AAA", "qty": "20", "ask_1030": "9.0", "locate_fee_bp": ""}
    orders = [{"client_order_id": f"gapshort-{DAY}-AAA", "filled_avg_price": 10.0, "filled_qty": 20.0},
              {"client_order_id": f"gapshort-cov-{DAY}-AAA", "filled_avg_price": 9.0, "filled_qty": 20.0}]
    out = gp.score_row(row, orders, {"open": 10.1})
    assert out["entry_fill"] == 10.0 and out["cover_fill"] == 9.0 and out["slip_bp_vs_ask"] == 0.0
    assert out["pnl_usd"] == 20.0 and out["net_bp"] == pytest.approx(R.net_bp(10.0, 9.0), abs=0.1)
    assert gp.score_row(row, [], None)["note"] == "no entry fill"


def test_score_writes_pause_flag_when_week_below_minus_300(paths):
    gp.write_ledger([{"date": DAY, "symbol": "AAA", "qty": "100", "ask_1030": "12", "locate_fee_bp": ""}])
    client = MagicMock(spec=AlpacaClient)
    client.get_snapshots.return_value = {"AAA": {"open": 10}}
    broker = _broker(orders=[{"client_order_id": f"gapshort-{DAY}-AAA", "filled_avg_price": 10.0, "filled_qty": 100.0},
                             {"client_order_id": f"gapshort-cov-{DAY}-AAA", "filled_avg_price": 14.0, "filled_qty": 100.0}])
    gp.run_score(False, client, broker, NOW)
    assert gp.PAUSE_FLAG.exists() and gp.read_ledger()[0]["pnl_usd"] == "-400.0"


def test_weekly_summary_and_dry_run_prints(paths, capsys):
    gp.write_ledger([{"date": DAY, "symbol": "A", "net_bp": "100", "pnl_usd": "3"}, {"date": DAY, "symbol": "B", "net_bp": "-50", "pnl_usd": "-1"}])
    s = gp.weekly_summary(gp.read_ledger())
    assert s["n"] == 2 and s["mean_net_bp"] == 25 and s["pnl_usd"] == 2
    gp.run_weekly(True)
    assert "[GAPSHORT]" in capsys.readouterr().out


def test_week_start():
    assert gp.week_start("2026-11-18") == "2026-11-16"


def test_stale_latest_trade_is_not_a_gapper(paths):
    stale = {k: dict(v, latest_trade_ts=datetime(2026, 11, 13, 15, 0, tzinfo=R.ET)) for k, v in SNAPS.items()}
    client = _client(stale, {"AAA": _bars(9e6)})
    gp.run_pick(True, client, _broker(), _session(), UNI, NOW)
    client.get_premarket_1min_bars_multi.assert_not_called()


def test_bars_fetched_only_for_gap_qualified_names(paths):
    client = _client(SNAPS, {"AAA": _bars(9e6)})
    gp.run_pick(True, client, _broker(), _session(), {**UNI, "LOW": 0.5}, NOW)
    assert set(client.get_premarket_1min_bars_multi.call_args.args[0]) == {"AAA", "BBB"}   # CCC gap 5 %, LOW prev close < $1


def test_partial_rejection_is_warning_not_error(paths, caplog):
    client = _client(SNAPS, {"AAA": _bars(9e6), "BBB": _bars(9e6)})
    broker = _broker()
    broker.short_opg.side_effect = [RuntimeError("HTB"), {"id": "o2", "status": "accepted"}]
    gp.run_pick(False, client, broker, _session(), UNI, NOW)
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
    assert any(r.levelname == "WARNING" and "rejected" in r.message for r in caplog.records)


def test_deadline_mid_loop_stops_submitting(paths, caplog):
    client = _client(SNAPS, {"AAA": _bars(9e6), "BBB": _bars(9e6)})
    broker = _broker()
    gp.run_pick(False, client, broker, _session(), UNI, NOW, clock=lambda: datetime(2026, 11, 16, 9, 27, 40, tzinfo=R.ET))
    broker.short_opg.assert_not_called()
    assert any("OPG deadline passed, 2 names not submitted" in r.message for r in caplog.records)


def test_qty_below_one_skipped(paths):
    snaps = {"AAA": {"latest_price": 400.0, "latest_trade_ts": TS}}
    client = _client(snaps, {"AAA": _bars(9e6)})
    broker = _broker()
    gp.run_pick(False, client, broker, _session(), {"AAA": 300.0}, NOW)
    broker.short_opg.assert_not_called()


def test_universe_loader_uses_prior_day_volume_not_dollar_volume(tmp_path):
    import sqlite3
    db = tmp_path / "c.db"
    con = sqlite3.connect(db)
    con.execute("create table daily_bars(symbol text, bar_date text, close real, volume real)")
    con.executemany("insert into daily_bars values(?,?,?,?)", [("OK", "2026-11-13", 4.0, 600000), ("LOWV", "2026-11-13", 4.0, 400000),
                                                              ("ZAZZT", "2026-11-13", 4.0, 900000), ("OLD", "2026-11-12", 4.0, 900000)])
    con.commit()
    con.close()
    assert gp.load_universe(db) == {"OK": 4.0}          # $2.4 M < nothing: no $1 M rule; <500 K and test ticker out


def test_price_band_excludes_candidates_before_bars(paths):
    snaps = {"AAA": {"latest_price": 35.0, "latest_trade_ts": TS}}
    client = _client(snaps, {})
    gp.run_pick(True, client, _broker(), _session(), {"AAA": 20.0}, NOW)
    client.get_premarket_1min_bars_multi.assert_not_called()


def test_pick_idempotent_skips_symbol_with_any_status_order_today(paths):
    client = _client(SNAPS, {"AAA": _bars(9e6)})
    broker = _broker(orders=[{"client_order_id": f"gapshort-{DAY}-AAA", "symbol": "AAA", "status": "filled"}])
    gp.run_pick(False, client, broker, _session(), UNI, NOW)
    broker.short_opg.assert_not_called()


def test_api_failure_skips_day_with_warning_and_ledger_row(paths, caplog, monkeypatch):
    monkeypatch.setattr(gp.time, "sleep", lambda s: None)
    client = _client(SNAPS, {})
    client.get_snapshots.side_effect = RuntimeError("503")
    assert gp.run_pick(False, client, _broker(), _session(), UNI, NOW) == 0
    assert gp.read_ledger()[0]["entry_status"] == "skipped_api"
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("skipped" in r.message for r in caplog.records if r.levelname == "WARNING")


def test_with_retries_succeeds_on_third(monkeypatch):
    monkeypatch.setattr(gp.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("x")
        return 7
    assert gp.with_retries(fn, "t") == 7


def test_reconcile_covers_leftover_after_1040_once_and_not_before(paths, caplog):
    orders = [{"client_order_id": "gapshort-2026-11-13-AAA", "symbol": "AAA", "side": "sell", "status": "filled"}]
    broker = _broker(positions={"AAA": (-20.0, 15.0)}, orders=orders)
    assert gp.reconcile(broker, NOW.replace(hour=10, minute=39), False, "score") == []
    left = gp.reconcile(broker, NOW.replace(hour=10, minute=41), False, "score")
    assert left == [("AAA", 20.0, 15.0)] and broker.cover_market.call_count == 1
    assert len([r for r in caplog.records if r.levelname == "ERROR"]) == 1


def test_cover_replaces_at_fresh_ask_then_markets(paths, monkeypatch):
    monkeypatch.setattr(gp.time, "sleep", lambda s: None)
    client = MagicMock(spec=AlpacaClient)
    client.get_latest_quote.side_effect = [{"ask_price": 10.0}, {"ask_price": 10.5}]
    broker = _broker(positions={"AAA": (-20.0, 15.0)}, orders=[{"client_order_id": f"gapshort-{DAY}-AAA", "symbol": "AAA"}])
    broker.cover_limit.side_effect = [{"id": "c1", "status": "accepted"}, {"id": "c2", "status": "accepted"}]
    broker.order.return_value = {"status": "new", "filled_qty": 0.0, "filled_avg_price": None}
    gp.write_ledger([{"date": DAY, "symbol": "AAA"}])
    gp.run_cover(False, client, broker, NOW.replace(hour=10, minute=30), poll_s=0)
    assert [c.args[2] for c in broker.cover_limit.call_args_list] == [10.01, 10.51]
    assert [c.args[0] for c in broker.cancel.call_args_list] == ["c1", "c2"]
    assert broker.cover_market.call_count == 1


def test_weekly_clears_pause_on_green_week_and_sets_on_two_red(paths):
    gp.PAUSE_FLAG.write_text("x")
    gp.update_pause_flag([], 5.0, DAY)
    assert not gp.PAUSE_FLAG.exists()
    rows = [{"date": "2026-11-02", "pnl_usd": "-120"}, {"date": "2026-11-09", "pnl_usd": "-130"}]
    gp.update_pause_flag(rows, -130.0, "2026-11-13")
    assert gp.PAUSE_FLAG.exists()

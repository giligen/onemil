"""MOM backtest-stays-current (spec docs/mom_bt_current_spec_20261031.md): weekly panel refresh, rule-vs-book
parity, the EOD lines.  Unit tests on fixtures; one integration test through refresh -> parity -> csv -> EOD line
on a tiny REAL parquet panel (the rule module REBUILD_1700tu runs for real) with a mocked Alpaca client.
Everything writes to tmp_path: the real logs/mom_bt_parity.csv and panel_ext.parquet are never touched."""
import hashlib
import logging
import math
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "research" / "momentum_weekly"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eod_report as er  # noqa: E402
import eod_sections as es  # noqa: E402
import mom_bt_parity as par  # noqa: E402
import mom_panel_refresh as ref  # noqa: E402
from data_sources.alpaca_client import AlpacaClient  # noqa: E402

COLS = ref.COLS


# ----------------------------------------------------------------------------- fixtures
def _bars(symbols, dates, factor=None, vol=None):
    """Deterministic daily bars: price = base(sym) + 0.01 * day index, x factor[sym] (a restatement)."""
    rows = []
    for k, s in enumerate(symbols):
        for i, d in enumerate(dates):
            px = (50.0 + 5 * k + 0.01 * i) * (factor or {}).get(s, 1.0)
            v = (vol or {}).get(s, 5_000_000.0 if s == "SPY" else 1000.0)
            rows.append((s, pd.Timestamp(d), px, px, px, px, v))
    return pd.DataFrame(rows, columns=COLS).astype({c: "float32" for c in COLS[2:]})


class _BarSet:
    """Stands in for the Alpaca BarSet: only `.df` (MultiIndex symbol/timestamp) is read by bars_to_df."""

    def __init__(self, df):
        d = df.rename(columns={"bar_date": "timestamp"}).set_index(["symbol", "timestamp"])
        d.index = d.index.set_levels(d.index.levels[1].tz_localize("UTC"), level=1)
        self.df = d


def _fake_fetch(universe_bars, removed=(), factor=None):
    """fetch_fn(client, batch, start, end) -> (barset, remaining, removed): the bars of ``universe_bars`` inside
    start..end for the batch's symbols, scaled by ``factor`` (restatement) -- the shape of fetch_batch_resilient."""
    def fn(client, batch, start, end):
        got = [s for s in batch if s not in removed]
        b = universe_bars[universe_bars["symbol"].isin(got)]
        b = b[(b["bar_date"] >= start) & (b["bar_date"] < end)].copy()
        for s, f in (factor or {}).items():
            m = b["symbol"] == s
            for c in ("open", "high", "low", "close"):
                b.loc[m, c] = (b.loc[m, c].astype("float64") * f).astype("float32")
        return (_BarSet(b) if len(b) else None), [s for s in got if s in set(b["symbol"])], list(removed)
    return fn


ASSETS = pd.DataFrame({"symbol": ["SPY", "AAA", "BBB", "FND", "ZAZZT"],
                       "name": ["SPDR S&P 500 ETF TRUST", "Alpha Inc", "Beta Corp", "Some Growth FUND", "Test Co"]})
SESS = list(pd.bdate_range("2026-09-21", "2026-10-23"))       # 25 sessions: 20 for the ADV window + asof
EXT0_DATES = [d for d in SESS if d <= pd.Timestamp("2026-10-21")]


def _mk_ext(tmp_path, dates=EXT0_DATES, syms=("SPY", "AAA", "BBB")):
    p = tmp_path / "panel_ext.parquet"
    ref.write_atomic(_bars(list(syms), dates), str(p))
    return p


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _refresh(tmp_path, ext, fetch_fn, **kw):
    kw.setdefault("today", date(2026, 10, 24))
    kw.setdefault("free_fn", lambda _p: 10 * 1024 ** 3)
    return ref.refresh(MagicMock(spec=AlpacaClient), ASSETS, ext_path=str(ext), fetch_fn=fetch_fn, **kw)


# ----------------------------------------------------------------------------- refresh: unit
def test_universe_is_common_stock_only_plus_spy():
    u = ref.sleeve_universe(ASSETS)
    assert u == ["AAA", "BBB", "SPY"]                      # ETF-named SPY kept explicitly; FUND and ZxZZT dropped
    assert ref.sleeve_universe(ASSETS, {"AAA"}) == ["AAA", "SPY"]      # only names the backtest tracks


def test_refresh_appends_new_bars_and_is_idempotent(tmp_path):
    ext = _mk_ext(tmp_path)
    full = _bars(["SPY", "AAA", "BBB"], SESS)
    n0 = len(pd.read_parquet(ext))
    assert _refresh(tmp_path, ext, _fake_fetch(full)) == 0
    after = pd.read_parquet(ext)
    assert len(after) == n0 + 3 * 2 and after["bar_date"].max() == pd.Timestamp("2026-10-23")
    assert list(after.columns) == COLS and after.duplicated(["symbol", "bar_date"]).sum() == 0
    digest = _sha(ext)
    assert _refresh(tmp_path, ext, _fake_fetch(full)) == 0         # second run, same day: nothing to append
    assert _sha(ext) == digest


def test_refresh_dry_run_writes_nothing(tmp_path, capsys):
    ext = _mk_ext(tmp_path)
    digest = _sha(ext)
    assert _refresh(tmp_path, ext, _fake_fetch(_bars(["SPY", "AAA", "BBB"], SESS)), dry_run=True) == 0
    assert _sha(ext) == digest and "DRY-RUN: would append 6 rows" in capsys.readouterr().out


def test_restated_symbol_rows_are_replaced_and_listed(tmp_path, caplog):
    ext = _mk_ext(tmp_path)
    full = _bars(["SPY", "AAA", "BBB"], SESS)
    with caplog.at_level(logging.INFO):
        assert _refresh(tmp_path, ext, _fake_fetch(full, factor={"AAA": 0.5})) == 0     # a 2:1 re-adjustment
    after = pd.read_parquet(ext)
    old_aaa = _bars(["SPY", "AAA", "BBB"], EXT0_DATES).query("symbol == 'AAA'").iloc[0]["close"]
    new_aaa = after[(after.symbol == "AAA")].sort_values("bar_date").iloc[0]["close"]
    assert new_aaa == pytest.approx(old_aaa * 0.5, rel=1e-5)                              # ALL history of AAA replaced
    assert after[after.symbol == "BBB"].sort_values("bar_date").iloc[0]["close"] == \
        pytest.approx(_bars(["SPY", "AAA", "BBB"], EXT0_DATES).query("symbol == 'BBB'").iloc[0]["close"], rel=1e-6)              # others untouched
    assert any("restated 1 symbols" in r.message and "AAA" in r.message and r.levelno == logging.INFO
               for r in caplog.records)


def test_gate_failure_leaves_file_untouched_and_logs_error(tmp_path, caplog):
    ext = _mk_ext(tmp_path)
    digest = _sha(ext)
    full = _bars(["SPY", "AAA", "BBB"], SESS)
    with caplog.at_level(logging.INFO):
        rc = _refresh(tmp_path, ext, _fake_fetch(full, removed=("AAA",)))      # 1 of 3 requested LOST = 33 % > 5 %
    assert rc == 1 and _sha(ext) == digest
    assert any(r.levelno == logging.ERROR and "GATE FAILED" in r.message for r in caplog.records)


def test_low_disk_and_missing_extension_are_errors(tmp_path, caplog):
    ext = _mk_ext(tmp_path)
    with caplog.at_level(logging.INFO):
        assert _refresh(tmp_path, ext, _fake_fetch(_bars(["SPY"], SESS)), free_fn=lambda _p: 1024 ** 3) == 1
        assert _refresh(tmp_path, tmp_path / "nope.parquet", _fake_fetch(_bars(["SPY"], SESS))) == 1
    msgs = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert any("disk free" in m for m in msgs) and any("missing or empty" in m for m in msgs)


# ----------------------------------------------------------------------------- parity: unit
def _w(monday, names, ret, spy, end):
    return {"monday": pd.Timestamp(monday), "names": names, "ret": ret, "spy_ret": spy,
            "end": pd.Timestamp(end), "complete": True}


NAMES = [f"N{i:02d}" for i in range(20)]


def _three_week_fixture():
    weeks = [_w("2026-10-12", NAMES, 0.015, 0.004, "2026-10-16"),
             _w("2026-10-19", NAMES, -0.024, -0.005, "2026-10-23"),
             _w("2026-10-26", NAMES[:17] + ["Y1", "Y2", "Y3"], 0.010, 0.002, "2026-10-30")]
    weekly = [{"date": "2026-10-12", "equity": "20000", "names_in": " ".join(NAMES), "names_out": ""},
              {"date": "2026-10-19", "equity": "20400", "names_in": "X1", "names_out": "N19"}]
    live = [{"date": "2026-10-12", "equity": "10000"}, {"date": "2026-10-19", "equity": "9800"}]
    # account equity per session (history stamp = next weekday): Fri 10/9 100000, 10/16 100400, 10/23 99912, 10/30 100100
    stamps = {date(2026, 10, 12): 100000.0, date(2026, 10, 19): 100400.0, date(2026, 10, 26): 99912.0,
              date(2026, 11, 2): 100100.0}
    hist = [{"stamp": k, "equity": v} for k, v in stamps.items()]
    return weeks, weekly, live, hist


def test_parity_row_math_on_three_weeks():
    from hourly_summary import session_date
    weeks, weekly, live, hist = _three_week_fixture()
    rows = par.build_rows(weeks, weekly, live, hist, 9500.0, session_date)
    r1, r2, r3 = rows
    assert (r1["bt_names_match"], r2["bt_names_match"], r3["bt_names_match"]) == (20, 19, 17)
    assert r1["paper_ret"] == pytest.approx((100400 - 100000) / 20000)           # +2.0 % on the slice mark
    assert r1["residual_pp"] == pytest.approx((0.02 - 0.015) * 100)
    assert r2["paper_ret"] == pytest.approx((99912 - 100400) / 20400)
    assert r2["residual_pp"] == pytest.approx((-488 / 20400 + 0.024) * 100)
    assert r1["live_ret"] == pytest.approx(9800 / 10000 - 1)
    assert math.isnan(r2["live_ret"]) and math.isnan(r3["live_ret"])             # no mark for the skipped 10/26 rotation
    assert r3["paper_ret"] == pytest.approx((100100 - 99912) / 20000)            # no 10/26 slice mark -> slice start
    assert r3["diff"]["bt_only"] == ["Y1", "Y2", "Y3"] and r3["diff"]["paper_only"] == ["N17", "N18", "X1"]


def test_live_week_ret_running_week_uses_sleeve_equity_now():
    live = [{"date": "2026-10-19", "equity": "10000"}]
    assert par.live_week_ret(live, pd.Timestamp("2026-10-19"), None, 9454.0) == pytest.approx(-0.0546)
    assert math.isnan(par.live_week_ret(live, pd.Timestamp("2026-10-19"), None, None))
    assert math.isnan(par.live_week_ret(live, pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-26"), 9454.0))


def test_mismatch_wording_and_ok_line():
    ok = {"date": "2026-10-26", "bt_names_match": 19, "bt_ret": -0.024, "paper_ret": -0.021, "live_ret": -0.054,
          "spy_ret": -0.005, "residual_pp": 0.3}
    assert par.format_line(ok) == ("BT parity wk 10/26: names 19/20 | BT -2.4 % | paper -2.1 % | live -5.4 % | "
                                   "SPY -0.5 % | resid +0.3 pp")
    assert par.is_mismatch(ok) is False
    assert par.format_line({**ok, "bt_names_match": 17}).startswith("MISMATCH BT parity wk 10/26: names 17/20")
    assert par.format_line({**ok, "residual_pp": -2.4}).startswith("MISMATCH ")
    nan = {**ok, "live_ret": float("nan"), "paper_ret": float("nan"), "residual_pp": float("nan")}
    assert "live n/a" in par.format_line(nan) and "resid n/a" in par.format_line(nan) and not par.is_mismatch(nan)


def test_mismatch_logs_warning_with_names_and_csv_upserts(tmp_path, caplog, monkeypatch):
    from hourly_summary import session_date
    weeks, weekly, live, hist = _three_week_fixture()
    monkeypatch.setattr(par, "bt_weeks", lambda R_, data, n=8: weeks)       # isolate the reconciliation from the rule
    wp, lp, path = tmp_path / "w.csv", tmp_path / "l.csv", tmp_path / "mom_bt_parity.csv"
    pd.DataFrame(weekly).to_csv(wp, index=False)
    pd.DataFrame(live).to_csv(lp, index=False)
    sent = []
    with caplog.at_level(logging.INFO):
        par.run(MagicMock(), {}, str(wp), str(lp), str(path), hist, 9500.0, session_date, send=sent.append)
        par.run(MagicMock(), {}, str(wp), str(lp), str(path), hist, 9500.0, session_date, send=sent.append)
    warn = [r for r in caplog.records if r.levelno == logging.WARNING and "MISMATCH wk 2026-10-26" in r.message]
    assert warn and "Y1" in warn[0].message and "X1" in warn[0].message                # WARNING, never ERROR
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(pd.read_csv(path)) == 3                                                 # second run upserts, no duplicates
    assert len(sent) == 2 and sent[0].startswith("MISMATCH BT parity wk 10/26: names 17/20")


# ----------------------------------------------------------------------------- EOD lines: unit
def test_eod_parity_line_no_data_and_populated(tmp_path):
    p = tmp_path / "mom_bt_parity.csv"
    assert par.latest_parity_line(str(p)).startswith("MOM BT parity: NO-DATA")
    par.write_parity_csv([{"date": "2026-10-26", "bt_names_match": 19, "bt_ret": -0.024, "paper_ret": -0.021,
                           "live_ret": float("nan"), "spy_ret": -0.005, "residual_pp": 0.3}], str(p))
    assert par.latest_parity_line(str(p)) == ("MOM BT parity: wk 10/26: names 19/20 | BT -2.4 % | paper -2.1 % | "
                                              "live n/a | SPY -0.5 % | resid +0.3 pp")
    par.write_parity_csv([{"date": "2026-10-26", "bt_names_match": 15, "bt_ret": -0.024, "paper_ret": -0.021,
                           "live_ret": float("nan"), "spy_ret": -0.005, "residual_pp": 0.3}], str(p))
    assert len(pd.read_csv(p)) == 1 and "MISMATCH wk 10/26" in par.latest_parity_line(str(p))


def test_history_period_reaches_before_the_sleeve_start():
    assert es.history_period(date(2026, 10, 31)) == "3M"           # was a fixed 1M: lost the 9/29 base -> 'since n/a'
    assert es.history_period(date(2027, 9, 1)) == "1A"


def test_mom_history_asks_for_the_long_period(monkeypatch):
    import hourly_summary
    seen = {}
    monkeypatch.setattr(hourly_summary, "fetch_history", lambda c, period="1W": seen.setdefault("p", period) and [{"stamp": date(2026, 9, 29), "equity": 1.0}])
    assert es.mom_history(MagicMock(), None) is not None and seen["p"] == es.history_period()


def _hod_db(tmp_path):
    p = tmp_path / "trades.db"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE trades (trade_date TEXT, strategy TEXT, account TEXT, order_status TEXT, "
                "exit_price REAL, pnl REAL)")
    con.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?)", [
        ("2026-09-29", "hod_break", "paper", "filled", 5.0, -20.0), ("2026-09-30", "hod_break", "paper", "filled", 5.0, 50.0),
        ("2026-09-30", "hod_break", "paper", "filled", None, None), ("2026-09-30", "hod_break", "paper", "cancelled", None, None),
        ("2026-09-30", "hod_break", "live", "filled", 5.0, -999.0), ("2026-09-30", "orb", "paper", "filled", 5.0, 999.0)])
    con.commit()
    con.close()
    return p


def test_hod_lines_read_the_paper_book_only(tmp_path):
    db = _hod_db(tmp_path)
    line = es.hod_paper_line(db)
    assert line.startswith("HOD (paper book): 3 fills since 2026-09-29 | closed 2 | total $+30")
    g = es.hod_guardrail_line(db)
    assert g.startswith("hod_break (paper): n=3 fills since 2026-09-29") and "worst session 2026-09-29 $-20" in g
    assert "NO-DATA" in es.hod_paper_line(tmp_path / "missing.db")
    out = es.promotion_section("2026-10-30", None, None, tmp_path / "s.json", closed=True, hod_db=db)
    assert out[-1] == "  " + line


def test_hod_summary_is_read_only(tmp_path):
    db = _hod_db(tmp_path)
    digest = _sha(db)
    es.hod_paper_summary(db)
    assert _sha(db) == digest


def test_bf_freeze_printed_only_when_bf_enabled(tmp_path):
    from trading import ramp_freeze as rf
    p = tmp_path / "ramp_freeze.json"
    rf.set_freeze("bf", "JAGX: LIVE_ONLY", day="2026-09-24", path=p, notify=False)
    assert er.freeze_summary(p, bf_on=False) == "freeze: none"
    assert "BF frozen since 2026-09-24" in er.freeze_summary(p, bf_on=True)


# ----------------------------------------------------------------------------- integration
def _tiny_panel(tmp_path, n_sym=22, n_days=330, last="2026-10-02"):
    """A real parquet panel in the frozen-panel schema: every symbol liquid (price ~50, $250M/day), distinct drift."""
    dates = pd.bdate_range(end=last, periods=n_days)
    rng = np.random.default_rng(7)
    rows = []
    for k in range(n_sym):
        drift = 0.0004 + 0.0002 * k
        px = 50 * np.cumprod(1 + drift + rng.normal(0, 0.004, n_days))
        for d, p in zip(dates, px):
            rows.append((f"S{k:02d}", d, p, p * 1.002, p * 0.998, p, 5_000_000.0))
    spy = 400 * np.cumprod(1 + 0.0003 + rng.normal(0, 0.003, n_days))
    rows += [("SPY", d, p, p, p, p, 80_000_000.0) for d, p in zip(dates, spy)]
    df = pd.DataFrame(rows, columns=COLS).astype({c: "float32" for c in COLS[2:]})
    panel = tmp_path / "panel.parquet"
    df.to_parquet(panel, index=False)
    names = pd.DataFrame({"symbol": sorted(df.symbol.unique()), "name": "Common Stock Inc"})
    assets = tmp_path / "assets.csv"
    names.to_csv(assets, index=False)
    return panel, assets, df


def test_refresh_parity_csv_eod_line_end_to_end(tmp_path, monkeypatch, caplog):
    from hourly_summary import session_date
    import REBUILD_1700tu as R
    panel, assets_csv, base = _tiny_panel(tmp_path)
    syms = sorted(base.symbol.unique())
    # the extension holds bars up to Fri 10/9; the Saturday 10/17 refresh must add 10/12..10/16
    full_ext = _bars_like(base, pd.bdate_range("2026-09-14", "2026-10-16"))
    ext = tmp_path / "panel_ext.parquet"
    ref.write_atomic(full_ext[full_ext.bar_date <= "2026-10-09"], str(ext))
    n0 = len(pd.read_parquet(ext))
    assets = pd.DataFrame({"symbol": syms, "name": "Common Stock Inc"})
    with caplog.at_level(logging.INFO):
        rc = ref.refresh(MagicMock(spec=AlpacaClient), assets, ext_path=str(ext), fetch_fn=_fake_fetch(full_ext),
                         today=date(2026, 10, 17), free_fn=lambda _p: 10 * 1024 ** 3)
    assert rc == 0
    e = pd.read_parquet(ext)
    assert len(e) == n0 + 5 * len(syms) and e["bar_date"].max() == pd.Timestamp("2026-10-16")        # boundary 1
    monkeypatch.setattr(R, "PANEL", str(panel))
    monkeypatch.setattr(R, "ASSETS", str(assets_csv))
    data = R.load()
    assert par.extend_data(data, e) == 10 * len(syms)             # only bars after the panel's last date (10/2)
    cal = pd.DatetimeIndex(data["SPY"][0])
    assert cal[-1] == pd.Timestamp("2026-10-16")                                                      # boundary 2
    last = par.bt_weeks(R, data)[-1]
    assert len(last["names"]) == 20 and last["monday"] == pd.Timestamp("2026-10-12") and last["end"] == cal[-1]
    assert not last["complete"] and math.isfinite(last["ret"]) and math.isfinite(last["spy_ret"])     # boundary 3
    weekly = tmp_path / "weekly.csv"                  # the paper book = the BT book except one name -> 19/20
    held = list(last["names"][:-1]) + ["ZZZZ"]
    pd.DataFrame([{"date": "2026-10-12", "equity": 20000.0, "names_in": " ".join(held), "names_out": ""}]
                 ).to_csv(weekly, index=False)
    hist = [{"stamp": date(2026, 10, 12), "equity": 100000.0}, {"stamp": date(2026, 10, 17), "equity": 100500.0}]
    parity, sent = tmp_path / "mom_bt_parity.csv", []
    rows = par.run(R, data, str(weekly), str(tmp_path / "nolive.csv"), str(parity), hist, None, session_date,
                   send=sent.append)
    r = rows[-1]
    assert len(rows) == 1 and r["date"] == "2026-10-12"
    assert r["bt_names_match"] == 19 and r["paper_ret"] == pytest.approx(0.025)    # (100500-100000)/20000
    assert r["residual_pp"] == pytest.approx((0.025 - last["ret"]) * 100) and math.isnan(r["live_ret"])
    csv = pd.read_csv(parity)                                                                         # boundary 4
    assert csv.columns.tolist() == par.FIELDS and len(csv) == 1      # the 7 earlier Mondays predate the paper sleeve
    assert csv.iloc[-1]["date"] == "2026-10-12" and csv.iloc[-1]["bt_names_match"] == 19
    assert len(sent) == 1 and sent[0] == par.format_line(r) and "wk 10/12" in sent[0]
    line = par.latest_parity_line(str(parity))                                                        # boundary 5
    assert line.startswith("MOM BT parity: ") and "names 19/20" in line and "live n/a" in line


def _bars_like(base, dates):
    """Extension bars for the tiny panel's symbols over ``dates`` (a continuation of each symbol's last close)."""
    rows = []
    last = base.sort_values("bar_date").groupby("symbol").tail(1).set_index("symbol")
    rng = np.random.default_rng(11)
    for s, r in last.iterrows():
        px = float(r["close"])
        for d in dates:
            px *= 1 + rng.normal(0.0005, 0.004)
            rows.append((s, pd.Timestamp(d), px, px * 1.002, px * 0.998, px, float(r["volume"])))
    return pd.DataFrame(rows, columns=COLS).astype({c: "float32" for c in COLS[2:]})

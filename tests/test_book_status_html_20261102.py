"""Tests for scripts/book_status_html.py (renderer) and the HTML-contract extensions of scripts/book_status.py."""
import copy
import datetime as dt
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import book_status as bs  # noqa: E402
import book_status_html as bh  # noqa: E402

DAY = "2026-11-02"  # Monday
TOKENS = ["bg", "panel", "panel-2", "line", "fg", "muted", "accent", "gain", "loss", "warn", "crit", "info"]


def _mom(eq, start):
    """Fully populated MOM book dict."""
    return {"date": DAY, "equity": eq, "start": start, "start_equity": start, "pct_vs_start": (eq / start - 1) * 100,
            "to_pause": eq - 0.75 * start, "to_halve": eq - 0.6 * start, "pause_rail": 0.75 * start, "halve_rail": 0.6 * start,
            "names_in": ["HPE", "MRNA"], "names_out": ["DINO"], "last_rebalance": DAY, "holdings_n": 20,
            "weekly": [{"date": "2026-10-26", "equity": eq * 1.02, "week_pct": None, "spy_pct": None},
                       {"date": DAY, "equity": eq, "week_pct": -2.0, "spy_pct": -1.0}],
            "parity": [{"date": "2026-10-19", "names_n": 20, "resid_pp": 0.4}, {"date": "2026-10-26", "names_n": 17, "resid_pp": 1.0}]}


def _fill(sym, t, pnl, why):
    """One fill dict."""
    return {"symbol": sym, "entry_ts_et": t, "fill_price": 10.5, "stop": 10.0, "target": 11.5, "shares": 100, "risk_usd": 50.0,
            "pnl": pnl, "exit_reason": why, "exit_ts_et": None, "entry_quote_spread_bps": 12.0, "pricing_method": None}


@pytest.fixture
def full():
    """All sections populated, mid-session on a Monday, with one fill symbol that needs escaping."""
    fills = [_fill("AAA", "09:45:10", -40.0, "stop_loss"), _fill("<X>", "10:30:00", 90.0, "target"), _fill("CCC", "13:10:30", None, None)]
    hod = {"today": {"fills": 3, "pnl": 50.0, "open": 1}, "wtd": {"fills": 3, "pnl": 50.0, "open": 1},
           "since_0929": {"fills": 40, "pnl": -256.0, "open": 1}, "exits": {"stops": 1, "targets": 1, "squeeze": 0, "eod": 0},
           "open_symbols": ["CCC"], "arm_skips": {"r_pct": 4, "min_price": 2, "other": 0}, "skip_symbols": 5, "cap_reached": 1,
           "squeeze_lines": 0, "fills": fills, "account": "paper",
           "daily": [{"date": f"2026-10-{d:02d}", "pnl": p, "fills": 3} for d, p in zip(range(12, 24), [5, -20, 30, -10, 40, 0, -60, 20, 10, -5, 15, 25])],
           "cum_since_start": [{"date": "2026-09-29", "cum_pnl": 0.0}, {"date": "2026-10-01", "cum_pnl": 40.0}, {"date": DAY, "cum_pnl": -256.0}],
           "gate": {"arm_skip_r_pct": 4, "arm_skip_min_price": 2, "arm_skip_symbols": 5, "cap_hits": 1, "squeeze_lines": 0,
                    "squeeze_enabled": True, "concurrency_cap": "4", "per_day_cap": "12", "risk_usd": 50.0, "dry_run": "False", "day_kill": -450.0}}
    orb = {"today": {"fills": 1, "pnl": -10.0}, "wtd": {"fills": 1, "pnl": -10.0}, "ramp": {"frozen": False, "reason": ""},
           "fills": [_fill("ORBX", "09:50:00", -10.0, "stop")], "account": "paper", "vetoes": [{"symbol": "XPEV", "reason": "PDR VETO quiet"}],
           "daily": [{"date": "2026-10-30", "pnl": 20.0, "fills": 2}]}
    return {"date": DAY, "session_date": DAY, "generated_at_et": f"{DAY}T15:20:00-05:00",
            "service": {"active": "active"}, "hod": hod, "orb": orb,
            "mom": {"paper": _mom(16400.0, 20000.0), "live": _mom(9173.0, 10000.0)},
            "paused": {"bf_enabled": False, "r2g_enabled": False},
            "ops": {"service_active": True, "service_state": "active", "boot_utc": "12:30", "errors_today": 0, "warnings_today": 4,
                    "journal_lines_today": 100, "repeated_warnings": [],
                    "crons": [{"name": "momentum_sleeve", "schedule": "50 11 * * 1-3", "log": "m.log", "last_write": "11-02 11:50",
                               "errors_today": 0, "last_run": "11-02 11:50", "overdue": False}]},
            "flags": [{"severity": "warn", "book": "mom_paper", "text": "MOM paper near rail", "why": "rule x"},
                      {"severity": "info", "book": "hod", "text": "137 ARM SKIP", "why": "rule y"}]}


@pytest.fixture
def empty():
    """Weekend, empty books, no flags."""
    return {"date": "2026-11-01", "session_date": "2026-11-01", "generated_at_et": "2026-11-01T10:00:00-05:00",
            "service": {"active": "active"},
            "hod": {"today": {"fills": 0, "pnl": 0.0, "open": 0}, "wtd": {"fills": 0, "pnl": 0.0}, "since_0929": {"fills": 0, "pnl": 0.0},
                    "exits": {}, "fills": [], "daily": [], "cum_since_start": [], "gate": {}},
            "orb": {"today": {"fills": 0, "pnl": 0.0}, "wtd": {"fills": 0, "pnl": 0.0}, "ramp": None, "fills": [], "vetoes": [], "daily": []},
            "mom": {"paper": {"unavailable": "x"}, "live": {"unavailable": "y"}}, "paused": {"bf_enabled": False, "r2g_enabled": False},
            "ops": {"service_active": True, "service_state": "active", "boot_utc": None, "errors_today": 0, "warnings_today": 0,
                    "journal_lines_today": 0, "repeated_warnings": [], "crons": []}, "flags": []}


def test_tokens_themes_and_page_contract(full):
    """Every token on bare :root, both theme blocks, title first, no document tags, size, only font hosts."""
    h = bh.render(full)
    assert h.startswith("<title>OneMil Books</title>")
    root = re.search(r":root\{(.*?)\}", h, re.S).group(1)
    for t in TOKENS:
        assert f"--{t}:" in root
    assert "color-scheme:dark" in root
    assert '@media (prefers-color-scheme:light){:root:not([data-theme="dark"])' in h
    assert ':root[data-theme="light"]' in h
    assert "body{margin:0;background:var(--bg)" in h
    assert not re.search(r"<(!doctype|html|head|body)\b", h, re.I)
    assert len(h.encode()) < 400_000
    hosts = set(re.findall(r"https?://([^/\"' )<]+)", h))
    assert hosts <= {"fonts.googleapis.com", "fonts.gstatic.com"}


def test_tiles_flags_and_escape(full):
    """Five tiles with their big numbers, flags rendered, symbol escaped."""
    h = bh.render(full)
    assert h.count('data-tile="') == 5
    assert "+$50" in h and "-$10" in h and "-8.3%" in h and "-18.0%" in h
    assert "HOD · PAPER $50 R" in h
    assert "MOM paper near rail" in h and "137 ARM SKIP" in h
    assert "&lt;X&gt;" in h and "<X>" not in h
    assert h.count('<tr class="') >= 4  # fills rows present


def test_timeline_circles_markers_and_details(full):
    """One circle per fill, the 14:00 marker, now marker; details open only where there was activity."""
    h = bh.render(full)
    tl = re.search(r'<svg id="timeline".*?</svg>', h, re.S).group(0)
    assert tl.count("<circle") == 4
    assert "mark-squeeze" in tl and "14:00 squeeze" in tl and 'class="now"' in tl
    opened = re.findall(r'<details class="book" data-key="(\w+)"( open)?', h)
    assert {k: bool(o) for k, o in opened} == {"hod": True, "orb": True, "mom": True, "paused": False, "ops": False}
    assert h.count("Expand all") == 1


def test_empty_weekend(empty):
    """Empty state text, weekend session line, no activity -> all details closed, MOM unavailable shown."""
    h = bh.render(empty)
    assert "Nothing needs action." in h
    assert "no session (weekend)" in h
    assert "<details class=\"book\" data-key=\"hod\" open" not in h
    assert not re.search(r"<details[^>]* open", h)
    assert h.count('data-tile="') == 5
    assert 'class="now"' not in h
    assert "unavailable" in h


def test_session_state_branches():
    """Clock words for pre-open, open, flat and after the close."""
    mon = dt.datetime(2026, 11, 2)
    assert "before the open" in bh.session_state(mon.replace(hour=8))
    assert "min to the flat" in bh.session_state(mon.replace(hour=15, minute=20))
    assert bh.session_state(mon.replace(hour=17)) == "after the close"


# ---- data-contract extensions (book_status.py) -------------------------------------------------------------------
@pytest.fixture
def db(tmp_path):
    """tmp trades.db with the production columns this contract reads."""
    p = tmp_path / "trades.db"
    con = sqlite3.connect(p)
    con.execute("create table trades (id integer primary key, trade_date text, symbol text, strategy text, fill_price real, pnl real, "
                "exit_reason text, filled_at text, exited_at text, stop_loss_price real, take_profit_price real, shares int, "
                "total_risk real, entry_quote_spread real, exit_pricing_method text, account text)")
    con.execute("insert into trades values (1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (DAY, "AAA", "hod_break", 100.0, -40.0, "stop_loss", "2026-11-02T14:46:37+00:00", "2026-11-02T15:10:00+00:00",
                 98.0, 104.0, 20, 40.0, 0.5, "stop_bid", "paper"))
    con.execute("insert into trades values (2,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (DAY, "BBB", "hod_break", 50.0, None, None, "2026-11-02T15:00:00+00:00", None, 49.0, 52.0, 10, 10.0, None, None, "paper"))
    con.execute("insert into trades values (3,'2026-10-30','CCC','orb',20.0,5.0,'target',null,null,19,22,5,5,null,null,'paper')")
    con.commit()
    con.close()
    return p


def test_fills_daily_and_utc_to_et(db):
    """Fill dicts: UTC -> ET, spread bps from dollars, open row has null pnl; daily and cumulative series."""
    f = bs.book_fills(db, DAY, "hod_break")
    assert [x["symbol"] for x in f] == ["AAA", "BBB"]
    assert f[0]["entry_ts_et"] == "09:46:37" and f[0]["exit_ts_et"] == "10:10:00"
    assert f[0]["entry_quote_spread_bps"] == pytest.approx(50.0) and f[0]["risk_usd"] == 40.0 and f[0]["pricing_method"] == "stop_bid"
    assert f[1]["pnl"] is None and f[1]["exit_reason"] is None
    d = bs.book_daily(db, DAY, "hod_break", "2026-09-29")
    assert d["daily"] == [{"date": DAY, "pnl": -40.0, "fills": 1}] and d["cum_since_start"][-1]["cum_pnl"] == -40.0 and d["account"] == "paper"
    assert bs.book_daily(db, DAY, "orb")["daily"][0]["pnl"] == 5.0


def test_fills_tolerate_reduced_schema(tmp_path):
    """A schema without the optional columns yields None fields instead of failing (columns come from pragma table_info)."""
    p = tmp_path / "t.db"
    con = sqlite3.connect(p)
    con.execute("create table trades (id integer primary key, trade_date text, symbol text, strategy text, fill_price real, pnl real, exit_reason text)")
    con.execute("insert into trades values (1,?,?,?,?,?,?)", (DAY, "ZZZ", "orb", 5.0, 1.0, "target"))
    con.commit()
    con.close()
    f = bs.book_fills(p, DAY, "orb")
    assert f[0]["symbol"] == "ZZZ" and f[0]["entry_ts_et"] is None and f[0]["risk_usd"] is None


def test_vetoes_gate_and_repeated_warnings():
    """PDR / Q1 vetoes parsed and deduped; gates line parsed; repeated WARNING text collapses digits."""
    jl = ["x [ORB PREPLACE] provisional PDR VETO: XPEV prev-day range 1.88% <= 11.0% quiet", "[ORB PREPLACE] provisional PDR VETO: XPEV again",
          "[ORB] Q1 filter dropped 2 candidate(s): AAA, BBB",
          "[HOD] engine gates: book=hod_break enabled=True dry_run=False risk=$50 kills=-450.0/-1500.0 per_day=12 concurrent=4 flat=955"]
    assert [v["symbol"] for v in bs.orb_vetoes(jl)] == ["XPEV", "AAA", "BBB"]
    g = bs.hod_gate(jl, {"arm_skips": {"r_pct": 1, "min_price": 2}, "skip_symbols": 3}, Path("/nonexistent.yaml"))
    assert g["risk_usd"] == 50.0 and g["day_kill"] == -450.0 and g["concurrency_cap"] == "4" and g["squeeze_enabled"] is None
    warns = [f"2026-11-02 12:00:0{i} | WARNING  | mod:1 | cache MISS {i}/99 symbols" for i in range(3)]
    assert bs.repeated_warnings(warns)[0]["count"] == 3


def test_cron_schedule_and_overdue(tmp_path):
    """last_scheduled_run honours dow ranges; a log older than the last scheduled run is overdue, a fresh one is not."""
    now = dt.datetime(2026, 11, 2, 15, 0)  # Monday
    assert bs.last_scheduled_run("50 11 * * 1-3", now) == dt.datetime(2026, 11, 2, 11, 50)
    assert bs.last_scheduled_run("0 13 * * 6", now) == dt.datetime(2026, 10, 31, 13, 0)
    (tmp_path / "fresh.log").write_text("ok\n")
    stale = tmp_path / "stale.log"
    stale.write_text("ok\n")
    import os
    os.utime(stale, (1, 1))
    lines = ["50 11 * * 1-3 cd /x && python3 scripts/job_a.py >> logs/stale.log 2>&1", "# comment", "MAILTO=x"]
    cr = bs.ops_crons(DAY, lambda: lines, tmp_path, now)
    assert len(cr) == 1 and cr[0]["overdue"] is True and cr[0]["name"] == "job_a"


def test_flags_rules():
    """Crit / warn / info rules fire on the pre-committed conditions and sort crit first."""
    data = {"ops": {"errors_today": 2, "service_active": False, "service_state": "inactive", "repeated_warnings": [{"text": "w", "count": 5}],
                    "crons": [{"name": "c", "log": "c.log", "schedule": "* * * * *", "overdue": True, "last_run": "x"}]},
            "hod": {"today": {"fills": 0, "pnl": -500.0, "open": 1}, "gate": {"day_kill": -450.0, "squeeze_enabled": True, "squeeze_lines": 0,
                                                                                "arm_skip_symbols": 150}},
            "orb": {"today": {"fills": 0, "pnl": 0.0}}, "mom": {"paper": _mom(14000.0, 20000.0), "live": _mom(8000.0, 10000.0)}}
    fl = bs.build_flags(data, dt.datetime(2026, 11, 2, 15, 0))
    sev = [(f["severity"], f["book"]) for f in fl]
    assert ("crit", "ops") in sev and ("crit", "mom_paper") in sev and ("crit", "hod") in sev
    assert ("warn", "hod") in sev and ("warn", "mom_live") in sev
    assert [f["severity"] for f in fl] == sorted([f["severity"] for f in fl], key=lambda s: {"crit": 0, "warn": 1, "info": 2}[s])
    assert any(f["severity"] == "info" and "0 fills" in f["text"] for f in fl)
    weekend = bs.build_flags({"ops": {"service_active": False}, "hod": {}, "orb": {}, "mom": {}}, dt.datetime(2026, 11, 1, 12, 0))
    assert weekend == []


def test_build_end_to_end_into_html(db, tmp_path):
    """build() with injected paths yields the contract keys and the renderer accepts them (DB -> JSON -> HTML boundary)."""
    wk = tmp_path / "w.csv"
    wk.write_text("date,equity,cash,n_names,turnover_usd,names_in,names_out,spy_close\n2026-10-26,10000,1,20,1,A B,,700\n2026-11-02,9500,1,20,1,C,A,693\n")
    par = tmp_path / "p.csv"
    par.write_text("date,bt_names_match,bt_ret,paper_ret,live_ret,spy_ret,residual_pp\n2026-10-26,17,0,0,nan,0,1.5\n")
    st = tmp_path / "s.json"
    st.write_text('{"equity_start": 10000}')
    cfg = tmp_path / "c.yaml"
    cfg.write_text("trading: {enabled: false}\nred_to_green: {enabled: false}\nhod_break: {squeeze_enabled: true}\n")
    d = bs.build(DAY, journal=["[HOD] engine gates: dry_run=False risk=$50 kills=-450.0/-1500.0 per_day=12 concurrent=4"], db=db,
                 state_fn=lambda: "active", config_path=cfg, cron_fn=lambda: [], log_dir=tmp_path, ramp_path=tmp_path / "nope.json",
                 now=dt.datetime(2026, 11, 2, 10, 0, tzinfo=bs.ET),
                 mom={"paper_csv": wk, "live_csv": wk, "live_state": st, "parity_csv": par, "log_glob": str(tmp_path / "*.log")})
    for k in ("generated_at_et", "session_date", "ops", "flags"):
        assert k in d
    assert d["hod"]["fills"][0]["symbol"] == "AAA" and d["hod"]["gate"]["squeeze_enabled"] is True
    assert d["mom"]["live"]["weekly"][-1]["week_pct"] == pytest.approx(-5.0) and d["mom"]["live"]["parity"][0]["names_n"] == 17
    h = bh.render(json.loads(json.dumps(d, default=str)))
    assert "AAA" in h and h.startswith("<title>OneMil Books</title>")

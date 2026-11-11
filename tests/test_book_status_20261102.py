"""Tests for scripts/book_status.py: sections take injected paths; journal/systemctl/crontab are stubbed."""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import book_status as bs  # noqa: E402

DAY = "2026-11-02"  # a Monday


@pytest.fixture
def trades_db(tmp_path):
    """tmp trades.db: 3 hod_break rows (1 open) + 2 orb rows (one open)."""
    p = tmp_path / "trades.db"
    con = sqlite3.connect(p)
    con.execute("create table trades (id integer primary key, trade_date text, symbol text, strategy text, "
                "fill_price real, pnl real, exit_reason text)")
    rows = [
        (DAY, "AAA", "hod_break", 10.0, -50.0, "stop_loss"),
        (DAY, "BBB", "hod_break", 20.0, 120.0, "target"),
        (DAY, "CCC", "hod_break", 30.0, None, None),
        ("2026-10-28", "DDD", "hod_break", 5.0, 10.0, "eod"),     # prior week
        (DAY, "EEE", "orb", 7.0, -100.0, "stop_loss"),
        (DAY, "FFF", "orb", 8.0, None, None),
    ]
    con.executemany("insert into trades (trade_date,symbol,strategy,fill_price,pnl,exit_reason) values (?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return p


JOURNAL = [
    "2026-11-02 12:30:08 | INFO     | x | [HOD] engine gates: book=hod_break enabled=True dry_run=False risk=$50 per_day=12 concurrent=4 flat=955",
    "2026-11-02 12:30:13 | INFO     | __main__:928 | Trading mode ACTIVE — LIVE",
    "2026-11-02 14:40:01 | INFO     | x | [HOD] ARM SKIP PRKS: r_pct=0.53% < 1.5%",
    "2026-11-02 14:41:01 | INFO     | x | [HOD] ARM SKIP PRKS: r_pct=0.53% < 1.5%",
    "2026-11-02 14:40:00 | INFO     | x | [HOD] ARM SKIP ATTO: trigger 16.23 < min_price 20.00",
    "2026-11-02 14:50:00 | INFO     | x | [HOD] LIVE cap reached (4/4)",
    "2026-11-02 14:51:00 | WARNING  | x | SQUEEZE detected ZZZ",
    "2026-11-02 14:35:10 | INFO     | x | [ORB] PDR VETO: BILI",
    "2026-11-02 14:35:11 | INFO     | x | [ORB] PDR VETO: HSAI",
    "2026-11-02 14:36:00 | INFO     | x | [ORB] PREPLACE PARITY ok",
    "2026-11-02 15:00:00 | ERROR    | x | boom",
]


@pytest.fixture
def mom_paths(tmp_path):
    """tmp weekly CSV pair, state_live.json, parity CSV and a momentum log."""
    hdr = "date,equity,cash,n_names,turnover_usd,names_in,names_out,spy_close"
    (tmp_path / "paper.csv").write_text(f"{hdr}\n2026-10-26,19000,0,20,1,A B,,1\n2026-11-02,16000,0,20,1,X Y Z,P Q,1\n")
    (tmp_path / "live.csv").write_text(hdr + ",week_r,weeks_since_strong\n2026-11-02,9000,0,20,1,X Y,P,1,-8.2,2\n")
    (tmp_path / "state_live.json").write_text(json.dumps({"equity_start": 10000.0}))
    (tmp_path / "parity.csv").write_text("date,bt_names_match,bt_ret,paper_ret,live_ret,spy_ret,residual_pp\n"
                                         "2026-10-26,17,0,0,nan,0,1.02\n")
    (tmp_path / "momentum_sleeve.log").write_text(f"{DAY} 13:45:03,673 INFO m: rotation due: first session\n"
                                                  f"{DAY} 14:45:58,444 WARNING m: state resynced\n"
                                                  "2026-10-30 09:00:00,000 ERROR m: old\n")
    return {"paper_csv": tmp_path / "paper.csv", "live_csv": tmp_path / "live.csv",
            "live_state": tmp_path / "state_live.json", "parity_csv": tmp_path / "parity.csv",
            "log_glob": str(tmp_path / "momentum_sleeve*.log")}


def test_hod_section_numbers(trades_db):
    h = bs.section_hod(trades_db, DAY, JOURNAL)
    assert h["today"] == {"fills": 3, "pnl": 70.0, "open": 1}
    assert h["wtd"]["fills"] == 3 and h["since_0929"]["fills"] == 4 and h["since_0929"]["pnl"] == 80.0
    assert h["exits"] == {"stops": 1, "targets": 1, "squeeze": 0, "eod": 0}
    assert h["open_symbols"] == ["CCC"]
    assert h["arm_skips"] == {"r_pct": 2, "min_price": 1, "other": 0} and h["skip_symbols"] == 2
    assert h["cap_reached"] == 1 and h["squeeze_lines"] == 1


def test_orb_section_numbers_and_ramp(trades_db, tmp_path):
    ramp = tmp_path / "ramp.json"
    ramp.write_text(json.dumps({"orb": {"frozen": True, "reason": "fill parity"}}))
    o = bs.section_orb(trades_db, DAY, JOURNAL, ramp)
    assert o["today"] == {"fills": 2, "pnl": -100.0} and o["pdr_veto"] == 2 and o["preplace_parity"] == 1
    assert o["ramp"] == {"frozen": True, "reason": "fill parity"}
    assert bs.section_orb(trades_db, DAY, JOURNAL, tmp_path / "none.json")["ramp"] is None


def test_service_section_parses_gates_and_errors():
    s = bs.section_service(JOURNAL, lambda: "active")
    assert s["active"] == "active" and s["boot"] == "2026-11-02 12:30:13" and s["errors_today"] == 1
    assert s["hod_gates"] == {"dry_run": "False", "risk": "$50", "per_day": "12", "concurrent": "4"}


def test_mom_section_rails_and_parity(mom_paths):
    m = bs.section_mom(DAY, **mom_paths)
    assert m["paper"]["equity"] == 16000 and m["paper"]["names_in"] == ["X", "Y", "Z"]
    assert m["paper"]["to_pause"] == pytest.approx(16000 - 15000)
    assert m["paper"]["to_halve"] == pytest.approx(16000 - 12000)
    assert m["live"]["to_pause"] == pytest.approx(9000 - 7500) and m["live"]["weeks_since_strong"] == "2"
    assert m["parity"]["bt_names_match"] == "17"
    assert "state resynced" in m["last_log"][1]          # the latest of today's lines, not the old ERROR


def test_paused_and_crons(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("trading:\n  enabled: false\nred_to_green:\n  enabled: false\n")
    assert bs.section_paused(cfg) == {"bf_enabled": False, "r2g_enabled": False}
    (tmp_path / "eod_report.log").write_text(f"{DAY} 21:58:00 ERROR x\n")
    c = bs.section_crons(DAY, lambda: ["# c", "57 21 * * 1-5 python scripts/eod_report.py", "0 13 * * 6 python scripts/mom_panel_refresh.py"],
                         tmp_path)
    assert c["eod"]["schedules"] == ["57 21 * * 1-5"] and c["eod"]["logs"][0]["errors_today"] == 1
    assert c["refresh"]["logs"] == [] and c["parity"]["schedules"] == []


def test_render_and_json_same_content(trades_db, mom_paths, tmp_path, monkeypatch):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("trading:\n  enabled: false\nred_to_green:\n  enabled: false\n")
    secs = bs.build(DAY, journal=JOURNAL, db=trades_db, state_fn=lambda: "active", mom=mom_paths, config_path=cfg,
                    ramp_path=tmp_path / "none.json", cron_fn=lambda: [], log_dir=tmp_path)
    text = bs.render(secs, DAY)
    assert len(text.splitlines()) <= 25
    assert "HOD paper: 3 fills, closed +$70, open 1 (CCC)" in text
    assert "ORB paper (since 11/12, A2): 2 fills -$100 today" in text and "ramp n/a" in text
    assert "MOM paper: 2026-11-02 eq $16,000" in text and "pause rail +$1,000" in text
    assert "BF trading.enabled=False" in text
    assert json.loads(json.dumps(secs, default=str))["hod"]["today"]["fills"] == 3


def test_unavailable_branches(tmp_path, mom_paths):
    secs = bs.build(DAY, journal=[], db=tmp_path / "missing.db", state_fn=lambda: "inactive",
                    mom={"paper_csv": tmp_path / "x.csv", "live_csv": tmp_path / "y.csv", "live_state": tmp_path / "z.json",
                         "parity_csv": tmp_path / "w.csv", "log_glob": str(tmp_path / "nope*.log")},
                    config_path=tmp_path / "nocfg.yaml", ramp_path=tmp_path / "r.json", cron_fn=lambda: [], log_dir=tmp_path)
    assert "unavailable" in secs["hod"] and "unavailable" in secs["orb"] and "unavailable" in secs["paused"]
    text = bs.render(secs, DAY)
    assert "hod: unavailable (FileNotFoundError" in text and "orb: unavailable" in text and "paused: unavailable" in text
    assert "MOM paper: unavailable" in text and "MOM live: unavailable" in text and "parity: unavailable" in text
    assert "HOD gates: line not found" in text


def test_main_exit_zero_json(monkeypatch, capsys):
    monkeypatch.setattr(bs, "build", lambda day, **k: {"service": {"active": "x"}})
    assert bs.main(["--date", DAY, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["date"] == DAY


@pytest.fixture(autouse=True)
def _legacy_fixtures_predate_orb_day_one(monkeypatch):
    """These fixtures are dated 10/26-11/02 (pre-parity); day one is pinned early so they keep exercising the aggregates.
    The day-one clamp itself is covered by tests/test_orb_day_one_20261111.py."""
    from datetime import date
    monkeypatch.setattr(bs, "ORB_DAY_ONE", date(2026, 1, 1))

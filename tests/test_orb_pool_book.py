"""Nightly add-on pool (P1) book: shared pool reader, pool universe SQL, marker rows, eod P1 lines (spec 2026-10-03)."""
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from trading.orb_pool_defs import find_pool, load_addon_pools, parse_addon_pools, pool_bounds, pool_matches

POOL = {"name": "addon_gap35_range5", "pool_id": "P1", "min_gap_pct": 3.0, "max_gap_pct": 5.0,
        "min_price": 3.0, "max_price": 30.0, "min_prev_volume": 500000}


def test_parse_defaults_off_and_dry():
    assert parse_addon_pools({}) == {"enabled": False, "dry_run": True, "pools": []}


def test_orb_yaml_p1_is_read_through_the_shared_loader():
    defs = load_addon_pools(str(ROOT / "orb.yaml"))
    p1 = find_pool(defs["pools"], "P1")
    assert p1 is not None and p1["min_gap_pct"] == 3.0 and p1["max_gap_pct"] == 5.0


def test_engine_membership_uses_the_shared_helper():
    src = (ROOT / "trading" / "orb_engine.py").read_text()
    assert "from trading.orb_pool_defs import pool_matches" in src
    assert "p_min_gap = float(pool.get" not in src, "a second copy of the pool thresholds is back in the engine"


@pytest.mark.parametrize("gap,expect", [(2.99, False), (3.0, True), (4.99, True), (5.0, True), (5.01, False)])
def test_pool_matches_inclusive_bounds(gap, expect):
    assert pool_matches(POOL, 10.0, gap, 600000, 500000) is expect


def _db(tmp_path):
    path = str(tmp_path / "c.db")
    c = sqlite3.connect(path)
    c.execute("CREATE TABLE daily_bars (symbol TEXT, bar_date TEXT, open REAL, high REAL, low REAL, close REAL, volume INTEGER)")
    rows = []
    for sym, gap in [("LOW", 2.0), ("P1A", 4.0), ("P1B", 3.0), ("EDGE", 5.0), ("PROD", 8.0)]:
        rows.append((sym, "2026-10-01", 10.0, 10, 10, 10.0, 1_000_000))
        rows.append((sym, "2026-10-02", 10.0 * (1 + gap / 100), 11, 10, 10, 1_000_000))
    c.executemany("INSERT INTO daily_bars VALUES (?,?,?,?,?,?,?)", rows)
    c.commit()
    c.close()
    return path


def test_pool_universe_pairs_match_pool_bounds(tmp_path):
    import orb_backtest as ob
    db = _db(tmp_path)
    bounds = pool_bounds(POOL, 500000)
    pool = ob._qualifying_pairs_for_dates(db, [date(2026, 10, 2)], bounds=bounds)["2026-10-02"]
    prod = ob._qualifying_pairs_for_dates(db, [date(2026, 10, 2)])["2026-10-02"]
    assert sorted(pool) == ["EDGE", "P1A", "P1B"]          # gap 3..5 inclusive, the engine's rule
    assert sorted(prod) == ["EDGE", "PROD"]                # production unchanged (gap >= 5)


def test_marker_row_zero_picks_and_upsert(tmp_path):
    import study_orb_pipeline_static_lock as pl
    out = str(tmp_path / "markers.csv")
    feats = pd.DataFrame({"date": pd.to_datetime(["2026-10-02", "2026-10-02"])})
    pl._write_markers(["2026-10-01", "2026-10-02"], "P1", feats, pd.DataFrame(), out)
    m = pd.read_csv(out)
    assert m[m.date == "2026-10-01"].iloc[0].picks == 0 and m[m.date == "2026-10-01"].iloc[0].candidates == 0
    assert m[m.date == "2026-10-02"].iloc[0].candidates == 2
    pl._write_markers(["2026-10-02"], "P1", feats, pd.DataFrame(), out)       # rerun: upsert, no duplicate
    assert len(pd.read_csv(out)) == 2


def test_no_marker_means_not_computed_not_zero(tmp_path):
    import eod_sections as es
    book = tmp_path / "b.csv"
    book.write_text("")
    mk = tmp_path / "m.csv"
    mk.write_text("date,pool_id,candidates,picks\n2026-10-01,P1,5,0\n")
    rows, why = es.p1_bt_book("2026-10-01", book, mk)   # 699fe0b: the note names the marker
    assert rows == [] and "marker: candidates 5, picks 0" in why
    rows, why = es.p1_bt_book("2026-10-02", book, mk)
    assert rows is None and "not computed" in why


def test_parse_and_p1_lines_own_counter(tmp_path):
    import eod_sections as es
    line = ("2026-10-01 13:35:02 | INFO | ORB SCORED: APPS comp=0.4 Q4 | gap=4.865 pool=addon_gap35_range5")
    parsed = es.parse_orb_log(line)
    assert parsed["p1_scored"] == ["APPS"] and parsed["scored"] == []
    bt = {"ranked": ["APPS"], "picks": [], "rows": 1, "pdr": 0, "g1": 0, "range": 0, "dedup": 0, "n": 999}
    st = tmp_path / "p1.json"
    out = es.p1_parity_lines("2026-10-01", [], parsed, [], "", bt, "", state_path=st)
    assert out[0].startswith("P1 ranked: engine 1 vs BT 1 | match 1")
    assert out[1].startswith("P1 picks/fills") and out[2].startswith("P1 P&L")
    assert out[3].startswith("P1 clean sessions 1")
    assert json.loads(st.read_text())["p1"]["2026-10-01"] is True


# ---------------------------------------------------------------- spec 2026-10-06: bars backfill + failed markers
import logging
import subprocess as _sp
import types

FEATURES_ROWS = [("AAA", "2026-10-02"), ("AAA", "2026-10-05"), ("NA", "2026-10-02")]   # NA = a real ticker


def _features_csv(path, rows=FEATURES_ROWS):
    pd.DataFrame({"symbol": [s for s, _ in rows], "date": [d for _, d in rows], "pnl": 0.0}).to_csv(path, index=False)
    return str(path)


class _FakeAlpaca:
    """Records every (symbol, day) bars request and returns two 1-min bars."""
    def __init__(self):
        self.calls = []

    def get_historical_1min_bars(self, sym, start, end):
        self.calls.append((sym, start.strftime("%Y-%m-%d")))
        ts = pd.to_datetime([start.replace(hour=14, minute=30), start.replace(hour=14, minute=31)], utc=True)
        return pd.DataFrame({"timestamp": ts, "open": 5.0, "high": 5.1, "low": 4.9, "close": 5.0, "volume": 1000})


@pytest.fixture
def side_store(tmp_path, monkeypatch):
    """A real bars store (persistence.Database schema) with (AAA, 10-05) cached and the module pointed at it."""
    import orb_backtest as ob
    from persistence.database import Database
    path = str(tmp_path / "bars.db")
    monkeypatch.setattr(ob, "CACHE_DB", path)
    db = Database(db_path=path)
    db.save_intraday_bars("AAA", "2026-10-05", [{"timestamp": pd.Timestamp("2026-10-05 14:30", tz="UTC"),
                                                 "open": 5, "high": 5, "low": 5, "close": 5, "volume": 1}])
    yield ob, db
    db.close()


def test_backfill_fetches_exactly_the_missing_pairs_then_zero(tmp_path, side_store):
    ob, db = side_store
    csv = _features_csv(tmp_path / "orb_features_x.csv")
    al = _FakeAlpaca()
    assert ob.backfill_features_bars(al, db, csv) == 4
    assert sorted(al.calls) == [("AAA", "2026-10-02"), ("NA", "2026-10-02")]   # (AAA, 10-05) was cached
    al2 = _FakeAlpaca()
    assert ob.backfill_features_bars(al2, db, csv) == 0 and al2.calls == []     # second run: nothing to fetch


def test_backfill_without_client_reports_and_fetches_nothing(tmp_path, side_store, capsys):
    ob, _db_ = side_store
    csv = _features_csv(tmp_path / "orb_features_x.csv")
    assert ob.backfill_features_bars(None, None, csv) == 0
    assert "WARNING: features-bars: 2/3 pairs lack bars" in capsys.readouterr().out   # reported, never fetched


def _pool_run(tmp_path, monkeypatch, run_stub):
    """build_pool_books for 10-02 with one P1 pool, stubbed subprocess, side output dir; returns the markers path."""
    import orb_backtest as ob
    import trading.orb_pool_defs as pd_mod
    store = _db(tmp_path)
    c = sqlite3.connect(store)   # the nightly store always has the bars table
    c.execute("CREATE TABLE intraday_bars_1min (symbol TEXT, bar_date TEXT, timestamp TEXT, open REAL, high REAL, "
              "low REAL, close REAL, volume INTEGER)")
    c.commit()
    c.close()
    monkeypatch.setattr(ob, "CACHE_DB", store)
    monkeypatch.setattr(ob, "OUT_DIR", str(tmp_path / "out"))
    markers = str(tmp_path / "out" / "orb_bplus_book_markers.csv")
    monkeypatch.setattr(ob, "PRODUCTION_MARKERS_CSV", markers)
    (tmp_path / "out" / "pool_P1").mkdir(parents=True)
    monkeypatch.setattr(pd_mod, "load_addon_pools",
                        lambda *a, **k: {"enabled": True, "dry_run": True, "pools": [POOL],
                                         "production": {"min_prev_volume": 500000}})
    monkeypatch.setattr(ob.subprocess, "run", run_stub)
    ob.build_pool_books(["2026-10-02"], None, fetch=False)
    return markers


def _completed(rc, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def test_features_failure_writes_a_failed_marker(tmp_path, monkeypatch, caplog):
    def run(cmd, **kw):
        return _completed(1, stderr="Traceback ...\nValueError: features boom")
    with caplog.at_level(logging.ERROR):
        markers = _pool_run(tmp_path, monkeypatch, run)
    m = pd.read_csv(markers, keep_default_na=False)
    row = m[(m.date == "2026-10-02") & (m.pool_id == "P1")].iloc[0]
    assert row.status == "failed" and "features rc=1" in row.note and "features boom" in row.note
    assert any(r.levelno >= logging.ERROR for r in caplog.records)


def test_pipeline_failure_writes_a_failed_marker_with_the_resim_reason(tmp_path, monkeypatch):
    def run(cmd, **kw):
        if "study_orb_features.py" in cmd[1]:
            _features_csv(tmp_path / "out" / "pool_P1" / "orb_features_20261002_1.csv")
            return _completed(0)
        return _completed(1, stderr="ERROR RESIM: n_missing_bars/n_entered=0.6364 > 0.02 -- bars source inadequate")
    markers = _pool_run(tmp_path, monkeypatch, run)
    row = pd.read_csv(markers, keep_default_na=False).iloc[0]
    assert row.status == "failed" and row.pool_id == "P1" and "pipeline rc=1" in row.note and "0.6364" in row.note


def test_pipeline_timeout_writes_a_failed_marker(tmp_path, monkeypatch):
    def run(cmd, **kw):
        if "study_orb_features.py" in cmd[1]:
            _features_csv(tmp_path / "out" / "pool_P1" / "orb_features_20261002_1.csv")
            return _completed(0)
        raise _sp.TimeoutExpired(cmd, 2400)
    markers = _pool_run(tmp_path, monkeypatch, run)
    row = pd.read_csv(markers, keep_default_na=False).iloc[0]
    assert row.status == "failed" and "timeout" in row.note


def test_success_leaves_no_failed_marker(tmp_path, monkeypatch):
    def run(cmd, **kw):
        if "study_orb_features.py" in cmd[1]:
            _features_csv(tmp_path / "out" / "pool_P1" / "orb_features_20261002_1.csv")
        return _completed(0, stdout="ok")
    markers = _pool_run(tmp_path, monkeypatch, run)
    assert not Path(markers).exists()   # the (stubbed) pipeline writes ok markers itself; the runner adds none


def test_write_markers_keeps_status_and_reads_old_files_as_ok(tmp_path):
    import study_orb_pipeline_static_lock as pl
    from trading.orb_markers import write_failed_markers
    out = str(tmp_path / "markers.csv")
    Path(out).write_text("date,pool_id,candidates,picks\n2026-10-01,P1,10,0\n2026-10-02,production,36,0\n")
    write_failed_markers(out, ["2026-10-05"], "P1", "pipeline rc=1: boom, with comma\nand newline")
    m = pd.read_csv(out, keep_default_na=False)
    assert list(m.columns) == ["date", "pool_id", "candidates", "picks", "status", "note"]
    assert m[m.date == "2026-10-01"].iloc[0].status == "ok"                      # old row defaults to ok
    assert m[m.date == "2026-10-05"].iloc[0].status == "failed" and "\n" not in m.iloc[-1].note
    feats = pd.DataFrame({"date": pd.to_datetime(["2026-10-05"])})
    pl._write_markers(["2026-10-05"], "P1", feats, pd.DataFrame(), out)           # a later good run replaces it
    m = pd.read_csv(out, keep_default_na=False)
    row = m[(m.date == "2026-10-05") & (m.pool_id == "P1")].iloc[0]
    assert row.status == "ok" and row.note == "" and len(m) == 3


def test_eod_prints_failed_not_nodata_for_a_failed_p1_night(tmp_path, caplog):
    import eod_sections as es
    book = tmp_path / "P1.csv"
    book.write_text("pool_id\n")
    mk = tmp_path / "m.csv"
    mk.write_text("date,pool_id,candidates,picks,status,note\n2026-10-05,P1,0,0,failed,pipeline rc=1: RESIM 0.6364 > 0.02\n")
    with caplog.at_level(logging.ERROR):
        rows, why = es.p1_bt_book("2026-10-05", book, mk)
    assert rows is None and why == "FAILED: pipeline rc=1: RESIM 0.6364 > 0.02"
    assert any(r.levelno == logging.ERROR for r in caplog.records)
    parsed = es.parse_orb_log("")
    bt = {"ranked": [], "picks": [], "rows": 0, "pdr": 0, "g1": 0, "range": 0, "dedup": 0, "n": 999}
    st = tmp_path / "p1.json"
    text = "\n".join(es.p1_parity_lines("2026-10-05", [], parsed, rows, why, bt, "", state_path=st))
    assert "P1 BT: FAILED (pipeline rc=1: RESIM 0.6364 > 0.02)" in text
    assert "NO-DATA" not in text
    assert not st.exists() or "2026-10-05" not in json.loads(st.read_text()).get("p1", {})   # neutral, not decided


def test_old_marker_format_still_reads_ok(tmp_path):
    import eod_sections as es
    mk = tmp_path / "m.csv"
    mk.write_text("date,pool_id,candidates,picks\n2026-10-01,P1,5,0\n")
    assert es.bt_marker("2026-10-01", "P1", mk) == {"candidates": 5, "picks": 0, "status": "ok", "note": ""}

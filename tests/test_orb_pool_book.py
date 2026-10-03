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
    assert es.p1_bt_book("2026-10-01", book, mk) == ([], "")
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

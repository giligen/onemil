"""Unit + parity tests for trading/gap_short_rules.py (parity vs research/signal_search/1776R_events.parquet picks)."""
import os
import random
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from trading import gap_short_rules as R

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "research/signal_search/1776R_events.parquet"
POP = ROOT / "research/hod_entry/cell_1428_population.csv"
PM_DIR = ROOT / "research/hod_entry/1762_cache/premarket"
CACHE_DB = ROOT / "data/cache.db"


def _bar(h, m, c, v):
    """1-min bar starting h:m ET on 2026-03-10."""
    return {"t": datetime(2026, 3, 10, h, m, tzinfo=R.ET), "c": c, "v": v}


def test_gap_fraction():
    assert R.gap_fraction(11, 10) == pytest.approx(0.1)
    assert R.gap_fraction(None, 10) is None and R.gap_fraction(5, 0) is None


def test_premarket_dollar_volume_window():
    bars = [_bar(3, 59, 1, 100), _bar(4, 0, 2, 10), _bar(9, 18, 3, 10), _bar(9, 19, 4, 1000), _bar(9, 24, 5, 10)]
    assert R.premarket_dollar_volume(bars) == pytest.approx(2 * 10 + 3 * 10)           # live window ends 09:19
    assert R.premarket_dollar_volume(bars, R.BT_PM_END_HM) == pytest.approx(20 + 30 + 4000 + 50)   # BT window 09:25


def test_price_band():
    assert R.price_ok(3.0) and R.price_ok(30.0) and not R.price_ok(2.99) and not R.price_ok(30.01) and not R.price_ok(None)


def test_is_todays_premarket_trade():
    assert R.is_todays_premarket_trade(datetime(2026, 3, 10, 5, 0, tzinfo=R.ET), "2026-03-10")
    assert not R.is_todays_premarket_trade(datetime(2026, 3, 10, 3, 59, tzinfo=R.ET), "2026-03-10")
    assert not R.is_todays_premarket_trade(datetime(2026, 3, 9, 15, 0, tzinfo=R.ET), "2026-03-10")
    assert not R.is_todays_premarket_trade(None, "2026-03-10")


def test_select_picks_order_is_rank_then_l1():
    cands = [{"symbol": f"S{i}", "gap": 1.0 - i * 0.01, "l1": 1e6 if i == 0 else 9e6} for i in range(12)]
    got = [p["symbol"] for p in R.select_picks(cands)]
    assert "S0" not in got and got[0] == "S1" and len(got) == 9      # S0 in top 10 but L1 low; S10/S11 not refilled


def test_select_picks_bounds_and_exclusions():
    cands = [{"symbol": "ZVZZT", "gap": .5, "l1": 9e6}, {"symbol": "A", "gap": 2.5, "l1": 9e6},
             {"symbol": "B", "gap": .03, "l1": 9e6}, {"symbol": "C", "gap": .2, "l1": float("nan")},
             {"symbol": "D", "gap": .3, "l1": 9e6}]
    assert [p["symbol"] for p in R.select_picks(cands)] == ["D"]
    assert [p["symbol"] for p in R.select_picks(cands, min_gap=None)] == ["D", "B"]


def test_sizing_and_cap_and_prices():
    assert R.position_qty(7.0) == 42 and R.position_qty(400) == 0 and R.position_qty(0) == 0
    picks = [{"qty": 100, "pm_last": 10.0}] * 4
    assert len(R.apply_gross_cap(picks, cap=3000)) == 3
    assert R.cover_limit_price(10.0) == 10.01
    assert R.backstop_hit(5, 10) and not R.backstop_hit(5, 9.9)
    assert R.net_bp(10, 10) == pytest.approx(-(R.FEES_BP + R.LOCATE_BP + R.BORROW_BP))


def test_rails_and_ids():
    assert R.day_killed(-300) and not R.day_killed(-299)
    assert R.week_paused(-300.01) and not R.week_paused(0)
    assert R.two_red_weeks_below_p10([0.1, -0.7, -0.8]) and not R.two_red_weeks_below_p10([-0.7, 0.2])
    assert R.client_order_id("2026-03-10", "ABC") == "gapshort-2026-03-10-ABC"
    assert R.client_order_id("2026-03-10", "ABC", "cov") == "gapshort-cov-2026-03-10-ABC"


@pytest.mark.skipif(not (EVENTS.exists() and POP.exists() and PM_DIR.exists() and CACHE_DB.exists()), reason="research inputs absent")
def test_parity_with_1776R_picks_on_10_sampled_2026_days():
    """Same (day, symbol) set as the BT book from the same inputs (population + daily_bars gap + 04:00-09:25 bar L1)."""
    ev = pd.read_parquet(EVENTS)
    ev["day"] = ev.day.astype(str).str[:10]
    cutoff = float(R.L1_MIN_USD)
    assert cutoff > 7.2e6       # frozen live cutoff = the BT cutoff (rounded down to 7,224,000)
    days = sorted(d for d in ev.day.unique() if d >= "2026-01-01")
    sample = random.Random(1776).sample(days, 10)
    pop = pd.read_csv(POP)
    db = sqlite3.connect(f"file:{CACHE_DB}?mode=ro", uri=True)
    d = pd.read_sql("select symbol,bar_date as day,open,close from daily_bars where bar_date>='2025-06-01'", db)
    d = d.sort_values(["symbol", "day"])
    d["pclose"] = d.groupby("symbol").close.shift(1)
    for day in sample:
        rows = pop[pop.day == day].merge(d[d.day == day][["symbol", "open", "pclose"]], on="symbol", how="left")
        pm = pd.read_parquet(PM_DIR / f"{day}.parquet")
        cands = []
        for r in rows.itertuples():
            g = R.gap_fraction(r.open, r.pclose)
            if g is None:
                continue
            b = pm[pm.symbol == r.symbol]
            bars = [{"t": pd.Timestamp(t).to_pydatetime(), "c": c, "v": v} for t, c, v in zip(pd.to_datetime(b.t, utc=True), b.c, b.v)]
            cands.append({"symbol": r.symbol, "gap": g, "l1": R.premarket_dollar_volume(bars, R.BT_PM_END_HM) if bars else None})
        got = {p["symbol"] for p in R.select_picks(cands, l1_cutoff=cutoff)}
        want = set(ev[ev.day == day].symbol)
        assert got == want, (day, got ^ want)


def test_frozen_constants():
    """No silent tuning: every strategy constant is pinned here."""
    assert (R.L1_MIN_USD, R.TOP_K, R.NOTIONAL_PER_NAME, R.GAP_MIN, R.GAP_MAX) == (
        7_224_000, 10, 300, 0.05, 2.0)
    assert (R.PRICE_LO, R.PRICE_HI, R.PRIOR_DAY_VOL_MIN) == (3.0, 30.0, 500_000)
    assert R.GROSS_CAP == 3000.0 and R.COVER_ASK_BP == 10.0 and R.PM_END_HM == (9, 19) and R.OPG_LAST_SUBMIT_HM == (9, 27, 30)


def test_previous_session_skips_weekends_and_nyse_holidays():
    assert R.previous_session("2026-11-16") == "2026-11-13"       # Monday -> Friday
    assert R.previous_session("2026-11-27") == "2026-11-25"       # day after Thanksgiving -> Wednesday
    assert R.previous_session("2026-11-24") == "2026-11-23"


def test_week_start_is_the_iso_monday():
    assert R.week_start("2026-11-18") == "2026-11-16" and R.week_start("2026-11-22") == "2026-11-16"

#!/usr/bin/env python3
"""Audit the touchgo BT<->LIVE breakout-bar parity gap.

BT (study_orb_pipeline_static_lock.py) evaluates touchgo Rule M on the first
1-min bar whose high > range_high (the MARKET breakout bar, fill assumed
instant). LIVE (trading/orb_engine.py:1726) evaluates it on
minute-floor(actual fill time). For slow stop-limit fills these are different
bars -> live touchgo can fire/not-fire opposite to what BT validated.

This script quantifies, over real live fills:
  - how many fills land in a LATER minute than the market breakout bar
  - the distribution of that lag (minutes)
  - how often the Rule M decision (bb_close_pos < 0.5) actually FLIPS between
    the BT bar and the live fill bar
  - cross-checks against the recorded exit_reason

Read-only. Run on a node with fresh data/cache.db + data/trades.db:
    python3 scripts/audit_touchgo_breakout_bar_gap.py
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
RULE_M_THRESH = 0.5

tc = sqlite3.connect("data/trades.db")
tc.row_factory = sqlite3.Row
cc = sqlite3.connect("data/cache.db")
cc.row_factory = sqlite3.Row


def parse_dt(s: str) -> datetime:
    """Parse an ISO timestamp (with or without tz) to UTC-aware."""
    s = s.replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def bb_close_pos(bar) -> float:
    """(close - low) / (high - low); 0.0 on degenerate bar."""
    rng = bar["high"] - bar["low"]
    if rng <= 0:
        return 0.0
    return (bar["close"] - bar["low"]) / rng


def range_end_utc(trade_date: str) -> datetime:
    """9:35 ET (end of the 5-min opening range) in UTC, DST-correct."""
    d = datetime.strptime(trade_date, "%Y-%m-%d")
    et = datetime(d.year, d.month, d.day, 9, 35, tzinfo=ET)
    return et.astimezone(UTC)


rows = tc.execute(
    """
    SELECT trade_date, symbol, filled_at, fill_price, exit_reason,
           submit_to_fill_ms, pattern_data, pnl_pct, order_submitted_at
    FROM trades
    WHERE strategy='orb' AND fill_price IS NOT NULL AND filled_at IS NOT NULL
    ORDER BY trade_date, symbol
    """
).fetchall()

total = 0
no_breakout = 0
aligned = 0          # fill bar == BT breakout bar
lagged = 0           # fill bar is a later minute
lag_hist: dict[int, int] = {}
flips = []           # Rule M decision differs between BT bar and live fill bar
detail_rows = []

for r in rows:
    pdata = json.loads(r["pattern_data"] or "{}")
    rh = pdata.get("range_high")
    if rh is None:
        continue
    total += 1

    re_utc = range_end_utc(r["trade_date"])
    bars = cc.execute(
        """
        SELECT timestamp, open, high, low, close, volume
        FROM intraday_bars_1min
        WHERE symbol = ? AND bar_date = ? AND timestamp >= ?
        ORDER BY timestamp
        """,
        (r["symbol"], r["trade_date"], re_utc.isoformat()),
    ).fetchall()
    if not bars:
        no_breakout += 1
        continue

    # BT breakout bar = first bar after range_end with high > range_high
    bt_bar = None
    for b in bars:
        if b["high"] > rh:
            bt_bar = b
            break
    if bt_bar is None:
        no_breakout += 1
        continue
    bt_bar_ts = parse_dt(bt_bar["timestamp"])

    # LIVE bar = bar at minute-floor(fill time)
    fill_dt = parse_dt(r["filled_at"])
    fill_min = fill_dt.replace(second=0, microsecond=0)
    live_bar = None
    for b in bars:
        if parse_dt(b["timestamp"]) == fill_min:
            live_bar = b
            break

    lag = int((fill_min - bt_bar_ts).total_seconds() // 60)
    if lag <= 0:
        aligned += 1
    else:
        lagged += 1
    lag_hist[lag] = lag_hist.get(lag, 0) + 1

    bt_pos = bb_close_pos(bt_bar)
    bt_fires = bt_pos < RULE_M_THRESH
    if live_bar is not None:
        live_pos = bb_close_pos(live_bar)
        live_fires = live_pos < RULE_M_THRESH
    else:
        live_pos = None
        live_fires = None

    flipped = (live_fires is not None) and (bt_fires != live_fires)
    if flipped:
        flips.append((r["trade_date"], r["symbol"]))

    detail_rows.append(
        (
            r["trade_date"], r["symbol"], lag,
            f"{bt_pos:.3f}", bt_fires,
            (f"{live_pos:.3f}" if live_pos is not None else "n/a"), live_fires,
            "FLIP" if flipped else "",
            r["exit_reason"] or "",
            round((r["submit_to_fill_ms"] or 0) / 1000.0, 1),
        )
    )

print(f"Live fills audited:            {total}")
print(f"  no breakout bar found:      {no_breakout}")
print(f"  fill bar == BT breakout bar:{aligned}")
print(f"  fill bar LAGS breakout bar: {lagged}  ({(lagged/max(total-no_breakout,1))*100:.0f}% of resolvable)")
print(f"Lag distribution (minutes -> #fills): {dict(sorted(lag_hist.items()))}")
print(f"Rule M decision FLIPS (BT bar vs live fill bar): {len(flips)}")
if flips:
    print("  flipped:", ", ".join(f"{d} {s}" for d, s in flips))
print()
hdr = ("date", "sym", "lagMin", "bt_pos", "bt_M?", "live_pos", "live_M?", "flag", "exit_reason", "fill_s")
print("{:<11} {:<6} {:>6} {:>7} {:>6} {:>8} {:>7} {:>5} {:<22} {:>7}".format(*hdr))
for row in detail_rows:
    print("{:<11} {:<6} {:>6} {:>7} {:>6} {:>8} {:>7} {:>5} {:<22} {:>7}".format(*[str(x) for x in row]))

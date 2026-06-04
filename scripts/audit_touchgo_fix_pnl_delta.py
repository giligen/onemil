#!/usr/bin/env python3
"""Counterfactual P&L of re-keying live touchgo to the market breakout bar.

For each real live ORB fill, compare the CURRENT live exit (recorded) against
the FIXED policy (touchgo Rule M/D evaluated on the market breakout bar = first
1-min bar with high > range_high, exactly as the BT validates). Only flipped
trades differ; aligned trades are zero-delta by construction.

Exit model replicates study_orb_pipeline_static_lock.simulate_static_lock:
  1. Rule M on breakout bar -> tag_bb at bar close * (1-10bps)
  2. Rule D on breakout+1   -> tag_b1 at (entry-0.5R) * (1-10bps)
  3. static lock: arm at entry+1.75R (stop ratchets to entry+0.5R), stop=range_low
  4. EOD force-close at 15:45 ET (last bar close * (1-10bps))

entry_price held = ACTUAL live fill_price (the fix changes only the exit
decision, not the fill). delta_$ uses ACTUAL live shares. Exit slippage modeled
at the BT 10bps assumption (true counterfactual fills are unknowable).

Read-only. Run on a node with fresh data/cache.db + data/trades.db.
"""
from __future__ import annotations
import json, sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York"); UTC = ZoneInfo("UTC")
RULE_M_THRESH = 0.5; RULE_D_REVERT_R = 0.75; RULE_D_EXIT_R = -0.5
LOCK_TRIGGER_R = 1.75; LOCK_STOP_R = 0.5; EXIT_SLIP = 10 / 10000.0

tc = sqlite3.connect("data/trades.db"); tc.row_factory = sqlite3.Row
cc = sqlite3.connect("data/cache.db"); cc.row_factory = sqlite3.Row


def pdt(s):
    s = s.replace("Z", "+00:00"); d = datetime.fromisoformat(s)
    return (d.replace(tzinfo=UTC) if d.tzinfo is None else d).astimezone(UTC)


def et_utc(trade_date, h, m):
    d = datetime.strptime(trade_date, "%Y-%m-%d")
    return datetime(d.year, d.month, d.day, h, m, tzinfo=ET).astimezone(UTC)


def pos_of(b):
    rng = b["high"] - b["low"]
    return 0.0 if rng <= 0 else (b["close"] - b["low"]) / rng


MAX_BREAKOUT_AGE_MIN = 15.0  # shipped late-fill guard (orb_touchgo_filter)


def simulate_fixed(bars, entry, rh, rl, breakout_ts, force_close_ts, fill_min=None):
    """Return (exit_price, reason) under the breakout-bar touchgo policy.

    Mirrors shipped logic: touchgo Rule M/D evaluate the market breakout bar,
    BUT are skipped (late-fill guard) when the fill lagged the breakout bar by
    more than MAX_BREAKOUT_AGE_MIN — a stale entry holds via static-lock only.
    """
    rs = rh - rl
    trigger = entry + LOCK_TRIGGER_R * rs
    lock_stop = entry + LOCK_STOP_R * rs
    stop = rl
    armed = False
    post = [b for b in bars if breakout_ts <= pdt(b["timestamp"]) <= force_close_ts]
    if not post:
        return None, "no_bars"
    stale = (fill_min is not None
             and (fill_min - breakout_ts).total_seconds() / 60.0 > MAX_BREAKOUT_AGE_MIN)
    # Rule M on breakout bar (skipped if late-fill guard trips)
    eb = post[0]
    if not stale and pos_of(eb) < RULE_M_THRESH:
        return eb["close"] * (1 - EXIT_SLIP), "tag_bb"
    # Rule D on breakout+1 (skipped if late-fill guard trips)
    if not stale and len(post) >= 2:
        if (entry - post[1]["low"]) / rs >= RULE_D_REVERT_R:
            return (entry + RULE_D_EXIT_R * rs) * (1 - EXIT_SLIP), "tag_b1"
    # static lock loop
    for b in post[1:]:
        if not armed and b["high"] >= trigger:
            armed = True; stop = max(stop, lock_stop)
        if b["low"] <= stop:
            return stop * (1 - EXIT_SLIP), ("lock" if armed else "stop")
    return post[-1]["close"] * (1 - EXIT_SLIP), "eod"


rows = tc.execute("""
    SELECT trade_date, symbol, filled_at, fill_price, exit_price, exit_reason,
           pnl, pnl_pct, shares, pattern_data
    FROM trades WHERE strategy='orb' AND fill_price IS NOT NULL AND filled_at IS NOT NULL
    ORDER BY trade_date, symbol""").fetchall()

flips = []
for r in rows:
    pdata = json.loads(r["pattern_data"] or "{}")
    rh = pdata.get("range_high"); rl = pdata.get("range_low")
    if rh is None or rl is None:
        continue
    re_utc = et_utc(r["trade_date"], 9, 35)
    fc_utc = et_utc(r["trade_date"], 15, 45)
    bars = cc.execute("""SELECT timestamp,open,high,low,close,volume FROM intraday_bars_1min
        WHERE symbol=? AND bar_date=? AND timestamp>=? ORDER BY timestamp""",
        (r["symbol"], r["trade_date"], re_utc.isoformat())).fetchall()
    if not bars:
        continue
    bt_bar = next((b for b in bars if b["high"] > rh), None)
    if bt_bar is None:
        continue
    bt_ts = pdt(bt_bar["timestamp"])
    fill_min = pdt(r["filled_at"]).replace(second=0, microsecond=0)
    live_bar = next((b for b in bars if pdt(b["timestamp"]) == fill_min), None)
    bt_fire = pos_of(bt_bar) < RULE_M_THRESH
    live_fire = (pos_of(live_bar) < RULE_M_THRESH) if live_bar else None
    if live_fire is None or bt_fire == live_fire:
        continue  # aligned (or unresolvable) -> zero delta

    entry = r["fill_price"]; shares = r["shares"] or 0
    fx_exit, fx_reason = simulate_fixed(bars, entry, rh, rl, bt_ts, fc_utc, fill_min=fill_min)
    if fx_exit is None:
        continue
    fixed_pct = (fx_exit - entry) / entry * 100
    fixed_pnl = (fx_exit - entry) * shares
    delta_pnl = fixed_pnl - (r["pnl"] or 0)
    flips.append({
        "date": r["trade_date"], "sym": r["symbol"], "shares": shares,
        "actual_reason": r["exit_reason"], "actual_pct": r["pnl_pct"], "actual_pnl": r["pnl"],
        "fixed_reason": fx_reason, "fixed_pct": fixed_pct, "fixed_pnl": fixed_pnl,
        "delta_pnl": delta_pnl, "delta_pct": fixed_pct - (r["pnl_pct"] or 0),
    })

print(f"Flipped trades: {len(flips)}\n")
h = ("date","sym","shr","actual_exit","act_%","act_$","fixed_exit","fix_%","fix_$","Δ$ (prod)")
print("{:<11}{:<6}{:>6}  {:<13}{:>7}{:>9}  {:<11}{:>7}{:>9}{:>11}".format(*h))
tot_actual = tot_fixed = tot_delta = 0.0
for f in flips:
    print("{:<11}{:<6}{:>6}  {:<13}{:>7.2f}{:>9.1f}  {:<11}{:>7.2f}{:>9.1f}{:>11.1f}".format(
        f["date"], f["sym"], f["shares"], f["actual_reason"], f["actual_pct"] or 0,
        f["actual_pnl"] or 0, f["fixed_reason"], f["fixed_pct"], f["fixed_pnl"], f["delta_pnl"]))
    tot_actual += f["actual_pnl"] or 0; tot_fixed += f["fixed_pnl"]; tot_delta += f["delta_pnl"]
print("-"*92)
print(f"Flipped actual P&L (prod $): {tot_actual:+.1f}")
print(f"Flipped fixed  P&L (prod $): {tot_fixed:+.1f}")
print(f"NET P&L CHANGE FROM FIX (prod $, {len(flips)} trades over live sample): {tot_delta:+.1f}")
print(f"Avg Δ per flipped trade (%): {sum(f['delta_pct'] for f in flips)/max(len(flips),1):+.2f}")

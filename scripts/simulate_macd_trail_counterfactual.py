#!/usr/bin/env python3
"""Simulate what MACD wave trades would have realized with 0.3% trail working.

Background
==========
The MACD wave 0.3% trail was broken in WS mode from inception until commit
9e160af (2026-04-28 evening). During that window the trail never ratcheted
in production — every trade exited via macd_flip, hard 2% stop, or
bracket_exit, but never via trail_stop.

For trades that made a meaningful HIGHER HIGH after entry, a working
0.3% trail would have ratcheted the stop above entry and locked in the
move when price retraced 0.3% from peak.

This script simulates the counterfactual exit per trade by:
1. Loading 1-min bars for each (symbol, trade_date) during the holding
   window from cache.db.
2. Walking forward from entry bar, tracking high_since_entry (HSE) and
   stop = max(prior_stop, HSE * (1 - 0.003)).
3. Trail fires when bar.low <= stop on a subsequent bar; estimate fill
   at min(stop, bar.open).
4. Trail competes with the ACTUAL exit (whichever comes first chrono-
   logically wins). If actual exit fires before trail level reached,
   trail counterfactual = actual.
5. Compares actual P&L to simulated trail-or-actual P&L; reports lift.

Caveats
=======
- The hard 2% stop and macd_flip signal would still fire under the fix.
  We only simulate the TRAIL becoming active and competing with them.
- Slippage modeled at 0bps for the trail fill — real fills would have
  some bid-side slippage. Estimates may be slightly optimistic.
- We use 1-min bars; actual ticks could trigger trail at different
  prices within a bar. Approximation, not exact replay.

Usage
=====
    python scripts/simulate_macd_trail_counterfactual.py 2026-04-20 2026-04-29
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DB = ROOT / 'data' / 'cache.db'
TRADES_DB = ROOT / 'data' / 'trades.db'

DEFAULT_TRAIL_PCT = 0.003  # MACD wave 0.3% trail per macd_wave.yaml
HARD_STOP_PCT = 0.02  # for reference; hard stop already fires correctly under bug


def load_bars(symbol: str, bar_date: str) -> list:
    """Return chronological list of 1-min bars for (symbol, date)."""
    con = sqlite3.connect(CACHE_DB)
    rows = con.execute(
        "SELECT timestamp, open, high, low, close, volume "
        "FROM intraday_bars_1min "
        "WHERE symbol = ? AND bar_date = ? ORDER BY timestamp",
        (symbol, bar_date),
    ).fetchall()
    con.close()
    return [
        {
            'ts': datetime.fromisoformat(r[0].replace('Z', '+00:00'))
            if isinstance(r[0], str) else r[0],
            'open': float(r[1]), 'high': float(r[2]),
            'low': float(r[3]), 'close': float(r[4]),
            'volume': int(r[5]),
        }
        for r in rows
    ]


def simulate_trail(bars, entry_ts, entry_price, exit_ts, actual_exit_price,
                   shares, trail_pct=DEFAULT_TRAIL_PCT):
    """Walk bars from entry to actual exit; return (trail_exit_ts, trail_fill).

    If trail level is hit before actual exit, return that. Otherwise return
    the actual exit (trail didn't fire first).
    """
    if not bars:
        return None, None
    # Filter bars to entry_ts <= bar.ts <= exit_ts
    in_window = [b for b in bars if entry_ts <= b['ts'] <= exit_ts]
    if not in_window:
        return None, None

    high_since_entry = entry_price
    stop = entry_price * (1 - HARD_STOP_PCT)  # initial hard stop
    # %-based trail activates immediately (matches add_watch semantics)
    trailing_active = True

    for i, bar in enumerate(in_window):
        # Skip the entry bar itself for trail check (assume mid-bar fill)
        if i == 0 and bar['ts'] == entry_ts:
            high_since_entry = max(high_since_entry, bar['high'])
            new_stop = high_since_entry * (1 - trail_pct)
            stop = max(stop, new_stop)
            continue

        # Update high first, then check stop against bar's low
        if bar['high'] > high_since_entry:
            high_since_entry = bar['high']
            new_stop = high_since_entry * (1 - trail_pct)
            if new_stop > stop:
                stop = new_stop

        # Trail check on this bar's low
        if trailing_active and bar['low'] <= stop:
            # Trail fires. Fill estimate: min(stop, bar.open).
            # bar.open captures market sentiment going into the bar; if
            # already at/below stop, fill is at open. Otherwise fill is
            # at stop (limit-style).
            fill = min(stop, bar['open'])
            return bar['ts'], fill

    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('start', help='YYYY-MM-DD')
    ap.add_argument('end', help='YYYY-MM-DD')
    ap.add_argument('--trail-pct', type=float, default=DEFAULT_TRAIL_PCT,
                    help='Trail percentage (default 0.003 = 0.3%%, matches macd_wave.yaml)')
    ap.add_argument('--sweep', action='store_true',
                    help='Run sweep over multiple trail_pct values; prints summary only')
    args = ap.parse_args()

    if args.sweep:
        run_sweep(args.start, args.end)
        return
    run_single(args.start, args.end, args.trail_pct)


def run_sweep(start: str, end: str):
    """Run sim for multiple trail_pct values; print one-line summary per."""
    sweep_values = [0.003, 0.005, 0.0075, 0.010, 0.015, 0.020, 0.030]
    print(f"=== MACD wave trail-pct sweep, {start} → {end} ===\n")
    print(f"{'trail_pct':>10} {'total P&L':>14} {'lift vs actual':>16} "
          f"{'fired':>7} {'no_fire':>8}")
    print("-" * 65)
    actual_total, _, _ = _compute_trades(start, end, trail_pct=0.0,
                                          dry_print=False)
    print(f"{'ACTUAL':>10} {actual_total:>+14,.2f} {'(baseline)':>16}")
    for tp in sweep_values:
        cf_total, fired, no_fire = _compute_trades(
            start, end, trail_pct=tp, dry_print=False
        )
        lift = cf_total - actual_total
        print(f"{tp:>10.4f} {cf_total:>+14,.2f} {lift:>+16,.2f} "
              f"{fired:>7d} {no_fire:>8d}")


def _compute_trades(start: str, end: str, trail_pct: float, dry_print: bool):
    """Inner runner. Returns (counterfactual_total, n_fired, n_no_fire).

    If trail_pct == 0, returns (actual_total, 0, n_filled) — the baseline.
    """
    con = sqlite3.connect(TRADES_DB)
    rows = con.execute(
        """
        SELECT id, trade_date, symbol, fill_price, shares, exit_price,
               exit_reason, pnl, filled_at, exited_at
        FROM trades
        WHERE strategy = 'macd_wave'
          AND trade_date BETWEEN ? AND ?
          AND fill_price IS NOT NULL
          AND exit_price IS NOT NULL
        ORDER BY trade_date, id
        """,
        (start, end),
    ).fetchall()
    con.close()

    cf_total = 0.0
    n_fired = 0
    n_no_fire = 0
    for (tid, tdate, sym, fill, sh, exit_p, reason, pnl, fa, ea) in rows:
        if trail_pct <= 0:
            cf_total += pnl
            n_no_fire += 1
            continue
        try:
            entry_ts = datetime.fromisoformat(str(fa).replace('Z', '+00:00')) \
                if fa else None
            exit_ts = datetime.fromisoformat(str(ea).replace('Z', '+00:00')) \
                if ea else None
        except Exception:
            entry_ts = exit_ts = None
        if entry_ts is None or exit_ts is None:
            cf_total += pnl
            n_no_fire += 1
            continue
        bars = load_bars(sym, tdate)
        trail_ts, trail_fill = simulate_trail(
            bars, entry_ts, fill, exit_ts, exit_p, sh, trail_pct=trail_pct,
        )
        if trail_ts is None:
            cf_total += pnl
            n_no_fire += 1
        else:
            cf_pnl = (trail_fill - fill) * sh
            cf_total += cf_pnl
            n_fired += 1
    return cf_total, n_fired, n_no_fire


def run_single(start: str, end: str, trail_pct: float):
    """Original detailed per-trade output, parameterized by trail_pct."""

    con = sqlite3.connect(TRADES_DB)
    rows = con.execute(
        """
        SELECT id, trade_date, symbol, fill_price, shares, exit_price,
               exit_reason, pnl, filled_at, exited_at
        FROM trades
        WHERE strategy = 'macd_wave'
          AND trade_date BETWEEN ? AND ?
          AND fill_price IS NOT NULL
          AND exit_price IS NOT NULL
        ORDER BY trade_date, id
        """,
        (start, end),
    ).fetchall()
    con.close()

    print(f"=== MACD wave counterfactual: trail={trail_pct*100:.2f}% FIXED, {start} → {end} ===\n")
    print(f"{'date':<12} {'sym':<6} {'fill':>8} {'actual_exit':>12} "
          f"{'actual_pnl':>11} {'trail_exit':>11} {'trail_pnl':>11} "
          f"{'lift':>10} {'reason':<20}")
    print("-" * 115)

    actual_total = 0.0
    counter_total = 0.0
    for (tid, tdate, sym, fill, sh, exit_p, reason, pnl, fa, ea) in rows:
        try:
            entry_ts = datetime.fromisoformat(str(fa).replace('Z', '+00:00')) \
                if fa else None
            exit_ts = datetime.fromisoformat(str(ea).replace('Z', '+00:00')) \
                if ea else None
        except Exception:
            entry_ts = exit_ts = None

        if entry_ts is None or exit_ts is None:
            print(f"{tdate:<12} {sym:<6} {fill:>8.2f} (skipped — missing timestamps)")
            actual_total += pnl
            counter_total += pnl
            continue

        bars = load_bars(sym, tdate)
        trail_ts, trail_fill = simulate_trail(
            bars, entry_ts, fill, exit_ts, exit_p, sh, trail_pct=trail_pct,
        )

        if trail_ts is None:
            cf_exit = exit_p
            cf_pnl = pnl
            note = "trail no-fire"
        else:
            cf_exit = trail_fill
            cf_pnl = (cf_exit - fill) * sh
            note = "trail fired"

        lift = cf_pnl - pnl
        actual_total += pnl
        counter_total += cf_pnl
        print(f"{tdate:<12} {sym:<6} {fill:>8.2f} {exit_p:>12.2f} "
              f"{pnl:>+11,.2f} {cf_exit:>11.2f} {cf_pnl:>+11,.2f} "
              f"{lift:>+10,.2f}  {reason}/{note}")

    print("-" * 115)
    print(f"{'TOTAL ACTUAL':<46} {actual_total:>+11,.2f}")
    print(f"{'TOTAL COUNTERFACTUAL (trail fixed)':<46} "
          f"{' ':>23} {counter_total:>+11,.2f}")
    print(f"{'LIFT FROM TRAIL FIX':<46} "
          f"{' ':>23} {' ':>11} {counter_total - actual_total:>+10,.2f}")


if __name__ == '__main__':
    main()

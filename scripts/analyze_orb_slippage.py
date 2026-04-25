"""Measure actual entry/exit slippage on live ORB trades vs BT assumptions.

BT assumes:
  entry_slip_bps: 30 (stop-limit BUY fills at trigger * 1.003 max)
  exit_slip_bps:  10

Live:
  entry actual = fill_price
  entry trigger = pattern_data.range_high (stop level)
  exit actual = exit_price
  exit trigger = exit_trigger_price (lock level or range_low)

Key question: is live slippage close to BT? If 2x BT, the BT projections
are over-optimistic. If 1x or better, BT is realistic.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(ROOT / 'data' / 'trades.db')
BT_ENTRY_BPS = 30.0
BT_EXIT_BPS = 10.0


def bps(num: float, denom: float) -> float:
    """Return basis points: (num / denom) * 10000."""
    if denom <= 0:
        return float('nan')
    return num / denom * 10000


def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT trade_date, symbol, side, entry_price, fill_price,
               exit_price, exit_trigger_price, exit_quote_bid, exit_quote_ask,
               exit_limit_price, pattern_data, shares, pnl
        FROM trades
        WHERE strategy = 'orb'
        ORDER BY trade_date, symbol
    """, conn)
    conn.close()

    print(f"ORB trades in DB: {len(df)}")

    # Parse pattern_data for range_high
    def parse_pd(s):
        if pd.isna(s) or not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    df['pd'] = df['pattern_data'].apply(parse_pd)
    df['range_high'] = df['pd'].apply(lambda p: p.get('range_high') if p else None)
    df['range_low'] = df['pd'].apply(lambda p: p.get('range_low') if p else None)
    df['quintile'] = df['pd'].apply(lambda p: p.get('quintile') if p else None)
    df['range_size'] = df['pd'].apply(lambda p: p.get('range_size') if p else None)

    # Entry slippage: actual fill vs range_high (the trigger)
    # BT assumes worst-case fill at range_high * 1.003 (30 bps above trigger)
    # Slippage in bps = (fill - range_high) / range_high * 10000
    # IMPORTANT: only use actual fill_price; if it's null the order didn't fill
    # so we can't measure slippage (entry_price column = limit price for unfilled).
    def entry_slip(row):
        fp = row['fill_price']
        rh = row['range_high']
        if rh is None or pd.isna(fp) or fp <= 0:
            return None
        return bps(fp - rh, rh)
    df['entry_slip_bps'] = df.apply(entry_slip, axis=1)

    # Exit slippage: actual exit vs exit_trigger_price
    # BT assumes worst-case fill at trigger * (1 - 10 bps)
    # Slippage = (trigger - actual) / trigger * 10000  (positive = sold below trigger)
    def exit_slip(row):
        tp = row['exit_trigger_price']
        ep = row['exit_price']
        if pd.isna(tp) or tp <= 0 or pd.isna(ep) or ep <= 0:
            return None
        return bps(tp - ep, tp)
    df['exit_slip_bps'] = df.apply(exit_slip, axis=1)

    # Filled / closed flags
    df['is_filled'] = df['fill_price'].notna() & (df['fill_price'] > 0)
    df['is_closed'] = df['exit_price'].notna() & (df['exit_price'] > 0)

    # Per-trade table
    print(f"\n{'='*120}")
    print("  Per-trade slippage (entry: live vs range_high; exit: live vs trigger)")
    print(f"{'='*120}")
    print(f"{'Date':<11} {'Symbol':<7} {'Q':<3} {'RangeH':>8} {'Fill':>9} "
          f"{'EntSlip':>9} {'ExTrig':>8} {'ExFill':>9} {'ExSlip':>9} "
          f"{'Spread':>8} {'Status':<12}")
    print('-' * 120)
    for _, r in df.iterrows():
        ent_str = f"{r['entry_slip_bps']:.1f}bps" if pd.notna(r['entry_slip_bps']) else '   --'
        exit_str = f"{r['exit_slip_bps']:.1f}bps" if pd.notna(r['exit_slip_bps']) else '   --'
        spread_bps = bps(r['exit_quote_ask'] - r['exit_quote_bid'], r['exit_quote_bid']) \
            if pd.notna(r['exit_quote_bid']) and pd.notna(r['exit_quote_ask']) else None
        spread_str = f"{spread_bps:.1f}bps" if spread_bps is not None else '   --'
        status = 'FILLED+CLOSED' if r['is_filled'] and r['is_closed'] else \
                 ('FILLED' if r['is_filled'] else \
                  ('OPEN/PENDING' if not r['is_filled'] else 'UNKNOWN'))
        rh = f"{r['range_high']:.2f}" if pd.notna(r['range_high']) else '--'
        fp = f"{r['fill_price']:.4f}" if pd.notna(r['fill_price']) else '--'
        et = f"{r['exit_trigger_price']:.2f}" if pd.notna(r['exit_trigger_price']) else '--'
        ep = f"{r['exit_price']:.4f}" if pd.notna(r['exit_price']) else '--'
        print(f"{r['trade_date']:<11} {r['symbol']:<7} {r['quintile'] or '--':<3} "
              f"{rh:>8} {fp:>9} {ent_str:>9} {et:>8} {ep:>9} {exit_str:>9} "
              f"{spread_str:>8} {status:<12}")

    # Summary
    filled = df[df['entry_slip_bps'].notna()].copy()
    closed = df[df['exit_slip_bps'].notna()].copy()

    print(f"\n{'='*70}")
    print("  ENTRY SLIPPAGE SUMMARY")
    print(f"{'='*70}")
    if len(filled):
        print(f"  Filled trades:      {len(filled)}")
        print(f"  BT assumption:       {BT_ENTRY_BPS:.0f} bps")
        print(f"  Live mean:          {filled['entry_slip_bps'].mean():.1f} bps")
        print(f"  Live median:        {filled['entry_slip_bps'].median():.1f} bps")
        print(f"  Live min:           {filled['entry_slip_bps'].min():.1f} bps")
        print(f"  Live max:           {filled['entry_slip_bps'].max():.1f} bps")
        print(f"  Live p90:           {filled['entry_slip_bps'].quantile(0.9):.1f} bps")
        bt_better = (filled['entry_slip_bps'] < BT_ENTRY_BPS).sum()
        bt_worse = (filled['entry_slip_bps'] > BT_ENTRY_BPS).sum()
        print(f"  Better than BT 30bps:  {bt_better}/{len(filled)}")
        print(f"  Worse than BT 30bps:   {bt_worse}/{len(filled)}")
    else:
        print("  (no filled trades)")

    print(f"\n{'='*70}")
    print("  EXIT SLIPPAGE SUMMARY")
    print(f"{'='*70}")
    if len(closed):
        print(f"  Closed trades:      {len(closed)}")
        print(f"  BT assumption:       {BT_EXIT_BPS:.0f} bps")
        print(f"  Live mean:          {closed['exit_slip_bps'].mean():.1f} bps")
        print(f"  Live median:        {closed['exit_slip_bps'].median():.1f} bps")
        print(f"  Live min:           {closed['exit_slip_bps'].min():.1f} bps")
        print(f"  Live max:           {closed['exit_slip_bps'].max():.1f} bps")
        print(f"  Live p90:           {closed['exit_slip_bps'].quantile(0.9):.1f} bps")
        bt_better = (closed['exit_slip_bps'] < BT_EXIT_BPS).sum()
        bt_worse = (closed['exit_slip_bps'] > BT_EXIT_BPS).sum()
        print(f"  Better than BT 10bps:  {bt_better}/{len(closed)}")
        print(f"  Worse than BT 10bps:   {bt_worse}/{len(closed)}")
    else:
        print("  (no closed trades)")

    # Combined slippage cost vs BT
    print(f"\n{'='*70}")
    print("  COMBINED ROUND-TRIP SLIPPAGE COST")
    print(f"{'='*70}")
    rt = df[df['entry_slip_bps'].notna() & df['exit_slip_bps'].notna()].copy()
    if len(rt):
        rt['rt_slip_bps'] = rt['entry_slip_bps'] + rt['exit_slip_bps']
        rt['bt_rt_bps'] = BT_ENTRY_BPS + BT_EXIT_BPS
        rt['extra_cost_bps'] = rt['rt_slip_bps'] - rt['bt_rt_bps']
        print(f"  Round-trip trades:  {len(rt)}")
        print(f"  BT round-trip cost:  {BT_ENTRY_BPS + BT_EXIT_BPS:.0f} bps")
        print(f"  Live round-trip mean: {rt['rt_slip_bps'].mean():.1f} bps")
        print(f"  Live round-trip max:  {rt['rt_slip_bps'].max():.1f} bps")
        print(f"  Extra cost vs BT mean: {rt['extra_cost_bps'].mean():+.1f} bps")
        print(f"  Trades costing > 50 bps extra: "
              f"{(rt['extra_cost_bps'] > 50).sum()}/{len(rt)}")

    # By price bucket — capacity check for Stage 4 sizing
    print(f"\n{'='*70}")
    print("  ENTRY SLIPPAGE BY PRICE BUCKET (capacity proxy)")
    print(f"{'='*70}")
    if len(filled):
        for lo, hi, label in [(0, 5, '< $5'), (5, 10, '$5-$10'),
                                (10, 20, '$10-$20'), (20, 1000, '$20+')]:
            sub = filled[(filled['fill_price'] >= lo) & (filled['fill_price'] < hi)]
            if len(sub) == 0: continue
            print(f"  {label:<10}: n={len(sub):>2}  mean={sub['entry_slip_bps'].mean():>5.1f}bps  "
                  f"max={sub['entry_slip_bps'].max():>5.1f}bps")

    # Spread summary at exit
    print(f"\n{'='*70}")
    print("  EXIT QUOTE SPREAD")
    print(f"{'='*70}")
    sp = df[(df['exit_quote_bid'].notna()) & (df['exit_quote_ask'].notna()) &
            (df['exit_quote_bid'] > 0)].copy()
    if len(sp):
        sp['spread_bps'] = sp.apply(
            lambda r: bps(r['exit_quote_ask'] - r['exit_quote_bid'], r['exit_quote_bid']),
            axis=1)
        print(f"  Trades with quote data: {len(sp)}")
        print(f"  Spread mean:  {sp['spread_bps'].mean():.1f} bps")
        print(f"  Spread max:   {sp['spread_bps'].max():.1f} bps")
        print(f"  Spread p90:   {sp['spread_bps'].quantile(0.9):.1f} bps")
        # Note: orb.yaml has max_spread_bps: 150 — we should never see this exceeded
        over_150 = (sp['spread_bps'] > 150).sum()
        if over_150:
            print(f"  WARNING: {over_150} trades had exit spread > 150 bps "
                  f"(this should have been blocked at entry, but exit time can be different)")


if __name__ == '__main__':
    main()

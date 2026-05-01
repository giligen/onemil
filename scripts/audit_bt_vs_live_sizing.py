#!/usr/bin/env python3
"""Audit BT vs LIVE sizing divergence — per-trade decomposition.

For every (symbol, date) that appears in BOTH the BT cache and the LIVE
trades DB, compute:
  - dollar_risk_BT   = bt_shares  × (bt_entry  - bt_stop)
  - dollar_risk_LIVE = live_shares × (live_entry - live_stop)
  - risk_ratio       = dollar_risk_LIVE / dollar_risk_BT

The decomposition tells us WHERE the divergence is:
  ratio ≈ 1.0  → same dollar risk, share diff is just price/stop noise
  ratio ≈ 0.5  → LIVE got half the risk budget — likely conviction or
                 regime mult difference, or DTBP cap, or marginability
  ratio ≈ 2.0  → LIVE got 2x — marginability flipped on, or BT under-
                 sized for some reason

We also extract conviction_mult and macd_zone_mult from both sides
(LIVE: pattern_data JSON; BT: cache columns) and report the components.

Usage: python3 scripts/audit_bt_vs_live_sizing.py [start_date] [end_date]
Default: April 2026.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
CACHE_CSV = ROOT / 'data' / 'bull_flag_cache_e50_x30.csv'
TRADES_DB = ROOT / 'data' / 'trades.db'


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else '2026-04-01'
    end = sys.argv[2] if len(sys.argv) > 2 else '2026-04-30'

    # Load BT cache
    bt = pd.read_csv(CACHE_CSV)
    bt['date'] = pd.to_datetime(bt['date']).dt.strftime('%Y-%m-%d')
    bt = bt[(bt['date'] >= start) & (bt['date'] <= end)].copy()
    bt['entry_price'] = pd.to_numeric(bt['entry_price'])
    bt['stop_loss'] = pd.to_numeric(bt['stop_loss'])
    bt['shares'] = pd.to_numeric(bt['shares'])
    bt['pnl'] = pd.to_numeric(bt['pnl'])
    bt['conviction_mult'] = pd.to_numeric(bt.get('conviction_mult'), errors='coerce')
    bt['macd_zone_mult'] = pd.to_numeric(bt.get('macd_zone_mult'), errors='coerce')

    # Load LIVE trades
    con = sqlite3.connect(str(TRADES_DB))
    live = pd.read_sql_query(
        """
        SELECT trade_date AS date, symbol, fill_price AS entry_price,
               real_stop_loss_price AS stop_loss, shares,
               exit_price, exit_reason, pnl, pattern_data,
               total_risk, risk_per_share, strategy
        FROM trades
        WHERE strategy='bull_flag' AND trade_date >= ? AND trade_date <= ?
              AND fill_price IS NOT NULL
        """,
        con, params=(start, end),
    )
    con.close()

    # Extract conviction_mult / macd_zone_mult from pattern_data JSON
    def parse_pd(s):
        try:
            return json.loads(s) if s else {}
        except Exception:
            return {}

    live['_pd'] = live['pattern_data'].apply(parse_pd)
    live['conviction_mult'] = live['_pd'].apply(
        lambda d: d.get('conviction_mult') or d.get('conviction') or 1.0
    )
    live['macd_zone_mult'] = live['_pd'].apply(
        lambda d: d.get('macd_zone_mult') or d.get('macd_zone') or 1.0
    )
    live = live.drop(columns=['_pd', 'pattern_data'])

    # Join on (symbol, date)
    merged = bt.merge(
        live, on=['symbol', 'date'], how='inner',
        suffixes=('_bt', '_live'),
    )
    print(f"BT trades: {len(bt)}  LIVE trades: {len(live)}  "
          f"matched: {len(merged)}")
    if merged.empty:
        print("No overlap. Try a wider date range.")
        return

    # Per-trade decomposition
    merged['risk_per_share_bt'] = merged['entry_price_bt'] - merged['stop_loss_bt']
    merged['risk_per_share_live'] = (
        merged['entry_price_live'] - merged['stop_loss_live']
    )
    merged['dollar_risk_bt'] = merged['shares_bt'] * merged['risk_per_share_bt']
    merged['dollar_risk_live'] = (
        merged['shares_live'] * merged['risk_per_share_live']
    )
    merged['risk_ratio'] = (
        merged['dollar_risk_live'] / merged['dollar_risk_bt']
    )
    merged['share_ratio'] = merged['shares_live'] / merged['shares_bt']
    merged['conv_ratio'] = (
        merged['conviction_mult_live'] / merged['conviction_mult_bt']
    )
    merged['macd_zone_ratio'] = (
        merged['macd_zone_mult_live'] / merged['macd_zone_mult_bt']
    )
    merged['pnl_diff'] = merged['pnl_live'] - merged['pnl_bt']

    # Sort by absolute pnl_diff (biggest divergences first)
    merged['abs_pnl_diff'] = merged['pnl_diff'].abs()
    merged = merged.sort_values('abs_pnl_diff', ascending=False)

    print()
    print("=" * 110)
    print("PER-TRADE BT vs LIVE SIZING AUDIT")
    print("=" * 110)
    cols_print = [
        ('symbol', '<7'), ('date', '<12'),
        ('shares_bt', '>7'), ('shares_live', '>7'),
        ('share_ratio', '>6.2f'),
        ('dollar_risk_bt', '>9,.0f'), ('dollar_risk_live', '>9,.0f'),
        ('risk_ratio', '>6.2f'),
        ('conv_ratio', '>6.2f'),
        ('macd_zone_ratio', '>6.2f'),
        ('pnl_bt', '>+9,.0f'), ('pnl_live', '>+9,.0f'),
        ('pnl_diff', '>+9,.0f'),
        ('exit_reason_live', '<22'),
    ]
    headers = [c[0] for c in cols_print]
    print(' '.join(f'{h:<7}' if i < 2 else f'{h:>7}' for i, h in enumerate(headers[:12])) +
          ' ' + headers[12] + ' ' + headers[13])

    for _, r in merged.iterrows():
        sym = r['symbol']
        dt = r['date']
        sb = int(r['shares_bt'])
        sl = int(r['shares_live'])
        sr = r['share_ratio']
        drb = r['dollar_risk_bt']
        drl = r['dollar_risk_live']
        rr = r['risk_ratio']
        cr = r['conv_ratio']
        mr = r['macd_zone_ratio']
        pb = r['pnl_bt']
        pl = r['pnl_live']
        pd_ = r['pnl_diff']
        er = r['exit_reason_live'][:22] if r['exit_reason_live'] else '?'
        print(
            f"{sym:<7} {dt:<11} {sb:>7,d} {sl:>7,d} {sr:>5.2f}x "
            f"${drb:>8,.0f} ${drl:>8,.0f} {rr:>5.2f}x {cr:>5.2f}x {mr:>5.2f}x "
            f"${pb:>+8,.0f} ${pl:>+8,.0f} ${pd_:>+8,.0f}  {er}"
        )

    print()
    print("=" * 110)
    print("DECOMPOSITION SUMMARY")
    print("=" * 110)
    print(f"  matched trades: {len(merged)}")
    print(f"  median share_ratio  : {merged['share_ratio'].median():.2f}x")
    print(f"  median risk_ratio   : {merged['risk_ratio'].median():.2f}x")
    print(f"  median conv_ratio   : {merged['conv_ratio'].median():.2f}x")
    print(f"  median macd_zone_r  : {merged['macd_zone_ratio'].median():.2f}x")
    print(f"  total BT pnl   : ${merged['pnl_bt'].sum():+,.0f}")
    print(f"  total LIVE pnl : ${merged['pnl_live'].sum():+,.0f}")
    print(f"  total pnl_diff : ${merged['pnl_diff'].sum():+,.0f}")
    print()

    # Group by ratio bucket to highlight where the divergence concentrates
    buckets = [
        ('LIVE 0-25% of BT', merged['risk_ratio'] < 0.25),
        ('LIVE 25-75% of BT', (merged['risk_ratio'] >= 0.25) & (merged['risk_ratio'] < 0.75)),
        ('LIVE ~equal (0.75-1.25)', (merged['risk_ratio'] >= 0.75) & (merged['risk_ratio'] < 1.25)),
        ('LIVE 1.25-2x', (merged['risk_ratio'] >= 1.25) & (merged['risk_ratio'] < 2.0)),
        ('LIVE 2x+', merged['risk_ratio'] >= 2.0),
    ]
    print("RISK_RATIO BUCKETS (LIVE / BT):")
    for label, mask in buckets:
        sub = merged[mask]
        if len(sub) == 0:
            continue
        print(f"  {label:<28} n={len(sub):>2}  "
              f"BT_pnl=${sub['pnl_bt'].sum():>+9,.0f}  "
              f"LIVE_pnl=${sub['pnl_live'].sum():>+9,.0f}  "
              f"diff=${sub['pnl_diff'].sum():>+9,.0f}")

    # Save CSV
    out = ROOT / f'audit_sizing_{start}_to_{end}.csv'
    merged_out = merged[[
        'symbol', 'date',
        'entry_price_bt', 'stop_loss_bt', 'shares_bt', 'pnl_bt',
        'entry_price_live', 'stop_loss_live', 'shares_live', 'pnl_live',
        'share_ratio', 'risk_ratio', 'conv_ratio', 'macd_zone_ratio',
        'pnl_diff', 'exit_reason_bt', 'exit_reason_live',
    ]].copy()
    merged_out.to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()

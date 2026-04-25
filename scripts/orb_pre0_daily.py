"""Pre-Stage-0 LIVE daily monitor.

Run after market close (or anytime). Reports:
  - Days into Pre-Stage-0 LIVE
  - Cumulative cushion (realized P&L since launch)
  - Today's P&L
  - Today's slippage vs BT (entry/exit/round-trip)
  - Promotion eligibility (Pre-0 → Stage 0)
  - Demotion triggers fired

Usage:
  python3 scripts/orb_pre0_daily.py [--launch-date YYYY-MM-DD]

If --launch-date not provided, infers from most recent commit matching
"ORB ramp: Pre-Stage-0 LIVE" or falls back to first ORB trade in DB at
half-size config (account_budget=15000).
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yaml


DB_PATH = '/home/ec2-user/onemil/data/trades.db'
ORB_YAML = '/home/ec2-user/onemil/orb.yaml'

# Pre-Stage-0 spec
PRE0_BUDGET = 15000
PRE0_RISK = 500
PRE0_DAILY_LOSS = -750

# Promotion thresholds
PROMOTE_MIN_DAYS = 10
PROMOTE_MIN_CUSHION = 1000
PROMOTE_MAX_RT_SLIP_BPS = 60
PROMOTE_MAX_ENTRY_BPS = 45
PROMOTE_MAX_EXIT_BPS = 25
PROMOTE_MAX_DRIFT = 0.05

# Demotion triggers
HARD_STOP_CUSHION = -3000
DEMOTE_DAILY_LOSS = -1000

# BT assumptions for comparison
BT_ENTRY_BPS = 30.0
BT_EXIT_BPS = 10.0


def bps(num: float, denom: float) -> float:
    if denom <= 0: return float('nan')
    return num / denom * 10000


def find_launch_date_from_git() -> Optional[date]:
    """Look in git log for the Pre-Stage-0 launch commit."""
    try:
        result = subprocess.run(
            ['git', 'log', '--all', '--pretty=format:%aI %s'],
            capture_output=True, text=True, cwd='/home/ec2-user/onemil',
        )
        for line in result.stdout.splitlines():
            if 'ORB ramp: Pre-Stage-0 LIVE' in line or \
               'ORB ramp: Pre-Stage-0 (live launch)' in line:
                ts = line.split(' ', 1)[0]
                return datetime.fromisoformat(ts.replace('Z', '+00:00')).date()
    except Exception:
        pass
    return None


def find_launch_date_from_db() -> Optional[date]:
    """Fallback: look for first ORB trade after orb.yaml had budget=15000.

    Approximation: first ORB trade in the DB. (We don't store config snapshot
    per trade, so this is best-effort.)
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(
            "SELECT MIN(trade_date) AS first FROM trades WHERE strategy = 'orb'",
            conn,
        )
        conn.close()
        first = df.iloc[0]['first']
        if first:
            return datetime.strptime(first, '%Y-%m-%d').date()
    except Exception:
        pass
    return None


def load_config_state() -> dict:
    """Read current orb.yaml to verify we're actually on Pre-Stage-0 config."""
    if not os.path.exists(ORB_YAML):
        return {}
    with open(ORB_YAML) as f:
        cfg = yaml.safe_load(f)
    return {
        'budget': cfg.get('sizing', {}).get('account_budget_usd'),
        'risk': cfg.get('sizing', {}).get('risk_per_trade_usd'),
        'daily_loss': cfg.get('risk', {}).get('daily_loss_limit_usd'),
        'tg_prefix': cfg.get('notifications', {}).get('telegram', {}).get('prefix'),
    }


def fetch_orb_trades_since(start_date: date) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT trade_date, symbol, entry_price, fill_price, exit_price,
               exit_trigger_price, exit_quote_bid, exit_quote_ask,
               pattern_data, shares, pnl
        FROM trades
        WHERE strategy = 'orb' AND trade_date >= ?
        ORDER BY trade_date, symbol
    """, conn, params=(start_date.isoformat(),))
    conn.close()
    if not len(df): return df
    df['pd'] = df['pattern_data'].apply(
        lambda s: json.loads(s) if pd.notna(s) and s else {})
    df['range_high'] = df['pd'].apply(lambda p: p.get('range_high'))
    return df


def count_trading_days(start: date, end: date) -> int:
    """Approximate count of trading days between start and end (inclusive)."""
    n = 0
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon-Fri
            n += 1
        d += timedelta(days=1)
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch-date', type=str, default=None)
    args = parser.parse_args()

    # Resolve launch date
    if args.launch_date:
        launch = datetime.strptime(args.launch_date, '%Y-%m-%d').date()
    else:
        launch = find_launch_date_from_git()
        if launch is None:
            launch = find_launch_date_from_db()
    if launch is None:
        print("ERROR: Cannot determine Pre-Stage-0 launch date.")
        print("Pass --launch-date YYYY-MM-DD or commit with message containing")
        print("'ORB ramp: Pre-Stage-0 LIVE'.")
        sys.exit(1)

    today = date.today()
    days_in_stage = count_trading_days(launch, today)

    print(f"{'='*72}")
    print(f"  ORB Pre-Stage-0 LIVE — Daily Monitor")
    print(f"  Generated: {datetime.now().isoformat(timespec='seconds')}")
    print(f"{'='*72}")
    print(f"  Launch date:        {launch}")
    print(f"  Days in stage:      {days_in_stage}")
    print()

    # Verify config is actually Pre-Stage-0
    cfg = load_config_state()
    print(f"  Current orb.yaml config:")
    print(f"    budget:       ${cfg.get('budget'):>7}  (expected ${PRE0_BUDGET})"
          if cfg.get('budget') is not None else f"    budget:       ?")
    print(f"    risk/trade:   ${cfg.get('risk'):>7}  (expected ${PRE0_RISK})"
          if cfg.get('risk') is not None else f"    risk/trade:   ?")
    print(f"    daily_loss:  ${cfg.get('daily_loss'):>7}  (expected ${PRE0_DAILY_LOSS})"
          if cfg.get('daily_loss') is not None else f"    daily_loss:   ?")
    print(f"    tg prefix:    {cfg.get('tg_prefix')!r}  (expected '[ORB-LIVE-PRE0]')")
    config_matches = (
        cfg.get('budget') == PRE0_BUDGET
        and cfg.get('risk') == PRE0_RISK
        and cfg.get('daily_loss') == PRE0_DAILY_LOSS
    )
    if not config_matches:
        print(f"\n  WARNING: orb.yaml does NOT match Pre-Stage-0 spec.")
        print(f"  Either you're not in Pre-Stage-0 yet, or the config has drifted.")

    # Fetch trades since launch
    trades = fetch_orb_trades_since(launch)
    if not len(trades):
        print(f"\n  No ORB trades since launch. Waiting for first trade.")
        sys.exit(0)

    # Cumulative cushion
    closed = trades[trades['pnl'].notna() & (trades['exit_price'].notna())].copy()
    cushion = closed['pnl'].sum()
    today_iso = today.isoformat()
    today_trades = closed[closed['trade_date'] == today_iso]
    today_pnl = today_trades['pnl'].sum()

    print(f"\n  P&L since Pre-Stage-0 launch:")
    print(f"    Trades closed:    {len(closed)}")
    print(f"    Cumulative cushion: ${cushion:+,.0f}")
    print(f"    Today ({today_iso}):  ${today_pnl:+,.0f}  ({len(today_trades)} trades)")

    # Daily breakdown (last 5 days)
    print(f"\n  Last 5 trading days:")
    for d in (closed.groupby('trade_date')['pnl'].sum()
              .reset_index().sort_values('trade_date').tail(5).iterrows()):
        idx, row = d
        marker = '  '
        if row['pnl'] < DEMOTE_DAILY_LOSS:
            marker = '⚠ '
        print(f"    {marker}{row['trade_date']}: ${row['pnl']:+8,.0f}")

    # Slippage today
    if len(today_trades):
        today_filled = today_trades[today_trades['fill_price'].notna()].copy()
        if len(today_filled):
            today_filled['entry_slip_bps'] = today_filled.apply(
                lambda r: bps(r['fill_price'] - r['range_high'], r['range_high'])
                if pd.notna(r['range_high']) else None, axis=1)
            today_closed = today_trades[
                today_trades['exit_trigger_price'].notna() &
                today_trades['exit_price'].notna()].copy()
            today_closed['exit_slip_bps'] = today_closed.apply(
                lambda r: bps(r['exit_trigger_price'] - r['exit_price'],
                              r['exit_trigger_price']), axis=1)
            entry_mean = today_filled['entry_slip_bps'].mean()
            exit_mean = today_closed['exit_slip_bps'].mean() if len(today_closed) else None
            print(f"\n  Today's slippage:")
            print(f"    Entry mean:  {entry_mean:.1f} bps  (BT {BT_ENTRY_BPS:.0f}, gate ≤ {PROMOTE_MAX_ENTRY_BPS:.0f})")
            if exit_mean is not None:
                print(f"    Exit mean:   {exit_mean:.1f} bps  (BT {BT_EXIT_BPS:.0f}, gate ≤ {PROMOTE_MAX_EXIT_BPS:.0f})")
                print(f"    Round-trip:  {entry_mean + exit_mean:.1f} bps  (BT 40, gate ≤ {PROMOTE_MAX_RT_SLIP_BPS})")

    # Cumulative slippage
    closed_all = closed[closed['fill_price'].notna() & closed['exit_price'].notna()].copy()
    if len(closed_all):
        closed_all['entry_slip_bps'] = closed_all.apply(
            lambda r: bps(r['fill_price'] - r['range_high'], r['range_high'])
            if pd.notna(r['range_high']) else None, axis=1)
        closed_all['exit_slip_bps'] = closed_all.apply(
            lambda r: bps(r['exit_trigger_price'] - r['exit_price'],
                          r['exit_trigger_price'])
            if pd.notna(r['exit_trigger_price']) else None, axis=1)
        em = closed_all['entry_slip_bps'].dropna().mean()
        xm = closed_all['exit_slip_bps'].dropna().mean()
        print(f"\n  Cumulative slippage (since launch):")
        print(f"    Entry mean:  {em:.1f} bps  (n={int(closed_all['entry_slip_bps'].notna().sum())})")
        print(f"    Exit mean:   {xm:.1f} bps  (n={int(closed_all['exit_slip_bps'].notna().sum())})")
        print(f"    Round-trip:  {em + xm:.1f} bps")

    # Demotion triggers
    print(f"\n{'='*72}")
    print(f"  DEMOTION TRIGGERS")
    print(f"{'='*72}")
    triggered = []
    if cushion <= HARD_STOP_CUSHION:
        triggered.append(f"HARD STOP: cushion ${cushion:+,.0f} ≤ ${HARD_STOP_CUSHION:+,.0f}")
    if today_pnl < DEMOTE_DAILY_LOSS:
        triggered.append(f"Daily loss: today ${today_pnl:+,.0f} < ${DEMOTE_DAILY_LOSS:+,.0f}")
    # Check 3 consecutive red days
    daily = closed.groupby('trade_date')['pnl'].sum().reset_index().sort_values('trade_date')
    if len(daily) >= 3:
        last3 = daily.tail(3)['pnl'].values
        if all(v < 0 for v in last3):
            triggered.append(f"3 consecutive red days: {[round(v,0) for v in last3]}")
    if triggered:
        for t in triggered:
            print(f"  ⚠ {t}")
        print(f"\n  ACTION: Halt Pre-Stage-0 LIVE. Revert orb.yaml to research config.")
    else:
        print(f"  ✓ No demotion triggers fired")

    # Promotion eligibility
    print(f"\n{'='*72}")
    print(f"  PROMOTION ELIGIBILITY (Pre-Stage-0 → Stage 0)")
    print(f"{'='*72}")
    days_ok = days_in_stage >= PROMOTE_MIN_DAYS
    cushion_ok = cushion >= PROMOTE_MIN_CUSHION
    slip_ok = True
    if len(closed_all):
        em = closed_all['entry_slip_bps'].dropna().mean()
        xm = closed_all['exit_slip_bps'].dropna().mean()
        slip_ok = (em <= PROMOTE_MAX_ENTRY_BPS and xm <= PROMOTE_MAX_EXIT_BPS
                   and (em + xm) <= PROMOTE_MAX_RT_SLIP_BPS)
    print(f"  ≥ {PROMOTE_MIN_DAYS} days in stage:    {'✓' if days_ok else '✗'}  ({days_in_stage} days)")
    print(f"  ≥ ${PROMOTE_MIN_CUSHION} cushion:      {'✓' if cushion_ok else '✗'}  (${cushion:+,.0f})")
    print(f"  Slippage gates met:  {'✓' if slip_ok else '✗'}")
    if days_ok and cushion_ok and slip_ok:
        print(f"\n  ✓✓ ELIGIBLE for promotion to Stage 0 ($30K, $1K risk).")
        print(f"     Edit orb.yaml: budget=30000, risk=1000, daily_loss=-1500")
        print(f"     Telegram prefix: '[ORB-LIVE]'")
        print(f"     Commit: 'ORB ramp: Pre-Stage-0 → Stage 0 (cushion $X built, Y days)'")
    else:
        print(f"\n  Not yet eligible. Continue Pre-Stage-0.")


if __name__ == '__main__':
    main()

"""Q1 2025 day-by-day under FINAL spec: static_lock_1R exit.

Full pipeline:
  - Filter: composite z >= 0 (TRAIN-fit on H1 2025)
  - Rank: Q4-preferred + family/super-group dedup
  - Cap: top 4 picks per day
  - Size: risk-parity $3K/trade, $25K per-position cap, adaptive Q5-capped mults
  - Exit: static lock at +1R after price touches +1.5R; else hold to EOD or stop at range_low
"""
from __future__ import annotations

import os, sys, glob
from datetime import timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database
from study_orb import _bars_to_df, _session_open_timestamp, EXIT_SLIP_BPS_DEFAULT
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN,
    fit_quintile_cutoffs, assign_quintile,
)
from study_orb_correlation_filter import symbol_family, symbol_super_group

ACCOUNT = 100_000
N = 4
RISK = 3000
OLD_POS = 50_000
MIN_STOP_PCT = 1.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}

# Static-lock exit parameters
LOCK_TRIGGER_R = 1.5  # arm lock when price touches +1.5R
LOCK_STOP_R = 1.0     # stop pinned at entry + 1R


def simulate_static_lock(bars, entry_price, range_high, range_low, entry_time,
                         exit_slip=EXIT_SLIP_BPS_DEFAULT):
    range_size = range_high - range_low
    trigger_lvl = entry_price + LOCK_TRIGGER_R * range_size
    lock_stop = entry_price + LOCK_STOP_R * range_size
    stop_price = range_low
    armed = False
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        if not armed and bar_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        if bar_low <= stop_price:
            return stop_price * (1 - exit_slip/10000), 'lock' if armed else 'stop'
    last = post.iloc[-1]
    return float(last['close']) * (1 - exit_slip/10000), 'eod'


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl','date','pnl_pct','range_size_pct','entry_price'])
    df['date'] = pd.to_datetime(df['date'])

    # Load bars for all trades
    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    print(f"Loading bars for {len(pairs)} pairs...")
    db_obj = Database(db_path='data/cache.db')
    raw_bars = db_obj.get_intraday_bars_bulk(pairs)
    db_obj.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}

    # Re-simulate all trades with static_lock_1R exit
    print("Re-simulating all trades with static_lock_1R exit...")
    new_pnls = []; new_pnl_pcts = []; new_reasons = []
    for idx, row in df.reset_index(drop=True).iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) & (bars['timestamp'] < range_end)]
        if len(range_bars) < 5:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh:
                entry_ts = b['timestamp']; break
        if entry_ts is None:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue

        entry_p = float(row['entry_price'])
        exit_p, reason = simulate_static_lock(bars, entry_p, rh, rl, entry_ts)
        shares = max(1, int(OLD_POS / entry_p))
        new_pnls.append((exit_p - entry_p) * shares)
        new_pnl_pcts.append((exit_p - entry_p) / entry_p * 100)
        new_reasons.append(reason)

    df = df.reset_index(drop=True)
    df['pnl'] = new_pnls
    df['pnl_pct'] = new_pnl_pcts
    df['exit_reason'] = new_reasons

    # Risk-parity sizing
    per_pos_cap = ACCOUNT / N
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS

    # Fit filter + quintile + adaptive mults on H1 2025 TRAIN
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    train_k = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    avg = float(train_k['_rp_pnl'].mean())
    mults = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = train_k[train_k['_quintile'] == q]
        mults[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], float(sub['_rp_pnl'].mean()) / avg))

    # Apply to Q1 2025
    q1 = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-03-31')]
    q1_k = q1[q1['_composite'] >= FILTER_THRESHOLD].copy()
    q1_k['_quintile'] = assign_quintile(q1_k['_composite'], cutoffs)

    print(f"\n{'='*125}")
    print(f"Q1 2025 — FINAL SPEC: static_lock_1R exit")
    print(f"  Pipeline: filter z>=0 → Q4-pref + family/super-group dedup → cap{N} → Q5-capped adaptive mults")
    print(f"  Exit: stop=range_low initially; if +{LOCK_TRIGGER_R}R touched, lock stop at +{LOCK_STOP_R}R; else hold to EOD")
    print(f"  Mults: " + " ".join(f"{q}={mults[q]:.2f}x" for q in ['Q1','Q2','Q3','Q4','Q5']))
    print(f"{'='*125}")
    print(f"\n{'date':<12} {'wd':<4} {'sig':>4} {'pick':>4} {'trades':<56} "
          f"{'day P&L':>10} {'equity':>10} {'DD':>9}")
    print('-' * 125)

    equity = 0.0; peak = 0.0; max_dd = 0.0
    daily_rows = []
    for day in sorted(q1_k['date'].unique()):
        dg = q1_k[q1_k['date'] == day]
        # Rank + dedup
        d = dg.copy()
        d['_q_rank'] = d['_quintile'].map(Q_ORDER)
        d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        seen_fam = set(); seen_sup = set()
        picked_rows = []
        for _, r in d.iterrows():
            fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_fam: continue
            if sup and sup in seen_sup: continue
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
            picked_rows.append(r)
            if len(picked_rows) >= N: break
        picked = pd.DataFrame(picked_rows).copy()
        picked['_sized_pnl'] = picked.apply(
            lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)

        parts = []
        for _, r in picked.iterrows():
            sup = symbol_super_group(r['symbol']); fam = symbol_family(r['symbol'])
            tag = ''
            if sup == 'lev_short': tag = '[SHORT]'
            elif sup == 'lev_long': tag = '[LONG]'
            elif fam: tag = f"[{fam[:4]}]"
            arrow = '✓' if r['_sized_pnl'] > 0 else '✗'
            xr = r['exit_reason'][:4] if 'exit_reason' in r else ''
            parts.append(f"{r['symbol']}{tag}({r['_quintile']},{xr}){arrow}${r['_sized_pnl']:+,.0f}")
        trade_str = ', '.join(parts)
        if len(trade_str) > 54: trade_str = trade_str[:51] + '…'

        day_pnl = float(picked['_sized_pnl'].sum())
        equity += day_pnl
        peak = max(peak, equity)
        dd_now = equity - peak
        max_dd = min(max_dd, dd_now)
        daily_rows.append({'date': day, 'day_pnl': day_pnl, 'equity': equity, 'dd_now': dd_now,
                          'trades': trade_str, 'n_sig': len(dg), 'n_pick': len(picked)})
        wd = day.strftime('%a')
        print(f"{day.date().isoformat():<12} {wd:<4} {len(dg):>4} {len(picked):>4} "
              f"{trade_str:<56} ${day_pnl:>+8,.0f} ${equity:>+8,.0f} ${dd_now:>+7,.0f}")

    daily = pd.DataFrame(daily_rows)
    print(f"\n{'='*125}")
    print(f"Q1 2025 SUMMARY — static_lock_1R")
    print(f"{'='*125}")
    print(f"  Final equity: ${equity:+,.0f}")
    print(f"  Peak equity:  ${peak:+,.0f}")
    print(f"  Max DD:       ${max_dd:+,.0f}")
    print(f"  Trading days: {len(daily)}")
    wins = (daily['day_pnl'] > 0).sum(); losses = (daily['day_pnl'] < 0).sum()
    print(f"  Winning days: {wins}  Losing days: {losses}  Daily WR: {100*wins/len(daily):.1f}%")

    print(f"\nTop 5 winning days:")
    for _, r in daily.nlargest(5, 'day_pnl').iterrows():
        print(f"  {r['date'].date()}  ${r['day_pnl']:>+8,.0f}  {r['trades']}")

    print(f"\nTop 5 losing days:")
    for _, r in daily.nsmallest(5, 'day_pnl').iterrows():
        print(f"  {r['date'].date()}  ${r['day_pnl']:>+8,.0f}  {r['trades']}")

    print(f"\nComparison across Q1 2025 under three exit variants:")
    print(f"  Original undefended N=3 r=$2K: final ${29878}, peak ${51076}, DD ${-21933}")
    print(f"  Defended N=4 r=$3K (target=2R): final ${32836}, peak ${38493}, DD ${-8248}")
    print(f"  Defended N=4 r=$3K (static_lock_1R):  final ${equity:+,.0f}, peak ${peak:+,.0f}, DD ${max_dd:+,.0f}")


if __name__ == '__main__':
    main()

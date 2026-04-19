"""WR analysis: psychological angle on ORB.

Computes:
  1. Overall WR at each pipeline stage (raw → filter → cap+rank → sized)
  2. WR per quintile
  3. WR per variant (V1..V5 exits)
  4. Daily WR and longest losing streaks
  5. Target-reduction sweep: what WR do we get at target=1R, 1.5R, 2R, 2.5R, 3R?
  6. Partial-exit sim: 50% at 1R + rest to 2R
"""
from __future__ import annotations

import os, sys, glob, sqlite3
from datetime import timedelta
from typing import Dict, List, Tuple

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
N_MAX = 4
RISK = 3000
OLD_POS = 50_000.0
MIN_STOP_PCT = 1.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}


def compute_wr(pnls) -> float:
    if len(pnls) == 0:
        return 0.0
    return float((pd.Series(pnls) > 0).mean() * 100)


def longest_losing_streak(pnl_series: pd.Series) -> int:
    """Longest run of consecutive losing days/trades."""
    longest = 0; current = 0
    for p in pnl_series:
        if p < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def simulate_with_target(bars, entry_price, range_high, range_low, entry_time,
                         target_r: float, exit_slip=EXIT_SLIP_BPS_DEFAULT):
    """Simulate single trade with variable target (in R-multiples of range_size)."""
    range_size = range_high - range_low
    target_price = range_high + target_r * range_size
    stop_price = range_low
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        if float(row['low']) <= stop_price:
            return stop_price * (1 - exit_slip/10000), 'stop'
        if float(row['high']) >= target_price:
            return target_price * (1 - exit_slip/10000), 'target'
    last = post.iloc[-1]
    return float(last['close']) * (1 - exit_slip/10000), 'eod'


def simulate_partial(bars, entry_price, range_high, range_low, entry_time,
                     partial_r=1.0, final_r=2.0, partial_frac=0.5,
                     exit_slip=EXIT_SLIP_BPS_DEFAULT):
    """Simulate: close partial_frac at +partial_r, remainder to +final_r target with range_low stop.
    Returns weighted exit price + reason."""
    range_size = range_high - range_low
    partial_price = entry_price + partial_r * range_size
    final_price = range_high + final_r * range_size
    stop_price = range_low
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)

    partial_hit = False
    partial_exit = 0.0
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        # Stop first
        if bar_low <= stop_price:
            # If we already harvested partial, blend; else pure stop
            if partial_hit:
                stop_exit = stop_price * (1 - exit_slip/10000)
                blended = partial_frac * partial_exit + (1 - partial_frac) * stop_exit
                return blended, 'stop_after_partial'
            return stop_price * (1 - exit_slip/10000), 'stop'
        # Partial hit
        if not partial_hit and bar_high >= partial_price:
            partial_exit = partial_price * (1 - exit_slip/10000)
            partial_hit = True
            # After partial, move stop to breakeven on remainder
            stop_price = max(stop_price, entry_price)
        # Final target
        if bar_high >= final_price:
            final_exit = final_price * (1 - exit_slip/10000)
            if partial_hit:
                return partial_frac * partial_exit + (1 - partial_frac) * final_exit, 'target_after_partial'
            return final_exit, 'target'
    # EOD
    last = post.iloc[-1]
    eod_exit = float(last['close']) * (1 - exit_slip/10000)
    if partial_hit:
        return partial_frac * partial_exit + (1 - partial_frac) * eod_exit, 'eod_after_partial'
    return eod_exit, 'eod'


def run_defended_pipeline(df):
    """Apply defended pipeline: filter → cap+dedup → Q5 cap → adaptive mults.
    Returns selected trades with _sized_pnl column."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    per_pos_cap = ACCOUNT / N_MAX
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS

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
        if len(sub) == 0 or avg <= 0:
            mults[q] = 1.0
            continue
        raw = float(sub['_rp_pnl'].mean()) / avg
        mults[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], raw))

    kept = df[df['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)

    sel_rows = []
    for date, dg in kept.groupby('date'):
        d = dg.copy()
        d['_q_rank'] = d['_quintile'].map(Q_ORDER)
        d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        seen_fam = set(); seen_sup = set()
        for _, r in d.iterrows():
            fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_fam: continue
            if sup and sup in seen_sup: continue
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
            sel_rows.append(r)
            if len([r for r in sel_rows if r['date'] == date]) >= N_MAX: break
    sel = pd.DataFrame(sel_rows)
    sel['_sized_pnl'] = sel.apply(lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    sel = sel[(sel['date'] >= '2025-01-01') & (sel['date'] <= '2026-04-30')]
    return sel, mults


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl','date','pnl_pct','range_size_pct','entry_price'])
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df):,} trades")

    # --- Step 1: WR at each pipeline stage ---
    print(f"\n{'='*90}")
    print("WR AT EACH PIPELINE STAGE (defended config N=4, risk=$3K)")
    print(f"{'='*90}")
    print(f"  Stage                                    n     WR      avg pnl_pct    sum pnl ($50K)")
    print('  ' + '-' * 85)
    all_raw = df
    print(f"  {'A. Raw all trades':<38} {len(all_raw):>6} {compute_wr(all_raw['pnl']):>5.1f}%  "
          f"{all_raw['pnl_pct'].mean():>+8.2f}%  ${all_raw['pnl'].sum():>+12,.0f}")

    # After filter (composite z >= 0 with full-period params)
    all_params = fit_z_params(df, FILTER_FEATURES)
    df_c = df.copy()
    df_c['_composite'] = composite_score(df_c, all_params)
    filtered = df_c[df_c['_composite'] >= 0]
    print(f"  {'B. After composite filter (z>=0)':<38} {len(filtered):>6} "
          f"{compute_wr(filtered['pnl']):>5.1f}%  "
          f"{filtered['pnl_pct'].mean():>+8.2f}%  ${filtered['pnl'].sum():>+12,.0f}")

    # Full defended pipeline
    sel, mults = run_defended_pipeline(df)
    print(f"  {'C. After cap+dedup+Q5cap+adaptive':<38} {len(sel):>6} "
          f"{compute_wr(sel['_sized_pnl']):>5.1f}%  "
          f"{sel['pnl_pct'].mean():>+8.2f}%  ${sel['_sized_pnl'].sum():>+12,.0f}")

    print(f"\n  Adaptive mults applied: " +
          " ".join(f"{q}={mults[q]:.2f}x" for q in ['Q1','Q2','Q3','Q4','Q5']))

    # --- Step 2: WR per quintile (test period — defended) ---
    print(f"\n{'='*90}")
    print("WR PER QUINTILE (defended pipeline, selected trades)")
    print(f"{'='*90}")
    print(f"  {'Q':<4} {'n':>5} {'WR':>7} {'avg sized $':>12} {'med sized $':>12}")
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = sel[sel['_quintile'] == q]
        if len(sub) == 0: continue
        print(f"  {q:<4} {len(sub):>5} {compute_wr(sub['_sized_pnl']):>6.1f}%  "
              f"${sub['_sized_pnl'].mean():>+10,.0f}  ${sub['_sized_pnl'].median():>+10,.0f}")

    # --- Step 3: DAILY WR and losing streaks ---
    print(f"\n{'='*90}")
    print("DAILY AGGREGATE WR AND LOSING STREAKS (defended pipeline)")
    print(f"{'='*90}")
    daily = sel.groupby('date')['_sized_pnl'].sum().reset_index()
    daily = daily.sort_values('date')
    daily_wr = compute_wr(daily['_sized_pnl'])
    worst_streak = longest_losing_streak(daily['_sized_pnl'])
    best_streak_wins = longest_losing_streak(-daily['_sized_pnl'])  # flip signs to count winning streaks
    print(f"  Total trading days: {len(daily)}")
    print(f"  Daily WR: {daily_wr:.1f}%  "
          f"({(daily['_sized_pnl']>0).sum()} winning days, "
          f"{(daily['_sized_pnl']<0).sum()} losing days)")
    print(f"  Longest losing streak: {worst_streak} days in a row")
    print(f"  Longest winning streak: {best_streak_wins} days in a row")
    print(f"  Worst losing streak identified:")
    # Find WHERE the longest losing streak happened
    streak_start = None; streak_len = 0; longest_so_far = 0; longest_range = None
    for i, (d, pnl) in enumerate(zip(daily['date'], daily['_sized_pnl'])):
        if pnl < 0:
            if streak_start is None:
                streak_start = d
            streak_len += 1
            if streak_len > longest_so_far:
                longest_so_far = streak_len
                longest_range = (streak_start, d)
        else:
            streak_start = None
            streak_len = 0
    if longest_range:
        print(f"    {longest_range[0].date()} → {longest_range[1].date()}  "
              f"({longest_so_far} consecutive losing days)")

    # Trade-level losing streak
    trade_streak = longest_losing_streak(sel.sort_values('date')['_sized_pnl'])
    print(f"\n  Longest losing TRADE streak: {trade_streak} trades in a row")

    # --- Step 4: target-R sweep (re-simulate with different targets) ---
    print(f"\n{'='*90}")
    print("TARGET-R SWEEP — trade-off between WR and P&L")
    print(f"{'='*90}")
    print(f"  Note: re-simulates all 3,258 trades with each target level.")
    print(f"  Load bars and re-simulate...")

    # Load bars
    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])[:10]),
        axis=1))
    db_obj = Database(db_path='data/cache.db')
    raw_bars = db_obj.get_intraday_bars_bulk(pairs)
    db_obj.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}

    # Pre-compute entry info
    entry_info_map = {}
    for idx, row in df.iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10])
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) & (bars['timestamp'] < range_end)]
        if len(range_bars) < 5:
            continue
        rh = float(range_bars['high'].max())
        rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh:
                entry_ts = b['timestamp']; break
        if entry_ts is None:
            continue
        entry_info_map[idx] = {'bars': bars, 'range_high': rh, 'range_low': rl,
                               'entry_time': entry_ts,
                               'entry_price': float(row['entry_price'])}
    print(f"  Entry info built for {len(entry_info_map)}/{len(df)} trades")

    target_results = []
    for target_r in [1.0, 1.5, 2.0, 2.5, 3.0]:
        # Rebuild df with new pnl for this target
        new_df = df.copy().reset_index(drop=True)
        new_pnls = []; new_pnl_pcts = []
        for idx in range(len(new_df)):
            if idx not in entry_info_map:
                new_pnls.append(df.iloc[idx]['pnl'])
                new_pnl_pcts.append(df.iloc[idx]['pnl_pct'])
                continue
            info = entry_info_map[idx]
            exit_p, _ = simulate_with_target(
                info['bars'], info['entry_price'], info['range_high'],
                info['range_low'], info['entry_time'], target_r)
            entry_p = info['entry_price']
            shares = max(1, int(OLD_POS / entry_p))
            new_pnls.append((exit_p - entry_p) * shares)
            new_pnl_pcts.append((exit_p - entry_p) / entry_p * 100)
        new_df['pnl'] = new_pnls
        new_df['pnl_pct'] = new_pnl_pcts

        # Run through pipeline
        sel_new, _ = run_defended_pipeline(new_df)
        trade_wr = compute_wr(sel_new['_sized_pnl'])
        daily_new = sel_new.groupby('date')['_sized_pnl'].sum().reset_index()
        daily_wr_new = compute_wr(daily_new['_sized_pnl'])
        streak_new = longest_losing_streak(daily_new.sort_values('date')['_sized_pnl'])
        cum = daily_new.sort_values('date')['_sized_pnl'].cumsum().tolist()
        running = -np.inf; dd = 0.0
        for c in cum:
            running = max(running, c)
            dd = min(dd, c - running)
        total_pnl = float(sel_new['_sized_pnl'].sum())
        target_results.append({
            'target_r': target_r, 'n': len(sel_new),
            'trade_wr': trade_wr, 'daily_wr': daily_wr_new,
            'longest_streak': streak_new,
            'total_pnl': total_pnl, 'dd': dd,
            'calmar': total_pnl/abs(dd) if dd < 0 else float('inf'),
        })

    print(f"\n  {'target':>7} {'n':>5} {'trade WR':>9} {'daily WR':>9} "
          f"{'longest streak':>16} {'Full P&L':>12} {'Full DD':>11} {'Calmar':>8}")
    for r in target_results:
        print(f"  {r['target_r']:>6.1f}R {r['n']:>5} {r['trade_wr']:>7.1f}% "
              f"{r['daily_wr']:>7.1f}% {r['longest_streak']:>10} days  "
              f"${r['total_pnl']:>+10,.0f} ${r['dd']:>+9,.0f} {r['calmar']:>6.2f}x")

    # --- Step 5: partial exit ---
    print(f"\n{'='*90}")
    print(f"PARTIAL EXITS: 50% at +1R + 50% runs to +2R (breakeven stop after partial)")
    print(f"{'='*90}")
    new_df = df.copy().reset_index(drop=True)
    new_pnls = []; new_pnl_pcts = []; new_reasons = []
    for idx in range(len(new_df)):
        if idx not in entry_info_map:
            new_pnls.append(df.iloc[idx]['pnl'])
            new_pnl_pcts.append(df.iloc[idx]['pnl_pct'])
            new_reasons.append(df.iloc[idx]['exit_reason'])
            continue
        info = entry_info_map[idx]
        exit_p, reason = simulate_partial(
            info['bars'], info['entry_price'], info['range_high'],
            info['range_low'], info['entry_time'],
            partial_r=1.0, final_r=2.0, partial_frac=0.5)
        entry_p = info['entry_price']
        shares = max(1, int(OLD_POS / entry_p))
        new_pnls.append((exit_p - entry_p) * shares)
        new_pnl_pcts.append((exit_p - entry_p) / entry_p * 100)
        new_reasons.append(reason)
    new_df['pnl'] = new_pnls
    new_df['pnl_pct'] = new_pnl_pcts
    new_df['exit_reason'] = new_reasons
    sel_p, _ = run_defended_pipeline(new_df)
    wr_p = compute_wr(sel_p['_sized_pnl'])
    daily_p = sel_p.groupby('date')['_sized_pnl'].sum().reset_index()
    daily_wr_p = compute_wr(daily_p['_sized_pnl'])
    streak_p = longest_losing_streak(daily_p.sort_values('date')['_sized_pnl'])
    cum = daily_p.sort_values('date')['_sized_pnl'].cumsum().tolist()
    running = -np.inf; dd = 0.0
    for c in cum:
        running = max(running, c)
        dd = min(dd, c - running)
    total_p = float(sel_p['_sized_pnl'].sum())
    calmar_p = total_p/abs(dd) if dd < 0 else float('inf')
    print(f"  n={len(sel_p)}  trade WR={wr_p:.1f}%  daily WR={daily_wr_p:.1f}%  "
          f"longest streak={streak_p} days")
    print(f"  Full P&L=${total_p:+,.0f}  DD=${dd:+,.0f}  Calmar={calmar_p:.2f}x")

    # Distribution of exit_reason under partial
    print(f"\n  Exit reasons (among selected):")
    for r, n in sel_p['exit_reason'].value_counts().items():
        print(f"    {r}: {n}")


if __name__ == '__main__':
    main()

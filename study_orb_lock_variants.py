"""Static-lock-after-1.5R variants for ORB.

User's question: after price touches +1.5R, can we lock the stop at +1.4R?
Or breakeven? Test both.

Two flavors of "lock":
  A. STATIC LOCK: once 1.5R touched, stop pinned at X forever. Runners continue
     to EOD. If price retraces to lock level, exit there.
  B. LOCK + TRAIL: lock floor + trailing component on top (take higher of lock
     and trailing stop).

Variants tested:
  static_lock_1_4R      pin stop at +1.4R  (user's suggestion)
  static_lock_1R        pin stop at +1R
  static_lock_BE        pin stop at entry (breakeven)
  static_lock_0_5R      pin stop at +0.5R (halfway)
  lock_1_4R_and_trail   pin at +1.4R floor, AND trail 0.5R behind high
  lock_BE_and_trail     pin at 0R floor, AND trail 0.5R behind high

Comparison baselines:
  target_1_5R_fixed        user's earlier choice
  trail_2R_0.5R            previous best (Calmar 16.63x)

WR accounting: we track TWO WR flavors:
  wr_strict  = pnl > 0                (breakeven counts as loss)
  wr_loose   = pnl >= -slippage_cost  (breakeven counts as NEITHER win nor loss)
"""
from __future__ import annotations

import os, sys, glob
from datetime import timedelta
from typing import Dict, List, Tuple, Optional

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


def longest_losing_streak(series) -> int:
    longest = 0; current = 0
    for p in series:
        if p < 0:
            current += 1; longest = max(longest, current)
        else:
            current = 0
    return longest


def simulate_lock_variant(
    bars, entry_price, range_high, range_low, entry_time,
    trigger_r: float,       # lock arms when high reaches entry + X × range_size
    lock_r: float,          # stop pinned at entry + X × range_size after arming
    add_trail_r: Optional[float] = None,  # if set, also trail Y × range_size behind high
    target_r: Optional[float] = None,     # optional hard target cap (None = EOD)
    exit_slip=EXIT_SLIP_BPS_DEFAULT,
) -> Tuple[float, str]:
    """Simulate with lock-after-trigger + optional trail. Returns (exit_price, reason)."""
    range_size = range_high - range_low
    trigger_level = entry_price + trigger_r * range_size
    lock_stop = entry_price + lock_r * range_size
    target_price = (range_high + target_r * range_size) if target_r is not None else float('inf')

    stop_price = range_low  # initial: range_low
    trail_high = entry_price
    armed = False

    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])

        if bar_high > trail_high:
            trail_high = bar_high

        # Arm lock when trigger reached
        if not armed and bar_high >= trigger_level:
            armed = True
            stop_price = max(stop_price, lock_stop)

        # If armed AND trailing, update trail
        if armed and add_trail_r is not None:
            new_trail = trail_high - add_trail_r * range_size
            stop_price = max(stop_price, new_trail)

        # Stop check (conservative)
        if bar_low <= stop_price:
            raw = stop_price
            reason = 'lock' if armed and abs(stop_price - lock_stop) < 1e-6 else ('trail' if armed else 'stop')
            return raw * (1 - exit_slip/10000), reason

        # Target (if set)
        if target_r is not None and bar_high >= target_price:
            return target_price * (1 - exit_slip/10000), 'target'

    last = post.iloc[-1]
    return float(last['close']) * (1 - exit_slip/10000), 'eod'


def simulate_fixed_target(bars, entry_price, range_high, range_low, entry_time,
                          target_r, exit_slip=EXIT_SLIP_BPS_DEFAULT):
    range_size = range_high - range_low
    target_price = range_high + target_r * range_size
    stop_price = range_low
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        if float(row['low']) <= stop_price:
            return stop_price * (1 - exit_slip/10000), 'stop'
        if float(row['high']) >= target_price:
            return target_price * (1 - exit_slip/10000), 'target'
    return float(post.iloc[-1]['close']) * (1 - exit_slip/10000), 'eod'


def simulate_trail(bars, entry_price, range_high, range_low, entry_time,
                   activate_r, distance_r, exit_slip=EXIT_SLIP_BPS_DEFAULT):
    range_size = range_high - range_low
    activate_level = entry_price + activate_r * range_size
    stop_price = range_low
    trail_high = entry_price
    armed = False
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        if bar_high > trail_high:
            trail_high = bar_high
        if not armed and bar_high >= activate_level:
            armed = True
        if armed:
            stop_price = max(stop_price, trail_high - distance_r * range_size)
        if bar_low <= stop_price:
            return stop_price * (1 - exit_slip/10000), 'trail' if armed else 'stop'
        # no target cap for trail variants
    return float(post.iloc[-1]['close']) * (1 - exit_slip/10000), 'eod'


def run_defended_pipeline(df):
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
    avg = float(train_k['_rp_pnl'].mean()) if len(train_k) else 1.0
    mults = {}
    for q in ['Q1','Q2','Q3','Q4','Q5']:
        sub = train_k[train_k['_quintile'] == q]
        if len(sub) == 0 or avg <= 0:
            mults[q] = 1.0; continue
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
        picked = 0
        for _, r in d.iterrows():
            fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_fam: continue
            if sup and sup in seen_sup: continue
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
            sel_rows.append(r); picked += 1
            if picked >= N_MAX: break
    sel = pd.DataFrame(sel_rows)
    sel['_sized_pnl'] = sel.apply(lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    sel = sel[(sel['date'] >= '2025-01-01') & (sel['date'] <= '2026-04-30')]
    return sel


def evaluate(entry_info_map, df, name, exit_fn):
    new_df = df.copy().reset_index(drop=True)
    pnls = []; pnl_pcts = []; reasons = []
    for idx in range(len(new_df)):
        if idx not in entry_info_map:
            pnls.append(df.iloc[idx]['pnl']); pnl_pcts.append(df.iloc[idx]['pnl_pct'])
            reasons.append(df.iloc[idx].get('exit_reason', 'eod')); continue
        info = entry_info_map[idx]
        exit_p, reason = exit_fn(info)
        entry_p = info['entry_price']
        shares = max(1, int(OLD_POS / entry_p))
        pnls.append((exit_p - entry_p) * shares)
        pnl_pcts.append((exit_p - entry_p) / entry_p * 100)
        reasons.append(reason)
    new_df['pnl'] = pnls; new_df['pnl_pct'] = pnl_pcts; new_df['exit_reason'] = reasons
    sel = run_defended_pipeline(new_df)

    # WR variants
    strict_wr = float((sel['_sized_pnl'] > 0).mean() * 100) if len(sel) else 0
    # "Loose" = non-loss (>= -small epsilon, accounts for slippage)
    be_threshold = 50.0  # $50 epsilon — accounts for rounding + slippage
    loose_wr = float((sel['_sized_pnl'] >= -be_threshold).mean() * 100) if len(sel) else 0
    near_be = int(((sel['_sized_pnl'] >= -be_threshold) & (sel['_sized_pnl'] < be_threshold)).sum())

    daily = sel.groupby('date')['_sized_pnl'].sum().reset_index()
    daily_wr = float((daily['_sized_pnl'] > 0).mean() * 100) if len(daily) else 0
    streak = longest_losing_streak(daily.sort_values('date')['_sized_pnl'])
    cum = daily.sort_values('date')['_sized_pnl'].cumsum().tolist()
    running = -np.inf; dd = 0.0
    for c in cum: running = max(running, c); dd = min(dd, c - running)
    total = float(sel['_sized_pnl'].sum())
    worst_trade = float(sel['_sized_pnl'].min()) if len(sel) else 0
    worst_day = float(daily['_sized_pnl'].min()) if len(daily) else 0
    calmar = total/abs(dd) if dd < 0 else float('inf')
    exit_counts = sel['exit_reason'].value_counts().to_dict() if 'exit_reason' in sel.columns else {}
    return {
        'name': name, 'n': len(sel),
        'strict_wr': strict_wr, 'loose_wr': loose_wr,
        'near_be': near_be,
        'daily_wr': daily_wr, 'streak': streak,
        'total_pnl': total, 'dd': dd,
        'worst_day': worst_day, 'worst_trade': worst_trade,
        'calmar': calmar, 'exit_counts': exit_counts,
    }


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl','date','pnl_pct','range_size_pct','entry_price'])
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df):,} trades")

    # Load bars
    print("Loading bars...")
    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    db_obj = Database(db_path='data/cache.db')
    raw_bars = db_obj.get_intraday_bars_bulk(pairs)
    db_obj.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}

    entry_info_map = {}
    for idx, row in df.iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty: continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None: continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) & (bars['timestamp'] < range_end)]
        if len(range_bars) < 5: continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh:
                entry_ts = b['timestamp']; break
        if entry_ts is None: continue
        entry_info_map[idx] = {'bars': bars, 'range_high': rh, 'range_low': rl,
                               'entry_time': entry_ts, 'entry_price': float(row['entry_price'])}
    print(f"  Entry info built for {len(entry_info_map)}/{len(df)} trades")

    # Variants
    def make_lock(trigger_r, lock_r, trail_r=None):
        return lambda info: simulate_lock_variant(
            info['bars'], info['entry_price'], info['range_high'],
            info['range_low'], info['entry_time'],
            trigger_r=trigger_r, lock_r=lock_r, add_trail_r=trail_r)

    def make_target(target_r):
        return lambda info: simulate_fixed_target(
            info['bars'], info['entry_price'], info['range_high'],
            info['range_low'], info['entry_time'], target_r=target_r)

    def make_trail(act_r, dist_r):
        return lambda info: simulate_trail(
            info['bars'], info['entry_price'], info['range_high'],
            info['range_low'], info['entry_time'],
            activate_r=act_r, distance_r=dist_r)

    variants = [
        # Baselines for comparison
        ('target_1_5R_fixed',    make_target(1.5)),
        ('target_2R_fixed',      make_target(2.0)),
        ('trail_2R_0.5R (prev winner)', make_trail(2.0, 0.5)),
        # LOCK VARIANTS (what user is asking)
        ('static_lock_1_4R',     make_lock(1.5, 1.4)),   # user's suggestion
        ('static_lock_1R',       make_lock(1.5, 1.0)),
        ('static_lock_BE',       make_lock(1.5, 0.0)),   # breakeven
        ('static_lock_0_5R',     make_lock(1.5, 0.5)),
        # LOCK + TRAIL (floor + runner capture)
        ('lock_1_4R_and_trail_0.5R', make_lock(1.5, 1.4, trail_r=0.5)),
        ('lock_BE_and_trail_0.5R',   make_lock(1.5, 0.0, trail_r=0.5)),
        ('lock_1R_and_trail_0.5R',   make_lock(1.5, 1.0, trail_r=0.5)),
    ]

    print(f"\nRunning {len(variants)} variants...")
    results = []
    for name, fn in variants:
        m = evaluate(entry_info_map, df, name, fn)
        results.append(m)

    print(f"\n{'='*140}")
    print("RESULTS — defended pipeline, Jan'25-Apr'26")
    print(f"{'='*140}")
    print(f"  {'Variant':<32} {'n':>5} {'strict WR':>10} {'loose WR':>9} {'near BE':>8} "
          f"{'daily WR':>9} {'streak':>7} {'P&L':>11} {'DD':>11} {'Calmar':>8}")
    print('  ' + '-' * 135)
    for m in results:
        print(f"  {m['name']:<32} {m['n']:>5} {m['strict_wr']:>8.1f}% {m['loose_wr']:>7.1f}% "
              f"{m['near_be']:>6} {m['daily_wr']:>7.1f}% {m['streak']:>5}d "
              f"${m['total_pnl']:>+9,.0f} ${m['dd']:>+9,.0f} {m['calmar']:>6.2f}x")

    print(f"\nTop 5 by Calmar:")
    for m in sorted(results, key=lambda r: r['calmar'], reverse=True)[:5]:
        print(f"  {m['name']}: Calmar {m['calmar']:.2f}x, P&L ${m['total_pnl']:+,.0f}, "
              f"DD ${m['dd']:+,.0f}, strict WR {m['strict_wr']:.1f}%, "
              f"loose WR {m['loose_wr']:.1f}%, near-BE {m['near_be']}")

    # Exit reason breakdown for top 5
    print(f"\n{'='*120}")
    print("EXIT REASON BREAKDOWN (top 5 by Calmar)")
    print(f"{'='*120}")
    for m in sorted(results, key=lambda r: r['calmar'], reverse=True)[:5]:
        print(f"\n  {m['name']}: n={m['n']}")
        for reason, count in sorted(m['exit_counts'].items(), key=lambda x: -x[1]):
            print(f"    {reason:<25} {count}")


if __name__ == '__main__':
    main()

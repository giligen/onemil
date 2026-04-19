"""Loose-trail variants: wide giveback from peak to survive momentum noise.

User's clarification: after +1.5R touched, allow 1.4R of giveback from the
running high. So if price hits 1.5R and reverses all the way, we exit near
FLAT (+0.1R). If price runs to 5R then pulls back, we exit at 3.6R. Gives
trades MORE ROOM to breathe before stopping.

Variants:
  trail_1_5R_1.4R     arm at +1.5R, trail 1.4R behind high (USER SUGGESTION)
  trail_1_5R_1.2R     arm at +1.5R, trail 1.2R behind high
  trail_1_5R_1.0R     arm at +1.5R, trail 1.0R behind high (tested before: 11.48x)
  trail_2R_1.4R       arm at +2R, trail 1.4R (more room at activation)
  trail_2R_1.2R       arm at +2R, trail 1.2R
  trail_2.5R_1.4R     arm later (+2.5R), trail 1.4R
  trail_3R_1.4R       arm very late (+3R), trail 1.4R

Baselines: trail_2R_0.5R (prev winner), target_1_5R_fixed, static_lock_1R
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
N_MAX = 4
RISK = 3000
OLD_POS = 50_000.0
MIN_STOP_PCT = 1.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}


def longest_losing_streak(series):
    longest = 0; current = 0
    for p in series:
        if p < 0:
            current += 1; longest = max(longest, current)
        else:
            current = 0
    return longest


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
    return float(post.iloc[-1]['close']) * (1 - exit_slip/10000), 'eod'


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

    strict_wr = float((sel['_sized_pnl'] > 0).mean() * 100) if len(sel) else 0
    be_threshold = 50.0
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
        'strict_wr': strict_wr, 'loose_wr': loose_wr, 'near_be': near_be,
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

    def tr(act_r, dist_r):
        return lambda info: simulate_trail(
            info['bars'], info['entry_price'], info['range_high'],
            info['range_low'], info['entry_time'],
            activate_r=act_r, distance_r=dist_r)

    def tg(target_r):
        return lambda info: simulate_fixed_target(
            info['bars'], info['entry_price'], info['range_high'],
            info['range_low'], info['entry_time'], target_r=target_r)

    variants = [
        # Baselines
        ('target_1_5R_fixed',  tg(1.5)),
        ('target_2R_fixed',    tg(2.0)),
        ('trail_2R_0.5R (prev winner)', tr(2.0, 0.5)),
        # USER SUGGESTION: loose trail so we're near-flat if it collapses
        ('trail_1_5R_1.4R (user ask)', tr(1.5, 1.4)),
        # Adjacent variants
        ('trail_1_5R_1.2R',    tr(1.5, 1.2)),
        ('trail_1_5R_1.0R',    tr(1.5, 1.0)),
        ('trail_1_5R_0.7R',    tr(1.5, 0.7)),
        # Also test wider distances from later activations
        ('trail_2R_1.4R',      tr(2.0, 1.4)),
        ('trail_2R_1.2R',      tr(2.0, 1.2)),
        ('trail_2R_1.0R',      tr(2.0, 1.0)),
        ('trail_2R_0.7R',      tr(2.0, 0.7)),
        ('trail_2.5R_1.4R',    tr(2.5, 1.4)),
        ('trail_3R_1.4R',      tr(3.0, 1.4)),
    ]

    print(f"\nRunning {len(variants)} variants...")
    results = []
    for name, fn in variants:
        m = evaluate(entry_info_map, df, name, fn)
        results.append(m)

    print(f"\n{'='*140}")
    print("RESULTS — defended pipeline, Jan'25-Apr'26")
    print(f"{'='*140}")
    print(f"  {'Variant':<34} {'n':>5} {'strict WR':>9} {'loose WR':>9} {'near BE':>7} "
          f"{'daily WR':>9} {'streak':>7} {'P&L':>11} {'DD':>11} {'Calmar':>8}")
    print('  ' + '-' * 135)
    for m in results:
        print(f"  {m['name']:<34} {m['n']:>5} {m['strict_wr']:>7.1f}% {m['loose_wr']:>7.1f}% "
              f"{m['near_be']:>5} {m['daily_wr']:>7.1f}% {m['streak']:>5}d "
              f"${m['total_pnl']:>+9,.0f} ${m['dd']:>+9,.0f} {m['calmar']:>6.2f}x")

    print(f"\nTop 5 by Calmar:")
    for m in sorted(results, key=lambda r: r['calmar'], reverse=True)[:5]:
        print(f"  {m['name']}: Calmar {m['calmar']:.2f}x, P&L ${m['total_pnl']:+,.0f}, "
              f"DD ${m['dd']:+,.0f}, WR {m['strict_wr']:.1f}%, daily {m['daily_wr']:.1f}%")

    print(f"\nTop 5 by P&L:")
    for m in sorted(results, key=lambda r: r['total_pnl'], reverse=True)[:5]:
        print(f"  {m['name']}: P&L ${m['total_pnl']:+,.0f}, Calmar {m['calmar']:.2f}x, "
              f"DD ${m['dd']:+,.0f}, WR {m['strict_wr']:.1f}%")


if __name__ == '__main__':
    main()

"""Trail-after-activation variants for ORB.

Hypothesis: a tight trail activated at +1.5R gives same WR as fixed 1.5R target
(any trade touching 1.5R becomes a win), but CAPTURES RUNNERS instead of capping
them. Compare multiple activation × trail-distance combinations.

Variants:
  target_1_5R      fixed target at 1.5R (baseline from target sweep)
  trail_1R_0.5R    activate at +1R, trail 0.5R behind high
  trail_1_5R_0.3R  activate at +1.5R, trail 0.3R behind high (tight, like MACD wave)
  trail_1_5R_0.5R  activate at +1.5R, trail 0.5R (user's suggestion)
  trail_1_5R_0.8R  activate at +1.5R, trail 0.8R (looser)
  trail_1_5R_1R    activate at +1.5R, trail 1R (like bull flag)
  trail_2R_0.5R    activate at +2R, trail 0.5R (let it run more)
  trail_2R_1R      activate at +2R, trail 1R (bull-flag style)

For each: run through defended pipeline, report trade WR, daily WR, P&L, DD, Calmar.
"""
from __future__ import annotations

import os, sys, glob
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


def longest_losing_streak(series) -> int:
    longest = 0; current = 0
    for p in series:
        if p < 0:
            current += 1; longest = max(longest, current)
        else:
            current = 0
    return longest


def simulate_with_trail(
    bars, entry_price, range_high, range_low, entry_time,
    target_r: float,              # fixed target (None = no cap)
    trail_activate_r: float,      # activate trail when high reaches entry + X × range_size
    trail_distance_r: float,      # trail stop X × range_size below running high
    exit_slip=EXIT_SLIP_BPS_DEFAULT,
) -> Tuple[float, str]:
    """Simulate with activate-trail-at-X-R, trail-Y-R-below. Returns (exit_price, reason)."""
    range_size = range_high - range_low
    activate_level = entry_price + trail_activate_r * range_size
    target_price = (range_high + target_r * range_size) if target_r is not None else float('inf')

    stop_price = range_low
    trail_high = entry_price
    trail_armed = False

    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])

        # Update running high
        if bar_high > trail_high:
            trail_high = bar_high

        # Arm trail when price first reaches activation level
        if not trail_armed and bar_high >= activate_level:
            trail_armed = True

        # Update trailing stop if armed
        if trail_armed:
            new_stop = trail_high - trail_distance_r * range_size
            stop_price = max(stop_price, new_stop)

        # Stop first (conservative)
        if bar_low <= stop_price:
            raw = stop_price
            reason = 'trail_stop' if trail_armed else 'stop'
            return raw * (1 - exit_slip/10000), reason

        # Target check (if target enabled)
        if target_r is not None and bar_high >= target_price:
            return target_price * (1 - exit_slip/10000), 'target'

    # EOD
    last = post.iloc[-1]
    return float(last['close']) * (1 - exit_slip/10000), 'eod'


def run_defended_pipeline(df):
    """Full defended pipeline. Returns selected trades with _sized_pnl."""
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
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
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
        picked_for_day = 0
        for _, r in d.iterrows():
            fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_fam: continue
            if sup and sup in seen_sup: continue
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
            sel_rows.append(r)
            picked_for_day += 1
            if picked_for_day >= N_MAX: break
    sel = pd.DataFrame(sel_rows)
    sel['_sized_pnl'] = sel.apply(lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    sel = sel[(sel['date'] >= '2025-01-01') & (sel['date'] <= '2026-04-30')]
    return sel


def evaluate_variant(entry_info_map, df, target_r, activate_r, trail_r, variant_name):
    """Re-simulate all trades with given exit rule, run defended pipeline, return metrics."""
    new_df = df.copy().reset_index(drop=True)
    new_pnls = []
    new_pnl_pcts = []
    new_reasons = []
    for idx in range(len(new_df)):
        if idx not in entry_info_map:
            new_pnls.append(df.iloc[idx]['pnl'])
            new_pnl_pcts.append(df.iloc[idx]['pnl_pct'])
            new_reasons.append(df.iloc[idx].get('exit_reason', 'eod'))
            continue
        info = entry_info_map[idx]
        exit_p, reason = simulate_with_trail(
            info['bars'], info['entry_price'], info['range_high'],
            info['range_low'], info['entry_time'],
            target_r=target_r, trail_activate_r=activate_r,
            trail_distance_r=trail_r)
        entry_p = info['entry_price']
        shares = max(1, int(OLD_POS / entry_p))
        new_pnls.append((exit_p - entry_p) * shares)
        new_pnl_pcts.append((exit_p - entry_p) / entry_p * 100)
        new_reasons.append(reason)
    new_df['pnl'] = new_pnls
    new_df['pnl_pct'] = new_pnl_pcts
    new_df['exit_reason'] = new_reasons

    sel = run_defended_pipeline(new_df)
    trade_wr = float((sel['_sized_pnl'] > 0).mean() * 100) if len(sel) else 0
    daily = sel.groupby('date')['_sized_pnl'].sum().reset_index()
    daily_wr = float((daily['_sized_pnl'] > 0).mean() * 100) if len(daily) else 0
    streak = longest_losing_streak(daily.sort_values('date')['_sized_pnl'])
    cum = daily.sort_values('date')['_sized_pnl'].cumsum().tolist()
    running = -np.inf; dd = 0.0
    for c in cum:
        running = max(running, c)
        dd = min(dd, c - running)
    total = float(sel['_sized_pnl'].sum())
    worst_trade = float(sel['_sized_pnl'].min()) if len(sel) else 0
    worst_day = float(daily['_sized_pnl'].min()) if len(daily) else 0
    calmar = total/abs(dd) if dd < 0 else float('inf')
    exit_counts = sel['exit_reason'].value_counts().to_dict() if 'exit_reason' in sel.columns else {}
    return {
        'name': variant_name,
        'n': len(sel),
        'trade_wr': trade_wr,
        'daily_wr': daily_wr,
        'longest_streak': streak,
        'total_pnl': total,
        'dd': dd,
        'worst_day': worst_day,
        'worst_trade': worst_trade,
        'calmar': calmar,
        'exit_counts': exit_counts,
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

    # Pre-compute entry info
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
        rh = float(range_bars['high'].max())
        rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh:
                entry_ts = b['timestamp']; break
        if entry_ts is None: continue
        entry_info_map[idx] = {'bars': bars, 'range_high': rh, 'range_low': rl,
                               'entry_time': entry_ts,
                               'entry_price': float(row['entry_price'])}
    print(f"  Entry info built for {len(entry_info_map)}/{len(df)} trades")

    # Variants — (name, target_r, activate_r, trail_r)
    # target_r=None means no hard target cap (let trail run indefinitely)
    variants = [
        ('target_1_5R_fixed',      1.5, 999,  999),   # baseline: fixed 1.5R target
        ('target_2R_fixed',        2.0, 999,  999),   # original 2R
        ('target_3R_fixed',        3.0, 999,  999),   # from target sweep
        ('trail_1R_0.5R',          None, 1.0, 0.5),   # arm at 1R, tight 0.5R trail
        ('trail_1_5R_0.3R',        None, 1.5, 0.3),   # arm at 1.5R, very tight 0.3R trail
        ('trail_1_5R_0.5R',        None, 1.5, 0.5),   # USER SUGGESTION
        ('trail_1_5R_0.8R',        None, 1.5, 0.8),   # arm at 1.5R, looser 0.8R
        ('trail_1_5R_1R',          None, 1.5, 1.0),   # arm at 1.5R, bull-flag-style 1R
        ('trail_2R_0.5R',          None, 2.0, 0.5),   # arm at 2R, tight
        ('trail_2R_1R',            None, 2.0, 1.0),   # bull flag style
    ]

    print(f"\nRunning {len(variants)} variants through defended pipeline...")
    results = []
    for name, tr, ar, tl in variants:
        m = evaluate_variant(entry_info_map, df, tr, ar, tl, name)
        results.append(m)
        print(f"  {name}: WR {m['trade_wr']:.1f}%, daily WR {m['daily_wr']:.1f}%, "
              f"P&L ${m['total_pnl']:+,.0f}, DD ${m['dd']:+,.0f}, Calmar {m['calmar']:.2f}x")

    # Table
    print(f"\n{'='*130}")
    print("RESULTS — full defended pipeline, Jan'25-Apr'26")
    print(f"{'='*130}")
    print(f"  {'Variant':<22} {'n':>5} {'trade WR':>9} {'daily WR':>9} {'streak':>7} "
          f"{'P&L':>11} {'DD':>11} {'Worst day':>11} {'Worst tr':>11} {'Calmar':>8}")
    print('  ' + '-' * 125)
    for m in results:
        print(f"  {m['name']:<22} {m['n']:>5} {m['trade_wr']:>7.1f}% {m['daily_wr']:>7.1f}% "
              f"{m['longest_streak']:>5}d ${m['total_pnl']:>+9,.0f} ${m['dd']:>+9,.0f} "
              f"${m['worst_day']:>+9,.0f} ${m['worst_trade']:>+9,.0f} {m['calmar']:>6.2f}x")

    # Top 3 by Calmar
    print(f"\nTop 3 by Calmar:")
    for m in sorted(results, key=lambda r: r['calmar'], reverse=True)[:3]:
        print(f"  {m['name']}: Calmar {m['calmar']:.2f}x  "
              f"(P&L ${m['total_pnl']:+,.0f}, DD ${m['dd']:+,.0f}, "
              f"WR {m['trade_wr']:.1f}%, daily {m['daily_wr']:.1f}%)")

    # Exit reason breakdown for top variants
    print(f"\n{'='*130}")
    print("EXIT REASON BREAKDOWN (top 5 variants)")
    print(f"{'='*130}")
    for m in sorted(results, key=lambda r: r['calmar'], reverse=True)[:5]:
        print(f"\n  {m['name']}: n={m['n']}")
        for reason, count in sorted(m['exit_counts'].items(), key=lambda x: -x[1]):
            print(f"    {reason:<25} {count}")


if __name__ == '__main__':
    main()

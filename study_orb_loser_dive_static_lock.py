"""Deep dive on Aug 2025 — the only negative month under shipped static_lock_1R.

Uses the CORRECT trade set from orb_static_lock_trades.csv (produced by
orb_monthly_static_lock.py). Per-losing-trade MFE/MAE analysis with
alt-exit simulations to identify salvageable trades.
"""
from __future__ import annotations

import sys, os, glob
from datetime import timedelta
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database
from study_orb import _bars_to_df


LOCK_TRIGGER_R = 1.5
LOCK_STOP_R = 1.0


def _session_open_timestamp(bars):
    if bars.empty:
        return None
    b = bars.copy()
    b['et'] = b['timestamp'].dt.tz_convert('America/New_York')
    m = b[b['et'].dt.time == pd.Timestamp('09:30').time()]
    return m.iloc[0]['timestamp'] if len(m) else None


def replay_one(bars, entry_price, range_high, range_low, entry_ts):
    one_r = range_high - range_low
    if one_r <= 0:
        return None
    # Walk POST-entry bars (skip entry bar — BT parity)
    post = bars[bars['timestamp'] >= entry_ts].reset_index(drop=True)
    if len(post) < 2:
        return None
    mfe = 0.0; mae = 0.0; peak_minute = 0
    armed = False
    stop = range_low
    actual_pnl = None; actual_reason = None; actual_exit_minute = None

    for m, (_, b) in enumerate(post.iloc[1:].iterrows(), start=1):
        high = float(b['high']); low = float(b['low']); close = float(b['close'])
        fav = (high - entry_price) / one_r
        adv = (low - entry_price) / one_r
        if fav > mfe:
            mfe = fav; peak_minute = m
        if adv < mae:
            mae = adv
        if not armed and high >= entry_price + LOCK_TRIGGER_R * one_r:
            armed = True; stop = max(stop, entry_price + LOCK_STOP_R * one_r)
        if low <= stop:
            actual_pnl = (stop - entry_price)
            actual_reason = 'lock' if armed else 'stop'
            actual_exit_minute = m
            break
    if actual_pnl is None:
        last = post.iloc[-1]
        actual_pnl = float(last['close']) - entry_price
        actual_reason = 'eod'
        actual_exit_minute = len(post) - 1

    # Alt-exits (simulated fresh)
    def sim_be(arm_R):
        st = range_low; be_armed = False; lock_armed = False
        for _, b in post.iloc[1:].iterrows():
            high, low = float(b['high']), float(b['low'])
            if not be_armed and high >= entry_price + arm_R * one_r:
                be_armed = True; st = max(st, entry_price)
            if not lock_armed and high >= entry_price + LOCK_TRIGGER_R * one_r:
                lock_armed = True; st = max(st, entry_price + LOCK_STOP_R * one_r)
            if low <= st:
                return st - entry_price
        return float(post.iloc[-1]['close']) - entry_price

    def sim_trail_after_1R(trail_r):
        """After MFE >= 1R, trail stop at trail_r below running peak."""
        st = range_low; peak = entry_price; trail_armed = False
        for _, b in post.iloc[1:].iterrows():
            high, low = float(b['high']), float(b['low'])
            if high > peak: peak = high
            if not trail_armed and high >= entry_price + 1.0 * one_r:
                trail_armed = True
            if trail_armed:
                st = max(st, peak - trail_r * one_r)
            if low <= st:
                return st - entry_price
        return float(post.iloc[-1]['close']) - entry_price

    def sim_time_exit(at_minute):
        st = range_low; lock_armed = False
        for m, (_, b) in enumerate(post.iloc[1:].iterrows(), start=1):
            high, low, close = float(b['high']), float(b['low']), float(b['close'])
            if not lock_armed and high >= entry_price + LOCK_TRIGGER_R * one_r:
                lock_armed = True; st = max(st, entry_price + LOCK_STOP_R * one_r)
            if low <= st:
                return st - entry_price
            if m >= at_minute:
                return close - entry_price
        return float(post.iloc[-1]['close']) - entry_price

    return {
        'one_R$': one_r,
        'mfe_R': mfe, 'mae_R': mae, 'peak_minute': peak_minute,
        'actual_R': actual_pnl / one_r,
        'actual_reason': actual_reason,
        'actual_exit_minute': actual_exit_minute,
        'be_0_5R_R': sim_be(0.5) / one_r,
        'be_1R_R': sim_be(1.0) / one_r,
        'trail_0_5_after_1R_R': sim_trail_after_1R(0.5) / one_r,
        'trail_1R_after_1R_R':  sim_trail_after_1R(1.0) / one_r,
        'time_30_R': sim_time_exit(30) / one_r,
        'time_60_R': sim_time_exit(60) / one_r,
        'time_120_R': sim_time_exit(120) / one_r,
    }


def main():
    sl_csv = 'analysis_results/orb_static_lock_trades.csv'
    if not os.path.exists(sl_csv):
        print("Run orb_monthly_static_lock.py first."); return
    trades = pd.read_csv(sl_csv)
    trades['date'] = pd.to_datetime(trades['date'])
    # Aug 2025
    aug = trades[(trades['date'] >= '2025-08-01') & (trades['date'] <= '2025-08-31')].copy()
    print(f"Aug 2025 picks: {len(aug)}   P&L ${aug['_sized_pnl'].sum():+,.0f}")
    losers = aug[aug['_sized_pnl'] < 0].copy()
    winners = aug[aug['_sized_pnl'] > 0].copy()
    print(f"  winners {len(winners)}  ({winners['_sized_pnl'].sum():+,.0f})")
    print(f"  losers  {len(losers)}  ({losers['_sized_pnl'].sum():+,.0f})")

    # Load bars for Aug trades
    db = Database(db_path='data/cache.db')
    pairs = [(r['symbol'], r['date'].strftime('%Y-%m-%d')) for _, r in aug.iterrows()]
    raw = db.get_intraday_bars_bulk(pairs)
    db.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw.items()}

    # Replay each loser
    rows = []
    for _, t in losers.iterrows():
        bars = bars_cache.get((t['symbol'], t['date'].strftime('%Y-%m-%d')))
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
        r = replay_one(bars, float(t['entry_price']), rh, rl, entry_ts)
        if r is None: continue
        rows.append({
            'date': t['date'].strftime('%Y-%m-%d'),
            'symbol': t['symbol'], 'q': t['_quintile'],
            'sized_pnl': float(t['_sized_pnl']),
            **r,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        print("no replay data"); return

    # Daily clustering (which day carried the pain?)
    print(f"\n{'='*60}")
    print("Aug 2025 — daily P&L breakdown")
    print(f"{'='*60}")
    daily = aug.groupby('date')['_sized_pnl'].agg(['sum', 'count']).reset_index()
    daily.columns = ['date', 'pnl', 'picks']
    daily = daily.sort_values('pnl')
    pd.set_option('display.float_format', '{:,.0f}'.format)
    pd.set_option('display.width', 180)
    print(daily.to_string(index=False))

    # Symbol repeats
    print(f"\n{'='*60}")
    print("Repeat-loser symbols (2+ losses in Aug)")
    print(f"{'='*60}")
    sym_stats = losers.groupby('symbol')['_sized_pnl'].agg(['sum', 'count']).sort_values('sum')
    sym_stats.columns = ['sum_pnl', 'n']
    print(sym_stats[sym_stats['n'] >= 2].to_string())

    # Bucket losers by MFE
    def bucket(m):
        if m < 0.25: return 'doomed(<0.25R)'
        if m < 0.75: return 'near_miss(0.25-0.75R)'
        if m < 1.5:  return 'runner_then_retraced(0.75-1.5R)'
        return 'should_have_locked(>=1.5R)'
    out['bucket'] = out['mfe_R'].apply(bucket)

    print(f"\n{'='*90}")
    print("Aug 2025 losers — MFE buckets")
    print(f"{'='*90}")
    g = out.groupby('bucket').agg(
        n=('sized_pnl', 'count'),
        total_loss=('sized_pnl', 'sum'),
        avg_mfe_R=('mfe_R', 'mean'),
        avg_peak_min=('peak_minute', 'mean'),
        avg_actual_R=('actual_R', 'mean'),
    )
    print(g.to_string())

    # Alt-exit $ savings — scale R-pnl by actual ratio
    out['$_per_R'] = out.apply(
        lambda r: r['sized_pnl']/r['actual_R'] if abs(r['actual_R']) > 1e-6 else 0.0, axis=1)
    alt_cols = {
        'be_0_5R_R': 'breakeven@+0.5R',
        'be_1R_R':   'breakeven@+1R',
        'trail_0_5_after_1R_R': 'trail 0.5R after +1R',
        'trail_1R_after_1R_R':  'trail 1.0R after +1R',
        'time_30_R': 'time-exit 30min',
        'time_60_R': 'time-exit 60min',
        'time_120_R': 'time-exit 120min',
    }
    print(f"\n{'='*100}")
    print("$ on Aug losers — shipped (static_lock) vs alt-exits")
    print(f"{'='*100}")
    actual_total = out['sized_pnl'].sum()
    print(f"Actual total loss: ${actual_total:+,.0f}\n")
    print(f"{'Alt exit':<30} {'Total $':>14} {'Δ':>12} {'avg/trade':>12}")
    print('-' * 75)
    for col, label in alt_cols.items():
        out[col + '_$'] = out[col] * out['$_per_R']
        s = out[col + '_$'].sum()
        print(f"{label:<30} ${s:>+13,.0f} ${s-actual_total:>+10,.0f} ${out[col + '_$'].mean():>+10,.0f}")

    # Full-month impact: apply best alt to Aug losers AND winners (to verify
    # we don't cap winners with the same rule)
    print(f"\n{'='*100}")
    print("Rule impact on ALL Aug trades (losers + winners)")
    print(f"{'='*100}")
    # Replay winners too for honest comparison
    rows_w = []
    for _, t in winners.iterrows():
        bars = bars_cache.get((t['symbol'], t['date'].strftime('%Y-%m-%d')))
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
        r = replay_one(bars, float(t['entry_price']), rh, rl, entry_ts)
        if r is None: continue
        rows_w.append({
            'sized_pnl': float(t['_sized_pnl']),
            'actual_R': r['actual_R'],
            **{k: r[k] for k in alt_cols.keys()},
        })
    wdf = pd.DataFrame(rows_w)
    all_df = pd.concat([
        out[['sized_pnl', 'actual_R'] + list(alt_cols.keys())],
        wdf,
    ], ignore_index=True) if len(wdf) else out[['sized_pnl', 'actual_R'] + list(alt_cols.keys())]
    all_df['$_per_R'] = all_df.apply(
        lambda r: r['sized_pnl']/r['actual_R'] if abs(r['actual_R']) > 1e-6 else 0.0, axis=1)

    all_actual = all_df['sized_pnl'].sum()
    print(f"Aug static_lock month P&L:  ${all_actual:+,.0f}\n")
    print(f"{'Rule':<30} {'Aug P&L':>14} {'Δ vs static_lock':>18}")
    print('-' * 70)
    for col, label in alt_cols.items():
        s = (all_df[col] * all_df['$_per_R']).sum()
        print(f"{label:<30} ${s:>+13,.0f} ${s - all_actual:>+17,.0f}")

    out.to_csv('analysis_results/orb_aug_2025_dive.csv', index=False)
    print("\nSaved: analysis_results/orb_aug_2025_dive.csv")


if __name__ == '__main__':
    main()

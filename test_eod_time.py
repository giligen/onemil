"""Test 15:45 ET vs 15:59 ET (default EOD) force-close under static_lock_1R."""
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

ACCOUNT = 100_000; N = 4; RISK = 3000; OLD_POS = 50_000
MIN_STOP_PCT = 1.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}


def is_force_close_bar(ts, hour_et, minute_et):
    """ET 15:45 → UTC 19:45 (EDT) or 20:45 (EST)."""
    # Convert ET to UTC offset: EDT = -4, EST = -5
    # Mar-Nov: EDT → UTC = ET+4. Nov-Mar: EST → UTC = ET+5.
    if hour_et == 15 and minute_et == 45:
        # EDT: 19:45 UTC; EST: 20:45 UTC
        return (ts.hour == 19 and ts.minute == 45) or (ts.hour == 20 and ts.minute == 45)
    if hour_et == 16 and minute_et == 0:
        return (ts.hour == 20 and ts.minute == 0) or (ts.hour == 21 and ts.minute == 0)
    return False


def simulate_static_lock_with_fc(bars, entry_price, range_high, range_low, entry_time,
                                  force_close_et=None, exit_slip=EXIT_SLIP_BPS_DEFAULT):
    """static_lock_1R with optional force-close at specific ET time."""
    range_size = range_high - range_low
    trigger_lvl = entry_price + 1.5 * range_size
    lock_stop = entry_price + 1.0 * range_size
    stop_price = range_low
    armed = False
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        ts = row['timestamp']
        if not armed and bar_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        if bar_low <= stop_price:
            return stop_price * (1 - exit_slip/10000), 'lock' if armed else 'stop'
        # Force close check
        if force_close_et is not None:
            hour, minute = force_close_et
            if is_force_close_bar(ts, hour, minute):
                return float(row['close']) * (1 - exit_slip/10000), 'force_close'
    last = post.iloc[-1]
    return float(last['close']) * (1 - exit_slip/10000), 'eod'


def run_pipeline_with_exit(df, exit_fn):
    """Run full defended pipeline with a custom exit function."""
    df = df.copy()
    # Re-simulate P&L with exit_fn
    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date'])[:10]),
        axis=1))
    db_obj = Database(db_path='data/cache.db')
    raw_bars = db_obj.get_intraday_bars_bulk(pairs)
    db_obj.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}

    new_pnls = []; new_reasons = []
    for idx, row in df.reset_index(drop=True).iterrows():
        d_str = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])[:10]
        key = (row['symbol'], d_str)
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            new_pnls.append(row['pnl']); new_reasons.append(row['exit_reason']); continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            new_pnls.append(row['pnl']); new_reasons.append(row['exit_reason']); continue
        range_end = open_ts + timedelta(minutes=5)
        rb = bars[(bars['timestamp'] >= open_ts) & (bars['timestamp'] < range_end)]
        if len(rb) < 5:
            new_pnls.append(row['pnl']); new_reasons.append(row['exit_reason']); continue
        rh = float(rb['high'].max()); rl = float(rb['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh:
                entry_ts = b['timestamp']; break
        if entry_ts is None:
            new_pnls.append(row['pnl']); new_reasons.append(row['exit_reason']); continue
        entry_p = float(row['entry_price'])
        exit_p, reason = exit_fn(bars, entry_p, rh, rl, entry_ts)
        shares = max(1, int(OLD_POS / entry_p))
        new_pnls.append((exit_p - entry_p) * shares)
        new_reasons.append(reason)
    df = df.reset_index(drop=True)
    df['pnl'] = new_pnls
    df['exit_reason'] = new_reasons
    df['date'] = pd.to_datetime(df['date'])

    # Apply pipeline
    per_pos_cap = ACCOUNT / N
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
        mults[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], float(sub['_rp_pnl'].mean()) / avg))
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
            if picked >= N: break
    sel = pd.DataFrame(sel_rows)
    sel['_sized_pnl'] = sel.apply(lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    sel = sel[(sel['date'] >= '2025-01-01') & (sel['date'] <= '2026-04-30')]

    strict_wr = float((sel['_sized_pnl'] > 0).mean() * 100) if len(sel) else 0
    daily = sel.groupby('date')['_sized_pnl'].sum().reset_index()
    daily_wr = float((daily['_sized_pnl'] > 0).mean() * 100) if len(daily) else 0
    cum = daily.sort_values('date')['_sized_pnl'].cumsum().tolist()
    running = -np.inf; dd = 0.0
    for c in cum: running = max(running, c); dd = min(dd, c - running)
    total = float(sel['_sized_pnl'].sum())
    exit_counts = sel['exit_reason'].value_counts().to_dict()
    return {
        'total_pnl': total, 'dd': dd, 'calmar': total/abs(dd) if dd < 0 else float('inf'),
        'strict_wr': strict_wr, 'daily_wr': daily_wr,
        'worst_day': float(daily['_sized_pnl'].min()) if len(daily) else 0,
        'worst_trade': float(sel['_sized_pnl'].min()) if len(sel) else 0,
        'n_trades': len(sel), 'exit_counts': exit_counts,
    }


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl','date','pnl_pct','range_size_pct','entry_price'])

    # Test 3 close times
    print("Testing static_lock_1R with different force-close times:")
    print("=" * 90)
    variants = [
        ('hold to EOD (last bar, ~15:59)',
            lambda b, e, rh, rl, et: simulate_static_lock_with_fc(b, e, rh, rl, et, force_close_et=None)),
        ('force close 15:45 ET (match production)',
            lambda b, e, rh, rl, et: simulate_static_lock_with_fc(b, e, rh, rl, et, force_close_et=(15, 45))),
        ('force close 16:00 ET (last bar)',
            lambda b, e, rh, rl, et: simulate_static_lock_with_fc(b, e, rh, rl, et, force_close_et=(16, 0))),
    ]
    results = []
    for name, fn in variants:
        m = run_pipeline_with_exit(df, fn)
        results.append((name, m))

    print(f"\n  {'Close time':<40} {'P&L':>10} {'DD':>10} {'Calmar':>8} {'WR':>7} {'dWR':>7}")
    print('  ' + '-' * 85)
    for name, m in results:
        print(f"  {name:<40} ${m['total_pnl']:>+8,.0f} ${m['dd']:>+8,.0f} "
              f"{m['calmar']:>6.2f}x {m['strict_wr']:>5.1f}% {m['daily_wr']:>5.1f}%")

    for name, m in results:
        print(f"\n  {name} — exit breakdown:")
        for r, c in sorted(m['exit_counts'].items(), key=lambda x: -x[1]):
            print(f"    {r:<15} {c}")


if __name__ == '__main__':
    main()

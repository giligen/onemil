"""Q1 2025 day-by-day under RECOMMENDED defended config.

⚠️  NOT PRODUCTION-PARITY — see study_orb_100k_defended.py header.
Uses fixed +2R target / -1R stop. Shipped exit is static_lock_1R.
For production-parity Q1 2025: use show_q1_2025_static_lock.py.

Config: N=4, risk=$3K, Q5 cap 1.5x, family+super-group dedup, $100K account.
Per-position cap = $25K.

Shows equity curve through Q1 2025 so the user can eyeball what happens
day-by-day with all defenses active. Q1 2025 is IN TRAIN for all 3 walk-forward
splits, but the PURPOSE here is mechanics visualization, not walk-forward test.
"""
from __future__ import annotations

import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN,
    fit_quintile_cutoffs, assign_quintile,
)
from study_orb_correlation_filter import (
    FAMILIES, symbol_family, symbol_super_group,
)

# RECOMMENDED DEFENDED CONFIG
ACCOUNT = 100_000
N = 4
RISK = 3000
OLD_POS = 50_000
MIN_STOP_PCT = 1.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}  # Q5 cap at 1.5x


def apply_rp(df, risk, per_pos_cap):
    df = df.copy()
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = risk / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS
    return df


def fit_mults_capped(tk):
    avg = float(tk['_rp_pnl'].mean())
    out = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = tk[tk['_quintile'] == q]
        if len(sub) == 0 or avg <= 0:
            out[q] = 1.0
            continue
        raw = float(sub['_rp_pnl'].mean()) / avg
        out[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], raw))
    return out


def select_defended(dg, k):
    d = dg.copy()
    d['_q_rank'] = d['_quintile'].map(Q_ORDER)
    d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
    seen_fam = set()
    seen_sup = set()
    kept = []
    for _, r in d.iterrows():
        sym = r['symbol']
        fam = symbol_family(sym)
        sup = symbol_super_group(sym)
        if fam and fam in seen_fam: continue
        if sup and sup in seen_sup: continue
        if fam: seen_fam.add(fam)
        if sup: seen_sup.add(sup)
        kept.append(r)
        if len(kept) >= k: break
    return pd.DataFrame(kept)


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])

    per_pos_cap = ACCOUNT / N
    df = apply_rp(df, RISK, per_pos_cap)

    # Train on H1 2025
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    train_k = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    mults = fit_mults_capped(train_k)

    # Apply to Q1 2025
    q1 = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-03-31')]
    q1_k = q1[q1['_composite'] >= FILTER_THRESHOLD].copy()
    q1_k['_quintile'] = assign_quintile(q1_k['_composite'], cutoffs)

    print(f"{'='*125}")
    print(f"Q1 2025 — DEFENDED CONFIG: N={N}, risk=${RISK:,}, per-pos cap ${per_pos_cap:,.0f}")
    print(f"Defenses: Q5 mult cap 1.5x + family dedup + super-group dedup")
    print(f"{'='*125}")
    print(f"Mults (Q5 capped at 1.5x): " +
          " ".join(f"{q}={mults[q]:.2f}x" for q in ['Q1','Q2','Q3','Q4','Q5']))
    print(f"\n{'date':<12} {'wd':<4} {'sig':>4} {'pick':>4} {'trades':<58} "
          f"{'day P&L':>10} {'equity':>10} {'DD':>9}")
    print('-' * 125)

    equity = 0.0; peak = 0.0; max_dd = 0.0
    daily_rows = []
    for day in sorted(q1_k['date'].unique()):
        dg = q1_k[q1_k['date'] == day]
        n_sig = len(dg)
        picked = select_defended(dg, N).copy()
        picked['_sized_pnl'] = picked.apply(
            lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)

        parts = []
        for _, r in picked.iterrows():
            sup = symbol_super_group(r['symbol'])
            fam = symbol_family(r['symbol'])
            tag = ''
            if sup == 'lev_short': tag = '[SHORT]'
            elif sup == 'lev_long': tag = '[LONG]'
            elif fam: tag = f"[{fam[:4]}]"
            arrow = '✓' if r['_sized_pnl'] > 0 else '✗'
            parts.append(f"{r['symbol']}{tag}({r['_quintile']}){arrow}${r['_sized_pnl']:+,.0f}")
        trade_str = ', '.join(parts)
        if len(trade_str) > 56: trade_str = trade_str[:53] + '…'

        day_pnl = float(picked['_sized_pnl'].sum())
        equity += day_pnl
        peak = max(peak, equity)
        dd_now = equity - peak
        max_dd = min(max_dd, dd_now)
        daily_rows.append({'date': day, 'day_pnl': day_pnl, 'equity': equity,
                          'dd': dd_now, 'n_sig': n_sig, 'n_pick': len(picked),
                          'trades': trade_str})
        wd = day.strftime('%a')
        print(f"{day.date().isoformat():<12} {wd:<4} {n_sig:>4} {len(picked):>4} "
              f"{trade_str:<58} ${day_pnl:>+8,.0f} ${equity:>+8,.0f} ${dd_now:>+7,.0f}")

    daily = pd.DataFrame(daily_rows)
    print(f"\n{'='*125}")
    print(f"Q1 2025 SUMMARY — defended config")
    print(f"{'='*125}")
    print(f"  Final equity: ${equity:+,.0f}")
    print(f"  Peak equity:  ${peak:+,.0f}")
    print(f"  Max DD:       ${max_dd:+,.0f}")
    print(f"  Trading days: {len(daily)}")
    print(f"  Winning days: {(daily['day_pnl'] > 0).sum()}  "
          f"Losing days: {(daily['day_pnl'] < 0).sum()}  "
          f"Flat: {(daily['day_pnl'] == 0).sum()}")

    print(f"\n  Top 5 winning days:")
    for _, r in daily.nlargest(5, 'day_pnl').iterrows():
        print(f"    {r['date'].date()}  ${r['day_pnl']:>+8,.0f}  picks: {r['trades']}")

    print(f"\n  Top 5 losing days:")
    for _, r in daily.nsmallest(5, 'day_pnl').iterrows():
        print(f"    {r['date'].date()}  ${r['day_pnl']:>+8,.0f}  picks: {r['trades']}")

    # Comparison vs BEFORE defenses (from prior run: Q1 final $+29,878, max DD $-21,933)
    print(f"\n{'='*125}")
    print(f"COMPARISON vs undefended N=3 r=$2K (original)")
    print(f"{'='*125}")
    print(f"  {'Metric':<25} {'Undefended (N=3, $2K)':>25} {'Defended (N=4, $3K)':>25} {'Change':>15}")
    print('  ' + '-' * 90)
    print(f"  {'Final equity':<25} {'$+29,878':>25} ${equity:>+24,.0f} ${equity-29878:>+13,.0f}")
    print(f"  {'Peak equity':<25} {'$+51,076':>25} ${peak:>+24,.0f} ${peak-51076:>+13,.0f}")
    print(f"  {'Max DD':<25} {'$-21,933':>25} ${max_dd:>+24,.0f} ${max_dd-(-21933):>+13,.0f}")

    # Show worst days under old vs new
    print(f"\n  Worst days under old config (N=3 r=$2K, no defenses):")
    print(f"    2025-03-06  $-6,831  (UVXY+TSLZ+SMST — vol/short cluster)")
    print(f"    2025-03-18  $-7,453  (MSTZ+SMST — short leveraged)")
    print(f"    2025-03-19  $-7,145  (SPIR+VG — uncorrelated but both lost)")
    print(f"\n  Worst days under defended config (from trace above):")
    # Pull from daily
    for _, r in daily.nsmallest(3, 'day_pnl').iterrows():
        print(f"    {r['date'].date()}  ${r['day_pnl']:>+8,.0f}  picks: {r['trades']}")


if __name__ == '__main__':
    main()

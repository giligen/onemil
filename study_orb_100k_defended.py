#!/usr/bin/env python3
"""ORB $100K account sweep — WITH Q5 mult cap + family/super-group dedup.

This is the integrated "production BT" version. Previously, study_orb_100k_account.py
ran the sweep WITHOUT the Q5 cap and WITHOUT dedup (those lived in separate files).
This file unifies them.

Defenses applied (each one validated in prior studies):
  A. Risk-parity sizing: position = risk / stop_pct, capped at per-pos cap
  B. Q5 adaptive-mult CAP at 1.5x (prevents Split A's anomalous TRAIN from over-sizing)
  C. Family dedup (e.g., TSLA leveraged pair, MSTR leveraged pair)
  D. Super-group dedup (lev_short, lev_long — cross-underlying directional correlation)

Sweep: max_concurrent ∈ {3, 4, 5} × risk_per_trade ∈ {$250, $500, $1K, $2K, $3K}

Reports:
  - Per-config per-split P&L / DD / worst-day / worst-trade (walk-forward 3 splits)
  - Per-config full-timeline Jan'25-Apr'26 continuous equity P&L and DD
  - Before/after comparison vs undefended sweep (study_orb_100k_account.py numbers)
"""
from __future__ import annotations

import os, sys, glob
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_orb import SPLITS, OUT_DIR
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN,
    fit_quintile_cutoffs, assign_quintile,
)
from study_orb_correlation_filter import (
    FAMILIES, LEV_SHORT_SET, LEV_LONG_SET,
    symbol_family, symbol_super_group,
)

ACCOUNT = 100_000
MIN_STOP_PCT = 1.0
OLD_POS = 50_000.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}

# Q5 cap is the key defense against Split A's anomalous TRAIN data
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}


# =========================================================================
# Pipeline with integrated defenses
# =========================================================================

def apply_rp(df, risk, per_pos_cap):
    df = df.copy()
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = risk / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS
    return df


def fit_mults_capped(tk, pnl_col='_rp_pnl'):
    """Adaptive mults with per-quintile caps. Q5 capped at 1.5x."""
    avg = float(tk[pnl_col].mean()) if len(tk) else 1.0
    out = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = tk[tk['_quintile'] == q]
        if len(sub) == 0 or avg <= 0:
            out[q] = 1.0
            continue
        raw = float(sub[pnl_col].mean()) / avg
        out[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], raw))
    return out


def select_top_k_defended(dg, k, use_dedup=True):
    """Q4-preferred ranking + optional family + super-group dedup."""
    d = dg.copy()
    d['_q_rank'] = d['_quintile'].map(Q_ORDER)
    d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
    if not use_dedup:
        return d.head(k)
    seen_fam = set()
    seen_sup = set()
    kept = []
    for _, r in d.iterrows():
        sym = r['symbol']
        fam = symbol_family(sym)
        sup = symbol_super_group(sym)
        if fam and fam in seen_fam:
            continue
        if sup and sup in seen_sup:
            continue
        if fam: seen_fam.add(fam)
        if sup: seen_sup.add(sup)
        kept.append(r)
        if len(kept) >= k: break
    return pd.DataFrame(kept)


def compute_daily(sel):
    if len(sel) == 0:
        return pd.DataFrame(columns=['date', 'daily_pnl', 'n_picks'])
    return sel.groupby('date').agg(
        daily_pnl=('_sized_pnl', 'sum'),
        n_picks=('_rp_pnl', 'count'),
    ).reset_index().sort_values('date').reset_index(drop=True)


def equity_stats(daily):
    """Max DD on a daily series (within a period or full timeline)."""
    if len(daily) == 0:
        return {'total_pnl': 0, 'max_dd': 0, 'worst_day': 0, 'n_days': 0,
                'peak_date': None, 'trough_date': None}
    d = daily.sort_values('date').reset_index(drop=True).copy()
    d['cum'] = d['daily_pnl'].cumsum()
    running_peak = -np.inf
    dd = 0.0
    peak_date = None; trough_date = None; cur_peak = None
    for i, c in enumerate(d['cum']):
        if c > running_peak:
            running_peak = c
            cur_peak = d.loc[i, 'date']
        dnow = c - running_peak
        if dnow < dd:
            dd = dnow
            peak_date = cur_peak
            trough_date = d.loc[i, 'date']
    return {
        'total_pnl': float(d['daily_pnl'].sum()),
        'max_dd': float(dd),
        'worst_day': float(d['daily_pnl'].min()),
        'n_days': len(d),
        'peak_date': peak_date,
        'trough_date': trough_date,
    }


def run_split(df, tr_s, tr_e, te_s, te_e, k, risk, use_dedup, use_q5_cap):
    """One walk-forward split with defenses applied."""
    global Q_CAPS
    saved = dict(Q_CAPS)
    Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0,
              'Q5': 1.5 if use_q5_cap else 3.0}

    per_pos_cap = ACCOUNT / k
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = apply_rp(df, risk, per_pos_cap)

    train = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= tr_s) & (df['date'] <= tr_e)]
    test = df[(df['date'] >= te_s) & (df['date'] <= te_e)]

    train_k = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    test_k = test[test['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    test_k['_quintile'] = assign_quintile(test_k['_composite'], cutoffs)
    mults = fit_mults_capped(train_k)

    sel = pd.concat([select_top_k_defended(dg, k, use_dedup)
                     for _, dg in test_k.groupby('date')])
    sel['_sized_pnl'] = sel.apply(
        lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)

    daily = compute_daily(sel)
    stats = equity_stats(daily)
    stats['sel'] = sel
    stats['mults'] = mults
    stats['worst_trade'] = float(sel['_sized_pnl'].min()) if len(sel) else 0
    Q_CAPS = saved
    return stats


def run_full_timeline(df, k, risk, use_dedup, use_q5_cap):
    """Full Jan'25-Apr'26 continuous equity curve.
    Uses Split A's TRAIN (H1 2025) for z-params and mults
    (same as our 3-way TRAIN/VAL/OOS), then applies across entire timeline."""
    global Q_CAPS
    saved = dict(Q_CAPS)
    Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0,
              'Q5': 1.5 if use_q5_cap else 3.0}

    per_pos_cap = ACCOUNT / k
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = apply_rp(df, risk, per_pos_cap)

    # Use H1 2025 as training period (same as 3-way TRAIN)
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    train_k = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    mults = fit_mults_capped(train_k)

    kept = df[df['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)
    sel = pd.concat([select_top_k_defended(dg, k, use_dedup)
                     for _, dg in kept.groupby('date')])
    sel['_sized_pnl'] = sel.apply(
        lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)

    daily = compute_daily(sel)
    # Continuous equity Jan 2025 - Apr 2026
    daily = daily[(daily['date'] >= '2025-01-01') & (daily['date'] <= '2026-04-30')].copy()
    stats = equity_stats(daily)
    stats['worst_trade'] = float(sel['_sized_pnl'].min()) if len(sel) else 0
    Q_CAPS = saved
    return stats


def run_sweep(df, use_dedup, use_q5_cap, label):
    print(f"\n{'='*130}")
    print(f"{label}  (dedup={'ON' if use_dedup else 'OFF'}, "
          f"Q5 cap={'ON (1.5x)' if use_q5_cap else 'OFF (3.0x)'})")
    print(f"{'='*130}")

    all_results = []
    for N in [3, 4, 5]:
        per_pos_cap = ACCOUNT / N
        print(f"\n--- MAX_CONCURRENT={N}   PER_POS_CAP=${per_pos_cap:,.0f} ---")
        print(f"  {'Risk/tr':>7} {'Sum P&L':>12} {'Min split':>12} {'Worst DD':>11} "
              f"{'Worst day':>11} {'Worst tr':>10} {'Full P&L':>11} {'Full DD':>11} "
              f"{'Calmar':>8}")
        for risk in [250, 500, 1000, 2000, 3000]:
            # 3-way walk-forward splits
            per_split = []
            for _, tr_s, tr_e, te_s, te_e in SPLITS:
                s = run_split(df, tr_s, tr_e, te_s, te_e, N, risk, use_dedup, use_q5_cap)
                per_split.append(s)
            sum_pnl = sum(s['total_pnl'] for s in per_split)
            min_pnl = min(s['total_pnl'] for s in per_split)
            worst_dd = min(s['max_dd'] for s in per_split)
            worst_day = min(s['worst_day'] for s in per_split)
            worst_trade = min(s['worst_trade'] for s in per_split)

            # Full-timeline (continuous Jan 2025 - Apr 2026)
            ft = run_full_timeline(df, N, risk, use_dedup, use_q5_cap)
            calmar = ft['total_pnl'] / abs(ft['max_dd']) if ft['max_dd'] < 0 else float('inf')
            all_results.append({
                'N': N, 'risk': risk, 'sum_pnl': sum_pnl, 'min_split': min_pnl,
                'worst_dd': worst_dd, 'worst_day': worst_day, 'worst_trade': worst_trade,
                'ft_pnl': ft['total_pnl'], 'ft_dd': ft['max_dd'], 'calmar': calmar,
                'ft_peak': ft['peak_date'], 'ft_trough': ft['trough_date'],
            })
            print(f"  ${risk:>5,} ${sum_pnl:>+10,.0f} ${min_pnl:>+10,.0f} "
                  f"${worst_dd:>+9,.0f} ${worst_day:>+9,.0f} "
                  f"${worst_trade:>+8,.0f} ${ft['total_pnl']:>+9,.0f} ${ft['max_dd']:>+9,.0f} "
                  f"{calmar:>6.2f}x")
    return all_results


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])
    print(f"Loaded {len(df):,} trades")

    # Run 4 quadrants: (dedup off, cap off) vs (dedup on, cap on) etc.
    before = run_sweep(df, use_dedup=False, use_q5_cap=False,
                        label="BEFORE: original BT (no defenses)")
    after = run_sweep(df, use_dedup=True, use_q5_cap=True,
                       label="AFTER: defended BT (Q5 cap + family/super-group dedup)")

    # Best config by Calmar in each
    before_best = max(before, key=lambda r: r['calmar'] if r['min_split'] > 0 else -1)
    after_best = max(after, key=lambda r: r['calmar'] if r['min_split'] > 0 else -1)

    print(f"\n{'='*130}")
    print("BEST CONFIG BY CALMAR — before vs after")
    print(f"{'='*130}")
    for label, r in [('BEFORE (no defenses)', before_best),
                     ('AFTER (Q5 cap + dedup)', after_best)]:
        print(f"\n  {label}: N={r['N']}, risk=${r['risk']:,}")
        print(f"    Walk-forward sum P&L: ${r['sum_pnl']:+,.0f}  "
              f"(min split ${r['min_split']:+,.0f}, worst DD ${r['worst_dd']:+,.0f}, "
              f"worst trade ${r['worst_trade']:+,.0f})")
        print(f"    Full-timeline Jan'25-Apr'26: "
              f"P&L ${r['ft_pnl']:+,.0f}, DD ${r['ft_dd']:+,.0f}, "
              f"Calmar {r['calmar']:.2f}x")
        if r['ft_peak']:
            print(f"    Peak→Trough: {r['ft_peak'].date()} → {r['ft_trough'].date()}")

    # Head-to-head at matched config (N=3, risk=2000 — the user's preferred)
    print(f"\n{'='*130}")
    print("HEAD-TO-HEAD at N=3, risk=$2,000 (recommended config)")
    print(f"{'='*130}")
    b = next(r for r in before if r['N'] == 3 and r['risk'] == 2000)
    a = next(r for r in after if r['N'] == 3 and r['risk'] == 2000)
    print(f"\n  {'Metric':<30} {'BEFORE':>15} {'AFTER':>15} {'Change':>15}")
    print('  ' + '-' * 77)
    metrics = [
        ('Walk-forward sum P&L',  'sum_pnl'),
        ('Min split P&L',         'min_split'),
        ('Worst split DD',        'worst_dd'),
        ('Worst single day',      'worst_day'),
        ('Worst single trade',    'worst_trade'),
        ('Full-timeline P&L',     'ft_pnl'),
        ('Full-timeline DD',      'ft_dd'),
        ('Calmar (P&L / |DD|)',   'calmar'),
    ]
    for label, key in metrics:
        bv = b[key]; av = a[key]
        change = av - bv
        if key == 'calmar':
            print(f"  {label:<30} {bv:>13.2f}x {av:>13.2f}x  "
                  f"{change:>+13.2f}x")
        else:
            pct = (change / abs(bv)) * 100 if bv != 0 else 0
            print(f"  {label:<30} ${bv:>+13,.0f} ${av:>+13,.0f}  "
                  f"${change:>+10,.0f} ({pct:+.1f}%)")

    # Save full sweep to CSV for follow-up analysis
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    rows = []
    for group, label in [(before, 'BEFORE'), (after, 'AFTER')]:
        for r in group:
            rows.append({'scenario': label, **{k: v for k, v in r.items()
                                                if k not in ('ft_peak', 'ft_trough')}})
    pd.DataFrame(rows).to_csv(f'{OUT_DIR}/orb_defended_sweep_{ts}.csv', index=False)
    print(f"\nSaved: {OUT_DIR}/orb_defended_sweep_{ts}.csv")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""FINAL framework: combine ALL validated defenses.

Pipeline:
  1. Composite filter (z >= 0) — already validated walk-forward
  2. cap3 + Q4-preferred ranking — already validated
  3. Risk-parity sizing ($2K risk/trade, $33K per-pos cap) — already validated
  4. Adaptive quintile mults with Q5 CAPPED at 1.5x (from correlation study)
  5. Family + super-group dedup (from correlation study)
  6. SPY 5-day vol chaos gate at 3% (from regime study, weak but free insurance)

Evaluation: strict 3-way TRAIN/VALIDATION/OOS + full-timeline DD.
"""
from __future__ import annotations

import os, sys, glob, sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN,
    fit_quintile_cutoffs, assign_quintile,
)
from study_orb_correlation_filter import (
    FAMILIES, LEV_SHORT_SET, LEV_LONG_SET,
    symbol_family, symbol_super_group,
)
from study_orb_regime_gates import (
    load_spy_daily, gate_spy_vol_chaos, apply_gate,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, OOS_START, OOS_END,
)

# Config (same as before)
ACCOUNT = 100_000
N = 3
RISK = 2000
OLD_POS = 50_000
MIN_STOP_PCT = 1.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}

# Per-quintile mult caps — Q5 capped at 1.5x
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}

# Regime gate
SPY_VOL_CHAOS_THRESHOLD = 3.0  # %


def apply_rp(df):
    df = df.copy()
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    cap = ACCOUNT / N
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS
    return df


def fit_mults_capped(tk):
    avg = float(tk['_rp_pnl'].mean()) if len(tk) else 1.0
    out = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = tk[tk['_quintile'] == q]
        if len(sub) == 0 or avg <= 0:
            out[q] = 1.0
            continue
        raw = float(sub['_rp_pnl'].mean()) / avg
        out[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], raw))
    return out


def select_with_all_defenses(dg, k, use_dedup=True):
    d = dg.copy()
    d['_q_rank'] = d['_quintile'].map(Q_ORDER)
    d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])

    if not use_dedup:
        return d.head(k)

    seen_fam = set(); seen_super = set()
    kept = []
    for _, r in d.iterrows():
        sym = r['symbol']
        fam = symbol_family(sym)
        sup = symbol_super_group(sym)
        if fam and fam in seen_fam: continue
        if sup and sup in seen_super: continue
        if fam: seen_fam.add(fam)
        if sup: seen_super.add(sup)
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


def full_timeline_metrics(daily):
    if len(daily) == 0:
        return {'total_pnl': 0, 'max_dd': 0, 'peak_date': None, 'trough_date': None}
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
        'peak_date': peak_date,
        'trough_date': trough_date,
        'n_days': len(d),
        'worst_day': float(d['daily_pnl'].min()),
    }


def period_metrics(daily, start, end):
    d = daily[(daily['date'] >= start) & (daily['date'] <= end)].copy()
    d = d.sort_values('date').reset_index(drop=True)
    if len(d) == 0:
        return {'total_pnl': 0, 'max_dd': 0, 'n_days': 0, 'worst_day': 0}
    d['cum'] = d['daily_pnl'].cumsum()
    running_peak = -np.inf; dd = 0.0
    for c in d['cum']:
        running_peak = max(running_peak, c)
        dd = min(dd, c - running_peak)
    return {
        'total_pnl': float(d['daily_pnl'].sum()),
        'max_dd': float(dd),
        'n_days': len(d),
        'worst_day': float(d['daily_pnl'].min()),
    }


def run_config(df, spy_df, config_name,
               use_q5_cap, use_dedup, use_regime_gate):
    global Q_CAPS
    # Save and restore
    saved_caps = dict(Q_CAPS)
    if not use_q5_cap:
        Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 3.0}
    else:
        Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = apply_rp(df)
    train = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)]
    train_k = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    mults = fit_mults_capped(train_k)

    kept = df[df['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)
    sel = pd.concat([select_with_all_defenses(dg, N, use_dedup)
                     for _, dg in kept.groupby('date')])
    sel['_sized_pnl'] = sel.apply(
        lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)

    if use_regime_gate:
        skip = gate_spy_vol_chaos(spy_df, SPY_VOL_CHAOS_THRESHOLD)
        sel = apply_gate(sel, skip)

    daily = compute_daily(sel)

    # Metrics
    ft = full_timeline_metrics(daily)
    tr = period_metrics(daily, TRAIN_START, TRAIN_END)
    va = period_metrics(daily, VAL_START, VAL_END)
    oo = period_metrics(daily, OOS_START, OOS_END)

    Q_CAPS = saved_caps
    return {
        'name': config_name,
        'mults': mults,
        'train': tr, 'val': va, 'oos': oo, 'full': ft,
    }


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl','date','range_size_pct','entry_price'])
    print(f"Loaded {len(df):,} trades")

    spy = load_spy_daily()

    # Run configurations: baseline → layer in fixes
    configs = [
        ('A. baseline',                    False, False, False),
        ('B. +Q5 cap 1.5x',                True,  False, False),
        ('C. +Q5 cap +family/super dedup', True,  True,  False),
        ('D. +Q5 cap +dedup +SPY gate',    True,  True,  True),
    ]

    results = []
    for name, q5, dd, rg in configs:
        results.append(run_config(df, spy, name, q5, dd, rg))

    # Report
    print(f"\n{'='*120}")
    print("LAYERED DEFENSE EVALUATION — TRAIN / VAL / OOS / FULL-TIMELINE DD")
    print(f"{'='*120}")
    print(f"\n{'Config':<36} {'TRAIN P&L':>11} {'TRAIN DD':>11} {'VAL P&L':>11} {'VAL DD':>11} "
          f"{'OOS P&L':>11} {'OOS DD':>11} {'FULL P&L':>11} {'FULL DD':>11}")
    print('-' * 150)
    for r in results:
        print(f"{r['name']:<36} "
              f"${r['train']['total_pnl']:>+9,.0f} ${r['train']['max_dd']:>+9,.0f} "
              f"${r['val']['total_pnl']:>+9,.0f} ${r['val']['max_dd']:>+9,.0f} "
              f"${r['oos']['total_pnl']:>+9,.0f} ${r['oos']['max_dd']:>+9,.0f} "
              f"${r['full']['total_pnl']:>+9,.0f} ${r['full']['max_dd']:>+9,.0f}")

    # Highlights
    base = results[0]; final = results[-1]
    pnl_chg = final['full']['total_pnl'] - base['full']['total_pnl']
    dd_chg = final['full']['max_dd'] - base['full']['max_dd']
    print(f"\n{'='*120}")
    print("IMPACT OF LAYERED DEFENSES (final vs baseline, full timeline Jan'25-Apr'26):")
    print(f"{'='*120}")
    print(f"  Total P&L: ${base['full']['total_pnl']:+,.0f} → ${final['full']['total_pnl']:+,.0f}  "
          f"(Δ ${pnl_chg:+,.0f}, {pnl_chg/abs(base['full']['total_pnl'])*100:+.1f}%)")
    print(f"  Max DD:    ${base['full']['max_dd']:+,.0f} → ${final['full']['max_dd']:+,.0f}  "
          f"(Δ ${dd_chg:+,.0f}, {dd_chg/abs(base['full']['max_dd'])*100:+.1f}%)")
    print(f"  Peak→Trough: {base['full']['peak_date'].date()} → {base['full']['trough_date'].date()} "
          f"(baseline)  vs  {final['full']['peak_date'].date() if final['full']['peak_date'] else '-'} → "
          f"{final['full']['trough_date'].date() if final['full']['trough_date'] else '-'} (final)")
    calmar_base = base['full']['total_pnl'] / abs(base['full']['max_dd'])
    calmar_final = final['full']['total_pnl'] / abs(final['full']['max_dd'])
    print(f"  Calmar (P&L / |DD|): {calmar_base:.2f}x → {calmar_final:.2f}x")


if __name__ == '__main__':
    main()

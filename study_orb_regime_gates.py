#!/usr/bin/env python3
"""ORB regime-gate framework — TRAIN / VALIDATION / OOS protocol.

Goal: prevent the -$42K DD we saw in Split C's 5-week grinding stretch
(Oct 29 - Dec 5, 2025). The DD is NOT a single catastrophic day — it's a
sustained regime of small-cap breakout failure. Correlation filters can't
help; we need to SKIP TRADING when regime signals warn.

3-way split protocol (strict, no peeking across boundaries):
  TRAIN      2025-01-01 → 2025-06-30  fit composite z-scores, quintile cutoffs, adaptive mults
  VALIDATION 2025-07-01 → 2025-12-31  tune gate thresholds (includes the bad stretch)
  OOS        2026-01-01 → 2026-04-30  final test — selected ONCE by VAL

Six gate families tested on validation, best-by-criterion selected, then
applied to OOS.

Gates (all no-look-ahead — computed from T-1 close or earlier):

  G1 SPY 5-day avg range > X%           (existing prod gate — wide-vol chaos)
  G2 SPY yesterday daily range > X%      (single-day chaos, e.g. 11/20 3.6%)
  G3 SPY trailing 5-day return < -X%     (bearish drift — breakouts fail)
  G4 SPY below N-day SMA                 (bear trend)
  G5 ORB self-pause after K losing days  (strategy knows it's broken)
  G6 ORB trailing M-day P&L < -$X         (rolling P&L filter)

All signals use ONLY data known BEFORE 9:35 ET on the trading day.

Selection criterion on VALIDATION:
  Prioritize DD reduction. Rank gates by VAL_min_DD (largest = least negative),
  filter to gates keeping P&L >= 50% of baseline VAL P&L.
"""
from __future__ import annotations

import os, sys, glob, sqlite3
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN, ADAPTIVE_MULT_MAX,
    fit_quintile_cutoffs, assign_quintile,
)

# Configuration
ACCOUNT = 100_000
N = 3
RISK = 2000
OLD_POS = 50_000
MIN_STOP_PCT = 1.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}

# 3-way split
TRAIN_START, TRAIN_END = '2025-01-01', '2025-06-30'
VAL_START, VAL_END     = '2025-07-01', '2025-12-31'
OOS_START, OOS_END     = '2026-01-01', '2026-04-30'


# =========================================================================
# SPY data + derived signals (no look-ahead)
# =========================================================================

def load_spy_daily(db_path='data/cache.db') -> pd.DataFrame:
    """Load SPY daily bars from cache.db."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT bar_date as date, open, high, low, close, volume "
        "FROM daily_bars WHERE symbol='SPY' ORDER BY bar_date",
        conn
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    # Derived signals - all use PAST data only
    df['range_pct'] = (df['high'] - df['low']) / df['close'] * 100
    df['return_pct'] = df['close'].pct_change() * 100
    # Shift so we use T-1 close data on day T (no look-ahead)
    df['prev_close'] = df['close'].shift(1)
    df['prev_range_pct'] = df['range_pct'].shift(1)
    df['prev_return_pct'] = df['return_pct'].shift(1)
    # Rolling signals (all as-of T-1 so they're safe for T's 9:35 decision)
    df['spy_5d_avg_range'] = df['range_pct'].shift(1).rolling(5).mean()
    df['spy_5d_return'] = df['close'].shift(1).pct_change(5) * 100  # T-6 to T-1 return
    df['spy_sma20'] = df['close'].shift(1).rolling(20).mean()
    df['spy_sma50'] = df['close'].shift(1).rolling(50).mean()
    df['spy_above_sma20'] = df['close'].shift(1) > df['spy_sma20']
    df['spy_above_sma50'] = df['close'].shift(1) > df['spy_sma50']
    return df


# =========================================================================
# ORB self-signals (no look-ahead)
# =========================================================================

def add_orb_self_signals(daily_orb: pd.DataFrame, max_window: int = 20
                         ) -> pd.DataFrame:
    """Compute trailing ORB P&L and consecutive-loss-days from PRIOR days only.

    daily_orb: DataFrame with columns [date, daily_pnl]. Must be sorted by date.

    Returns same df with added columns:
      orb_trailing_3d_pnl, orb_trailing_5d_pnl, orb_trailing_10d_pnl
      orb_consec_loss_days (number of consecutive losing days immediately before today)
    """
    d = daily_orb.sort_values('date').reset_index(drop=True).copy()
    # Trailing sums — use .shift(1) so current day is NOT included
    d['orb_trailing_3d_pnl'] = d['daily_pnl'].shift(1).rolling(3).sum()
    d['orb_trailing_5d_pnl'] = d['daily_pnl'].shift(1).rolling(5).sum()
    d['orb_trailing_10d_pnl'] = d['daily_pnl'].shift(1).rolling(10).sum()
    # Consecutive loss days (length of tail of losing days ending at T-1)
    consec = []
    count = 0
    for pnl in d['daily_pnl']:
        consec.append(count)  # count as-of start of day T (uses only T-1 and earlier)
        if pnl < 0:
            count += 1
        else:
            count = 0
    d['orb_consec_loss_days'] = consec
    return d


# =========================================================================
# Base ORB pipeline (cap3 + Q4-pref + adaptive + risk-parity)
# =========================================================================

def apply_rp(df):
    df = df.copy()
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    cap = ACCOUNT / N
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS
    return df


def fit_mults(tk):
    avg = float(tk['_rp_pnl'].mean())
    out = {}
    for q in ['Q1','Q2','Q3','Q4','Q5']:
        sub = tk[tk['_quintile'] == q]
        if len(sub) == 0 or avg <= 0:
            out[q] = 1.0
            continue
        out[q] = max(ADAPTIVE_MULT_MIN, min(ADAPTIVE_MULT_MAX,
                                             float(sub['_rp_pnl'].mean()) / avg))
    return out


def pick_top_n(dg):
    d = dg.copy()
    d['_q_rank'] = d['_quintile'].map(Q_ORDER)
    return d.sort_values(['_q_rank', '_composite'], ascending=[True, False]).head(N)


def compute_daily(sel: pd.DataFrame) -> pd.DataFrame:
    if len(sel) == 0:
        return pd.DataFrame(columns=['date', 'daily_pnl', 'n_picks'])
    return sel.groupby('date').agg(
        daily_pnl=('_sized_pnl', 'sum'),
        n_picks=('_rp_pnl', 'count'),
    ).reset_index().sort_values('date').reset_index(drop=True)


def metrics(daily: pd.DataFrame) -> Dict[str, float]:
    if len(daily) == 0:
        return {'total_pnl': 0, 'max_dd': 0, 'worst_day': 0, 'n_days': 0,
                'n_win_days': 0, 'n_loss_days': 0}
    d = daily.sort_values('date').reset_index(drop=True).copy()
    d['cum'] = d['daily_pnl'].cumsum()
    running_peak = -np.inf
    dd = 0.0
    for c in d['cum']:
        running_peak = max(running_peak, c)
        dd = min(dd, c - running_peak)
    return {
        'total_pnl': float(d['daily_pnl'].sum()),
        'max_dd': float(dd),
        'worst_day': float(d['daily_pnl'].min()),
        'n_days': len(d),
        'n_win_days': int((d['daily_pnl'] > 0).sum()),
        'n_loss_days': int((d['daily_pnl'] < 0).sum()),
    }


def run_base_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline without regime gates. Returns per-trade selected df
    with _sized_pnl column (sized P&L after cap3 + Q4-pref + adaptive).
    z-params fit on TRAIN only, applied to all dates."""
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = apply_rp(df)
    train = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    # Re-extract train now with composite
    train = df[(df['date'] >= TRAIN_START) & (df['date'] <= TRAIN_END)]
    train_k = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    mults = fit_mults(train_k)

    # Apply to all dates
    kept = df[df['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)
    sel = pd.concat([pick_top_n(dg) for _, dg in kept.groupby('date')])
    sel['_sized_pnl'] = sel.apply(
        lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    return sel, mults


# =========================================================================
# Gate definitions — each returns a boolean Series indexed by date.
# True = SKIP this day, False = trade normally.
# =========================================================================

def gate_spy_vol_chaos(spy_df: pd.DataFrame, threshold_pct: float) -> pd.Series:
    """G1: skip if SPY 5-day avg range > threshold_pct."""
    s = spy_df.set_index('date')['spy_5d_avg_range']
    return s > threshold_pct


def gate_spy_yesterday_range(spy_df: pd.DataFrame, threshold_pct: float) -> pd.Series:
    """G2: skip if SPY yesterday's range > threshold_pct."""
    s = spy_df.set_index('date')['prev_range_pct']
    return s > threshold_pct


def gate_spy_5d_return(spy_df: pd.DataFrame, threshold_pct: float) -> pd.Series:
    """G3: skip if SPY trailing 5-day return < -threshold_pct (bearish drift)."""
    s = spy_df.set_index('date')['spy_5d_return']
    return s < -threshold_pct


def gate_spy_below_sma(spy_df: pd.DataFrame, sma_period: int = 20) -> pd.Series:
    """G4: skip if SPY close < SMA-N."""
    col = f'spy_above_sma{sma_period}'
    s = spy_df.set_index('date')[col]
    return ~s.fillna(True)  # trade when NaN (insufficient history)


def gate_orb_consec_losses(orb_daily: pd.DataFrame, k: int) -> pd.Series:
    """G5: skip if trailing consecutive losses >= k."""
    s = orb_daily.set_index('date')['orb_consec_loss_days']
    return s >= k


def gate_orb_trailing_pnl(orb_daily: pd.DataFrame, window: str, threshold: float
                          ) -> pd.Series:
    """G6: skip if trailing-M-day ORB P&L < -threshold."""
    col = f'orb_trailing_{window}_pnl'
    s = orb_daily.set_index('date')[col]
    return (s < -threshold) & s.notna()


# =========================================================================
# Apply a gate to selected trades, compute metrics
# =========================================================================

def apply_gate(sel: pd.DataFrame, gate_skip: pd.Series) -> pd.DataFrame:
    """Return sel with trades on SKIP days removed."""
    if len(sel) == 0:
        return sel
    # gate_skip is indexed by date; look up each trade's date
    # Map: True = skip = drop
    date_skip = gate_skip.reindex(sel['date']).fillna(False).values
    return sel[~date_skip].copy()


def slice_by_period(sel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return sel[(sel['date'] >= start) & (sel['date'] <= end)].copy()


# =========================================================================
# Gate sweep + 3-way evaluation
# =========================================================================

def sweep_gates(sel: pd.DataFrame, spy_df: pd.DataFrame,
                orb_daily: pd.DataFrame) -> List[Dict]:
    """Run each gate-variant, evaluate on all 3 periods."""
    variants = []

    # No-gate baseline
    variants.append(('no_gate', None))

    # G1: SPY 5-day vol chaos
    for x in [3.0, 4.0, 5.0, 6.0]:
        variants.append((f'G1_spy5d_vol>{x}%', gate_spy_vol_chaos(spy_df, x)))

    # G2: SPY yesterday range
    for x in [2.0, 2.5, 3.0, 3.5]:
        variants.append((f'G2_spy1d_vol>{x}%', gate_spy_yesterday_range(spy_df, x)))

    # G3: SPY 5d bearish
    for x in [1.0, 2.0, 3.0]:
        variants.append((f'G3_spy5d_ret<-{x}%', gate_spy_5d_return(spy_df, x)))

    # G4: SPY below SMA
    for p in [20, 50]:
        variants.append((f'G4_spy_below_sma{p}', gate_spy_below_sma(spy_df, p)))

    # G5: ORB consecutive losses
    for k in [2, 3, 4, 5]:
        variants.append((f'G5_orb_consec_loss>={k}', gate_orb_consec_losses(orb_daily, k)))

    # G6: ORB trailing P&L
    for (w, t) in [('3d', 5000), ('5d', 5000), ('5d', 10000),
                    ('10d', 10000), ('10d', 15000)]:
        variants.append((f'G6_orb_{w}_pnl<-${t/1000:.0f}K',
                        gate_orb_trailing_pnl(orb_daily, w, t)))

    results = []
    for name, skip_series in variants:
        if skip_series is None:
            gated = sel
        else:
            gated = apply_gate(sel, skip_series)

        train = slice_by_period(gated, TRAIN_START, TRAIN_END)
        val = slice_by_period(gated, VAL_START, VAL_END)
        oos = slice_by_period(gated, OOS_START, OOS_END)

        m_train = metrics(compute_daily(train))
        m_val = metrics(compute_daily(val))
        m_oos = metrics(compute_daily(oos))

        n_skipped_days = 0
        if skip_series is not None:
            # count days in val+oos where gate fires (and we had data)
            val_dates = set(val['date'].unique()) | set(
                slice_by_period(sel, VAL_START, VAL_END)['date'].unique())
            oos_dates = set(oos['date'].unique()) | set(
                slice_by_period(sel, OOS_START, OOS_END)['date'].unique())
            all_active_dates = val_dates | oos_dates
            for d in all_active_dates:
                if skip_series.get(d, False):
                    n_skipped_days += 1

        results.append({
            'name': name,
            'train_pnl': m_train['total_pnl'],
            'train_dd': m_train['max_dd'],
            'val_pnl': m_val['total_pnl'],
            'val_dd': m_val['max_dd'],
            'val_worst_day': m_val['worst_day'],
            'val_days': m_val['n_days'],
            'oos_pnl': m_oos['total_pnl'],
            'oos_dd': m_oos['max_dd'],
            'oos_worst_day': m_oos['worst_day'],
            'oos_days': m_oos['n_days'],
            'skipped_days_val_oos': n_skipped_days,
        })
    return results


# =========================================================================
# Main
# =========================================================================

def main():
    # Load trades + features
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl','date','range_size_pct','entry_price'])
    print(f"Loaded {len(df):,} trades")

    # Load SPY
    spy = load_spy_daily()
    print(f"Loaded {len(spy)} SPY daily bars ({spy['date'].min().date()} → {spy['date'].max().date()})")

    # Run base pipeline (cap3 + Q4-pref + adaptive + risk-parity $2K)
    sel, mults = run_base_pipeline(df)
    sel['date'] = pd.to_datetime(sel['date'])
    print(f"Adaptive mults: " +
          " ".join(f"{q}={mults[q]:.2f}x" for q in ['Q1','Q2','Q3','Q4','Q5']))

    # Base daily P&L (for computing ORB self-signals)
    base_daily = compute_daily(sel)
    base_daily = add_orb_self_signals(base_daily)

    # Baseline metrics on each period
    print(f"\n{'='*100}")
    print(f"BASELINE (no regime gate) — cap{N}, risk=${RISK:,.0f}, adaptive sizing")
    print(f"{'='*100}")
    for name, s, e in [('TRAIN', TRAIN_START, TRAIN_END),
                        ('VALIDATION', VAL_START, VAL_END),
                        ('OOS', OOS_START, OOS_END)]:
        period_sel = slice_by_period(sel, s, e)
        m = metrics(compute_daily(period_sel))
        print(f"  {name:<12} {s} → {e}   n_days={m['n_days']:<3} "
              f"P&L=${m['total_pnl']:>+10,.0f}  DD=${m['max_dd']:>+10,.0f}  "
              f"worst_day=${m['worst_day']:>+9,.0f}")

    # Sweep gates
    print(f"\n{'='*120}")
    print(f"GATE SWEEP — each tested on TRAIN/VAL/OOS independently (no peeking)")
    print(f"{'='*120}")
    print(f"  {'Gate':<28} {'TR P&L':>10} {'TR DD':>10} | "
          f"{'VAL P&L':>11} {'VAL DD':>11} {'VAL worst':>11} | "
          f"{'OOS P&L':>11} {'OOS DD':>10} {'OOS worst':>11} | "
          f"{'skip d':>6}")
    results = sweep_gates(sel, spy, base_daily)
    baseline = results[0]
    for r in results:
        print(f"  {r['name']:<28} "
              f"${r['train_pnl']:>+8,.0f} ${r['train_dd']:>+8,.0f} | "
              f"${r['val_pnl']:>+9,.0f} ${r['val_dd']:>+9,.0f} ${r['val_worst_day']:>+9,.0f} | "
              f"${r['oos_pnl']:>+9,.0f} ${r['oos_dd']:>+8,.0f} ${r['oos_worst_day']:>+9,.0f} | "
              f"{r['skipped_days_val_oos']:>6}")

    # Selection on VALIDATION only
    print(f"\n{'='*120}")
    print(f"GATE SELECTION — based on VALIDATION only (OOS never peeked)")
    print(f"{'='*120}")

    # Criterion: minimize |VAL DD| subject to VAL_pnl >= 50% of baseline
    val_pnl_floor = 0.5 * baseline['val_pnl']
    print(f"Selection rule: VAL P&L >= 50% baseline (${val_pnl_floor:+,.0f}), "
          f"then rank by smallest |VAL DD|")
    eligible = [r for r in results if r['val_pnl'] >= val_pnl_floor and r['name'] != 'no_gate']
    eligible.sort(key=lambda r: r['val_dd'], reverse=True)  # least negative first

    print(f"\nTop 10 eligible gates (ranked by VAL DD, best first):")
    print(f"  {'Gate':<28} {'VAL P&L':>11} {'VAL DD':>11} {'VAL/Base P&L':>12} {'OOS P&L':>11} {'OOS DD':>11}")
    for r in eligible[:10]:
        val_ratio = r['val_pnl'] / baseline['val_pnl'] * 100 if baseline['val_pnl'] else 0
        print(f"  {r['name']:<28} ${r['val_pnl']:>+9,.0f} ${r['val_dd']:>+9,.0f} "
              f"{val_ratio:>10.0f}% ${r['oos_pnl']:>+9,.0f} ${r['oos_dd']:>+9,.0f}")

    # Pick winner: best VAL DD reduction
    if eligible:
        winner = eligible[0]
        print(f"\n*** WINNER by VAL criterion: {winner['name']} ***")
        print(f"  VAL:  P&L ${winner['val_pnl']:+,.0f} (baseline ${baseline['val_pnl']:+,.0f}), "
              f"DD ${winner['val_dd']:+,.0f} (baseline ${baseline['val_dd']:+,.0f})")
        print(f"  OOS:  P&L ${winner['oos_pnl']:+,.0f} (baseline ${baseline['oos_pnl']:+,.0f}), "
              f"DD ${winner['oos_dd']:+,.0f} (baseline ${baseline['oos_dd']:+,.0f})")
        oos_dd_improvement = winner['oos_dd'] - baseline['oos_dd']
        oos_pnl_cost = winner['oos_pnl'] - baseline['oos_pnl']
        print(f"  OOS IMPROVEMENT: DD reduced by ${oos_dd_improvement:+,.0f}, "
              f"P&L changed by ${oos_pnl_cost:+,.0f}")

    # Test COMBINATIONS of DISTINCT gate families (union — skip if any fires)
    print(f"\n{'='*120}")
    print(f"COMBINATION GATES — union across DISTINCT gate families")
    print(f"{'='*120}")

    # Hand-picked distinct combos that make strategic sense
    combos_to_test = [
        ('G1+G2: SPY chaos stack',
         [gate_spy_vol_chaos(spy, 3.0), gate_spy_yesterday_range(spy, 3.0)]),
        ('G1+G3: SPY vol + bearish 5d',
         [gate_spy_vol_chaos(spy, 3.0), gate_spy_5d_return(spy, 3.0)]),
        ('G1+G5: SPY vol + 3 loss days',
         [gate_spy_vol_chaos(spy, 3.0), gate_orb_consec_losses(base_daily, 3)]),
        ('G1+G3+G5',
         [gate_spy_vol_chaos(spy, 3.0), gate_spy_5d_return(spy, 3.0),
          gate_orb_consec_losses(base_daily, 3)]),
        ('G2+G6: SPY 1d chaos + ORB 10d pnl<-$15K',
         [gate_spy_yesterday_range(spy, 3.0),
          gate_orb_trailing_pnl(base_daily, '10d', 15000)]),
        ('G3+G4: SPY bearish stack',
         [gate_spy_5d_return(spy, 3.0), gate_spy_below_sma(spy, 50)]),
    ]
    for name, skip_list in combos_to_test:
        skip = skip_list[0].copy()
        for s in skip_list[1:]:
            # Align on union
            skip = skip.combine(s, lambda a, b: bool(a) or bool(b), fill_value=False)
        gated = apply_gate(sel, skip)
        t_m = metrics(compute_daily(slice_by_period(gated, TRAIN_START, TRAIN_END)))
        v_m = metrics(compute_daily(slice_by_period(gated, VAL_START, VAL_END)))
        o_m = metrics(compute_daily(slice_by_period(gated, OOS_START, OOS_END)))
        n_skip = skip.sum()
        print(f"\n  {name}   (skipped {n_skip} days total)")
        print(f"    TRAIN: P&L=${t_m['total_pnl']:+,.0f} DD=${t_m['max_dd']:+,.0f} "
              f"worst=${t_m['worst_day']:+,.0f}")
        print(f"    VAL:   P&L=${v_m['total_pnl']:+,.0f} DD=${v_m['max_dd']:+,.0f} "
              f"worst=${v_m['worst_day']:+,.0f}")
        print(f"    OOS:   P&L=${o_m['total_pnl']:+,.0f} DD=${o_m['max_dd']:+,.0f} "
              f"worst=${o_m['worst_day']:+,.0f}")

    # === FULL-TIMELINE equity curve (TRAIN → VAL → OOS continuous) ===
    print(f"\n{'='*120}")
    print(f"FULL-TIMELINE EQUITY DD — baseline vs best OOS gate")
    print(f"{'='*120}")
    print("This is what matters in live: equity doesn't reset between periods.")
    print("Max DD across the ENTIRE Jan'25-Apr'26 timeline.")

    def full_timeline_dd(gated_sel: pd.DataFrame, label: str) -> Dict:
        all_daily = compute_daily(gated_sel).sort_values('date').reset_index(drop=True)
        all_daily['cum'] = all_daily['daily_pnl'].cumsum()
        running_peak = -np.inf
        dd = 0.0
        peak_date = None
        trough_date = None
        cur_peak_date = None
        for i, c in enumerate(all_daily['cum']):
            if c > running_peak:
                running_peak = c
                cur_peak_date = all_daily.loc[i, 'date']
            dd_now = c - running_peak
            if dd_now < dd:
                dd = dd_now
                peak_date = cur_peak_date
                trough_date = all_daily.loc[i, 'date']
        total = float(all_daily['daily_pnl'].sum())
        print(f"\n  {label}:")
        print(f"    Total P&L (TRAIN+VAL+OOS): ${total:+,.0f}")
        print(f"    Max DD (full timeline):    ${dd:+,.0f}")
        print(f"    Peak→Trough:               {peak_date.date() if peak_date else '-'} → "
              f"{trough_date.date() if trough_date else '-'}")
        return {'pnl': total, 'dd': dd}

    full_timeline_dd(sel, 'Baseline (no gate)')

    # Best gate by VAL criterion on OOS
    if eligible:
        best_name = winner['name']
        # Build skip series
        if 'G1_' in best_name:
            x = float(best_name.split('>')[1].rstrip('%'))
            skip = gate_spy_vol_chaos(spy, x)
        elif 'G2_' in best_name:
            x = float(best_name.split('>')[1].rstrip('%'))
            skip = gate_spy_yesterday_range(spy, x)
        elif 'G3_' in best_name:
            x = float(best_name.split('<-')[1].rstrip('%'))
            skip = gate_spy_5d_return(spy, x)
        elif 'G4_' in best_name:
            p = int(best_name.split('sma')[1])
            skip = gate_spy_below_sma(spy, p)
        else:
            skip = None
        if skip is not None:
            full_timeline_dd(apply_gate(sel, skip), f'Best-by-VAL gate: {best_name}')

    # And the combo that gave best OOS DD
    combo_skip = gate_spy_vol_chaos(spy, 3.0) | gate_spy_yesterday_range(spy, 3.0) | gate_orb_consec_losses(base_daily, 3)
    full_timeline_dd(apply_gate(sel, combo_skip), 'Combo G1+G2+G5 (chaos + consec losses)')


if __name__ == '__main__':
    main()

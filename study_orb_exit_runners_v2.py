"""Round 2 runner-capture exit benchmark.

Round 1 (`study_orb_exit_runners.py`) found that V0 static_lock_1R's hidden
edge is NOT a +1R ceiling — it's a +1R FLOOR that lets winners ride to EOD
when they don't pull back. Tight 0.5R trails exit prematurely on intra-bar
volatility and kill the ride-to-EOD tail ($130K/yr loss).

Round 2 tests variants designed to preserve V0's ride-to-EOD behavior while
capturing the specific "pullback early → run late" shape (ATOM 2026-04-24):

  V0   static_lock_1R (shipped baseline)
  V1b  trail_after_arm_1.0R   — wider trail to survive noise
  V1c  trail_after_arm_1.5R   — even wider
  V4a  late_arm_3R_trail_0.5R — let trade breathe to +3R before tightening
  V4b  late_arm_4R_trail_0.5R — only tighten on true winners
  V6a  partial_50_trail_2.0R  — wider runner trail than V2's 1.0R
  V6b  partial_30_trail_1.5R  — less floor, more runner (70% runs)
  V9   milestone_ratchet       — discrete stop raises at +1.5R / +3R / +5R MFE

Each variant plugs into the SAME defended pipeline as V0. All use the same
bar-loop convention (arm/stop update first, then exit check) — intra-bar
bias is applied uniformly so the comparison is fair.
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import timedelta
from typing import Callable, Optional, Tuple, List

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database
from study_orb import _bars_to_df
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, fit_quintile_cutoffs, assign_quintile,
    ADAPTIVE_MULT_MIN,
)
from study_orb_correlation_filter import symbol_family, symbol_super_group


ACCOUNT = 100_000.0
N = 4
RISK = 3000.0
MIN_STOP_PCT = 1.0
OLD_POS = 50_000.0
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
EXIT_SLIP_BPS = 10.0


def _session_open_timestamp(bars):
    if bars.empty:
        return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning):
        return morning.iloc[0]['timestamp']
    return None


# ---------------------------------------------------------------------------
# General-purpose single-leg simulator.
#
# Modes (mutually exclusive — exactly one of lock_stop_r / trail_r /
# milestone_stops must be set):
#   - lock_stop_r=X         : fixed stop at entry+X*range_size once armed
#   - trail_r=X             : stop = peak_high - X*range_size once armed
#   - milestone_stops=[...] : discrete (mfe_r_reached, stop_r) pairs; stop
#                             ratchets up as each MFE milestone is hit
#
# All modes arm when bar_high >= entry + arm_at_r * range_size.
# ---------------------------------------------------------------------------


def simulate_single(bars, entry_price, range_high, range_low, entry_time,
                    arm_at_r: float = 1.5,
                    lock_stop_r: Optional[float] = None,
                    trail_r: Optional[float] = None,
                    milestone_stops: Optional[List[Tuple[float, float]]] = None,
                    ) -> Tuple[float, str, float]:
    """Single-leg simulator. Returns (exit_price, reason, mfe_r).

    mfe_r tracks full-bar-range excursion (ALL bars, not just pre-exit) so
    variant-to-variant capture comparisons are comparable.
    """
    modes_set = sum(x is not None for x in (lock_stop_r, trail_r, milestone_stops))
    if modes_set != 1:
        raise ValueError("Exactly one of lock_stop_r / trail_r / milestone_stops required")

    range_size = range_high - range_low
    trigger_lvl = entry_price + arm_at_r * range_size
    stop_price = range_low
    armed = False
    peak_high = 0.0
    milestones_hit = 0  # next milestone index to check
    mfe_abs_pre_exit = 0.0
    mfe_abs_full = 0.0
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        mfe_abs_full = max(mfe_abs_full, bar_high - entry_price)

        if exit_price is not None:
            continue  # still track MFE for remaining bars

        mfe_abs_pre_exit = max(mfe_abs_pre_exit, bar_high - entry_price)
        if bar_high > peak_high:
            peak_high = bar_high

        if armed or peak_high >= trigger_lvl:
            was_just_armed = not armed
            armed = True
            if lock_stop_r is not None:
                stop_price = max(stop_price, entry_price + lock_stop_r * range_size)
            elif trail_r is not None:
                stop_price = max(stop_price, peak_high - trail_r * range_size)
            elif milestone_stops is not None:
                mfe_r_now = (peak_high - entry_price) / range_size if range_size > 0 else 0
                while (milestones_hit < len(milestone_stops)
                       and mfe_r_now >= milestone_stops[milestones_hit][0]):
                    _, new_stop_r = milestone_stops[milestones_hit]
                    stop_price = max(stop_price, entry_price + new_stop_r * range_size)
                    milestones_hit += 1

        if bar_low <= stop_price:
            exit_price = stop_price * (1 - EXIT_SLIP_BPS/10000)
            if lock_stop_r is not None:
                exit_reason = 'lock' if armed else 'stop'
            elif trail_r is not None:
                exit_reason = 'trail' if armed else 'stop'
            else:
                exit_reason = f'mstone_{milestones_hit}' if armed else 'stop'

    if exit_price is None:
        last = post.iloc[-1]
        exit_price = float(last['close']) * (1 - EXIT_SLIP_BPS/10000)
        exit_reason = 'eod'

    return exit_price, exit_reason, mfe_abs_full / range_size if range_size > 0 else 0


def simulate_partial(bars, entry_price, range_high, range_low, entry_time,
                      partial_pct: float = 0.5,
                      arm_at_r: float = 1.5,
                      partial_stop_r: float = 1.0,
                      runner_trail_r: float = 1.0,
                      ) -> Tuple[float, str, float]:
    """Partial exit: `partial_pct` exits via fixed lock (V0-style), remainder
    trails `runner_trail_r` behind peak. Both halves arm at the same level.
    """
    range_size = range_high - range_low
    trigger_lvl = entry_price + arm_at_r * range_size
    fixed_stop_lvl = entry_price + partial_stop_r * range_size

    # Static half
    stop_1 = range_low; armed_1 = False
    exit_1: Optional[float] = None; reason_1: Optional[str] = None
    # Runner half
    stop_2 = range_low; armed_2 = False
    peak_high = 0.0
    exit_2: Optional[float] = None; reason_2: Optional[str] = None

    mfe_abs_full = 0.0
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        mfe_abs_full = max(mfe_abs_full, bar_high - entry_price)
        if bar_high > peak_high:
            peak_high = bar_high

        if exit_1 is None:
            if not armed_1 and bar_high >= trigger_lvl:
                armed_1 = True
                stop_1 = max(stop_1, fixed_stop_lvl)
            if bar_low <= stop_1:
                exit_1 = stop_1 * (1 - EXIT_SLIP_BPS/10000)
                reason_1 = 'lock' if armed_1 else 'stop'

        if exit_2 is None:
            if armed_2 or peak_high >= trigger_lvl:
                armed_2 = True
                stop_2 = max(stop_2, peak_high - runner_trail_r * range_size)
            if bar_low <= stop_2:
                exit_2 = stop_2 * (1 - EXIT_SLIP_BPS/10000)
                reason_2 = 'trail' if armed_2 else 'stop'

    if exit_1 is None:
        exit_1 = float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000)
        reason_1 = 'eod'
    if exit_2 is None:
        exit_2 = float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000)
        reason_2 = 'eod'

    blended = partial_pct * exit_1 + (1 - partial_pct) * exit_2
    reason = f"{reason_1}|{reason_2}"
    return blended, reason, mfe_abs_full / range_size if range_size > 0 else 0


# ---------------------------------------------------------------------------
# Variant registry
# ---------------------------------------------------------------------------

VARIANTS: List[Tuple[str, Callable]] = [
    ('V0 static_lock_1R (shipped)',
     lambda *a: simulate_single(*a, arm_at_r=1.5, lock_stop_r=1.0)),
    ('V1b trail_1.0R',
     lambda *a: simulate_single(*a, arm_at_r=1.5, trail_r=1.0)),
    ('V1c trail_1.5R',
     lambda *a: simulate_single(*a, arm_at_r=1.5, trail_r=1.5)),
    ('V4a late_arm_3R_trail_0.5R',
     lambda *a: simulate_single(*a, arm_at_r=3.0, trail_r=0.5)),
    ('V4b late_arm_4R_trail_0.5R',
     lambda *a: simulate_single(*a, arm_at_r=4.0, trail_r=0.5)),
    ('V6a partial_50_runner_trail_2.0R',
     lambda *a: simulate_partial(*a, partial_pct=0.5, runner_trail_r=2.0)),
    ('V6b partial_30_runner_trail_1.5R',
     lambda *a: simulate_partial(*a, partial_pct=0.3, runner_trail_r=1.5)),
    ('V9 milestone_ratchet_1.5_3_5',
     lambda *a: simulate_single(*a, arm_at_r=1.5,
                                 milestone_stops=[(1.5, 1.0), (3.0, 2.0), (5.0, 3.0)])),
]


# ---------------------------------------------------------------------------
# Pipeline wiring
# ---------------------------------------------------------------------------


def _simulate_all(df, bars_cache, exit_fn):
    pnls, pcts, reasons, mfes = [], [], [], []
    for _, row in df.reset_index(drop=True).iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); mfes.append(0.0); continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); mfes.append(0.0); continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) &
                          (bars['timestamp'] < range_end)]
        if len(range_bars) < 5:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); mfes.append(0.0); continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh:
                entry_ts = b['timestamp']; break
        if entry_ts is None:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct'])
            reasons.append(row['exit_reason']); mfes.append(0.0); continue
        entry_p = float(row['entry_price'])
        exit_p, reason, mfe_r = exit_fn(bars, entry_p, rh, rl, entry_ts)
        shares = max(1, int(OLD_POS / entry_p))
        pnls.append((exit_p - entry_p) * shares)
        pcts.append((exit_p - entry_p) / entry_p * 100)
        reasons.append(reason); mfes.append(mfe_r)
    out = df.reset_index(drop=True).copy()
    out['pnl'] = pnls; out['pnl_pct'] = pcts
    out['exit_reason'] = reasons; out['mfe_r'] = mfes
    return out


def _run_pipeline(df_with_pnl, label):
    df = df_with_pnl.copy()
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
        m = float(sub['_rp_pnl'].mean()) / avg if len(sub) else 1.0
        mults[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], m))

    kept = df[df['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)

    sel_rows = []
    for _, dg in kept.groupby('date'):
        d = dg.copy()
        d['_q_rank'] = d['_quintile'].map(Q_ORDER)
        d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        seen_fam = set(); seen_sup = set()
        today = []
        for _, r in d.iterrows():
            fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_fam: continue
            if sup and sup in seen_sup: continue
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
            today.append(r)
            if len(today) >= N: break
        sel_rows.extend(today)
    sel = pd.DataFrame(sel_rows)
    sel['_sized_pnl'] = sel.apply(lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    sel['date'] = pd.to_datetime(sel['date'])
    sel['month'] = sel['date'].dt.to_period('M').astype(str)
    return sel, mults


def _metrics(sel, label):
    daily = sel.groupby('date').agg(
        pnl=('_sized_pnl', 'sum'),
        picks=('_sized_pnl', 'count'),
    ).reset_index().sort_values('date').reset_index(drop=True)

    daily['cum'] = daily['pnl'].cumsum()
    peak = -1e18; mdd = 0.0; mdd_peak_date = None; mdd_trough_date = None
    cur_peak_date = None
    for _, r in daily.iterrows():
        if r['cum'] > peak:
            peak = r['cum']; cur_peak_date = r['date']
        dd = r['cum'] - peak
        if dd < mdd:
            mdd = dd; mdd_peak_date = cur_peak_date; mdd_trough_date = r['date']

    total_pnl = float(daily['pnl'].sum())
    calmar = total_pnl / abs(mdd) if mdd < 0 else float('inf')

    daily['month'] = daily['date'].dt.to_period('M')
    monthly = daily.groupby('month')['pnl'].sum()
    neg_months = int((monthly < 0).sum())
    worst_month = float(monthly.min()); best_month = float(monthly.max())

    # Capture ratios from trade-level MFE
    total_mfe_r = 0.0; total_realized_r = 0.0
    runner_realized = 0.0; runner_mfe = 0.0; runner_count = 0
    for _, r in sel.iterrows():
        rsp = float(r.get('range_size_pct', 0))
        if rsp <= 0: continue
        realized_r = float(r.get('pnl_pct', 0)) / rsp
        mfe_r = float(r.get('mfe_r', 0.0))
        total_mfe_r += mfe_r; total_realized_r += realized_r
        if mfe_r >= 3.0:
            runner_count += 1
            runner_mfe += mfe_r; runner_realized += realized_r

    return {
        'label': label,
        'trades': len(sel),
        'pnl': total_pnl,
        'max_dd': float(mdd),
        'calmar': calmar,
        'neg_months': neg_months,
        'worst_month': worst_month,
        'best_month': best_month,
        'capture_overall_pct': (total_realized_r / total_mfe_r * 100) if total_mfe_r > 0 else 0,
        'capture_runners_pct': (runner_realized / runner_mfe * 100) if runner_mfe > 0 else 0,
        'runner_count': runner_count,
        'avg_mfe_r': float(sel['mfe_r'].mean()) if 'mfe_r' in sel else 0.0,
        'dd_peak': str(mdd_peak_date.date()) if mdd_peak_date is not None else '',
        'dd_trough': str(mdd_trough_date.date()) if mdd_trough_date is not None else '',
    }


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    print(f"Loading features from {csv}")
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'pnl_pct', 'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])

    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    print(f"Loading bars for {len(pairs)} pairs...")
    db = Database(db_path='data/cache.db')
    raw_bars = db.get_intraday_bars_bulk(pairs)
    db.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}

    # Run all variants
    all_metrics = []
    all_sel = {}
    for label, exit_fn in VARIANTS:
        print(f"\nSimulating {label}...")
        df_v = _simulate_all(df, bars_cache, exit_fn)
        sel, mults = _run_pipeline(df_v, label)
        m = _metrics(sel, label)
        all_metrics.append(m)
        all_sel[label] = sel
        print(f"  P&L ${m['pnl']:+,.0f}  DD ${m['max_dd']:+,.0f}  "
              f"Calmar {m['calmar']:.2f}x  neg_mo {m['neg_months']}  "
              f"mults {{Q4:{mults['Q4']:.2f}, Q5:{mults['Q5']:.2f}}}")

    # Comparison table
    print(f"\n{'='*100}")
    print("  COMPARISON (all variants, full Jan'25 → Apr'26 timeline)")
    print(f"{'='*100}")
    base = all_metrics[0]
    print(f"{'Variant':<40} {'P&L':>11} {'Δ P&L':>10} {'Max DD':>10} "
          f"{'Calmar':>8} {'Neg mo':>7} {'Capt%':>7}")
    print('-' * 100)
    for m in all_metrics:
        delta = m['pnl'] - base['pnl']
        print(f"{m['label']:<40} "
              f"${m['pnl']:>+9,.0f}  "
              f"${delta:>+8,.0f}  "
              f"${m['max_dd']:>+7,.0f}  "
              f"{m['calmar']:>6.2f}x  "
              f"{m['neg_months']:>5}   "
              f"{m['capture_overall_pct']:>5.1f}%")

    # Winner callouts
    winners = [m for m in all_metrics if m['pnl'] > base['pnl']]
    if winners:
        winners.sort(key=lambda x: x['pnl'], reverse=True)
        print(f"\n{'='*100}")
        print(f"  ✓ VARIANTS THAT BEAT BASELINE ({len(winners)} of {len(all_metrics)-1})")
        print(f"{'='*100}")
        for m in winners:
            delta = m['pnl'] - base['pnl']
            print(f"  {m['label']}: +${delta:,.0f} "
                  f"({delta/abs(base['pnl'])*100:+.1f}%) | "
                  f"Calmar {m['calmar']:.2f}x vs {base['calmar']:.2f}x")
    else:
        print(f"\n{'='*100}")
        print("  ✗ NO variant beat the baseline on total P&L")
        print(f"{'='*100}")

    # Deep dive: top-10 runners per variant
    print(f"\n{'='*100}")
    print("  TOP-10 RUNNERS (by V0 MFE) — per-variant sized P&L")
    print(f"{'='*100}")
    v0_sel = all_sel[base['label']]
    runners = v0_sel[v0_sel['mfe_r'] >= 3.0].copy().sort_values('mfe_r', ascending=False).head(10)
    cols = ['symbol', 'date_str', 'mfe_r']
    out = runners[['symbol', 'date', 'mfe_r']].copy()
    out['date_str'] = out['date'].dt.strftime('%Y-%m-%d')
    out = out[['symbol', 'date_str', 'mfe_r']]
    for m in all_metrics:
        sel = all_sel[m['label']]
        sel_idx = sel.set_index(['symbol', 'date'])
        pnls = []
        for _, r in runners.iterrows():
            k = (r['symbol'], r['date'])
            v = sel_idx.loc[k, '_sized_pnl'] if k in sel_idx.index else None
            if hasattr(v, 'iloc'): v = v.iloc[0]
            pnls.append(v)
        out[m['label'].split()[0]] = pnls
    pd.set_option('display.width', 220)
    pd.set_option('display.float_format', '{:,.0f}'.format)
    print(out.to_string(index=False))

    # Save monthly comparison csv
    print(f"\nSaving analysis_results/orb_exit_v2_summary.csv")
    df_sum = pd.DataFrame(all_metrics)
    df_sum.to_csv('analysis_results/orb_exit_v2_summary.csv', index=False)


if __name__ == '__main__':
    main()

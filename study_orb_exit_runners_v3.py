"""Round 3: MFE-conditional exit variants.

Round 1+2 showed that V0 static_lock_1R has TWO runner archetypes:
  A. "EOD-rider" — never retraces past +1R, rides to EOD close. V0 wins big.
  B. "Pullback-runner" — retraces to +1R (V0 exits), then continues huge.
                         V0 captures only +1R, loses the tail.

All unconditional trail variants sacrifice archetype A to help B, and A's
absolute contribution dominates → net loss vs V0.

Round 3 tests MFE-conditional variants: keep V0's behavior as DEFAULT, and
only modify the exit once the trade has demonstrated runner potential via
high MFE. Goal: preserve A, improve B.

  V0   static_lock_1R (baseline)
  V10  V0 + conditional trail: stop = max(+1R lock, peak-2R) once MFE >= +3R
  V11  V0 + remove stop: once MFE >= +3R, no stop until EOD (riskier but rides)
  V12  V0 + delayed trail: once MFE >= +3R, replace +1R lock with peak-1R trail
  V13  V0 + step-up lock: once MFE >= +3R, raise lock to +2R; +5R → +3R lock
  V4b  late_arm_4R_trail_0.5R (best Calmar from round 2, included for reference)
  V4c  late_arm_4R_trail_1.0R (wider trail after late arm)

All use same bar-loop convention and plug into the same defended pipeline.
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import timedelta
from typing import Callable, List, Optional, Tuple

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
# MFE-conditional simulators
#
# Shared skeleton:
#   - Arm at +1.5R (like V0)
#   - Baseline stop = range_low before arm, entry + 1R after arm (V0 behavior)
#   - When MFE crosses runner_trigger_r, apply mode-specific modification
# ---------------------------------------------------------------------------


def simulate_v0_baseline(bars, entry_price, range_high, range_low, entry_time
                          ) -> Tuple[float, str, float]:
    range_size = range_high - range_low
    trigger_lvl = entry_price + 1.5 * range_size
    lock_stop = entry_price + 1.0 * range_size
    stop_price = range_low
    armed = False
    mfe_abs = 0.0
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        mfe_abs = max(mfe_abs, bar_high - entry_price)
        if not armed and bar_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        if bar_low <= stop_price:
            px = stop_price * (1 - EXIT_SLIP_BPS/10000)
            return px, 'lock' if armed else 'stop', mfe_abs / range_size
    last = post.iloc[-1]
    return float(last['close']) * (1 - EXIT_SLIP_BPS/10000), 'eod', mfe_abs / range_size


def simulate_v10_conditional_trail(bars, entry_price, range_high, range_low, entry_time,
                                     runner_trigger_r: float = 3.0,
                                     trail_r: float = 2.0
                                     ) -> Tuple[float, str, float]:
    """V10: V0 until MFE >= runner_trigger_r, then stop = max(+1R lock, peak - trail_r)."""
    range_size = range_high - range_low
    trigger_lvl = entry_price + 1.5 * range_size
    lock_stop = entry_price + 1.0 * range_size
    runner_trigger_abs = runner_trigger_r * range_size
    stop_price = range_low
    armed = False
    runner_triggered = False
    peak_high = 0.0
    mfe_abs = 0.0
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        mfe_abs = max(mfe_abs, bar_high - entry_price)
        if bar_high > peak_high:
            peak_high = bar_high
        if not armed and bar_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        if armed and mfe_abs >= runner_trigger_abs:
            runner_triggered = True
        if runner_triggered:
            stop_price = max(stop_price, peak_high - trail_r * range_size)
        if bar_low <= stop_price:
            px = stop_price * (1 - EXIT_SLIP_BPS/10000)
            reason = 'runner_trail' if runner_triggered else ('lock' if armed else 'stop')
            return px, reason, mfe_abs / range_size
    last = post.iloc[-1]
    return float(last['close']) * (1 - EXIT_SLIP_BPS/10000), 'eod', mfe_abs / range_size


def simulate_v11_remove_stop(bars, entry_price, range_high, range_low, entry_time,
                              runner_trigger_r: float = 3.0
                              ) -> Tuple[float, str, float]:
    """V11: V0 until MFE >= runner_trigger_r, then REMOVE stop entirely. Ride to EOD."""
    range_size = range_high - range_low
    trigger_lvl = entry_price + 1.5 * range_size
    lock_stop = entry_price + 1.0 * range_size
    runner_trigger_abs = runner_trigger_r * range_size
    stop_price = range_low
    armed = False
    runner_triggered = False
    mfe_abs = 0.0
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        mfe_abs = max(mfe_abs, bar_high - entry_price)
        if not armed and bar_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        if armed and mfe_abs >= runner_trigger_abs:
            runner_triggered = True
        if not runner_triggered and bar_low <= stop_price:
            px = stop_price * (1 - EXIT_SLIP_BPS/10000)
            return px, 'lock' if armed else 'stop', mfe_abs / range_size
    last = post.iloc[-1]
    reason = 'eod_runner' if runner_triggered else 'eod'
    return float(last['close']) * (1 - EXIT_SLIP_BPS/10000), reason, mfe_abs / range_size


def simulate_v12_delayed_trail(bars, entry_price, range_high, range_low, entry_time,
                                 runner_trigger_r: float = 3.0,
                                 trail_r: float = 1.0
                                 ) -> Tuple[float, str, float]:
    """V12: V0 until MFE >= runner_trigger_r, then REPLACE +1R lock with peak-trail_r."""
    range_size = range_high - range_low
    trigger_lvl = entry_price + 1.5 * range_size
    lock_stop = entry_price + 1.0 * range_size
    runner_trigger_abs = runner_trigger_r * range_size
    stop_price = range_low
    armed = False
    runner_triggered = False
    peak_high = 0.0
    mfe_abs = 0.0
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        mfe_abs = max(mfe_abs, bar_high - entry_price)
        if bar_high > peak_high:
            peak_high = bar_high
        if not armed and bar_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        if armed and mfe_abs >= runner_trigger_abs:
            runner_triggered = True
        if runner_triggered:
            # Replace: stop is now peak-trail_r (which may be lower than +1R
            # if peak == trigger. But since trigger_r > arm_r+trail_r typically
            # won't be an issue with default 3/1. Still max(range_low,...) for safety.)
            stop_price = max(range_low, peak_high - trail_r * range_size)
        if bar_low <= stop_price:
            px = stop_price * (1 - EXIT_SLIP_BPS/10000)
            reason = 'runner_trail' if runner_triggered else ('lock' if armed else 'stop')
            return px, reason, mfe_abs / range_size
    last = post.iloc[-1]
    return float(last['close']) * (1 - EXIT_SLIP_BPS/10000), 'eod', mfe_abs / range_size


def simulate_v13_step_lock(bars, entry_price, range_high, range_low, entry_time
                            ) -> Tuple[float, str, float]:
    """V13: V0 with step-ups. +1R lock at arm; if MFE>=+3R, lock to +2R; if MFE>=+5R, lock to +3R."""
    range_size = range_high - range_low
    trigger_lvl = entry_price + 1.5 * range_size
    lock_stop_1r = entry_price + 1.0 * range_size
    lock_stop_2r = entry_price + 2.0 * range_size
    lock_stop_3r = entry_price + 3.0 * range_size
    trig_3r = 3.0 * range_size
    trig_5r = 5.0 * range_size
    stop_price = range_low
    armed = False
    mfe_abs = 0.0
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        mfe_abs = max(mfe_abs, bar_high - entry_price)
        if not armed and bar_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop_1r)
        if armed:
            if mfe_abs >= trig_5r:
                stop_price = max(stop_price, lock_stop_3r)
            elif mfe_abs >= trig_3r:
                stop_price = max(stop_price, lock_stop_2r)
        if bar_low <= stop_price:
            px = stop_price * (1 - EXIT_SLIP_BPS/10000)
            return px, 'step_lock' if armed else 'stop', mfe_abs / range_size
    last = post.iloc[-1]
    return float(last['close']) * (1 - EXIT_SLIP_BPS/10000), 'eod', mfe_abs / range_size


def simulate_v4_late_arm(bars, entry_price, range_high, range_low, entry_time,
                          arm_r: float = 4.0, trail_r: float = 0.5
                          ) -> Tuple[float, str, float]:
    """V4: arm late, then trail. (For reference from round 2.)"""
    range_size = range_high - range_low
    trigger_lvl = entry_price + arm_r * range_size
    stop_price = range_low
    armed = False
    peak_high = 0.0
    mfe_abs = 0.0
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        mfe_abs = max(mfe_abs, bar_high - entry_price)
        if bar_high > peak_high:
            peak_high = bar_high
        if armed or peak_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, peak_high - trail_r * range_size)
        if bar_low <= stop_price:
            px = stop_price * (1 - EXIT_SLIP_BPS/10000)
            return px, 'trail' if armed else 'stop', mfe_abs / range_size
    last = post.iloc[-1]
    return float(last['close']) * (1 - EXIT_SLIP_BPS/10000), 'eod', mfe_abs / range_size


VARIANTS: List[Tuple[str, Callable]] = [
    ('V0  static_lock_1R (shipped)', simulate_v0_baseline),
    ('V10 conditional_trail (MFE>=3R → peak-2R)',
     lambda *a: simulate_v10_conditional_trail(*a, runner_trigger_r=3.0, trail_r=2.0)),
    ('V10b conditional_trail_tighter (MFE>=3R → peak-1R)',
     lambda *a: simulate_v10_conditional_trail(*a, runner_trigger_r=3.0, trail_r=1.0)),
    ('V10c conditional_trail_mfe5 (MFE>=5R → peak-2R)',
     lambda *a: simulate_v10_conditional_trail(*a, runner_trigger_r=5.0, trail_r=2.0)),
    ('V11 remove_stop_after_MFE3R',
     lambda *a: simulate_v11_remove_stop(*a, runner_trigger_r=3.0)),
    ('V11b remove_stop_after_MFE5R',
     lambda *a: simulate_v11_remove_stop(*a, runner_trigger_r=5.0)),
    ('V12 delayed_trail (MFE>=3R replace lock → peak-1R)',
     lambda *a: simulate_v12_delayed_trail(*a, runner_trigger_r=3.0, trail_r=1.0)),
    ('V13 step_lock (+1R→+2R@3R→+3R@5R)', simulate_v13_step_lock),
    ('V4b late_arm_4R_trail_0.5R (ref)',
     lambda *a: simulate_v4_late_arm(*a, arm_r=4.0, trail_r=0.5)),
    ('V4c late_arm_4R_trail_1.0R',
     lambda *a: simulate_v4_late_arm(*a, arm_r=4.0, trail_r=1.0)),
]


# ---------------------------------------------------------------------------
# Pipeline wiring (identical to rounds 1-2)
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


def _run_pipeline(df_with_pnl):
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
        seen_fam = set(); seen_sup = set(); today = []
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
    return sel, mults


def _metrics(sel, label):
    daily = sel.groupby('date').agg(pnl=('_sized_pnl', 'sum')).reset_index().sort_values('date').reset_index(drop=True)
    daily['cum'] = daily['pnl'].cumsum()
    peak = -1e18; mdd = 0.0
    for _, r in daily.iterrows():
        peak = max(peak, r['cum'])
        mdd = min(mdd, r['cum'] - peak)
    total_pnl = float(daily['pnl'].sum())
    calmar = total_pnl / abs(mdd) if mdd < 0 else float('inf')
    daily['month'] = daily['date'].dt.to_period('M')
    monthly = daily.groupby('month')['pnl'].sum()
    neg_months = int((monthly < 0).sum())
    worst_month = float(monthly.min())
    return {
        'label': label, 'trades': len(sel), 'pnl': total_pnl,
        'max_dd': float(mdd), 'calmar': calmar,
        'neg_months': neg_months, 'worst_month': worst_month,
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

    all_m = []; all_sel = {}
    for label, fn in VARIANTS:
        print(f"\nSimulating {label}...")
        df_v = _simulate_all(df, bars_cache, fn)
        sel, mults = _run_pipeline(df_v)
        m = _metrics(sel, label)
        all_m.append(m); all_sel[label] = sel
        print(f"  P&L ${m['pnl']:+,.0f}  DD ${m['max_dd']:+,.0f}  "
              f"Calmar {m['calmar']:.2f}x  neg_mo {m['neg_months']}  "
              f"mults {{Q4:{mults['Q4']:.2f}, Q5:{mults['Q5']:.2f}}}")

    print(f"\n{'='*100}")
    print("  COMPARISON (full timeline Jan'25 → Apr'26)")
    print(f"{'='*100}")
    base = all_m[0]
    print(f"{'Variant':<55} {'P&L':>11} {'Δ P&L':>10} {'Max DD':>10} {'Calmar':>8} {'Neg mo':>7}")
    print('-' * 110)
    for m in all_m:
        delta = m['pnl'] - base['pnl']
        marker = ' ★' if m['pnl'] > base['pnl'] else ''
        print(f"{m['label']:<55} "
              f"${m['pnl']:>+9,.0f}  "
              f"${delta:>+8,.0f}  "
              f"${m['max_dd']:>+7,.0f}  "
              f"{m['calmar']:>6.2f}x  "
              f"{m['neg_months']:>5}{marker}")

    winners = [m for m in all_m if m['pnl'] > base['pnl']]
    if winners:
        winners.sort(key=lambda x: x['pnl'], reverse=True)
        print(f"\n{'='*100}\n  ★ {len(winners)} VARIANT(S) BEAT BASELINE ON TOTAL P&L\n{'='*100}")
        for m in winners:
            delta = m['pnl'] - base['pnl']
            dd_delta = m['max_dd'] - base['max_dd']
            print(f"  {m['label']}")
            print(f"    P&L:  ${m['pnl']:+,.0f} (Δ ${delta:+,.0f}, {delta/abs(base['pnl'])*100:+.1f}%)")
            print(f"    DD:   ${m['max_dd']:+,.0f} (Δ ${dd_delta:+,.0f} vs baseline)")
            print(f"    Calmar: {m['calmar']:.2f}x vs baseline {base['calmar']:.2f}x")
            print(f"    Neg months: {m['neg_months']} (baseline: {base['neg_months']})")
    else:
        print(f"\n{'='*100}\n  ✗ No variant beat baseline on total P&L\n{'='*100}")

    # Runner table — how did each variant handle the top-10 V0 MFE runners
    print(f"\n{'='*100}")
    print("  TOP-12 RUNNERS — per-variant sized P&L ($)")
    print(f"{'='*100}")
    v0_sel = all_sel[base['label']]
    runners = v0_sel[v0_sel['mfe_r'] >= 3.0].copy().sort_values('mfe_r', ascending=False).head(12)
    out = runners[['symbol', 'date', 'mfe_r']].copy()
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    for m in all_m:
        sel = all_sel[m['label']].set_index(['symbol', 'date'])
        pnls = []
        for _, r in runners.iterrows():
            k = (r['symbol'], r['date'])
            v = sel.loc[k, '_sized_pnl'] if k in sel.index else None
            if hasattr(v, 'iloc'): v = v.iloc[0]
            pnls.append(v)
        # Short column name for table
        short = m['label'].split()[0]
        out[short] = pnls
    pd.set_option('display.width', 220)
    pd.set_option('display.float_format', '{:,.0f}'.format)
    print(out.to_string(index=False))

    # Total runner P&L per variant
    print(f"\nTotal runner (MFE>=3R) P&L by variant:")
    for m in all_m:
        sel = all_sel[m['label']]
        rr = sel[sel['mfe_r'] >= 3.0]
        print(f"  {m['label']:<55} ${rr['_sized_pnl'].sum():>+11,.0f}  ({len(rr)} runners)")

    pd.DataFrame(all_m).to_csv('analysis_results/orb_exit_v3_summary.csv', index=False)
    print(f"\nSaved analysis_results/orb_exit_v3_summary.csv")


if __name__ == '__main__':
    main()

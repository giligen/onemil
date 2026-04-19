#!/usr/bin/env python3
"""Test exit-rule variants. Re-simulate every entered trade under 5 variants:

  V1  base            stop=range_low, target=range_high + 2*range_size, else EOD
  V2  breakeven_1r    V1 + move stop to entry after price reaches +1R
  V3  trail_1_5r      V1 + after +1.5R, set stop to entry+0.5R, then trail 1R behind high
  V4  time_exit_11    V1 but force-close at 11:00 ET bar if still in trade
  V5  combined        breakeven_1r + trail_1_5r + time_exit_11

Each variant re-runs through the full defended pipeline (filter + cap + dedup +
Q5 cap + adaptive) so we compare apples-to-apples at the portfolio level.

For speed, we REUSE the existing features CSV (entry-time features don't change
between variants) and only re-simulate the POST-ENTRY exit sequence for each
trade. Bars loaded in bulk from cache.db.

Outputs: per-variant P&L (full timeline and per-split), DD, Calmar, worst day,
worst trade. Compared against each other and against the prior defended
baseline (V1).
"""
from __future__ import annotations

import os, sys, glob, sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database
from study_orb import (
    SPLITS, OUT_DIR, _bars_to_df, _session_open_timestamp,
    ENTRY_SLIP_BPS_DEFAULT, EXIT_SLIP_BPS_DEFAULT,
)
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, ADAPTIVE_MULT_MIN,
    fit_quintile_cutoffs, assign_quintile,
)
from study_orb_correlation_filter import (
    symbol_family, symbol_super_group,
)

# Pipeline config (matches defended recommendation)
ACCOUNT = 100_000
N_MAX = 4
RISK = 3000
OLD_POS = 50_000
MIN_STOP_PCT = 1.0
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}


# =========================================================================
# Simulator — replays exit logic on bars, given a known entry
# =========================================================================

def simulate_exit(
    bars_df: pd.DataFrame,
    entry_price: float,
    range_high: float,
    range_low: float,
    entry_time: pd.Timestamp,
    variant: str,
    exit_slip_bps: float = EXIT_SLIP_BPS_DEFAULT,
) -> Tuple[float, str, pd.Timestamp]:
    """Replay exit sequence given variant. Returns (exit_price, exit_reason, exit_time)."""
    range_size = range_high - range_low
    target_price = range_high + 2.0 * range_size

    # Initial stop
    stop_price = range_low
    trail_high = entry_price  # track running high for trailing
    breakeven_armed = False
    trail_armed = False

    # Find 11:00 ET bar timestamp — UTC is either 15:00 (EDT) or 16:00 (EST)
    # We handle both by looking for the bar with hour in (15, 16) and minute 0
    def is_11am_et(ts: pd.Timestamp) -> bool:
        return ts.minute == 0 and ts.hour in (15, 16)

    post_entry = bars_df[bars_df['timestamp'] >= entry_time].reset_index(drop=True)

    # R-multiples for trigger levels
    one_r_level = entry_price + range_size          # +1R above entry
    one_five_r_level = entry_price + 1.5 * range_size  # +1.5R above entry

    for _, row in post_entry.iloc[1:].iterrows():
        bar_high = float(row['high'])
        bar_low = float(row['low'])
        bar_close = float(row['close'])
        ts = row['timestamp']

        # Update trailing high for trail-variants
        if bar_high > trail_high:
            trail_high = bar_high

        # VARIANT-specific stop adjustments (before checking stop hit)
        if variant in ('V2_breakeven_1r', 'V5_combined'):
            if not breakeven_armed and bar_high >= one_r_level:
                # Arm breakeven — move stop to entry
                stop_price = max(stop_price, entry_price)
                breakeven_armed = True

        if variant in ('V3_trail_1_5r', 'V5_combined'):
            if not trail_armed and bar_high >= one_five_r_level:
                # Arm trail: move stop to entry + 0.5R
                stop_price = max(stop_price, entry_price + 0.5 * range_size)
                trail_armed = True
            if trail_armed:
                # Trail 1R behind running high
                new_trail = trail_high - range_size
                stop_price = max(stop_price, new_trail)

        # Check exits (stop first — conservative)
        if bar_low <= stop_price:
            raw = stop_price
            return (raw * (1 - exit_slip_bps / 10000), 'stop' if stop_price <= range_low else 'trail_stop', ts)
        if bar_high >= target_price:
            raw = target_price
            return (raw * (1 - exit_slip_bps / 10000), 'target', ts)

        # Variant-specific time exit
        if variant in ('V4_time_exit_11', 'V5_combined'):
            if is_11am_et(ts):
                raw = bar_close
                return (raw * (1 - exit_slip_bps / 10000), 'time_11', ts)

    # EOD exit
    last = post_entry.iloc[-1]
    raw = float(last['close'])
    return (raw * (1 - exit_slip_bps / 10000), 'eod', last['timestamp'])


# =========================================================================
# Main: load trades + bars, re-simulate each variant
# =========================================================================

def resimulate_all(features_df: pd.DataFrame, bars_cache: Dict[Tuple[str, str], pd.DataFrame],
                   variants: List[str]) -> Dict[str, pd.DataFrame]:
    """For each variant, produce a features-like DataFrame with new pnl/pnl_pct."""
    # Sanity check: need entry_price and the bars to replay
    results = {v: features_df.copy() for v in variants}

    # For each row, we need to re-derive the entry_time from bars
    # and then replay. To avoid re-deriving entry_time every variant,
    # we do one pass computing (entry_time, range_high, range_low) per row.
    entry_info = []
    skipped = 0
    for idx, row in features_df.iterrows():
        key = (row['symbol'], row['date'])
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            entry_info.append(None)
            skipped += 1
            continue
        # Find open_ts and range
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            entry_info.append(None)
            skipped += 1
            continue
        range_end = open_ts + timedelta(minutes=5)
        range_mask = (bars['timestamp'] >= open_ts) & (bars['timestamp'] < range_end)
        range_bars = bars.loc[range_mask]
        if len(range_bars) < 5:
            entry_info.append(None)
            skipped += 1
            continue
        rh = float(range_bars['high'].max())
        rl = float(range_bars['low'].min())
        # Re-derive entry_time (find first bar after range where high > rh)
        search_start = range_end
        search_end = range_end + timedelta(minutes=60)
        srch = bars[(bars['timestamp'] >= search_start) & (bars['timestamp'] < search_end)]
        entry_ts = None
        for _, b in srch.iterrows():
            if float(b['high']) > rh:
                entry_ts = b['timestamp']
                break
        if entry_ts is None:
            entry_info.append(None)
            skipped += 1
            continue
        entry_info.append({
            'bars': bars, 'range_high': rh, 'range_low': rl,
            'entry_time': entry_ts, 'entry_price': float(row['entry_price']),
        })

    print(f"  Entry info reconstructed for {len(entry_info) - skipped}/{len(features_df)} trades "
          f"(skipped {skipped})")

    # For each variant, replay
    for v in variants:
        pnls = []
        pnl_pcts = []
        reasons = []
        for idx, info in enumerate(entry_info):
            if info is None:
                pnls.append(features_df.iloc[idx]['pnl'])
                pnl_pcts.append(features_df.iloc[idx]['pnl_pct'])
                reasons.append(features_df.iloc[idx]['exit_reason'])
                continue
            exit_price, exit_reason, _ = simulate_exit(
                info['bars'], info['entry_price'],
                info['range_high'], info['range_low'],
                info['entry_time'], v,
            )
            # Compute shares and P&L ($50K position baseline — downstream
            # pipeline will re-scale to risk-parity)
            entry_p = info['entry_price']
            shares = max(1, int(OLD_POS / entry_p))
            pnl = (exit_price - entry_p) * shares
            pnl_pct = (exit_price - entry_p) / entry_p * 100
            pnls.append(pnl)
            pnl_pcts.append(pnl_pct)
            reasons.append(exit_reason)

        df = features_df.copy()
        df['pnl'] = pnls
        df['pnl_pct'] = pnl_pcts
        df['exit_reason'] = reasons
        df['win'] = (df['pnl'] > 0).astype(int)
        results[v] = df
    return results


# =========================================================================
# Pipeline (reuse from defended framework)
# =========================================================================

def apply_rp(df, risk, per_pos_cap):
    df = df.copy()
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = risk / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS
    return df


def fit_mults_capped(tk):
    avg = float(tk['_rp_pnl'].mean()) if len(tk) else 1.0
    out = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = tk[tk['_quintile'] == q]
        if len(sub) == 0 or avg <= 0:
            out[q] = 1.0; continue
        raw = float(sub['_rp_pnl'].mean()) / avg
        out[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], raw))
    return out


def select_defended(dg, k):
    d = dg.copy()
    d['_q_rank'] = d['_quintile'].map(Q_ORDER)
    d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
    seen_fam = set(); seen_sup = set()
    kept = []
    for _, r in d.iterrows():
        sym = r['symbol']
        fam = symbol_family(sym); sup = symbol_super_group(sym)
        if fam and fam in seen_fam: continue
        if sup and sup in seen_sup: continue
        if fam: seen_fam.add(fam)
        if sup: seen_sup.add(sup)
        kept.append(r)
        if len(kept) >= k: break
    return pd.DataFrame(kept)


def full_pipeline_metrics(df: pd.DataFrame, k: int, risk: float) -> Dict:
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    per_pos_cap = ACCOUNT / k
    df = apply_rp(df, risk, per_pos_cap)

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
    sel = pd.concat([select_defended(dg, k) for _, dg in kept.groupby('date')])
    sel['_sized_pnl'] = sel.apply(
        lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)

    # Restrict to Jan'25-Apr'26 and compute daily + DD
    sel_time = sel[(sel['date'] >= '2025-01-01') & (sel['date'] <= '2026-04-30')]
    daily = sel_time.groupby('date').agg(
        pnl=('_sized_pnl', 'sum'),
        n=('_rp_pnl', 'count'),
    ).reset_index().sort_values('date').reset_index(drop=True)
    if len(daily) == 0:
        return {'pnl': 0, 'dd': 0, 'worst_day': 0, 'worst_trade': 0, 'n_days': 0}
    daily['cum'] = daily['pnl'].cumsum()
    running = -np.inf; dd = 0.0
    for c in daily['cum']:
        running = max(running, c)
        dd = min(dd, c - running)
    return {
        'pnl': float(daily['pnl'].sum()),
        'dd': float(dd),
        'worst_day': float(daily['pnl'].min()),
        'worst_trade': float(sel_time['_sized_pnl'].min()) if len(sel_time) else 0,
        'n_days': len(daily),
        'exit_reason_counts': dict(sel_time.pivot_table(
            values='_sized_pnl', index='exit_reason', aggfunc='count').iloc[:, 0]) if 'exit_reason' in sel_time.columns else {},
    }


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    print(f"Loading features CSV: {csv}")
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl','date','range_size_pct','entry_price'])
    print(f"Features rows: {len(df):,}")

    # Get unique (symbol, date) pairs to load bars for
    pairs = list(df[['symbol', 'date']].drop_duplicates().itertuples(index=False, name=None))
    print(f"Unique (symbol, date) pairs: {len(pairs)}")

    # Bulk load bars
    print("Loading bars from cache.db...")
    t0 = datetime.now()
    db_obj = Database(db_path='data/cache.db')
    raw_bars = db_obj.get_intraday_bars_bulk(pairs)
    db_obj.close()
    print(f"  Loaded {len(raw_bars)} bar sets in {(datetime.now()-t0).total_seconds():.1f}s")

    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}

    variants = ['V1_base', 'V2_breakeven_1r', 'V3_trail_1_5r',
                'V4_time_exit_11', 'V5_combined']

    print(f"\nRe-simulating {len(variants)} exit variants over {len(df):,} trades...")
    t0 = datetime.now()
    per_variant = resimulate_all(df, bars_cache, variants)
    print(f"  Done in {(datetime.now()-t0).total_seconds():.1f}s")

    # Verify V1 matches original baseline (sanity check)
    v1_pnl_diff = (per_variant['V1_base']['pnl'] - df['pnl']).abs().mean()
    print(f"\nV1_base vs original CSV: mean |pnl diff| = ${v1_pnl_diff:.2f} "
          f"(should be ~0, small rounding OK)")

    # Per-variant exit reason distribution
    print(f"\n{'='*100}")
    print("PER-VARIANT: exit reason distribution (all 3,258 entered trades)")
    print(f"{'='*100}")
    print(f"  {'Variant':<20} "
          + " ".join(f"{r:>10}" for r in ['stop', 'target', 'trail_stop', 'time_11', 'eod']))
    for v in variants:
        d = per_variant[v]
        counts = d['exit_reason'].value_counts().to_dict()
        row = [f"{counts.get(r, 0):>10}" for r in ['stop', 'target', 'trail_stop', 'time_11', 'eod']]
        print(f"  {v:<20} {' '.join(row)}")

    # Per-variant pipeline metrics at recommended config
    print(f"\n{'='*120}")
    print(f"PIPELINE METRICS — N={N_MAX}, risk=${RISK:,}, defended (Q5 cap + dedup), Jan'25-Apr'26")
    print(f"{'='*120}")
    print(f"  {'Variant':<20} {'Full P&L':>12} {'Full DD':>11} {'Worst day':>11} "
          f"{'Worst trade':>12} {'Calmar':>8}")
    print('  ' + '-' * 85)
    per_v_metrics = {}
    for v in variants:
        m = full_pipeline_metrics(per_variant[v], N_MAX, RISK)
        per_v_metrics[v] = m
        calmar = m['pnl'] / abs(m['dd']) if m['dd'] < 0 else float('inf')
        print(f"  {v:<20} ${m['pnl']:>+10,.0f} ${m['dd']:>+9,.0f} "
              f"${m['worst_day']:>+9,.0f} ${m['worst_trade']:>+10,.0f} {calmar:>7.2f}x")

    # Head-to-head: best variant vs V1
    print(f"\n{'='*120}")
    print("BEST BY CALMAR — full-timeline comparison vs V1_base")
    print(f"{'='*120}")
    best_v = max(variants, key=lambda v: per_v_metrics[v]['pnl'] / max(abs(per_v_metrics[v]['dd']), 1))
    v1 = per_v_metrics['V1_base']
    bv = per_v_metrics[best_v]
    print(f"\n  V1_base:    P&L ${v1['pnl']:+,.0f}  DD ${v1['dd']:+,.0f}  "
          f"Calmar {v1['pnl']/abs(v1['dd']):.2f}x")
    print(f"  {best_v}:    P&L ${bv['pnl']:+,.0f}  DD ${bv['dd']:+,.0f}  "
          f"Calmar {bv['pnl']/abs(bv['dd']):.2f}x")
    print(f"\n  Impact: P&L ${bv['pnl']-v1['pnl']:+,.0f}, DD ${bv['dd']-v1['dd']:+,.0f}")

    # Save per-variant pnl tables
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    for v in variants:
        path = f'{OUT_DIR}/orb_exit_{v}_{ts}.csv'
        per_variant[v].to_csv(path, index=False)
    print(f"\nSaved per-variant trade CSVs to {OUT_DIR}/orb_exit_V*_{ts}.csv")


if __name__ == '__main__':
    main()

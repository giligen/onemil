"""ORB p20h floor filter — walk-forward validation.

PRE-REGISTERED HYPOTHESIS
-------------------------
H1: ORB entries with `price_vs_20d_high_pct < THRESHOLD` (deep drawdown
    setups) underperform other entries. Filter improves all 3 splits
    (TRAIN H1 2025 / VAL H2 2025 / OOS 2026) by >=5%, MDD increase <=10%,
    no period turns negative.

H0: p20h is uncorrelated with ORB outcome at any cutoff. Today's losers
    were random draws from a positive-EV population.

PRELIMINARY EVIDENCE (refuting H1 before formal test)
-----------------------------------------------------
Raw 3,371-trade distribution shows the OPPOSITE: trades with p20h <= -50%
are the BEST mean-P&L bucket (+$467/trade vs -$161 for trades with
p20h >= 0). Existing composite already gives credit for low p20h
(price_vs_20d_high_pct sign=-1).

This script formalises the negative result: sweeps p20h floor against
the production-parity static_lock_1R pipeline and confirms.

USAGE
-----
    python3 study_orb_p20h_filter.py
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import timedelta

import numpy as np
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
LOCK_TRIGGER_R = 1.5
LOCK_STOP_R = 1.0
EXIT_SLIP_BPS = 10.0

P20H_THRESHOLDS = [None, -60.0, -50.0, -40.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0]

SPLITS_ANALYSIS = [
    ('TRAIN', '2025-01-01', '2025-06-30'),
    ('VAL',   '2025-07-01', '2025-12-31'),
    ('OOS',   '2026-01-01', '2026-12-31'),
]


def _session_open_timestamp(bars):
    if bars.empty:
        return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning):
        return morning.iloc[0]['timestamp']
    return None


def simulate_static_lock(bars, entry_price, range_high, range_low, entry_time):
    range_size = range_high - range_low
    trigger_lvl = entry_price + LOCK_TRIGGER_R * range_size
    lock_stop = entry_price + LOCK_STOP_R * range_size
    stop_price = range_low
    armed = False
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        bar_high = float(row['high']); bar_low = float(row['low'])
        if not armed and bar_high >= trigger_lvl:
            armed = True
            stop_price = max(stop_price, lock_stop)
        if bar_low <= stop_price:
            return stop_price * (1 - EXIT_SLIP_BPS/10000), 'lock' if armed else 'stop'
    last = post.iloc[-1]
    return float(last['close']) * (1 - EXIT_SLIP_BPS/10000), 'eod'


def resimulate_static_lock(df, bars_cache):
    new_pnls = []; new_pnl_pcts = []; new_reasons = []
    for _, row in df.reset_index(drop=True).iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) & (bars['timestamp'] < range_end)]
        if len(range_bars) < 5:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh:
                entry_ts = b['timestamp']; break
        if entry_ts is None:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue
        entry_p = float(row['entry_price'])
        exit_p, reason = simulate_static_lock(bars, entry_p, rh, rl, entry_ts)
        shares = max(1, int(OLD_POS / entry_p))
        new_pnls.append((exit_p - entry_p) * shares)
        new_pnl_pcts.append((exit_p - entry_p) / entry_p * 100)
        new_reasons.append(reason)
    return new_pnls, new_pnl_pcts, new_reasons


def run_pipeline(df, p20h_floor):
    """Run the full production-parity pipeline with optional p20h floor.
    Returns (sel_df, mults).
    """
    df = df.copy()
    if p20h_floor is not None:
        before = len(df)
        df = df[df['price_vs_20d_high_pct'] >= p20h_floor].copy()
        # diagnostic stored as attr
        df.attrs['n_before_filter'] = before
        df.attrs['n_after_filter'] = len(df)

    # Risk-parity sizing
    per_pos_cap = ACCOUNT / N
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS

    # Composite — fit on TRAIN H1 2025 ONLY
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    if len(train) < 50:
        return None, None
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    train_k = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    if len(train_k) < 20:
        return None, None
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    avg = float(train_k['_rp_pnl'].mean())
    if avg == 0:
        return None, None
    mults = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = train_k[train_k['_quintile'] == q]
        if len(sub) == 0:
            mults[q] = ADAPTIVE_MULT_MIN
        else:
            mults[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], float(sub['_rp_pnl'].mean()) / avg))

    # Apply pipeline
    kept = df[df['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)
    kept = kept[kept['_quintile'] != 'Q1'].copy()  # Q1 filter on (default-ON)

    # Per-day top-N + family/super dedup
    sel_rows = []
    for day, dg in kept.groupby('date'):
        d = dg.copy()
        d['_q_rank'] = d['_quintile'].map(Q_ORDER)
        d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        seen_fam = set(); seen_sup = set()
        kept_today = []
        for _, r in d.iterrows():
            fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_fam: continue
            if sup and sup in seen_sup: continue
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
            kept_today.append(r)
            if len(kept_today) >= N: break
        sel_rows.extend(kept_today)
    if not sel_rows:
        return None, mults
    sel = pd.DataFrame(sel_rows)
    sel['_sized_pnl'] = sel.apply(lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    sel['date'] = pd.to_datetime(sel['date'])
    return sel, mults


def period_metrics(sel, lo, hi):
    sub = sel[(sel['date'] >= lo) & (sel['date'] <= hi)].copy()
    if len(sub) == 0:
        return {'n': 0, 'pnl': 0.0, 'mdd': 0.0, 'wr': 0.0, 'calmar': 0.0}
    sub = sub.sort_values('date').reset_index(drop=True)
    daily = sub.groupby('date')['_sized_pnl'].sum().reset_index().sort_values('date')
    daily['cum'] = daily['_sized_pnl'].cumsum()
    peak = -1e18; mdd = 0.0
    for c in daily['cum']:
        peak = max(peak, c); mdd = min(mdd, c - peak)
    pnl = float(sub['_sized_pnl'].sum())
    wr = float((daily['_sized_pnl'] > 0).mean() * 100)
    calmar = (pnl / abs(mdd)) if mdd != 0 else 0.0
    return {'n': len(sub), 'pnl': pnl, 'mdd': mdd, 'wr': wr, 'calmar': calmar}


def bootstrap_diff(arr_a, arr_b, n_iter=2000, seed=42):
    """Return (mean_diff, lo95, hi95) for arr_a.mean() - arr_b.mean()."""
    if len(arr_a) == 0 or len(arr_b) == 0:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.RandomState(seed)
    diffs = []
    a = np.array(arr_a); b = np.array(arr_b)
    for _ in range(n_iter):
        ar = rng.choice(a, size=len(a), replace=True)
        br = rng.choice(b, size=len(b), replace=True)
        diffs.append(ar.mean() - br.mean())
    diffs = np.sort(diffs)
    return float(np.mean(diffs)), float(diffs[int(0.025*n_iter)]), float(diffs[int(0.975*n_iter)])


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    print(f"Features: {csv}")
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'pnl_pct', 'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])
    print(f"Rows after dropna: {len(df)}, range {df['date'].min().date()} → {df['date'].max().date()}")

    # Re-simulate with static_lock_1R
    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    print(f"Loading bars for {len(pairs)} pairs...")
    db_obj = Database(db_path='data/cache.db')
    raw_bars = db_obj.get_intraday_bars_bulk(pairs)
    db_obj.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}
    print("Re-simulating exits with static_lock_1R...")
    new_pnls, new_pnl_pcts, new_reasons = resimulate_static_lock(df, bars_cache)
    df = df.reset_index(drop=True)
    df['pnl'] = new_pnls
    df['pnl_pct'] = new_pnl_pcts
    df['exit_reason'] = new_reasons

    # Run sweep
    print()
    print(f"{'='*100}")
    print("p20h FLOOR SWEEP — production-parity pipeline + static_lock_1R")
    print(f"{'='*100}")

    rows = []
    base_sel = None
    for thr in P20H_THRESHOLDS:
        sel, mults = run_pipeline(df, thr)
        if sel is None:
            print(f"  threshold={thr}: pipeline aborted (insufficient TRAIN trades)")
            continue
        if thr is None:
            base_sel = sel.copy()
        full = period_metrics(sel, '2025-01-01', '2026-12-31')
        train = period_metrics(sel, '2025-01-01', '2025-06-30')
        val = period_metrics(sel, '2025-07-01', '2025-12-31')
        oos = period_metrics(sel, '2026-01-01', '2026-12-31')
        rows.append({
            'p20h_floor': 'none' if thr is None else f'{thr:+.0f}%',
            'n_total': full['n'],
            'TRAIN_pnl': train['pnl'], 'TRAIN_n': train['n'],
            'VAL_pnl': val['pnl'], 'VAL_n': val['n'],
            'OOS_pnl': oos['pnl'], 'OOS_n': oos['n'],
            'full_pnl': full['pnl'],
            'full_mdd': full['mdd'],
            'full_calmar': full['calmar'],
            'full_wr': full['wr'],
        })

    out = pd.DataFrame(rows)
    print()
    print(f"{'p20h_floor':>11} | {'n':>4} | {'TRAIN $':>10} ({'n':>3}) | {'VAL $':>10} ({'n':>3}) | {'OOS $':>10} ({'n':>3}) | {'FULL $':>11} | {'MDD':>9} | {'Calmar':>6} | {'WR%':>5}")
    print('-'*145)
    base_full = float(out[out['p20h_floor']=='none']['full_pnl'].iloc[0])
    for _, r in out.iterrows():
        delta = r['full_pnl'] - base_full
        print(f"{r['p20h_floor']:>11} | {r['n_total']:>4} | "
              f"${r['TRAIN_pnl']:>+9,.0f} ({r['TRAIN_n']:>3}) | "
              f"${r['VAL_pnl']:>+9,.0f} ({r['VAL_n']:>3}) | "
              f"${r['OOS_pnl']:>+9,.0f} ({r['OOS_n']:>3}) | "
              f"${r['full_pnl']:>+10,.0f} | "
              f"${r['full_mdd']:>+8,.0f} | {r['full_calmar']:>6.2f} | {r['full_wr']:>4.1f}%   "
              f"Δ vs base ${delta:+,.0f}")

    # Counterfactual: what did each filter drop?
    if base_sel is not None:
        print()
        print(f"{'='*100}")
        print("COUNTERFACTUAL — distribution of trades dropped by each filter (in base pipeline)")
        print(f"{'='*100}")
        for thr in P20H_THRESHOLDS:
            if thr is None: continue
            dropped = base_sel[base_sel['price_vs_20d_high_pct'] < thr]
            kept = base_sel[base_sel['price_vs_20d_high_pct'] >= thr]
            if len(dropped) == 0:
                print(f"  floor={thr:+.0f}%: drops 0 trades")
                continue
            n = len(dropped)
            mean_d = dropped['_sized_pnl'].mean()
            sum_d = dropped['_sized_pnl'].sum()
            wr_d = (dropped['_sized_pnl'] > 0).mean() * 100
            mean_diff, lo95, hi95 = bootstrap_diff(
                dropped['_sized_pnl'].values, kept['_sized_pnl'].values)
            sig = '*' if (lo95 < 0 and hi95 < 0) or (lo95 > 0 and hi95 > 0) else ' '
            print(f"  floor={thr:+.0f}%: n_drop={n:>3} ({n/len(base_sel)*100:>4.1f}%)  "
                  f"mean_dropped=${mean_d:>+8.0f}  sum_dropped=${sum_d:>+10,.0f}  "
                  f"WR_dropped={wr_d:>4.1f}%  "
                  f"Δmean(drop-keep)=${mean_diff:>+7.0f}  95% CI [${lo95:>+7.0f}, ${hi95:>+7.0f}] {sig}")

    out.to_csv('analysis_results/orb_p20h_filter_sweep.csv', index=False)
    print()
    print("Saved: analysis_results/orb_p20h_filter_sweep.csv")

    # Decision
    print()
    print(f"{'='*100}")
    print("DECISION (per pre-registered rejection rule)")
    print(f"{'='*100}")
    print(f"Rejection rule: ship only if all of TRAIN/VAL/OOS lift >=5% AND no period turns negative")
    print(f"               AND MDD increase <=10%.")
    base = out.iloc[0]
    base_t = base['TRAIN_pnl']; base_v = base['VAL_pnl']; base_o = base['OOS_pnl']; base_m = abs(base['full_mdd'])
    qualifying = []
    for _, r in out.iloc[1:].iterrows():
        t_lift = (r['TRAIN_pnl'] - base_t) / abs(base_t) * 100 if base_t != 0 else 0
        v_lift = (r['VAL_pnl'] - base_v) / abs(base_v) * 100 if base_v != 0 else 0
        o_lift = (r['OOS_pnl'] - base_o) / abs(base_o) * 100 if base_o != 0 else 0
        mdd_change = (abs(r['full_mdd']) - base_m) / base_m * 100 if base_m != 0 else 0
        all_pos = (r['TRAIN_pnl'] > 0 and r['VAL_pnl'] > 0 and r['OOS_pnl'] > 0)
        if t_lift >= 5 and v_lift >= 5 and o_lift >= 5 and all_pos and mdd_change <= 10:
            qualifying.append(r['p20h_floor'])
    if qualifying:
        print(f"  Thresholds passing all gates: {qualifying}")
        print(f"  → consider shipping, with default-OFF feature flag")
    else:
        print(f"  NO threshold passes all gates.")
        print(f"  → DO NOT SHIP. Hypothesis rejected by walk-forward + MDD check.")
        print(f"  → Today's RDWU was a single bad draw from a positive-EV population.")


if __name__ == '__main__':
    main()

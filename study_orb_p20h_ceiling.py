"""ORB p20h CEILING filter (symmetric opposite of floor test).

PRE-REGISTERED HYPOTHESIS
-------------------------
H1: ORB entries with `price_vs_20d_high_pct >= THRESHOLD` (stocks at or
    above their 20-day high) underperform other entries. The raw unfiltered
    distribution shows the >=0% bucket averaging -$161/trade vs +$300+ for
    deep-drawdown buckets. A ceiling filter would drop these.

H0: After the existing composite filter (which already penalizes high p20h
    via sign=-1), the residual high-p20h trades that survive are the
    "compensated" subset and perform on par with kept trades.

REJECTION RULE (same as #2 / #3)
--------------------------------
Ship only if a threshold satisfies all 4 gates simultaneously:
  - TRAIN/VAL/OOS lift each >=5%
  - No period turns negative
  - MDD increase <=10%

Sweep ceilings: [None, +20%, +10%, +5%, 0%, -5%, -10%]
  None = baseline (no ceiling)
  +20% = drop trades with p20h >= +20%
  -10% = aggressive: drop trades within 10% of 20d high

USAGE
-----
    python3 study_orb_p20h_ceiling.py
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

CEILINGS = [None, 50.0, 30.0, 20.0, 10.0, 5.0, 0.0, -5.0, -10.0]


def _session_open_timestamp(bars):
    if bars.empty: return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning): return morning.iloc[0]['timestamp']
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


def run_pipeline(df, ceiling=None):
    df = df.copy()
    if ceiling is not None:
        df = df[df['price_vs_20d_high_pct'] < ceiling].copy()

    per_pos_cap = ACCOUNT / N
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS

    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    if len(train) < 50: return None
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-06-30')]
    train_k = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    if len(train_k) < 20: return None
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    avg = float(train_k['_rp_pnl'].mean())
    if avg == 0: return None
    mults = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = train_k[train_k['_quintile'] == q]
        mults[q] = (max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], float(sub['_rp_pnl'].mean()) / avg))
                    if len(sub) else ADAPTIVE_MULT_MIN)

    kept = df[df['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)
    kept = kept[kept['_quintile'] != 'Q1'].copy()

    sel_rows = []
    for day, dg in kept.groupby('date'):
        d = dg.copy()
        d['_q_rank'] = d['_quintile'].map(Q_ORDER)
        d = d.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        seen_fam = set(); seen_sup = set(); kept_today = []
        for _, r in d.iterrows():
            fam = symbol_family(r['symbol']); sup = symbol_super_group(r['symbol'])
            if fam and fam in seen_fam: continue
            if sup and sup in seen_sup: continue
            if fam: seen_fam.add(fam)
            if sup: seen_sup.add(sup)
            kept_today.append(r)
            if len(kept_today) >= N: break
        sel_rows.extend(kept_today)
    if not sel_rows: return None
    sel = pd.DataFrame(sel_rows)
    sel['_sized_pnl'] = sel.apply(lambda r: r['_rp_pnl'] * mults[r['_quintile']], axis=1)
    sel['date'] = pd.to_datetime(sel['date'])
    return sel


def period_metrics(sel, lo, hi):
    sub = sel[(sel['date'] >= lo) & (sel['date'] <= hi)].copy()
    if len(sub) == 0: return {'n': 0, 'pnl': 0.0, 'mdd': 0.0}
    sub = sub.sort_values('date').reset_index(drop=True)
    daily = sub.groupby('date')['_sized_pnl'].sum().reset_index().sort_values('date')
    daily['cum'] = daily['_sized_pnl'].cumsum()
    peak = -1e18; mdd = 0.0
    for c in daily['cum']:
        peak = max(peak, c); mdd = min(mdd, c - peak)
    return {'n': len(sub), 'pnl': float(sub['_sized_pnl'].sum()), 'mdd': mdd}


def bootstrap_diff(arr_a, arr_b, n_iter=2000, seed=42):
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
    print(f"Rows: {len(df)}")

    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    print(f"Loading bars for {len(pairs)} pairs...")
    db_obj = Database(db_path='data/cache.db')
    raw_bars = db_obj.get_intraday_bars_bulk(pairs)
    db_obj.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}
    print("Re-simulating static_lock_1R...")
    new_pnls, new_pnl_pcts, new_reasons = resimulate_static_lock(df, bars_cache)
    df = df.reset_index(drop=True)
    df['pnl'] = new_pnls
    df['pnl_pct'] = new_pnl_pcts
    df['exit_reason'] = new_reasons

    print()
    print(f"{'='*100}")
    print("p20h CEILING SWEEP")
    print(f"{'='*100}")

    rows = []
    base_sel = None
    for c in CEILINGS:
        sel = run_pipeline(df, ceiling=c)
        if sel is None:
            print(f"  ceiling={c}: pipeline aborted"); continue
        if c is None:
            base_sel = sel.copy()
        full = period_metrics(sel, '2025-01-01', '2026-12-31')
        train = period_metrics(sel, '2025-01-01', '2025-06-30')
        val = period_metrics(sel, '2025-07-01', '2025-12-31')
        oos = period_metrics(sel, '2026-01-01', '2026-12-31')
        rows.append({
            'ceiling': 'none' if c is None else f'<{c:+.0f}%',
            'n_total': full['n'],
            'TRAIN_pnl': train['pnl'], 'TRAIN_n': train['n'],
            'VAL_pnl': val['pnl'], 'VAL_n': val['n'],
            'OOS_pnl': oos['pnl'], 'OOS_n': oos['n'],
            'full_pnl': full['pnl'],
            'full_mdd': full['mdd'],
            'full_calmar': (full['pnl']/abs(full['mdd'])) if full['mdd'] else 0,
        })

    base_pnl = rows[0]['full_pnl']
    print()
    print(f"{'ceiling':>10} | {'n':>4} | {'TRAIN $':>10} ({'n':>3}) | {'VAL $':>10} ({'n':>3}) | {'OOS $':>10} ({'n':>3}) | {'FULL $':>11} | {'MDD':>9} | {'Calmar':>6} | {'Δ vs base':>11}")
    print('-' * 145)
    for r in rows:
        delta = r['full_pnl'] - base_pnl
        print(f"{r['ceiling']:>10} | {r['n_total']:>4} | "
              f"${r['TRAIN_pnl']:>+9,.0f} ({r['TRAIN_n']:>3}) | "
              f"${r['VAL_pnl']:>+9,.0f} ({r['VAL_n']:>3}) | "
              f"${r['OOS_pnl']:>+9,.0f} ({r['OOS_n']:>3}) | "
              f"${r['full_pnl']:>+10,.0f} | "
              f"${r['full_mdd']:>+8,.0f} | {r['full_calmar']:>6.2f} | ${delta:>+10,.0f}")

    # Counterfactual on dropped trades
    if base_sel is not None:
        print()
        print(f"{'='*100}")
        print("COUNTERFACTUAL — dropped trades")
        print(f"{'='*100}")
        for c in CEILINGS:
            if c is None: continue
            dropped = base_sel[base_sel['price_vs_20d_high_pct'] >= c]
            kept = base_sel[base_sel['price_vs_20d_high_pct'] < c]
            if len(dropped) == 0 or len(kept) == 0:
                print(f"  ceiling=<{c:+.0f}%: drops 0"); continue
            mean_d = dropped['_sized_pnl'].mean()
            sum_d = dropped['_sized_pnl'].sum()
            wr_d = (dropped['_sized_pnl'] > 0).mean() * 100
            mean_diff, lo95, hi95 = bootstrap_diff(
                dropped['_sized_pnl'].values, kept['_sized_pnl'].values)
            sig = '*' if (lo95 < 0 and hi95 < 0) or (lo95 > 0 and hi95 > 0) else ' '
            print(f"  ceiling=<{c:+.0f}%: n_drop={len(dropped):>3} ({len(dropped)/len(base_sel)*100:>4.1f}%)  "
                  f"mean_dropped=${mean_d:>+8.0f}  sum_dropped=${sum_d:>+10,.0f}  "
                  f"WR_dropped={wr_d:>4.1f}%  "
                  f"Δmean(drop-keep)=${mean_diff:>+7.0f}  CI [${lo95:>+7.0f}, ${hi95:>+7.0f}] {sig}")

    pd.DataFrame(rows).to_csv('analysis_results/orb_p20h_ceiling_sweep.csv', index=False)

    # Decision
    print()
    print(f"{'='*100}")
    print("DECISION")
    print(f"{'='*100}")
    base_t = rows[0]['TRAIN_pnl']; base_v = rows[0]['VAL_pnl']; base_o = rows[0]['OOS_pnl']; base_m = abs(rows[0]['full_mdd'])
    qualifying = []
    for r in rows[1:]:
        if base_t == 0 or base_v == 0 or base_o == 0 or base_m == 0: continue
        t_lift = (r['TRAIN_pnl'] - base_t) / abs(base_t) * 100
        v_lift = (r['VAL_pnl'] - base_v) / abs(base_v) * 100
        o_lift = (r['OOS_pnl'] - base_o) / abs(base_o) * 100
        mdd_change = (abs(r['full_mdd']) - base_m) / base_m * 100
        all_pos = (r['TRAIN_pnl'] > 0 and r['VAL_pnl'] > 0 and r['OOS_pnl'] > 0)
        if t_lift >= 5 and v_lift >= 5 and o_lift >= 5 and all_pos and mdd_change <= 10:
            qualifying.append((r['ceiling'], t_lift, v_lift, o_lift, r['full_pnl'] - base_pnl))
    if qualifying:
        for arm, t, v, o, d in qualifying:
            print(f"  PASS: {arm}: TRAIN+{t:.1f}% VAL+{v:.1f}% OOS+{o:.1f}% (Δ ${d:+,.0f})")
    else:
        print(f"  NO ceiling passes all gates → DO NOT SHIP.")


if __name__ == '__main__':
    main()

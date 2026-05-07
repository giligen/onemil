"""ORB filter research — 3 new features (delay, PM-high proximity, vol confirm).

PRE-REGISTERED HYPOTHESES
-------------------------
H1 ENTRY_DELAY: trades triggering >X min after 9:35 underperform.
   Filter: skip if entry_delay_min > THRESHOLD.

H2 PM_HIGH: trades where range_high < pm_high underperform (overhead
   resistance still in place). Filter: skip if range_high / pm_high < 1.0.

H3 VOL_CONFIRM: trades where entry-bar volume < X * range_avg_bar_volume
   underperform (no real demand). Filter: skip if ratio < THRESHOLD.

REJECTION RULE (Bonferroni-strict for multi-hypothesis)
-------------------------------------------------------
Ship only if ALL of:
  - TRAIN/VAL/OOS each lift >= 5%
  - No period turns negative
  - MDD increase <= 10%
  - Bootstrap 95% CI on Δmean(drop - keep) excludes zero in correct direction
  - sum_dropped <= 0 (filter actually drops net-losers, not just lower-mean)

USAGE
-----
    python3 study_orb_new_features.py
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
            armed = True; stop_price = max(stop_price, lock_stop)
        if bar_low <= stop_price:
            return stop_price * (1 - EXIT_SLIP_BPS/10000), 'lock' if armed else 'stop'
    last = post.iloc[-1]
    return float(last['close']) * (1 - EXIT_SLIP_BPS/10000), 'eod'


def enrich_with_new_features(df, bars_cache):
    """Re-simulate exits AND compute 3 new features per trade."""
    new_pnls = []; new_pnl_pcts = []; new_reasons = []
    delay_min = []; pm_high = []; pm_high_ratio = []
    range_high_list = []; range_total_vol_list = []
    entry_bar_vol = []; vol_ratio = []

    for _, row in df.reset_index(drop=True).iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)

        # defaults
        delay_min.append(np.nan); pm_high.append(np.nan); pm_high_ratio.append(np.nan)
        range_high_list.append(np.nan); range_total_vol_list.append(np.nan)
        entry_bar_vol.append(np.nan); vol_ratio.append(np.nan)

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
        range_total = float(range_bars['volume'].sum())

        # Pre-market high (any bar in [open_ts - 5h30m, open_ts))
        pm_window_start = open_ts - timedelta(hours=5, minutes=30)
        pm = bars[(bars['timestamp'] >= pm_window_start) & (bars['timestamp'] < open_ts)]
        if len(pm) > 0:
            pm_h = float(pm['high'].max())
            pm_high[-1] = pm_h
            if pm_h > 0:
                pm_high_ratio[-1] = rh / pm_h

        # Entry bar lookup
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None; entry_bar_idx = None
        for idx, b in search.iterrows():
            if float(b['high']) > rh:
                entry_ts = b['timestamp']; entry_bar_idx = idx; break

        if entry_ts is None:
            new_pnls.append(row['pnl']); new_pnl_pcts.append(row['pnl_pct'])
            new_reasons.append(row['exit_reason']); continue

        # Time delay (minutes from 9:35 to entry bar timestamp)
        delay_min[-1] = (entry_ts - range_end).total_seconds() / 60.0

        # Entry bar volume / 5-min avg
        entry_bar = bars.loc[entry_bar_idx]
        eb_vol = float(entry_bar['volume'])
        entry_bar_vol[-1] = eb_vol
        avg_5min = range_total / 5.0
        if avg_5min > 0:
            vol_ratio[-1] = eb_vol / avg_5min

        range_high_list[-1] = rh
        range_total_vol_list[-1] = range_total

        # Static-lock exit
        entry_p = float(row['entry_price'])
        exit_p, reason = simulate_static_lock(bars, entry_p, rh, rl, entry_ts)
        shares = max(1, int(OLD_POS / entry_p))
        new_pnls.append((exit_p - entry_p) * shares)
        new_pnl_pcts.append((exit_p - entry_p) / entry_p * 100)
        new_reasons.append(reason)

    df = df.reset_index(drop=True).copy()
    df['pnl'] = new_pnls
    df['pnl_pct'] = new_pnl_pcts
    df['exit_reason'] = new_reasons
    df['_entry_delay_min'] = delay_min
    df['_pm_high'] = pm_high
    df['_pm_high_ratio'] = pm_high_ratio
    df['_entry_bar_vol'] = entry_bar_vol
    df['_vol_ratio'] = vol_ratio
    return df


def run_pipeline(df, *, mask=None):
    """Run pipeline with optional pre-filter mask (True = keep)."""
    df = df.copy()
    if mask is not None:
        df = df[mask].copy()

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
    mults = {q: (max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], float(train_k[train_k['_quintile']==q]['_rp_pnl'].mean()) / avg))
                 if (train_k['_quintile']==q).sum() else ADAPTIVE_MULT_MIN)
             for q in ['Q1','Q2','Q3','Q4','Q5']}

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
    a = np.array(arr_a); b = np.array(arr_b)
    diffs = []
    for _ in range(n_iter):
        ar = rng.choice(a, size=len(a), replace=True)
        br = rng.choice(b, size=len(b), replace=True)
        diffs.append(ar.mean() - br.mean())
    diffs = np.sort(diffs)
    return float(np.mean(diffs)), float(diffs[int(0.025*n_iter)]), float(diffs[int(0.975*n_iter)])


def evaluate_filter(df, base_pnl_metrics, name, sweep_values, mask_fn, base_sel,
                    direction='drop_below'):
    """Run sweep, print table, return rows.
    direction: 'drop_below' or 'drop_above' — for counterfactual semantics.
    """
    print()
    print(f"{'='*100}")
    print(f"H{name} SWEEP")
    print(f"{'='*100}")
    print()
    print(f"{'thr':>10} | {'n':>4} | {'TRAIN $':>10} ({'n':>3}) | {'VAL $':>10} ({'n':>3}) | {'OOS $':>10} ({'n':>3}) | {'FULL $':>11} | {'MDD':>9} | {'Δ vs base':>11}")
    print('-' * 145)
    base_pnl = base_pnl_metrics['full']['pnl']
    rows = []
    for thr in sweep_values:
        m = mask_fn(df, thr)
        sel = run_pipeline(df, mask=m)
        if sel is None:
            print(f"  thr={thr}: aborted"); continue
        full = period_metrics(sel, '2025-01-01', '2026-12-31')
        train = period_metrics(sel, '2025-01-01', '2025-06-30')
        val = period_metrics(sel, '2025-07-01', '2025-12-31')
        oos = period_metrics(sel, '2026-01-01', '2026-12-31')
        delta = full['pnl'] - base_pnl
        thr_str = 'none' if thr is None else f'{thr:.2f}'
        print(f"  {thr_str:>8} | {full['n']:>4} | "
              f"${train['pnl']:>+9,.0f} ({train['n']:>3}) | "
              f"${val['pnl']:>+9,.0f} ({val['n']:>3}) | "
              f"${oos['pnl']:>+9,.0f} ({oos['n']:>3}) | "
              f"${full['pnl']:>+10,.0f} | "
              f"${full['mdd']:>+8,.0f} | ${delta:>+10,.0f}")
        rows.append({'name': name, 'thr': thr,
                     'TRAIN_pnl': train['pnl'], 'VAL_pnl': val['pnl'], 'OOS_pnl': oos['pnl'],
                     'TRAIN_n': train['n'], 'VAL_n': val['n'], 'OOS_n': oos['n'],
                     'full_pnl': full['pnl'], 'full_mdd': full['mdd']})
    return rows


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
    print(f"Loading bars for {len(pairs)} pairs (PM window included)...")
    db_obj = Database(db_path='data/cache.db')
    raw_bars = db_obj.get_intraday_bars_bulk(pairs)
    db_obj.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}
    print("Re-sim + computing new features...")
    df = enrich_with_new_features(df, bars_cache)

    # Diagnostic on new features
    print()
    print("New feature distributions (post-resim, all rows):")
    for col in ['_entry_delay_min', '_pm_high_ratio', '_vol_ratio']:
        s = df[col].dropna()
        if len(s) == 0:
            print(f"  {col}: all NaN")
            continue
        print(f"  {col}: n_valid={len(s):>4}  min={s.min():>7.2f}  "
              f"5%={s.quantile(0.05):>7.2f}  med={s.median():>7.2f}  "
              f"95%={s.quantile(0.95):>7.2f}  max={s.max():>7.2f}")

    # Baseline pipeline
    base_sel = run_pipeline(df, mask=None)
    base_full = period_metrics(base_sel, '2025-01-01', '2026-12-31')
    base_train = period_metrics(base_sel, '2025-01-01', '2025-06-30')
    base_val = period_metrics(base_sel, '2025-07-01', '2025-12-31')
    base_oos = period_metrics(base_sel, '2026-01-01', '2026-12-31')
    base_metrics = {'full': base_full, 'train': base_train, 'val': base_val, 'oos': base_oos}
    print(f"\nBaseline: TRAIN ${base_train['pnl']:+,.0f}  VAL ${base_val['pnl']:+,.0f}  "
          f"OOS ${base_oos['pnl']:+,.0f}  FULL ${base_full['pnl']:+,.0f}  MDD ${base_full['mdd']:+,.0f}")

    all_rows = []
    # H1 ENTRY DELAY — drop trades with delay > X (None = baseline keep all)
    delay_thrs = [None, 30.0, 20.0, 15.0, 10.0, 5.0, 3.0, 1.0]
    rows1 = evaluate_filter(
        df, base_metrics, 'H1_entry_delay', delay_thrs,
        mask_fn=lambda d, t: pd.Series(True, index=d.index) if t is None
                              else (d['_entry_delay_min'].isna() | (d['_entry_delay_min'] <= t)),
        base_sel=base_sel, direction='drop_above')
    all_rows.extend(rows1)

    # H2 PM HIGH PROXIMITY — drop trades with rh / pm_high < X (None = keep all)
    pm_thrs = [None, 0.85, 0.90, 0.95, 1.00, 1.02, 1.05]
    rows2 = evaluate_filter(
        df, base_metrics, 'H2_pm_high_ratio', pm_thrs,
        mask_fn=lambda d, t: pd.Series(True, index=d.index) if t is None
                              else (d['_pm_high_ratio'].isna() | (d['_pm_high_ratio'] >= t)),
        base_sel=base_sel, direction='drop_below')
    all_rows.extend(rows2)

    # H3 VOL CONFIRM — drop trades where entry_bar_vol / 5min_avg < X
    vol_thrs = [None, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    rows3 = evaluate_filter(
        df, base_metrics, 'H3_vol_confirm', vol_thrs,
        mask_fn=lambda d, t: pd.Series(True, index=d.index) if t is None
                              else (d['_vol_ratio'].isna() | (d['_vol_ratio'] >= t)),
        base_sel=base_sel, direction='drop_below')
    all_rows.extend(rows3)

    # Counterfactual on base_sel for each feature
    print()
    print(f"{'='*100}")
    print("COUNTERFACTUAL (against base pipeline trades)")
    print(f"{'='*100}")
    def cf(name, col, thrs, drop_below=True):
        print(f"\n{name}:")
        for t in thrs:
            if t is None: continue
            if drop_below:
                dropped = base_sel[base_sel[col] < t].dropna(subset=[col])
                kept = base_sel[base_sel[col] >= t].dropna(subset=[col])
            else:
                dropped = base_sel[base_sel[col] > t].dropna(subset=[col])
                kept = base_sel[base_sel[col] <= t].dropna(subset=[col])
            if len(dropped) == 0 or len(kept) == 0:
                print(f"  thr={t}: 0 dropped or 0 kept (n_drop={len(dropped)}, n_keep={len(kept)})"); continue
            mean_d = dropped['_sized_pnl'].mean(); sum_d = dropped['_sized_pnl'].sum()
            wr_d = (dropped['_sized_pnl'] > 0).mean() * 100
            md, lo, hi = bootstrap_diff(dropped['_sized_pnl'].values, kept['_sized_pnl'].values)
            sig = '*' if (lo < 0 and hi < 0) or (lo > 0 and hi > 0) else ' '
            print(f"  thr={t:>6}: n_drop={len(dropped):>3} ({len(dropped)/len(base_sel)*100:>4.1f}%)  "
                  f"mean_drop=${mean_d:>+7.0f}  sum_drop=${sum_d:>+9,.0f}  "
                  f"WR={wr_d:>4.1f}%  Δmean=${md:>+6.0f}  CI [${lo:>+5.0f}, ${hi:>+5.0f}] {sig}")

    cf('H1 entry_delay (drop > thr)', '_entry_delay_min', delay_thrs, drop_below=False)
    cf('H2 pm_high_ratio (drop < thr)', '_pm_high_ratio', pm_thrs, drop_below=True)
    cf('H3 vol_ratio (drop < thr)', '_vol_ratio', vol_thrs, drop_below=True)

    # Decision
    print()
    print(f"{'='*100}")
    print("DECISION (Bonferroni-strict)")
    print(f"{'='*100}")
    base_t = base_train['pnl']; base_v = base_val['pnl']; base_o = base_oos['pnl']
    base_m = abs(base_full['mdd']); base_full_pnl = base_full['pnl']
    qualifying = []
    for r in all_rows:
        if r['thr'] is None: continue
        if base_t == 0 or base_v == 0 or base_o == 0 or base_m == 0: continue
        t_lift = (r['TRAIN_pnl'] - base_t) / abs(base_t) * 100
        v_lift = (r['VAL_pnl'] - base_v) / abs(base_v) * 100
        o_lift = (r['OOS_pnl'] - base_o) / abs(base_o) * 100
        mdd_change = (abs(r['full_mdd']) - base_m) / base_m * 100
        all_pos = (r['TRAIN_pnl'] > 0 and r['VAL_pnl'] > 0 and r['OOS_pnl'] > 0)
        if t_lift >= 5 and v_lift >= 5 and o_lift >= 5 and all_pos and mdd_change <= 10:
            qualifying.append((r['name'], r['thr'], t_lift, v_lift, o_lift, r['full_pnl'] - base_full_pnl))
    if qualifying:
        for name, thr, t, v, o, d in qualifying:
            print(f"  PASS lift gates: {name} thr={thr}: TRAIN+{t:.1f}% VAL+{v:.1f}% OOS+{o:.1f}% (Δ ${d:+,.0f})")
        print("  → Verify CI gate via counterfactual table above before shipping.")
    else:
        print(f"  NO threshold passes lift gates.")
        print(f"  → No new filter to ship.")


if __name__ == '__main__':
    main()

"""H3 Option A — vol_confirm POST-FILL ABORT (production-faithful BT).

DIFFERENCE FROM H3 (the filter test)
------------------------------------
H3 (filter): trade is SKIPPED entirely if vol_ratio < 0.5. P&L = $0.
            Population = trades with vol_ratio >= 0.5 (or NaN).
            Pipeline refit on filtered population.

H3-A (abort): trade enters normally. After 1 min, bar closes; if
             vol_ratio < threshold, market-sell at bar close - 10bps.
             Population is IDENTICAL to baseline (no exclusion).
             Only the P&L of low-vol trades is modified.
             No composite refit cascade.

This is the production-faithful BT — in live, we cannot skip the buy-stop
fire (it triggers intra-bar). We can only abort post-fill with slippage.
H3-A measures the actual realisable lift.

PRE-REGISTERED HYPOTHESIS
-------------------------
H3-A: replacing static_lock P&L with abort_pnl for trades where
      vol_ratio < THRESHOLD lifts FULL P&L, with TRAIN/VAL/OOS each
      lifting >=5%, no period negative, MDD increase <=10%.

Rejection: standard 5-gate as before.

ABORT P&L MODEL
---------------
On low-vol fill:
  entry_price = range_high * 1.003 (BT buy-stop limit)
  entry_bar.close = closing price of the 1-min bar that triggered fill
  abort_exit = entry_bar.close * (1 - 10bps) [market-sell at bid]
  abort_pnl = (abort_exit - entry_price) * shares

Where shares = OLD_POS / entry_price (same sizing as static_lock leg).

USAGE
-----
    python3 study_orb_h3_option_a.py
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import timedelta
import glob as _glob_alias  # noqa

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from persistence.database import Database
from study_orb import _bars_to_df
from study_orb_filter import FILTER_FEATURES, fit_z_params, composite_score
from study_orb_sizing import (
    FILTER_THRESHOLD, fit_quintile_cutoffs, assign_quintile, ADAPTIVE_MULT_MIN,
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
ABORT_SLIP_BPS = 10.0    # market-sell exit on abort
ENTRY_BUFFER_BPS = 30.0  # buy-stop limit = range_high * (1 + 30bps)


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


def enrich(df, bars_cache):
    """Compute static_lock pnl + vol_ratio + entry_bar_close per trade."""
    sl_pnl = []; sl_pnl_pct = []; sl_reason = []
    vol_ratio = []; entry_bar_close = []; entry_price_used = []
    abort_pnl = []  # what abort would have produced

    for _, row in df.reset_index(drop=True).iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        sl_pnl.append(row['pnl']); sl_pnl_pct.append(row['pnl_pct'])
        sl_reason.append(row['exit_reason'])
        vol_ratio.append(np.nan); entry_bar_close.append(np.nan)
        entry_price_used.append(np.nan); abort_pnl.append(np.nan)

        if bars is None or bars.empty: continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None: continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) & (bars['timestamp'] < range_end)]
        if len(range_bars) < 5: continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        range_total = float(range_bars['volume'].sum())

        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None; entry_bar_idx = None
        for idx, b in search.iterrows():
            if float(b['high']) > rh:
                entry_ts = b['timestamp']; entry_bar_idx = idx; break
        if entry_ts is None: continue

        entry_p = float(row['entry_price'])
        entry_price_used[-1] = entry_p

        # Update with static_lock pnl (overrides CSV's wrong 2R/-1R pnl)
        exit_p, reason = simulate_static_lock(bars, entry_p, rh, rl, entry_ts)
        shares = max(1, int(OLD_POS / entry_p))
        sl_pnl[-1] = (exit_p - entry_p) * shares
        sl_pnl_pct[-1] = (exit_p - entry_p) / entry_p * 100
        sl_reason[-1] = reason

        # vol ratio + entry bar close
        eb = bars.loc[entry_bar_idx]
        ebvol = float(eb['volume']); ebclose = float(eb['close'])
        avg_5min = range_total / 5.0
        if avg_5min > 0:
            vol_ratio[-1] = ebvol / avg_5min
        entry_bar_close[-1] = ebclose

        # Abort P&L (post-fill abort at entry bar close - 10bps)
        abort_exit = ebclose * (1 - ABORT_SLIP_BPS/10000)
        abort_pnl[-1] = (abort_exit - entry_p) * shares

    df = df.reset_index(drop=True).copy()
    df['_sl_pnl'] = sl_pnl
    df['pnl_pct'] = sl_pnl_pct
    df['exit_reason'] = sl_reason
    df['_vol_ratio'] = vol_ratio
    df['_entry_bar_close'] = entry_bar_close
    df['_abort_pnl'] = abort_pnl
    return df


def run_pipeline_with_pnl_override(df, *, abort_threshold):
    """Run the standard pipeline, but for each trade with vol_ratio < threshold
    (and not NaN), substitute abort_pnl for static_lock_pnl.
    Population is unchanged from baseline.
    """
    df = df.copy()
    if abort_threshold is None:
        df['pnl'] = df['_sl_pnl']
    else:
        # Apply abort to trades where vol_ratio is known and below threshold
        abort_mask = df['_vol_ratio'].notna() & (df['_vol_ratio'] < abort_threshold)
        # Where abort applies AND abort_pnl is computable
        abort_applicable = abort_mask & df['_abort_pnl'].notna()
        df['pnl'] = np.where(abort_applicable, df['_abort_pnl'], df['_sl_pnl'])

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
    a = np.array(arr_a); b = np.array(arr_b); diffs = []
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
    print("Computing static_lock pnl + vol_ratio + entry_bar_close + abort_pnl...")
    df = enrich(df, bars_cache)

    # Diagnostic: show abort vs static_lock distribution at thr=0.5
    print()
    print("Abort vs static_lock comparison at thr=0.5 (per-trade):")
    sub = df[df['_vol_ratio'].notna() & (df['_vol_ratio'] < 0.5)].copy()
    print(f"  n trades with vol_ratio < 0.5: {len(sub)}")
    if len(sub) > 0:
        print(f"  static_lock pnl: mean=${sub['_sl_pnl'].mean():+,.0f}  "
              f"sum=${sub['_sl_pnl'].sum():+,.0f}  WR={(sub['_sl_pnl']>0).mean()*100:.1f}%")
        print(f"  abort pnl:       mean=${sub['_abort_pnl'].mean():+,.0f}  "
              f"sum=${sub['_abort_pnl'].sum():+,.0f}  WR={(sub['_abort_pnl']>0).mean()*100:.1f}%")
        delta = (sub['_abort_pnl'] - sub['_sl_pnl']).sum()
        print(f"  abort - static_lock delta (per trade level, ungeared): ${delta:+,.0f}")

    # Sweep abort thresholds
    sweep = [None, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]
    print()
    print(f"{'='*100}")
    print(f"H3-A SWEEP — abort if vol_ratio < threshold (population unchanged)")
    print(f"{'='*100}")
    print()
    print(f"{'thr':>6} | {'n':>4} | {'TRAIN $':>10} ({'n':>3}) | {'VAL $':>10} ({'n':>3}) | {'OOS $':>10} ({'n':>3}) | {'FULL $':>11} | {'MDD':>9} | {'Calmar':>6} | {'Δ vs base':>11}")
    print('-' * 145)

    rows = []
    base_sel = None
    for thr in sweep:
        sel = run_pipeline_with_pnl_override(df, abort_threshold=thr)
        if sel is None:
            print(f"  thr={thr}: aborted"); continue
        if thr is None: base_sel = sel.copy()
        full = period_metrics(sel, '2025-01-01', '2026-12-31')
        train = period_metrics(sel, '2025-01-01', '2025-06-30')
        val = period_metrics(sel, '2025-07-01', '2025-12-31')
        oos = period_metrics(sel, '2026-01-01', '2026-12-31')
        calmar = full['pnl']/abs(full['mdd']) if full['mdd'] else 0
        thr_str = 'none' if thr is None else f'{thr:.2f}'
        rows.append({
            'thr': thr_str, 'n_total': full['n'],
            'TRAIN_pnl': train['pnl'], 'TRAIN_n': train['n'],
            'VAL_pnl': val['pnl'], 'VAL_n': val['n'],
            'OOS_pnl': oos['pnl'], 'OOS_n': oos['n'],
            'full_pnl': full['pnl'], 'full_mdd': full['mdd'], 'calmar': calmar,
        })

    base_pnl = rows[0]['full_pnl']
    for r in rows:
        delta = r['full_pnl'] - base_pnl
        print(f"  {r['thr']:>4} | {r['n_total']:>4} | "
              f"${r['TRAIN_pnl']:>+9,.0f} ({r['TRAIN_n']:>3}) | "
              f"${r['VAL_pnl']:>+9,.0f} ({r['VAL_n']:>3}) | "
              f"${r['OOS_pnl']:>+9,.0f} ({r['OOS_n']:>3}) | "
              f"${r['full_pnl']:>+10,.0f} | "
              f"${r['full_mdd']:>+8,.0f} | {r['calmar']:>6.2f} | ${delta:>+10,.0f}")

    # Counterfactual: which trades got abort treatment, and was it better?
    print()
    print(f"{'='*100}")
    print("COUNTERFACTUAL — for each threshold, look at the trades where abort fires")
    print(f"{'='*100}")
    if base_sel is not None:
        for thr in sweep:
            if thr is None: continue
            # In the BASE selection, which trades have vol_ratio < thr?
            cand = base_sel[base_sel['_vol_ratio'].notna() & (base_sel['_vol_ratio'] < thr)].copy()
            others = base_sel[~(base_sel['_vol_ratio'].notna() & (base_sel['_vol_ratio'] < thr))].copy()
            if len(cand) == 0:
                print(f"  thr={thr}: 0 trades affected"); continue
            # NB: _sized_pnl in base_sel is the static_lock geared pnl
            n = len(cand)
            sl_sum = cand['_sized_pnl'].sum()
            sl_mean = cand['_sized_pnl'].mean()
            sl_wr = (cand['_sized_pnl'] > 0).mean() * 100
            # ungeared per-trade abort delta (rough)
            cand['_abort_geared_pnl'] = cand['_abort_pnl'] * cand['_rp_position'] / OLD_POS
            ab_sum = cand['_abort_geared_pnl'].sum()
            ab_mean = cand['_abort_geared_pnl'].mean()
            ab_wr = (cand['_abort_geared_pnl'] > 0).mean() * 100
            saved = ab_sum - sl_sum
            print(f"  thr={thr}: n_affected={n:>3} ({n/len(base_sel)*100:>4.1f}%)  "
                  f"static_lock=${sl_sum:>+9,.0f} (mean ${sl_mean:>+5.0f}, WR {sl_wr:>4.1f}%)  "
                  f"abort=${ab_sum:>+9,.0f} (mean ${ab_mean:>+5.0f}, WR {ab_wr:>4.1f}%)  "
                  f"saved=${saved:>+9,.0f}")

    # Save + decision
    pd.DataFrame(rows).to_csv('analysis_results/orb_h3a_sweep.csv', index=False)
    print()
    print(f"{'='*100}")
    print("DECISION (5-gate strict)")
    print(f"{'='*100}")
    base_t = rows[0]['TRAIN_pnl']; base_v = rows[0]['VAL_pnl']; base_o = rows[0]['OOS_pnl']
    base_m = abs(rows[0]['full_mdd'])
    qualifying = []
    for r in rows[1:]:
        if base_t == 0 or base_v == 0 or base_o == 0 or base_m == 0: continue
        t_lift = (r['TRAIN_pnl'] - base_t) / abs(base_t) * 100
        v_lift = (r['VAL_pnl'] - base_v) / abs(base_v) * 100
        o_lift = (r['OOS_pnl'] - base_o) / abs(base_o) * 100
        mdd_change = (abs(r['full_mdd']) - base_m) / base_m * 100
        all_pos = (r['TRAIN_pnl'] > 0 and r['VAL_pnl'] > 0 and r['OOS_pnl'] > 0)
        if t_lift >= 5 and v_lift >= 5 and o_lift >= 5 and all_pos and mdd_change <= 10:
            qualifying.append((r['thr'], t_lift, v_lift, o_lift, r['full_pnl'] - base_pnl))
    if qualifying:
        for thr, t, v, o, d in qualifying:
            print(f"  PASS lift gates: thr={thr}: TRAIN+{t:.1f}% VAL+{v:.1f}% OOS+{o:.1f}% (Δ ${d:+,.0f})")
    else:
        print(f"  NO threshold passes lift gates.")


if __name__ == '__main__':
    main()

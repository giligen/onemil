"""Classifier-based exit selection.

Concept: rule-based variants (V0-V13) each win on some trades and lose on
others because they're applying one rule to all trades. A classifier can
pick the BEST exit rule per trade based on entry-time features, which in
principle captures more edge than any single rule.

Pipeline:
  1. For each of 1,021 trades, simulate N candidate exit rules and compute
     the per-trade sized_pnl for each. The "best" exit is argmax_sized_pnl.
  2. Split by time: TRAIN (H1 2025), VAL (H2 2025), HOQ1 (Q1 2026+).
  3. Fit a classifier on TRAIN: entry features → best_exit.
  4. Score VAL + HOQ1 by simulating the predicted exit per trade and
     aggregating P&L.
  5. Compare to V0 baseline total P&L on same OOS slice.

Honest caveats:
  - Best-exit label is a NOISY supervision signal — reasonable alternative
    exits often differ by pennies, so the classifier is learning to
    separate cases where one rule is clearly dominant.
  - 4-class target on ~500 TRAIN trades is data-hungry. We use a simple
    logistic model + class_weight='balanced' and cap complexity to avoid
    memorizing.
  - We exclude V0 from "candidate exits" to force the classifier to pick a
    NON-V0 rule — then blend: use V0 unless the classifier has high
    confidence for a specific non-V0 rule. This gives us a soft overlay
    on V0.

Candidate exits (non-V0):
  V1b   trail_1.0R              — survives noise better than 0.5R
  V4b   late_arm_4R_trail_0.5R  — best Calmar from round 2
  V11b  remove_stop_after_MFE5R — round-3 winner
  V12   replace_lock_MFE3R      — aggressive peak-1R trail
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import timedelta
from typing import Callable, Dict, List, Tuple

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
EXIT_SLIP_BPS = 10.0

TRAIN_END = '2025-06-30'
VAL_START = '2025-07-01'
VAL_END   = '2025-12-31'
HOQ1_START = '2026-01-01'


def _session_open_timestamp(bars):
    if bars.empty: return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning): return morning.iloc[0]['timestamp']
    return None


# ---------------------------------------------------------------------------
# Candidate exit simulators (copy from earlier rounds)
# ---------------------------------------------------------------------------

def sim_v0(bars, entry, rh, rl, et):
    rs = rh - rl; trig = entry + 1.5 * rs; lock = entry + 1.0 * rs
    stop = rl; armed = False
    post = bars[bars['timestamp'] >= et].reset_index(drop=True)
    for _, b in post.iloc[1:].iterrows():
        h = float(b['high']); lo = float(b['low'])
        if not armed and h >= trig: armed = True; stop = max(stop, lock)
        if lo <= stop: return stop * (1 - EXIT_SLIP_BPS/10000)
    return float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000)


def sim_v1b(bars, entry, rh, rl, et):
    rs = rh - rl; trig = entry + 1.5 * rs
    stop = rl; armed = False; peak = 0.0
    post = bars[bars['timestamp'] >= et].reset_index(drop=True)
    for _, b in post.iloc[1:].iterrows():
        h = float(b['high']); lo = float(b['low'])
        if h > peak: peak = h
        if armed or peak >= trig:
            armed = True
            stop = max(stop, peak - 1.0 * rs)
        if lo <= stop: return stop * (1 - EXIT_SLIP_BPS/10000)
    return float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000)


def sim_v4b(bars, entry, rh, rl, et):
    rs = rh - rl; trig = entry + 4.0 * rs
    stop = rl; armed = False; peak = 0.0
    post = bars[bars['timestamp'] >= et].reset_index(drop=True)
    for _, b in post.iloc[1:].iterrows():
        h = float(b['high']); lo = float(b['low'])
        if h > peak: peak = h
        if armed or peak >= trig:
            armed = True
            stop = max(stop, peak - 0.5 * rs)
        if lo <= stop: return stop * (1 - EXIT_SLIP_BPS/10000)
    return float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000)


def sim_v11b(bars, entry, rh, rl, et):
    rs = rh - rl; trig = entry + 1.5 * rs; lock = entry + 1.0 * rs
    runner_abs = 5.0 * rs
    stop = rl; armed = False; runner = False; mfe = 0.0
    post = bars[bars['timestamp'] >= et].reset_index(drop=True)
    for _, b in post.iloc[1:].iterrows():
        h = float(b['high']); lo = float(b['low'])
        mfe = max(mfe, h - entry)
        if not armed and h >= trig: armed = True; stop = max(stop, lock)
        if armed and mfe >= runner_abs: runner = True
        if not runner and lo <= stop: return stop * (1 - EXIT_SLIP_BPS/10000)
    return float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000)


def sim_v12(bars, entry, rh, rl, et):
    rs = rh - rl; trig = entry + 1.5 * rs; lock = entry + 1.0 * rs
    runner_abs = 3.0 * rs
    stop = rl; armed = False; runner = False; peak = 0.0; mfe = 0.0
    post = bars[bars['timestamp'] >= et].reset_index(drop=True)
    for _, b in post.iloc[1:].iterrows():
        h = float(b['high']); lo = float(b['low'])
        mfe = max(mfe, h - entry)
        if h > peak: peak = h
        if not armed and h >= trig: armed = True; stop = max(stop, lock)
        if armed and mfe >= runner_abs: runner = True
        if runner: stop = max(rl, peak - 1.0 * rs)
        if lo <= stop: return stop * (1 - EXIT_SLIP_BPS/10000)
    return float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000)


EXITS: Dict[str, Callable] = {
    'v0': sim_v0,
    'v1b': sim_v1b,
    'v4b': sim_v4b,
    'v11b': sim_v11b,
    'v12': sim_v12,
}


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _simulate_all_exits(df, bars_cache):
    """For every trade, compute exit price under all candidate exits. Returns
    df enriched with columns `pnl_<exit>`, `pct_<exit>` for each candidate."""
    out = df.reset_index(drop=True).copy()
    results = {name: [] for name in EXITS}
    for _, row in out.iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            for name in EXITS: results[name].append(row['pnl'])
            continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            for name in EXITS: results[name].append(row['pnl'])
            continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) &
                          (bars['timestamp'] < range_end)]
        if len(range_bars) < 5:
            for name in EXITS: results[name].append(row['pnl'])
            continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh: entry_ts = b['timestamp']; break
        if entry_ts is None:
            for name in EXITS: results[name].append(row['pnl'])
            continue
        entry_p = float(row['entry_price'])
        shares = max(1, int(OLD_POS / entry_p))
        for name, fn in EXITS.items():
            exit_p = fn(bars, entry_p, rh, rl, entry_ts)
            results[name].append((exit_p - entry_p) * shares)
    for name in EXITS:
        out[f'pnl_{name}'] = results[name]
    return out


def _apply_pipeline_to_variant(df, pnl_col):
    """Run defended pipeline with `pnl_col` as the per-trade pnl.

    Returns selected-trade df with _sized_pnl column computed using the
    adaptive mults fit on TRAIN subset. Mults are VARIANT-SPECIFIC.
    """
    d = df.copy()
    d['pnl'] = d[pnl_col]
    d['pnl_pct'] = d['pnl'] / (d['entry_price'] * (OLD_POS / d['entry_price']).clip(lower=1).astype(int)) * 100
    stop = d['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = RISK / (stop / 100.0)
    d['_rp_position'] = uncap.clip(upper=ACCOUNT/N)
    d['_rp_pnl'] = d['pnl'] * d['_rp_position'] / OLD_POS

    train = d[(d['date'] >= '2025-01-01') & (d['date'] <= TRAIN_END)]
    params = fit_z_params(train, FILTER_FEATURES)
    d['_composite'] = composite_score(d, params)
    train = d[(d['date'] >= '2025-01-01') & (d['date'] <= TRAIN_END)]
    train_k = train[train['_composite'] >= FILTER_THRESHOLD].copy()
    cutoffs = fit_quintile_cutoffs(train_k['_composite'])
    train_k['_quintile'] = assign_quintile(train_k['_composite'], cutoffs)
    avg = float(train_k['_rp_pnl'].mean())
    mults = {}
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = train_k[train_k['_quintile'] == q]
        m = float(sub['_rp_pnl'].mean()) / avg if len(sub) else 1.0
        mults[q] = max(ADAPTIVE_MULT_MIN, min(Q_CAPS[q], m))

    kept = d[d['_composite'] >= FILTER_THRESHOLD].copy()
    kept['_quintile'] = assign_quintile(kept['_composite'], cutoffs)
    sel_rows = []
    for _, dg in kept.groupby('date'):
        dd = dg.copy()
        dd['_q_rank'] = dd['_quintile'].map(Q_ORDER)
        dd = dd.sort_values(['_q_rank', '_composite'], ascending=[True, False])
        seen_fam = set(); seen_sup = set(); today = []
        for _, r in dd.iterrows():
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
    return sel, cutoffs, mults, params


def _calmar(pnls: pd.Series) -> Tuple[float, float, float]:
    """Returns (total_pnl, max_dd, calmar) from a day-ordered pnl series."""
    cum = pnls.cumsum()
    peak = -1e18; mdd = 0.0
    for v in cum:
        peak = max(peak, v)
        mdd = min(mdd, v - peak)
    tp = float(pnls.sum())
    return tp, float(mdd), (tp / abs(mdd) if mdd < 0 else float('inf'))


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

# Entry-time features available for classification (from CSV + quintile + SPY)
CLF_FEATURES = [
    'range_size_pct', 'range_total_volume', 'range_avg_bar_range_pct',
    'range_volume_stddev_pct', 'bars_green_in_range',
    'range_close_position', 'range_return_pct', 'last_bar_green',
    'range_vwap_distance_pct', 'gap_pct',
    'prev_day_range_pct', 'prev_day_close_position',
    'avg_daily_volume_20d', 'avg_daily_range_pct_20d',
    'price_vs_20d_high_pct', 'return_volatility_20d',
    'prev_day_volume_vs_20d',
    'spy_range_pct_5min', 'spy_return_5min_pct',
    'spy_gap_pct', 'spy_3d_range_pct',
    'day_of_week', 'days_since_month_start',
    '_composite',
]


def _fit_classifier(train_df, pnl_cols):
    """Fit a simple classifier: best exit per trade given features.

    Uses sklearn if available; falls back to a simple per-feature rule.
    """
    X = train_df[CLF_FEATURES].fillna(0).values
    best_idx = train_df[pnl_cols].values.argmax(axis=1)
    y = np.array([pnl_cols[i].replace('pnl_', '') for i in best_idx])

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.08,
            random_state=42, min_samples_leaf=20,
        )
        clf.fit(X, y)
        return clf, 'gb'
    except ImportError:
        pass
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced')
        clf.fit(sc.transform(X), y)
        return (clf, sc), 'lr'
    except ImportError:
        return None, 'fallback'


def _predict_classifier(clf_obj, kind: str, df):
    X = df[CLF_FEATURES].fillna(0).values
    if kind == 'gb':
        return clf_obj.predict(X), clf_obj.predict_proba(X).max(axis=1)
    if kind == 'lr':
        clf, sc = clf_obj
        preds = clf.predict(sc.transform(X))
        probs = clf.predict_proba(sc.transform(X)).max(axis=1)
        return preds, probs
    # Fallback: always V0
    return np.array(['v0'] * len(df)), np.ones(len(df))


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

    print(f"Simulating {len(EXITS)} candidate exits for {len(df)} trades...")
    df_exits = _simulate_all_exits(df, bars_cache)

    # We need _composite and _quintile for selection. Fit once on V0 pnl for
    # the COMMON selection pass (all variants select the same trades so the
    # comparison is variant-difference-only).
    print("\nBuilding per-exit selected trade sets...")
    sel_by_exit = {}
    for name in EXITS:
        sel, cutoffs, mults, params = _apply_pipeline_to_variant(
            df_exits, f'pnl_{name}')
        sel_by_exit[name] = sel

    # Baseline V0 metrics (reference)
    v0_daily = sel_by_exit['v0'].groupby('date')['_sized_pnl'].sum().sort_index()
    v0_tp, v0_dd, v0_calmar = _calmar(v0_daily)
    print(f"\n{'='*80}")
    print(f"V0 baseline (full timeline):  ${v0_tp:+,.0f}  DD ${v0_dd:+,.0f}  "
          f"Calmar {v0_calmar:.2f}x")

    # Classifier: train on TRAIN trades, predict on VAL+HOQ1
    # Feature frame: augment df_exits with _composite (from V0 pipeline)
    v0_sel = sel_by_exit['v0']
    comp_map = dict(zip(
        v0_sel.apply(lambda r: (r['symbol'], r['date']), axis=1),
        v0_sel['_composite']
    ))
    df_exits['_composite'] = df_exits.apply(
        lambda r: comp_map.get((r['symbol'], r['date']), 0.0), axis=1
    )

    # Restrict classifier to trades that passed composite filter (same set
    # as pipeline selects) — irrelevant exits on filtered-out trades don't
    # matter for final P&L.
    clf_df = df_exits[df_exits['_composite'] >= FILTER_THRESHOLD].copy()

    train_mask = clf_df['date'] <= TRAIN_END
    val_mask = (clf_df['date'] >= VAL_START) & (clf_df['date'] <= VAL_END)
    hoq1_mask = clf_df['date'] >= HOQ1_START

    train_df = clf_df[train_mask]
    print(f"\nClassifier TRAIN:  {len(train_df)} trades")
    print(f"Classifier VAL:    {val_mask.sum()} trades")
    print(f"Classifier HOQ1:   {hoq1_mask.sum()} trades")

    pnl_cols = [f'pnl_{n}' for n in EXITS]

    # Best-exit distribution in TRAIN
    train_best = train_df[pnl_cols].values.argmax(axis=1)
    train_best_names = [pnl_cols[i].replace('pnl_', '') for i in train_best]
    print(f"\nTRAIN best-exit distribution:")
    for name in EXITS:
        c = train_best_names.count(name)
        print(f"  {name:<5} {c:>4}  ({c/len(train_df)*100:>5.1f}%)")

    # Fit classifier
    clf, kind = _fit_classifier(train_df, pnl_cols)
    print(f"\nClassifier type: {kind}")

    # Predict on full clf_df
    preds, probs = _predict_classifier(clf, kind, clf_df)
    clf_df['_predicted_exit'] = preds
    clf_df['_pred_prob'] = probs

    # Confidence-based blending: use predicted exit if prob > threshold, else fall back to V0
    for conf_thresh in [0.0, 0.40, 0.50, 0.60]:
        chosen_pnl = []
        for _, r in clf_df.iterrows():
            if r['_pred_prob'] >= conf_thresh:
                chosen = r['_predicted_exit']
            else:
                chosen = 'v0'
            chosen_pnl.append(r[f'pnl_{chosen}'])
        clf_df[f'_chosen_pnl_c{int(conf_thresh*100)}'] = chosen_pnl

    # Re-run pipeline using the chosen-per-trade pnl, in each confidence bucket
    for conf_thresh in [0.0, 0.40, 0.50, 0.60]:
        col = f'_chosen_pnl_c{int(conf_thresh*100)}'
        chosen_df = df_exits.copy()
        chosen_df = chosen_df.merge(
            clf_df[['symbol', 'date', col]],
            on=['symbol', 'date'], how='left'
        )
        chosen_df[col] = chosen_df[col].fillna(chosen_df['pnl_v0'])
        chosen_df = chosen_df.rename(columns={col: 'pnl_clf_tmp'})
        sel, _, _, _ = _apply_pipeline_to_variant(chosen_df, 'pnl_clf_tmp')
        # Eval OOS slices
        for slice_name, lo, hi in [
            ('TRAIN',  '2025-01-01', TRAIN_END),
            ('VAL',    VAL_START, VAL_END),
            ('HOQ1+',  HOQ1_START, '2030-12-31'),
            ('FULL',   '2025-01-01', '2030-12-31'),
        ]:
            sl = sel[(sel['date'] >= lo) & (sel['date'] <= hi)]
            daily = sl.groupby('date')['_sized_pnl'].sum().sort_index()
            tp, dd, cal = _calmar(daily)
            # V0 reference for same slice
            v0_sl = sel_by_exit['v0']
            v0sl = v0_sl[(v0_sl['date'] >= lo) & (v0_sl['date'] <= hi)]
            v0_daily = v0sl.groupby('date')['_sized_pnl'].sum().sort_index()
            v0_tp, v0_dd, v0_cal = _calmar(v0_daily)
            delta = tp - v0_tp
            if conf_thresh == 0.0 and slice_name == 'FULL':
                print(f"\n{'='*80}")
            print(f"  conf>={conf_thresh:.2f}  {slice_name:<6} "
                  f"CLF ${tp:>+9,.0f}  (V0 ${v0_tp:>+9,.0f})  "
                  f"Δ ${delta:>+8,.0f}  Cal {cal:.2f}x  (V0 {v0_cal:.2f}x)")

    # Feature importance (GB only)
    if kind == 'gb':
        importances = dict(zip(CLF_FEATURES, clf.feature_importances_))
        top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:12]
        print(f"\nTop-12 feature importances (GB):")
        for f, w in top:
            print(f"  {f:<35} {w:.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()

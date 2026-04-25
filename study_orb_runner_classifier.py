"""EOD-Rider entry classifier.

Shifts from "fix exits" (saturated, V0 is optimal) to "predict runners at entry
and size accordingly." A "runner" is a trade that reaches MFE >= 5R — these
are the trades V0 captures best via its ride-to-EOD behavior.

Hypothesis: some entry-time feature combinations predict runners better than
the composite_score alone. If yes, a probability-weighted sizing multiplier
could lift total P&L without changing exits.

Pipeline:
  1. Simulate V0 exit on all trades → get per-trade MFE (label).
  2. Define runner = (MFE >= 5R).
  3. Descriptive: runner rate by quintile, gap, SPY regime, etc.
  4. Train binary GBM classifier on TRAIN (H1 2025).
  5. Evaluate OOS precision/recall on VAL (H2 2025) + HOQ1 (Q1 2026+).
  6. Sizing experiment: scale position by (1.0 + alpha * P(runner)) and
     measure total P&L lift per alpha on OOS.
  7. Compare to V0 baseline.

Honest checks:
  - Strict train/val/hoq1 split — no leakage.
  - Classifier must beat trivial "always predict Q5" baseline.
  - Sizing must hold up on BOTH val AND hoq1 (not just one).
  - Check class imbalance (expect ~9% runner rate).
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
EXIT_SLIP_BPS = 10.0
RUNNER_MFE_R = 5.0

TRAIN_END = '2025-06-30'
VAL_END = '2025-12-31'


def _session_open_timestamp(bars):
    if bars.empty: return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning): return morning.iloc[0]['timestamp']
    return None


def sim_v0_with_mfe(bars, entry, rh, rl, et):
    """V0 exit + full-trade MFE (tracked across all bars, not just pre-exit)."""
    rs = rh - rl; trig = entry + 1.5 * rs; lock = entry + 1.0 * rs
    stop = rl; armed = False
    exit_price = None
    mfe_abs = 0.0  # FULL-BAR MFE — track across all bars
    post = bars[bars['timestamp'] >= et].reset_index(drop=True)
    for _, b in post.iloc[1:].iterrows():
        h = float(b['high']); lo = float(b['low'])
        mfe_abs = max(mfe_abs, h - entry)
        if exit_price is None:
            if not armed and h >= trig: armed = True; stop = max(stop, lock)
            if lo <= stop:
                exit_price = stop * (1 - EXIT_SLIP_BPS/10000)
    if exit_price is None:
        exit_price = float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000)
    return exit_price, mfe_abs / rs if rs > 0 else 0


def _simulate(df, bars_cache):
    pnls, pcts, mfes = [], [], []
    for _, row in df.reset_index(drop=True).iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct']); mfes.append(0.0)
            continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct']); mfes.append(0.0)
            continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) &
                          (bars['timestamp'] < range_end)]
        if len(range_bars) < 5:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct']); mfes.append(0.0)
            continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh: entry_ts = b['timestamp']; break
        if entry_ts is None:
            pnls.append(row['pnl']); pcts.append(row['pnl_pct']); mfes.append(0.0)
            continue
        entry_p = float(row['entry_price'])
        shares = max(1, int(OLD_POS / entry_p))
        exit_p, mfe_r = sim_v0_with_mfe(bars, entry_p, rh, rl, entry_ts)
        pnls.append((exit_p - entry_p) * shares)
        pcts.append((exit_p - entry_p) / entry_p * 100)
        mfes.append(mfe_r)
    out = df.reset_index(drop=True).copy()
    out['pnl'] = pnls; out['pnl_pct'] = pcts; out['mfe_r'] = mfes
    return out


def _run_pipeline(df_with_pnl, sizing_override=None):
    """Standard defended pipeline.

    `sizing_override`: optional dict {(symbol, date) → mult} that multiplies
    the base adaptive mult. Used to apply classifier-predicted scaling.
    """
    df = df_with_pnl.copy()
    per_pos_cap = ACCOUNT / N
    stop = df['range_size_pct'].clip(lower=MIN_STOP_PCT)
    uncap = RISK / (stop / 100.0)
    df['_rp_position'] = uncap.clip(upper=per_pos_cap)
    df['_rp_pnl'] = df['pnl'] * df['_rp_position'] / OLD_POS

    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= TRAIN_END)]
    params = fit_z_params(train, FILTER_FEATURES)
    df['_composite'] = composite_score(df, params)
    train = df[(df['date'] >= '2025-01-01') & (df['date'] <= TRAIN_END)]
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

    def _sized_pnl(r):
        base = r['_rp_pnl'] * mults[r['_quintile']]
        if sizing_override is not None:
            k = (r['symbol'], r['date'])
            extra = sizing_override.get(k, 1.0)
            return base * extra
        return base
    sel['_sized_pnl'] = sel.apply(_sized_pnl, axis=1)
    sel['date'] = pd.to_datetime(sel['date'])
    return sel, cutoffs, mults


def _calmar(pnls):
    cum = pnls.cumsum()
    peak = -1e18; mdd = 0.0
    for v in cum:
        peak = max(peak, v); mdd = min(mdd, v - peak)
    tp = float(pnls.sum())
    return tp, float(mdd), (tp / abs(mdd) if mdd < 0 else float('inf'))


# ---------------------------------------------------------------------------
# Classifier feature set + wiring
# ---------------------------------------------------------------------------

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


def _runner_rate_breakdown(df, label):
    """Descriptive: runner rate by various feature buckets."""
    print(f"\n{'='*70}")
    print(f"  Runner rate ({label})  —  runner := MFE >= {RUNNER_MFE_R}R")
    print(f"{'='*70}")
    total = len(df); runners = (df['_runner'] == 1).sum()
    print(f"  Overall runner rate: {runners}/{total} = {runners/total*100:.1f}%")

    print(f"\n  By quintile:")
    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        sub = df[df['_quintile'] == q]
        if not len(sub): continue
        rr = (sub['_runner'] == 1).mean() * 100
        print(f"    {q}: {int((sub['_runner']==1).sum()):>3}/{len(sub):<3} "
              f"({rr:.1f}%)  avg MFE={sub['mfe_r'].mean():.2f}R")

    print(f"\n  By SPY opening (spy_return_5min_pct):")
    for name, cond in [
        ('neg',   df['spy_return_5min_pct'] < -0.1),
        ('flat',  (df['spy_return_5min_pct'] >= -0.1) & (df['spy_return_5min_pct'] <= 0.1)),
        ('pos',   df['spy_return_5min_pct'] > 0.1),
    ]:
        sub = df[cond]
        if not len(sub): continue
        rr = (sub['_runner'] == 1).mean() * 100
        print(f"    {name}: {int((sub['_runner']==1).sum()):>3}/{len(sub):<3} ({rr:.1f}%)")

    print(f"\n  By gap_pct bucket:")
    bins = [(-100,-5,'< -5%'), (-5,0,'-5 to 0'), (0,5,'0 to 5%'),
            (5,15,'5-15%'), (15,50,'15-50%'), (50,200,'50%+')]
    for lo, hi, name in bins:
        sub = df[(df['gap_pct'] > lo) & (df['gap_pct'] <= hi)]
        if not len(sub): continue
        rr = (sub['_runner'] == 1).mean() * 100
        print(f"    {name:<10}: {int((sub['_runner']==1).sum()):>3}/{len(sub):<3} ({rr:.1f}%)")


def _fit_runner_classifier(train_df):
    X = train_df[CLF_FEATURES].fillna(0).values
    y = train_df['_runner'].values
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            random_state=42, min_samples_leaf=30, subsample=0.8,
        )
        clf.fit(X, y)
        return clf
    except ImportError:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced')
        clf.fit(sc.transform(X), y)
        return (clf, sc)


def _predict_runner_prob(clf_obj, df):
    X = df[CLF_FEATURES].fillna(0).values
    if isinstance(clf_obj, tuple):
        clf, sc = clf_obj
        return clf.predict_proba(sc.transform(X))[:, 1]
    return clf_obj.predict_proba(X)[:, 1]


def _apply_sizing(sel_df, probs_map, scheme: str, param: float):
    """Apply sizing scheme to selected trades.

    Schemes:
      - 'linear'  : multiplier = 1.0 + param * P(runner)  (alpha scaling)
      - 'threshold': multiplier = param if P(runner) > 0.5 else 1.0
      - 'three_tier': 0.7 if P<0.2; 1.0 if 0.2<=P<0.5; param if P>=0.5
    """
    overrides = {}
    for _, r in sel_df.iterrows():
        k = (r['symbol'], r['date'])
        p = probs_map.get(k, 0.0)
        if scheme == 'linear':
            m = 1.0 + param * p
        elif scheme == 'threshold':
            m = param if p > 0.5 else 1.0
        elif scheme == 'three_tier':
            if p < 0.2: m = 0.7
            elif p < 0.5: m = 1.0
            else: m = param
        else:
            m = 1.0
        # Cap mult to same range as quintile mults to avoid blowups
        m = max(0.25, min(3.0, m))
        overrides[k] = m
    return overrides


def main():
    csv = sorted(glob.glob('analysis_results/orb_features_*.csv'))
    csv = [p for p in csv if 'corrmatrix' not in p][-1]
    print(f"Loading features from {csv}")
    df = pd.read_csv(csv)
    needed = [f for f, _ in FILTER_FEATURES]
    df = df.dropna(subset=needed + ['pnl', 'date', 'pnl_pct',
                                     'range_size_pct', 'entry_price'])
    df['date'] = pd.to_datetime(df['date'])
    pairs = list(df[['symbol', 'date']].drop_duplicates().apply(
        lambda r: (r['symbol'], r['date'].strftime('%Y-%m-%d')), axis=1))
    print(f"Loading bars for {len(pairs)} pairs...")
    db = Database(db_path='data/cache.db')
    raw_bars = db.get_intraday_bars_bulk(pairs)
    db.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}

    print("Simulating V0 + tracking MFE...")
    df_v = _simulate(df, bars_cache)

    # Run pipeline with V0 pnl to attach quintile + composite to every trade
    print("Attaching quintile + composite...")
    sel, cutoffs, mults = _run_pipeline(df_v)

    # Merge quintile/composite onto full df (not just selected) so we can
    # fit the classifier on only selected trades — which is what the live
    # system would actually see.
    info = sel[['symbol', 'date', '_composite', '_quintile', '_sized_pnl']].copy()
    info['_runner'] = (sel['mfe_r'] >= RUNNER_MFE_R).astype(int)
    info['mfe_r'] = sel['mfe_r']
    info['date'] = pd.to_datetime(info['date'])

    # Attach all CLF features by joining back to df_v
    df_v['date'] = pd.to_datetime(df_v['date'])
    info = info.merge(df_v, on=['symbol', 'date'], suffixes=('', '_d'))

    # Descriptive
    _runner_rate_breakdown(info, 'ALL selected trades (1,021)')

    # Split
    train_m = info['date'] <= TRAIN_END
    val_m   = (info['date'] > TRAIN_END) & (info['date'] <= VAL_END)
    hoq1_m  = info['date'] > VAL_END

    train = info[train_m].copy()
    val   = info[val_m].copy()
    hoq1  = info[hoq1_m].copy()

    print(f"\n{'='*70}")
    print(f"  Splits")
    print(f"{'='*70}")
    for name, d in [('TRAIN', train), ('VAL', val), ('HOQ1+', hoq1)]:
        runners = int((d['_runner'] == 1).sum())
        print(f"  {name:<6} {len(d):>4} trades  ({runners} runners = "
              f"{runners/len(d)*100 if len(d) else 0:.1f}%)")

    _runner_rate_breakdown(train, 'TRAIN only')

    # Fit classifier
    clf = _fit_runner_classifier(train)

    # Score all splits
    for name, d in [('TRAIN', train), ('VAL', val), ('HOQ1+', hoq1)]:
        if not len(d): continue
        probs = _predict_runner_prob(clf, d)
        actual = d['_runner'].values
        # AUC-ish: rank runners by prob, compute what fraction land in top-N
        thresholds = [0.10, 0.20, 0.30, 0.50]
        print(f"\n  {name} precision/recall by threshold:")
        for t in thresholds:
            pred = (probs > t).astype(int)
            if pred.sum() == 0:
                print(f"    p>{t}: 0 predictions")
                continue
            prec = (pred & actual).sum() / pred.sum()
            rec = (pred & actual).sum() / max(actual.sum(), 1)
            print(f"    p>{t}: pred {pred.sum():>3}, prec {prec*100:>5.1f}%, "
                  f"rec {rec*100:>5.1f}%, actual runners in bucket "
                  f"{int((pred & actual).sum())}/{int(actual.sum())}")
        # Top-k capture
        print(f"  {name} top-k capture (% of runners in top-N predictions):")
        order = np.argsort(-probs)
        for k_frac in [0.10, 0.20, 0.30]:
            k = max(1, int(len(d) * k_frac))
            top_k_idx = order[:k]
            hit = actual[top_k_idx].sum()
            total = actual.sum()
            print(f"    top {k_frac*100:.0f}% ({k} trades): {hit}/{total} "
                  f"runners captured ({hit/max(total,1)*100:.0f}%)")

    # Sizing experiments: apply probability-based multiplier to VAL + HOQ1
    print(f"\n{'='*70}")
    print(f"  Sizing experiments (apply P(runner) to position)")
    print(f"{'='*70}")

    # Build prob maps for each slice
    slices = [('TRAIN', train), ('VAL', val), ('HOQ1+', hoq1)]
    probs_by_split = {}
    for name, d in slices:
        if not len(d): continue
        probs_by_split[name] = dict(zip(
            d.apply(lambda r: (r['symbol'], r['date']), axis=1),
            _predict_runner_prob(clf, d)
        ))

    # Baseline P&L per split
    print(f"\n  V0 baseline P&L per split:")
    for name in ['TRAIN', 'VAL', 'HOQ1+']:
        sub = info[(info['date'] > ('2025-06-30' if name == 'VAL' else
                                     '2025-12-31' if name == 'HOQ1+' else '1900-01-01'))
                    & (info['date'] <= ('2025-06-30' if name == 'TRAIN' else
                                         '2025-12-31' if name == 'VAL' else
                                         '2030-12-31'))] if False else (
            info[(info['date'] <= TRAIN_END) if name == 'TRAIN' else
                 ((info['date'] > TRAIN_END) & (info['date'] <= VAL_END)) if name == 'VAL' else
                 (info['date'] > VAL_END)])
        daily = sub.groupby('date')['_sized_pnl'].sum().sort_index()
        tp, dd, cal = _calmar(daily)
        print(f"    {name:<6} ${tp:>+9,.0f}  DD ${dd:>+7,.0f}  Cal {cal:.2f}x")

    # Try each sizing scheme
    schemes = [
        ('linear_a0.5',     'linear',    0.5),
        ('linear_a1.0',     'linear',    1.0),
        ('linear_a1.5',     'linear',    1.5),
        ('threshold_1.5x',  'threshold', 1.5),
        ('threshold_2.0x',  'threshold', 2.0),
        ('3tier_2.0x',      'three_tier',2.0),
        ('3tier_1.5x',      'three_tier',1.5),
    ]

    for sname, scheme, param in schemes:
        print(f"\n  Scheme: {sname}")
        for split_name in ['TRAIN', 'VAL', 'HOQ1+']:
            probs_map = probs_by_split.get(split_name, {})
            sub = info[(info['date'] <= TRAIN_END) if split_name == 'TRAIN' else
                        ((info['date'] > TRAIN_END) & (info['date'] <= VAL_END)) if split_name == 'VAL' else
                        (info['date'] > VAL_END)]
            overrides = _apply_sizing(sub, probs_map, scheme, param)
            # Apply overrides as multiplier on _sized_pnl
            sub = sub.copy()
            sub['_mult'] = sub.apply(
                lambda r: overrides.get((r['symbol'], r['date']), 1.0), axis=1)
            sub['_new_pnl'] = sub['_sized_pnl'] * sub['_mult']
            daily_new = sub.groupby('date')['_new_pnl'].sum().sort_index()
            daily_old = sub.groupby('date')['_sized_pnl'].sum().sort_index()
            tp_n, dd_n, cal_n = _calmar(daily_new)
            tp_o, dd_o, cal_o = _calmar(daily_old)
            delta = tp_n - tp_o
            print(f"    {split_name:<6} base ${tp_o:>+9,.0f} → new ${tp_n:>+9,.0f}  "
                  f"(Δ ${delta:>+7,.0f})  Cal {cal_o:.1f}→{cal_n:.1f}x")

    # Feature importance
    try:
        importances = dict(zip(CLF_FEATURES, clf.feature_importances_))
        top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
        print(f"\n  Top-15 feature importances (GB runner classifier):")
        for f, w in top:
            print(f"    {f:<35} {w:.4f}")
    except AttributeError:
        pass

    print("\nDone.")


if __name__ == '__main__':
    main()

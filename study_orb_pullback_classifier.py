"""Post-+1R pullback classifier.

Previous attempts at exit-side edge over V0:
  - Pre-trade runner classifier: TRAIN overfit, OOS marginal
  - Rule-based "stay if quintile ≥ Q3/Q4/Q5": fails HOQ1+

This is the last untested decision surface: at the MOMENT of the +1R pullback
(after arming at +1.5R), use entry features PLUS newly-available in-trade
features to decide stay-vs-exit per trade.

In-trade features available at the +1R moment:
  peak_mfe_pre_pullback_r    — how high MFE reached before pulling back
  bars_since_entry_at_pullback — time to develop
  bars_from_peak_to_pullback  — speed of retrace
  pullback_bar_vol_ratio      — pullback volume vs median pre-pullback vol
  pullback_bar_close_position — where did pullback bar close in its range
  cum_vol_ratio_pre_pullback  — cum volume vs entry-time avg_daily_volume_20d

Economics (from diagnostic):
  144 stay-wins, avg Δ +$2,009
  229 stay-losses, avg Δ -$924
  Break-even threshold: P(stay_wins) ≈ 0.315 — so stay any time P > 32%.

OOS protocol:
  - Train on TRAIN decision trades only
  - Evaluate at thresholds 0.30, 0.35, 0.40, 0.50
  - Baseline: V0 and SB_Q5 (round-2 best rule-based)
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import timedelta
from typing import Optional, Tuple

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
VAL_END = '2025-12-31'


def _session_open_timestamp(bars):
    if bars.empty: return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning): return morning.iloc[0]['timestamp']
    return None


def replay_trade(bars, entry_price, range_high, range_low, entry_time) -> dict:
    """Walk bars, compute:
      - exit outcomes: V0 (static_lock_1R) and SB (stay-BE)
      - category: never_armed / armed_no_pullback / armed_and_pulled_back
      - in-trade features at the pullback moment (if applicable)
    """
    rs = range_high - range_low
    out = {'category': None, 'rs': rs,
           'v0_exit_price': None, 'sb_exit_price': None}
    if rs <= 0:
        return out

    trig_lvl = entry_price + 1.5 * rs
    lock_lvl = entry_price + 1.0 * rs
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    if len(post) < 2:
        return out
    walk = post.iloc[1:].reset_index(drop=True)

    # Pass 1: identify arm + pullback, gather pre-pullback features
    armed = False
    pullback_bar = None
    peak_high_pre = 0.0
    peak_bar_idx = None
    vols_pre = []  # for computing pullback_vol_ratio
    # Walk
    for i, row in walk.iterrows():
        h = float(row['high']); lo = float(row['low'])
        v = int(row.get('volume', 0) or 0)
        if h > peak_high_pre:
            peak_high_pre = h
            peak_bar_idx = i
        if not armed and h >= trig_lvl:
            armed = True
        vols_pre.append(v)
        if armed and lo <= lock_lvl:
            pullback_bar = (i, row)
            break

    if not armed:
        # V0: hit range_low or EOD
        exit_price, _, _ = _v0_simulate(walk, entry_price, range_high, range_low)
        out['category'] = 'never_armed'
        out['v0_exit_price'] = exit_price
        out['sb_exit_price'] = exit_price  # same — never armed
        return out

    if pullback_bar is None:
        # Armed but no pullback — ride to EOD
        last = walk.iloc[-1]
        exit_price = float(last['close']) * (1 - EXIT_SLIP_BPS/10000)
        out['category'] = 'armed_no_pullback'
        out['v0_exit_price'] = exit_price
        out['sb_exit_price'] = exit_price  # same — no decision
        return out

    pullback_idx, pullback_row = pullback_bar
    out['category'] = 'armed_and_pulled_back'

    # V0 = exit at lock_lvl
    v0_exit = lock_lvl * (1 - EXIT_SLIP_BPS/10000)
    out['v0_exit_price'] = v0_exit

    # In-trade features at pullback moment
    bars_since_entry = pullback_idx + 1
    bars_from_peak = bars_since_entry - (peak_bar_idx + 1) if peak_bar_idx is not None else 0
    # pullback bar volume ratio: pullback vol / median of pre-pullback vols (excl last)
    pullback_vol = int(pullback_row.get('volume', 0) or 0)
    pre_vols = vols_pre[:-1]  # exclude the pullback bar itself
    med_pre_vol = float(np.median(pre_vols)) if pre_vols else 1.0
    pullback_vol_ratio = pullback_vol / max(med_pre_vol, 1.0)
    # cumulative volume up to pullback
    cum_vol_pre = sum(vols_pre)
    # pullback bar close position: (close-low)/(high-low)
    p_h = float(pullback_row['high']); p_l = float(pullback_row['low'])
    p_c = float(pullback_row['close'])
    close_pos = (p_c - p_l) / max(p_h - p_l, 1e-6)

    out.update({
        'peak_mfe_pre_pullback_r': (peak_high_pre - entry_price) / rs,
        'bars_since_entry_at_pullback': bars_since_entry,
        'bars_from_peak_to_pullback': bars_from_peak,
        'pullback_vol_ratio': pullback_vol_ratio,
        'pullback_close_position': close_pos,
        'cum_vol_pre_pullback': cum_vol_pre,
    })

    # SB = stay, BE stop, ride to EOD
    # Continue from pullback bar (since we're assuming we stayed)
    remaining = walk.iloc[pullback_idx:]
    sb_exit = None
    for _, row in remaining.iterrows():
        lo = float(row['low'])
        if lo <= entry_price:
            sb_exit = entry_price * (1 - EXIT_SLIP_BPS/10000)
            break
    if sb_exit is None:
        sb_exit = float(remaining.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000)
    out['sb_exit_price'] = sb_exit
    return out


def _v0_simulate(walk, entry, rh, rl):
    """V0 baseline exit — used for never_armed trades."""
    rs = rh - rl
    trig = entry + 1.5 * rs; lock = entry + 1.0 * rs
    stop = rl; armed = False
    for _, b in walk.iterrows():
        h = float(b['high']); lo = float(b['low'])
        if not armed and h >= trig: armed = True; stop = max(stop, lock)
        if lo <= stop:
            return stop * (1 - EXIT_SLIP_BPS/10000), armed, 'lock' if armed else 'stop'
    return float(walk.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000), armed, 'eod'


# ---------------------------------------------------------------------------
# Pipeline scaffolding
# ---------------------------------------------------------------------------

def _run_pipeline_base(df):
    """Select same trades as V0 would; returns sel with _composite, _quintile, _rp_*.

    This version doesn't need variant-specific pnl — it selects trades using
    V0 as the reference and returns the selection + mults (also fit on V0).
    """
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
    sel['date'] = pd.to_datetime(sel['date'])
    return sel, mults, cutoffs, params


ENTRY_FEATURES = [
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
IN_TRADE_FEATURES = [
    'peak_mfe_pre_pullback_r', 'bars_since_entry_at_pullback',
    'bars_from_peak_to_pullback', 'pullback_vol_ratio',
    'pullback_close_position', 'cum_vol_pre_pullback',
]
CLF_FEATURES = ENTRY_FEATURES + IN_TRADE_FEATURES


def _fit_classifier(X, y):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    # Try both GB and LR; pick the one with higher OOS log-loss on a 20% internal validation split
    n = len(X)
    perm = np.random.RandomState(42).permutation(n)
    split = int(n * 0.8)
    Xi, Xv = X[perm[:split]], X[perm[split:]]
    yi, yv = y[perm[:split]], y[perm[split:]]

    # LR
    sc = StandardScaler().fit(Xi)
    lr = LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced')
    lr.fit(sc.transform(Xi), yi)
    pv_lr = lr.predict_proba(sc.transform(Xv))[:, 1]
    from sklearn.metrics import log_loss
    ll_lr = log_loss(yv, pv_lr, labels=[0, 1]) if len(set(yv)) > 1 else 1e9

    # GB (small, regularized)
    gb = GradientBoostingClassifier(
        n_estimators=60, max_depth=2, learning_rate=0.05,
        random_state=42, min_samples_leaf=25, subsample=0.8,
    )
    gb.fit(Xi, yi)
    pv_gb = gb.predict_proba(Xv)[:, 1]
    ll_gb = log_loss(yv, pv_gb, labels=[0, 1]) if len(set(yv)) > 1 else 1e9

    if ll_lr < ll_gb:
        print(f"  Picked LR (val log-loss {ll_lr:.3f} vs GB {ll_gb:.3f})")
        # Refit on full training set
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(max_iter=2000, C=0.5, class_weight='balanced')
        lr.fit(sc.transform(X), y)
        return ('lr', (lr, sc))
    else:
        print(f"  Picked GB (val log-loss {ll_gb:.3f} vs LR {ll_lr:.3f})")
        gb.fit(X, y)
        return ('gb', gb)


def _predict(clf_tuple, X):
    kind, model = clf_tuple
    if kind == 'lr':
        lr, sc = model
        return lr.predict_proba(sc.transform(X))[:, 1]
    return model.predict_proba(X)[:, 1]


def _calmar(pnls):
    cum = pnls.cumsum()
    peak = -1e18; mdd = 0.0
    for v in cum:
        peak = max(peak, v); mdd = min(mdd, v - peak)
    tp = float(pnls.sum())
    return tp, float(mdd), (tp / abs(mdd) if mdd < 0 else float('inf'))


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

    sel, mults, cutoffs, params = _run_pipeline_base(df)
    print(f"Selected trades: {len(sel)}")
    print(f"Adaptive mults: { {q: round(v,3) for q,v in mults.items()} }")

    # Replay each selected trade
    print("Replaying trades to compute in-trade features + stay/exit outcomes...")
    recs = []
    for _, row in sel.iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None: continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = bars[(bars['timestamp'] >= open_ts) &
                          (bars['timestamp'] < range_end)]
        if len(range_bars) < 5: continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh: entry_ts = b['timestamp']; break
        if entry_ts is None: continue
        entry_p = float(row['entry_price'])
        res = replay_trade(bars, entry_p, rh, rl, entry_ts)
        if res['category'] is None:
            continue
        rec = dict(row)
        rec.update(res)
        rec['entry_price'] = entry_p
        rec['shares'] = max(1, int(OLD_POS / entry_p))
        # Compute per-trade pnls for V0 and SB (sized using quintile mult)
        mult = mults[row['_quintile']]
        v0_pnl_per_share = rec['v0_exit_price'] - entry_p
        sb_pnl_per_share = rec['sb_exit_price'] - entry_p
        rec['v0_raw_pnl'] = v0_pnl_per_share * rec['shares']
        rec['sb_raw_pnl'] = sb_pnl_per_share * rec['shares']
        rec['v0_sized_pnl'] = rec['v0_raw_pnl'] * rec['_rp_position'] / OLD_POS * mult
        rec['sb_sized_pnl'] = rec['sb_raw_pnl'] * rec['_rp_position'] / OLD_POS * mult
        rec['stay_wins'] = bool(sb_pnl_per_share > v0_pnl_per_share)
        recs.append(rec)

    rdf = pd.DataFrame(recs)
    print(f"\nReplayed {len(rdf)} trades")
    for cat, c in rdf['category'].value_counts().items():
        print(f"  {cat:<25} {c:>4} ({c/len(rdf)*100:.1f}%)")

    # Split
    rdf['date'] = pd.to_datetime(rdf['date'])
    train_m = rdf['date'] <= TRAIN_END
    val_m   = (rdf['date'] > TRAIN_END) & (rdf['date'] <= VAL_END)
    hoq1_m  = rdf['date'] > VAL_END

    # Decision trades (where classifier operates): armed_and_pulled_back
    dec = rdf[rdf['category'] == 'armed_and_pulled_back'].copy()
    train_dec = dec[dec['date'] <= TRAIN_END]
    val_dec   = dec[(dec['date'] > TRAIN_END) & (dec['date'] <= VAL_END)]
    hoq1_dec  = dec[dec['date'] > VAL_END]

    print(f"\nDecision trades (armed → pulled back to +1R):")
    print(f"  TRAIN: {len(train_dec):>3} ({train_dec['stay_wins'].mean()*100:.1f}% stay-wins)")
    print(f"  VAL:   {len(val_dec):>3} ({val_dec['stay_wins'].mean()*100:.1f}% stay-wins)")
    print(f"  HOQ1+: {len(hoq1_dec):>3} ({hoq1_dec['stay_wins'].mean()*100:.1f}% stay-wins)")

    # Classifier: predict stay_wins on decision trades
    print(f"\n{'='*70}")
    print("  Training classifier on TRAIN decision trades...")
    print(f"{'='*70}")
    X_train = train_dec[CLF_FEATURES].fillna(0).values
    y_train = train_dec['stay_wins'].astype(int).values
    if len(set(y_train)) < 2:
        print("  Can't train — single class in TRAIN. Abort.")
        return
    clf = _fit_classifier(X_train, y_train)

    # Predict on each split
    for label, d in [('TRAIN', train_dec), ('VAL', val_dec), ('HOQ1+', hoq1_dec)]:
        if len(d) == 0: continue
        X = d[CLF_FEATURES].fillna(0).values
        d['_p_stay_wins'] = _predict(clf, X)

    # Evaluate at multiple thresholds
    print(f"\n{'='*100}")
    print("  CLASSIFIER RESULTS — apply to each split, measure vs V0")
    print(f"{'='*100}")
    print(f"{'Threshold':<11} {'Split':<7} {'N stays':>9} {'Stay win%':>11} "
          f"{'P&L':>12} {'Δ vs V0':>11} {'Δ vs SB(all)':>13}")
    print('-' * 100)

    for th in [0.30, 0.35, 0.40, 0.50]:
        for split_name, d, all_d in [
            ('TRAIN', train_dec, rdf[train_m]),
            ('VAL',   val_dec,   rdf[val_m]),
            ('HOQ1+', hoq1_dec,  rdf[hoq1_m]),
        ]:
            if len(d) == 0: continue
            d_cp = d.copy()
            d_cp['_p'] = d_cp.get('_p_stay_wins', 0.5)
            # Choose exit per trade
            d_cp['_chosen_pnl'] = d_cp.apply(
                lambda r: r['sb_sized_pnl'] if r.get('_p_stay_wins', 0.5) >= th
                           else r['v0_sized_pnl'], axis=1
            )
            stays = (d_cp.get('_p_stay_wins', pd.Series([0]*len(d_cp))) >= th).sum()
            stay_wr = (d_cp[d_cp.get('_p_stay_wins', pd.Series([0]*len(d_cp))) >= th]['stay_wins']).mean() * 100 \
                if stays > 0 else 0
            # Reconstruct full sized_pnl for entire split (decision + non-decision)
            non_dec = all_d[all_d['category'] != 'armed_and_pulled_back']
            # For non-decision trades, V0 and SB produce same outcome
            non_dec_pnl = non_dec['v0_sized_pnl'].sum()
            total_pnl = non_dec_pnl + d_cp['_chosen_pnl'].sum()
            v0_total = (all_d['v0_sized_pnl']).sum()
            sb_total = (all_d['sb_sized_pnl']).sum()
            delta_v0 = total_pnl - v0_total
            delta_sb = total_pnl - sb_total
            print(f"  p>={th:<6} {split_name:<7} {stays:>7}   "
                  f"{stay_wr:>8.1f}%   "
                  f"${total_pnl:>+9,.0f}  "
                  f"${delta_v0:>+9,.0f}  "
                  f"${delta_sb:>+11,.0f}")
        print('-' * 100)

    # Final comparison at best threshold (judge by VAL + HOQ1+ Δ sum)
    print(f"\n{'='*100}")
    print("  SHIP RUBRIC — best threshold = max(VAL Δ + HOQ1+ Δ) with both positive")
    print(f"{'='*100}")
    best = None
    for th in [0.30, 0.35, 0.40, 0.45, 0.50]:
        def slice_pnl(d, all_d):
            if len(d) == 0: return 0, 0, 0
            d_cp = d.copy()
            d_cp['_chosen_pnl'] = d_cp.apply(
                lambda r: r['sb_sized_pnl'] if r.get('_p_stay_wins', 0) >= th
                           else r['v0_sized_pnl'], axis=1
            )
            non_dec = all_d[all_d['category'] != 'armed_and_pulled_back']
            tot = non_dec['v0_sized_pnl'].sum() + d_cp['_chosen_pnl'].sum()
            v0 = all_d['v0_sized_pnl'].sum()
            return tot, v0, tot - v0
        _, _, val_d = slice_pnl(val_dec, rdf[val_m])
        _, _, hoq1_d = slice_pnl(hoq1_dec, rdf[hoq1_m])
        both_pos = val_d > 5000 and hoq1_d > 5000
        marker = ' ← SHIP' if both_pos else ''
        print(f"  p>={th:<5} VAL Δ ${val_d:>+9,.0f}   HOQ1+ Δ ${hoq1_d:>+9,.0f}"
              f"   sum ${val_d+hoq1_d:>+9,.0f}{marker}")
        if best is None or (both_pos and (val_d + hoq1_d) > best[1]):
            if both_pos:
                best = (th, val_d + hoq1_d, val_d, hoq1_d)

    print(f"\n  Baselines for comparison:")
    v0_val = rdf[val_m]['v0_sized_pnl'].sum()
    v0_hoq = rdf[hoq1_m]['v0_sized_pnl'].sum()
    sb_val = rdf[val_m]['sb_sized_pnl'].sum()
    sb_hoq = rdf[hoq1_m]['sb_sized_pnl'].sum()
    print(f"    V0:       VAL ${v0_val:>+9,.0f}  HOQ1+ ${v0_hoq:>+9,.0f}")
    print(f"    SB (all): VAL ${sb_val:>+9,.0f}  HOQ1+ ${sb_hoq:>+9,.0f}  "
          f"(Δ vs V0: VAL ${sb_val-v0_val:+,.0f}, HOQ1+ ${sb_hoq-v0_hoq:+,.0f})")

    if best:
        print(f"\n  ★ BEST THRESHOLD (both OOS splits positive): p>={best[0]}, "
              f"combined OOS Δ = ${best[1]:+,.0f}")
    else:
        print(f"\n  ✗ No threshold produces both VAL Δ > $5K AND HOQ1+ Δ > $5K.")
        print(f"     Classifier does not generalize well enough to ship.")

    # Feature importance for GB model
    kind, model = clf
    if kind == 'gb':
        importances = dict(zip(CLF_FEATURES, model.feature_importances_))
        top = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
        print(f"\n  Top-15 feature importances:")
        for f, w in top:
            marker = '  (in-trade)' if f in IN_TRADE_FEATURES else ''
            print(f"    {f:<35} {w:.4f}{marker}")
    elif kind == 'lr':
        # Show signs/magnitudes of coefficients (standardized input)
        lr, sc = model
        coefs = dict(zip(CLF_FEATURES, lr.coef_[0]))
        top = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        print(f"\n  Top-15 LR coefficients (standardized):")
        for f, w in top:
            marker = '  (in-trade)' if f in IN_TRADE_FEATURES else ''
            sign = '+' if w > 0 else '-'
            print(f"    {f:<35} {sign}{abs(w):.3f}{marker}")

    print("\nDone.")


if __name__ == '__main__':
    main()

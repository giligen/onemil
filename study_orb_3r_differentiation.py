"""Stage 0 — Differentiation analysis for ORB +3R add-to-winners.

Goal: at the moment a trade crosses +3R MFE, find which intraday-bar features
separate WINNERS (forward bars reach >= +5R BEFORE retracing to +2R) from
LOSERS (forward bars touch +2R BEFORE reaching +5R).

Sample: TRAIN trades only (date <= 2025-06-30) that hit +3R MFE. We freeze
threshold-fitting on TRAIN to avoid OOS leakage when Stage 2 applies the rule.

Output: per-trade feature CSV + ranked feature analysis with AUC per feature
and recommended thresholds for the simple rule.
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import datetime, timedelta
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


# ---------------------------------------------------------------------------
# Constants — match shipping pipeline
# ---------------------------------------------------------------------------
ACCOUNT = 100_000.0
N = 4
RISK = 3000.0
MIN_STOP_PCT = 1.0
OLD_POS = 50_000.0
Q_CAPS = {'Q1': 3.0, 'Q2': 3.0, 'Q3': 3.0, 'Q4': 3.0, 'Q5': 1.5}
Q_ORDER = {'Q4': 0, 'Q5': 1, 'Q3': 2, 'Q2': 3, 'Q1': 4}
EXIT_SLIP_BPS = 10.0
TRAIN_END = '2025-06-30'

# Stage 0 specifics
ADD_TRIGGER_R = 3.0   # Add at +3R MFE
WIN_TARGET_R = 5.0    # Forward bars must reach this to label WIN
LOSS_STOP_R = 2.0     # Forward bars touching this label LOSS
SKIP_Q1 = True        # Match shipping pipeline (Q1 filter on)


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
# Feature computation at the +3R bar
# ---------------------------------------------------------------------------

def compute_features_at_3r(session_bars: pd.DataFrame, idx: int,
                             entry_price: float, range_size: float,
                             avg_daily_volume_20d: float,
                             entry_idx_in_session: int) -> dict:
    """Compute the 14 candidate features at session-frame bar index `idx`.

    `session_bars`: bars with RangeIndex starting at session open (idx=0 = 9:30 bar).
    `idx`: position of the +3R bar in session_bars.
    `entry_idx_in_session`: position of the entry bar (used for range_expansion).
    """
    bar = session_bars.iloc[idx]
    so_far = session_bars.iloc[:idx + 1]
    closes = so_far['close'].astype(float)
    highs = so_far['high'].astype(float)
    lows = so_far['low'].astype(float)
    volumes = so_far['volume'].astype(float)

    feats = {}

    # 1. VWAP distance %  (uses session-cumulative VWAP — consistent with trading.indicators.vwap)
    typical = (highs + lows + closes) / 3.0
    cum_pv = (typical * volumes).cumsum()
    cum_v = volumes.cumsum().replace(0, 1e-9)
    vwap_series = cum_pv / cum_v
    last_close = closes.iloc[-1]
    last_vwap = vwap_series.iloc[-1]
    feats['vwap_distance_pct'] = ((last_close - last_vwap) / last_vwap * 100
                                    if last_vwap > 0 else 0.0)

    # 2. VWAP slope over last 5 bars (normalized by VWAP level)
    if len(vwap_series) >= 5 and last_vwap > 0:
        last5 = vwap_series.iloc[-5:].values
        slope, _ = np.polyfit(np.arange(5), last5, 1)
        feats['vwap_slope_5bar_norm'] = slope / last_vwap
    else:
        feats['vwap_slope_5bar_norm'] = 0.0

    # 3-5. MACD on closes (12/26/9, like macd_wave_engine)
    if len(closes) >= 26:
        ema_fast = closes.ewm(span=12, adjust=False).mean()
        ema_slow = closes.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        hist = macd_line - signal_line
        feats['macd_hist'] = float(hist.iloc[-1])
        # Consecutive positive bars back from end
        consec = 0
        for h in reversed(hist.values):
            if h > 0:
                consec += 1
            else:
                break
        feats['macd_consecutive_pos'] = consec
        # Rising = last 3 bars strictly increasing
        if len(hist) >= 3:
            feats['macd_rising'] = int(
                hist.iloc[-1] > hist.iloc[-2] > hist.iloc[-3])
        else:
            feats['macd_rising'] = 0
    else:
        feats['macd_hist'] = 0.0
        feats['macd_consecutive_pos'] = 0
        feats['macd_rising'] = 0

    # 6. SMA20 distance %
    if len(closes) >= 20:
        sma20 = closes.iloc[-20:].mean()
        feats['sma20_distance_pct'] = ((last_close - sma20) / sma20 * 100
                                         if sma20 > 0 else 0.0)
    else:
        feats['sma20_distance_pct'] = 0.0

    # 7. Cumulative volume ratio (cum from 9:30 / avg_daily_20d)
    cum_vol = float(volumes.sum())
    feats['cum_vol_ratio'] = (cum_vol / avg_daily_volume_20d
                                if avg_daily_volume_20d > 0 else 0.0)

    # 8. Volume expansion (last 3 vs preceding 3)
    if len(volumes) >= 6:
        last3 = float(volumes.iloc[-3:].mean())
        prev3 = float(volumes.iloc[-6:-3].mean())
        feats['vol_expansion_3v6'] = last3 / prev3 if prev3 > 0 else 1.0
    else:
        feats['vol_expansion_3v6'] = 1.0

    # 9. Price convergence (5-bar stddev of closes / current close)
    if len(closes) >= 5 and last_close > 0:
        feats['price_convergence_5bar'] = float(
            closes.iloc[-5:].std() / last_close)
    else:
        feats['price_convergence_5bar'] = 0.0

    # 10. Bar color streak — consecutive green bars walking back from idx
    streak = 0
    for j in range(idx, -1, -1):
        b = session_bars.iloc[j]
        if float(b['close']) > float(b['open']):
            streak += 1
        else:
            break
    feats['bar_color_streak'] = streak

    # 11. High-close ratio at the +3R bar
    h = float(bar['high']); l = float(bar['low']); c = float(bar['close'])
    feats['high_close_ratio'] = (c - l) / max(h - l, 1e-6)

    # 12. Minutes since 9:30 ET open (idx 0 is the 9:30 bar)
    feats['minutes_since_open'] = idx

    # 13. Range expansion — last 3 bars avg range / first 3 post-entry bars avg range
    post_entry = session_bars.iloc[entry_idx_in_session:idx + 1]
    if len(post_entry) >= 6:
        last3_range = float((post_entry['high'].iloc[-3:].astype(float) -
                              post_entry['low'].iloc[-3:].astype(float)).mean())
        first3_range = float((post_entry['high'].iloc[:3].astype(float) -
                                post_entry['low'].iloc[:3].astype(float)).mean())
        feats['range_expansion_3v3'] = last3_range / first3_range if first3_range > 0 else 1.0
    else:
        feats['range_expansion_3v3'] = 1.0

    # 14. Pullback from peak %  (within recent 5 bars)
    recent_high = float(highs.iloc[-5:].max() if len(highs) >= 5 else highs.max())
    feats['pullback_from_peak_pct'] = ((recent_high - last_close) / recent_high * 100
                                         if recent_high > 0 else 0.0)

    return feats


# ---------------------------------------------------------------------------
# Outcome labeling — walk forward from +3R bar
# ---------------------------------------------------------------------------

def label_outcome_after_3r(session_bars: pd.DataFrame, idx_3r: int,
                             entry_price: float, range_size: float
                             ) -> Tuple[str, float, float]:
    """Walk forward from idx_3r+1 to EOD. Return (label, forward_peak_r, eod_close_r).

    label: 'WIN' (high >= +5R first), 'LOSS' (low <= +2R first), 'NEUTRAL' (neither).
    """
    target_5r = entry_price + WIN_TARGET_R * range_size
    stop_2r = entry_price + LOSS_STOP_R * range_size
    forward = session_bars.iloc[idx_3r + 1:]
    if len(forward) == 0:
        return 'NEUTRAL', 0.0, 0.0

    forward_peak = float(forward['high'].astype(float).max())
    eod_close = float(forward.iloc[-1]['close'])

    for _, row in forward.iterrows():
        h = float(row['high']); lo = float(row['low'])
        win_hit = h >= target_5r
        loss_hit = lo <= stop_2r
        if win_hit and loss_hit:
            # Same-bar ambiguous — drop from analysis
            label = 'NEUTRAL'
            break
        if win_hit:
            label = 'WIN'
            break
        if loss_hit:
            label = 'LOSS'
            break
    else:
        # Reached EOD without hitting either
        if eod_close >= target_5r:
            label = 'WIN'
        elif eod_close <= stop_2r:
            label = 'LOSS'
        else:
            label = 'NEUTRAL'

    forward_peak_r = (forward_peak - entry_price) / range_size if range_size > 0 else 0.0
    eod_close_r = (eod_close - entry_price) / range_size if range_size > 0 else 0.0
    return label, forward_peak_r, eod_close_r


# ---------------------------------------------------------------------------
# Pipeline scaffolding (mirrors shipping pipeline incl. Q1 filter)
# ---------------------------------------------------------------------------

def _run_pipeline(df_v0: pd.DataFrame):
    df = df_v0.copy()
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

    if SKIP_Q1:
        kept = kept[kept['_quintile'] != 'Q1'].copy()

    sel_rows = []
    for day, dg in kept.groupby('date'):
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
    return sel, mults


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def _compute_auc_simple(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUC for binary labels (1=WIN, 0=LOSS). Both ranking directions tested."""
    try:
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) < 2:
            return 0.5
        # Try both directions, return whichever has higher AUC (feature may be
        # negatively correlated with WIN — we report magnitude of separation)
        auc1 = roc_auc_score(labels, scores)
        return auc1
    except Exception:
        return 0.5


def _welch_t(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Welch's t-test (unequal variance). Returns (t_stat, p_value)."""
    try:
        from scipy.stats import ttest_ind
        if len(a) < 2 or len(b) < 2:
            return 0.0, 1.0
        result = ttest_ind(a, b, equal_var=False, nan_policy='omit')
        return float(result.statistic), float(result.pvalue)
    except Exception:
        return 0.0, 1.0


def _spearman(scores: np.ndarray, continuous_target: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        if len(scores) < 3:
            return 0.0
        rho, _ = spearmanr(scores, continuous_target, nan_policy='omit')
        return float(rho) if not np.isnan(rho) else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    print(f"Loading bars for {len(pairs)} (symbol, date) pairs...")
    db = Database(db_path='data/cache.db')
    raw_bars = db.get_intraday_bars_bulk(pairs)
    db.close()
    bars_cache = {k: _bars_to_df(v) for k, v in raw_bars.items()}

    print("Building V0 trade set (with shipping Q1 filter)...")
    sel, mults = _run_pipeline(df)
    print(f"Selected trades total: {len(sel)}")
    train_sel = sel[sel['date'] <= TRAIN_END].copy()
    print(f"TRAIN trades: {len(train_sel)}  (we analyze only these for differentiation)")

    # Replay each TRAIN trade to find +3R bar and compute features
    print("\nReplaying TRAIN trades to find +3R cross moments...")
    records = []
    n_no_3r = 0  # trades that never hit +3R
    n_no_bars = 0  # bars missing or malformed
    for _, row in train_sel.iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty:
            n_no_bars += 1; continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None:
            n_no_bars += 1; continue
        # Build session frame starting at 9:30
        session = bars[bars['timestamp'] >= open_ts].reset_index(drop=True)
        if len(session) < 10:
            n_no_bars += 1; continue
        # Compute range from 9:30-9:35 (first 5 bars)
        range_end = open_ts + timedelta(minutes=5)
        range_bars = session[(session['timestamp'] >= open_ts) &
                              (session['timestamp'] < range_end)]
        if len(range_bars) < 5:
            n_no_bars += 1; continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        range_size = rh - rl
        if range_size <= 0:
            n_no_bars += 1; continue
        # Find entry bar (first post-9:35 bar where high > rh, within 60 min)
        post_range = session[(session['timestamp'] >= range_end) &
                              (session['timestamp'] < range_end + timedelta(minutes=60))]
        entry_idx_in_session = None
        for i, b in post_range.iterrows():
            if float(b['high']) > rh:
                entry_idx_in_session = int(i)
                break
        if entry_idx_in_session is None:
            n_no_bars += 1; continue
        entry_price = float(row['entry_price'])
        # Find +3R bar — first bar at or after entry where high >= entry + 3R
        target_3r = entry_price + ADD_TRIGGER_R * range_size
        idx_3r = None
        for i in range(entry_idx_in_session, len(session)):
            if float(session.iloc[i]['high']) >= target_3r:
                idx_3r = i
                break
        if idx_3r is None:
            n_no_3r += 1; continue
        # Need enough bars after for outcome labeling
        if idx_3r >= len(session) - 1:
            n_no_3r += 1; continue

        # Compute features at +3R bar
        try:
            feats = compute_features_at_3r(
                session, idx_3r, entry_price, range_size,
                float(row.get('avg_daily_volume_20d', 0) or 0),
                entry_idx_in_session,
            )
        except Exception as e:
            print(f"  Feature compute failed for {row['symbol']} {row['date']}: {e}")
            continue

        # Label outcome
        label, fpr, eodr = label_outcome_after_3r(
            session, idx_3r, entry_price, range_size)

        rec = {
            'symbol': row['symbol'],
            'date': row['date'].strftime('%Y-%m-%d'),
            'quintile': row['_quintile'],
            'composite': row['_composite'],
            'range_size': range_size,
            'range_size_pct': float(row['range_size_pct']),
            'entry_price': entry_price,
            'idx_3r': idx_3r,
            'entry_idx': entry_idx_in_session,
            'label': label,
            'forward_peak_r': fpr,
            'eod_close_r': eodr,
            **feats,
        }
        records.append(rec)

    if not records:
        print("No records — analysis impossible. Exit.")
        return

    rdf = pd.DataFrame(records)
    print(f"\nReplay complete:")
    print(f"  TRAIN trades total:           {len(train_sel)}")
    print(f"  Skipped (missing bars):       {n_no_bars}")
    print(f"  Skipped (never hit +3R):      {n_no_3r}")
    print(f"  Records analyzed:             {len(rdf)}")
    print(f"\nLabel distribution on +3R-crossing trades:")
    for lbl in ['WIN', 'LOSS', 'NEUTRAL']:
        c = (rdf['label'] == lbl).sum()
        print(f"  {lbl:<8} {c:>4}  ({c/len(rdf)*100:.1f}%)")

    # Outcome distribution context
    print(f"\nForward peak R (all +3R trades):")
    for pct in [10, 25, 50, 75, 90, 99]:
        print(f"  p{pct}: {rdf['forward_peak_r'].quantile(pct/100):.2f}R")

    # Tail-dependence preview: WIN bucket P&L distribution
    wins = rdf[rdf['label'] == 'WIN'].copy()
    losses = rdf[rdf['label'] == 'LOSS'].copy()
    if len(wins):
        wins['add_r_gain'] = wins['forward_peak_r'] - ADD_TRIGGER_R
        print(f"\nWIN bucket — add_R_gain (peak_R - 3R) distribution:")
        for pct in [25, 50, 75, 90, 99]:
            print(f"  p{pct}: {wins['add_r_gain'].quantile(pct/100):.2f}R")
        print(f"  mean: {wins['add_r_gain'].mean():.2f}R")

    # Per-feature analysis
    feature_names = [
        'vwap_distance_pct', 'vwap_slope_5bar_norm',
        'macd_hist', 'macd_consecutive_pos', 'macd_rising',
        'sma20_distance_pct', 'cum_vol_ratio', 'vol_expansion_3v6',
        'price_convergence_5bar', 'bar_color_streak', 'high_close_ratio',
        'minutes_since_open', 'range_expansion_3v3', 'pullback_from_peak_pct',
    ]

    # Restrict to WIN vs LOSS for AUC analysis (drop NEUTRAL)
    wl = rdf[rdf['label'].isin(['WIN', 'LOSS'])].copy()
    if len(wl) < 10:
        print("\nWARNING: too few WIN+LOSS trades for stable AUC. Skipping ranking.")
        return
    wl['is_win'] = (wl['label'] == 'WIN').astype(int)

    print(f"\n{'='*90}")
    print(f"  FEATURE RANKING — WIN ({wl['is_win'].sum()}) vs LOSS ({(1-wl['is_win']).sum()})")
    print(f"{'='*90}")
    print(f"{'Feature':<28} {'WIN mean':>12} {'LOSS mean':>12} {'AUC':>7} "
          f"{'t-stat':>8} {'p-val':>8} {'Spearman':>10}")
    print('-' * 90)
    feat_stats = []
    for f in feature_names:
        if f not in wl.columns:
            continue
        scores = wl[f].astype(float).values
        labels = wl['is_win'].values
        win_vals = wl.loc[wl['is_win'] == 1, f].astype(float).values
        loss_vals = wl.loc[wl['is_win'] == 0, f].astype(float).values
        auc = _compute_auc_simple(scores, labels)
        # AUC < 0.5 means feature is INVERSELY related to WIN — record max for ranking
        auc_eff = max(auc, 1 - auc)
        t_stat, p_val = _welch_t(win_vals, loss_vals)
        spear = _spearman(scores, rdf.loc[rdf['label'].isin(['WIN', 'LOSS']), 'forward_peak_r'].values)
        feat_stats.append({
            'feature': f,
            'win_mean': float(np.mean(win_vals)) if len(win_vals) else 0,
            'loss_mean': float(np.mean(loss_vals)) if len(loss_vals) else 0,
            'auc': auc,
            'auc_eff': auc_eff,
            'inverse': auc < 0.5,
            't_stat': t_stat,
            'p_val': p_val,
            'spearman_with_peak': spear,
        })
    fdf = pd.DataFrame(feat_stats).sort_values('auc_eff', ascending=False)
    for _, r in fdf.iterrows():
        inv_marker = ' (INV)' if r['inverse'] else ''
        print(f"{r['feature']:<28} {r['win_mean']:>+11.3f}  {r['loss_mean']:>+11.3f}  "
              f"{r['auc_eff']:>5.3f}  {r['t_stat']:>+7.2f}  {r['p_val']:>7.4f}  "
              f"{r['spearman_with_peak']:>+9.3f}{inv_marker}")

    # Recommended thresholds for top features (AUC >= 0.60)
    print(f"\n{'='*90}")
    print(f"  RECOMMENDED THRESHOLDS (features with AUC >= 0.60)")
    print(f"{'='*90}")
    top = fdf[fdf['auc_eff'] >= 0.60].head(5)
    if len(top) == 0:
        print("  No features with AUC >= 0.60. Project may not be feasible —")
        print("  +3R moment is not discriminable from intraday features tested.")
        print("  Highest AUC observed: {:.3f}".format(fdf['auc_eff'].max()))
    else:
        for _, r in top.iterrows():
            f = r['feature']
            win_vals = wl.loc[wl['is_win'] == 1, f].astype(float).values
            loss_vals = wl.loc[wl['is_win'] == 0, f].astype(float).values
            # Suggest threshold = midway between WIN median and LOSS median (in correct direction)
            if r['inverse']:
                # LOWER values predict WIN — threshold = win_median + (loss_median - win_median) * 0.5
                threshold = float(np.percentile(win_vals, 75))  # be permissive: top-quartile of WIN
                op = '<='
                rule = f"{f} {op} {threshold:.4f}  (lower = bullish)"
            else:
                threshold = float(np.percentile(win_vals, 25))  # bottom-quartile of WIN values
                op = '>='
                rule = f"{f} {op} {threshold:.4f}  (higher = bullish)"
            # WIN-rate for trades passing this threshold
            if op == '>=':
                pass_mask = wl[f].astype(float) >= threshold
            else:
                pass_mask = wl[f].astype(float) <= threshold
            n_pass = int(pass_mask.sum())
            if n_pass > 0:
                pass_winrate = (wl.loc[pass_mask, 'is_win']).mean() * 100
            else:
                pass_winrate = 0.0
            print(f"  {rule}")
            print(f"    AUC={r['auc_eff']:.3f}  passes {n_pass}/{len(wl)} TRAIN trades  "
                  f"win-rate-after-pass={pass_winrate:.1f}%")
            print()

    # Save outputs
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_csv = f'analysis_results/orb_3r_features_train_{ts}.csv'
    rdf.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}  ({len(rdf)} per-trade records)")

    out_md = f'analysis_results/orb_3r_differentiation_train_{ts}.md'
    with open(out_md, 'w') as f:
        f.write(f"# Stage 0 — ORB +3R Differentiation Analysis\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"## Sample\n")
        f.write(f"- TRAIN trades total: {len(train_sel)}\n")
        f.write(f"- Records analyzed (hit +3R): {len(rdf)}\n")
        f.write(f"- Skipped (missing bars): {n_no_bars}\n")
        f.write(f"- Skipped (never hit +3R): {n_no_3r}\n\n")
        f.write(f"## Label distribution\n")
        for lbl in ['WIN', 'LOSS', 'NEUTRAL']:
            c = (rdf['label'] == lbl).sum()
            f.write(f"- {lbl}: {c} ({c/len(rdf)*100:.1f}%)\n")
        f.write(f"\n## Feature ranking (WIN vs LOSS)\n\n")
        f.write(f"| Feature | WIN mean | LOSS mean | AUC | t-stat | p-val |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for _, r in fdf.iterrows():
            inv = ' (inv)' if r['inverse'] else ''
            f.write(f"| {r['feature']}{inv} | {r['win_mean']:+.3f} | {r['loss_mean']:+.3f} "
                     f"| {r['auc_eff']:.3f} | {r['t_stat']:+.2f} | {r['p_val']:.4f} |\n")
        f.write(f"\n## Recommended thresholds (AUC ≥ 0.60)\n\n")
        if len(top) == 0:
            f.write(f"None. Highest AUC: {fdf['auc_eff'].max():.3f}.\n")
        else:
            for _, r in top.iterrows():
                f.write(f"- `{r['feature']}` AUC={r['auc_eff']:.3f}\n")
    print(f"Saved {out_md}")


if __name__ == '__main__':
    main()

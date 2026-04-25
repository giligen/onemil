"""Stage 2.1 — Bull-flag-gated add with population segmentation + strict detector.

Refinement of study_orb_add_bullflag.py based on user direction:
> "Bull-flag makes the winners worth more, but I don't want it to impact the
>  others. Split and see which stocks benefit more based on price, vol, etc.,
>  and have a very strict bull-flag detection for very clean flags."

Goal: find the SUB-POPULATION + STRICT-PATTERN combination where bull-flag-
gated adds clear the tail-robustness bar.

Approach:
  1. Run +2.5R trigger with 3 detector configs: default, loose (current best),
     strict (textbook tight).
  2. Capture rich per-fire metadata for stratification.
  3. Stratify TRAIN+VAL combined fires by 5 dimensions:
     - price bucket
     - quintile
     - pole_gain_pct of detected flag (real strength)
     - breakout volume ratio (conviction)
     - minutes since open at breakout
  4. Identify favorable strata (WR ≥ 45% AND positive tail-3).
  5. Test "favorable strata + strict detector" combo on HOQ1+ blindly.

Honest caveat (encoded in the report): with ~90 total fires at loose detector
and ~20-40 at strict, the OOS sample on HOQ1+ alone (~15-20 fires) is not
big enough for ship confirmation. This is hypothesis-generation, not
ship-validation. Output should be treated as "candidate for paper-validation".
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict

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
from trading.pattern_detector import BullFlagDetector, BullFlagPattern


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
ADD_PCT = 0.5
SAFETY_STOP_R = 2.0
ADD_TRIGGER_R = 2.5  # +2.5R as user requested


def _session_open_timestamp(bars):
    if bars.empty: return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning): return morning.iloc[0]['timestamp']
    return None


def sim_v0(bars, entry_price, range_high, range_low, entry_time):
    rs = range_high - range_low
    if rs <= 0:
        return float(bars.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000), 'no_range'
    trig = entry_price + 1.5 * rs
    lock = entry_price + 1.0 * rs
    stop = range_low; armed = False
    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, b in post.iloc[1:].iterrows():
        h = float(b['high']); lo = float(b['low'])
        if not armed and h >= trig: armed = True; stop = max(stop, lock)
        if lo <= stop:
            return stop * (1 - EXIT_SLIP_BPS/10000), ('lock' if armed else 'stop')
    return float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000), 'eod'


def find_xr_bar_idx(session_bars, entry_idx, entry_price, range_size, x_r):
    target = entry_price + x_r * range_size
    for i in range(entry_idx, len(session_bars)):
        if float(session_bars.iloc[i]['high']) >= target:
            return i
    return None


def detect_flag_breakout_after(session_bars, start_idx, detector, symbol):
    for i in range(max(start_idx, 5), len(session_bars)):
        try:
            pattern = detector.detect(symbol, session_bars, end_idx=i + 1)
        except Exception:
            continue
        if pattern is not None:
            return i, pattern
    return None, None


def sim_add_with_pattern(session_bars, breakout_idx, breakout_close, pattern,
                          entry_price, range_size):
    add_entry_price = breakout_close * (1 + EXIT_SLIP_BPS/10000)
    safety_stop = entry_price + SAFETY_STOP_R * range_size
    add_stop = max(float(pattern.flag_low), safety_stop)
    forward = session_bars.iloc[breakout_idx + 1:]
    if len(forward) == 0:
        return add_entry_price * (1 - EXIT_SLIP_BPS/10000), 'no_room'
    for _, row in forward.iterrows():
        if float(row['low']) <= add_stop:
            return add_stop * (1 - EXIT_SLIP_BPS/10000), 'stop'
    return float(forward.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000), 'eod'


def _run_pipeline(df_v0):
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
    kept = kept[kept['_quintile'] != 'Q1'].copy()
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
    return sel, mults


def _calmar(pnls):
    cum = pnls.cumsum()
    peak = -1e18; mdd = 0.0
    for v in cum:
        peak = max(peak, v); mdd = min(mdd, v - peak)
    tp = float(pnls.sum())
    return tp, float(mdd), (tp / abs(mdd) if mdd < 0 else float('inf'))


def replay_with_detector(sel, bars_cache, mults, detector_cfg, label):
    """Replay all selected trades with bull-flag-gated add + capture rich
    per-fire metadata for stratification analysis.
    """
    detector = BullFlagDetector(**detector_cfg)
    records = []
    for _, row in sel.iterrows():
        key = (row['symbol'], row['date'].strftime('%Y-%m-%d'))
        bars = bars_cache.get(key)
        if bars is None or bars.empty: continue
        open_ts = _session_open_timestamp(bars)
        if open_ts is None: continue
        session = bars[bars['timestamp'] >= open_ts].reset_index(drop=True)
        if len(session) < 10: continue
        range_end = open_ts + timedelta(minutes=5)
        range_bars = session[(session['timestamp'] >= open_ts) &
                              (session['timestamp'] < range_end)]
        if len(range_bars) < 5: continue
        rh = float(range_bars['high'].max()); rl = float(range_bars['low'].min())
        rs = rh - rl
        if rs <= 0: continue
        post_range = session[(session['timestamp'] >= range_end) &
                              (session['timestamp'] < range_end + timedelta(minutes=60))]
        entry_idx_in_session = None
        for i, b in post_range.iterrows():
            if float(b['high']) > rh:
                entry_idx_in_session = int(i); break
        if entry_idx_in_session is None: continue
        entry_p = float(row['entry_price'])
        shares = max(1, int(OLD_POS / entry_p))
        # V0
        v0_exit_price, v0_reason = sim_v0(bars, entry_p, rh, rl,
                                            session.iloc[entry_idx_in_session]['timestamp'])
        per_pos_cap = ACCOUNT / N
        stop_pct = max(float(row['range_size_pct']), MIN_STOP_PCT)
        rp_pos = min(RISK / (stop_pct / 100.0), per_pos_cap)
        v0_pnl_per_share = v0_exit_price - entry_p
        v0_sized_pnl = (v0_pnl_per_share * shares) * rp_pos / OLD_POS * mults[row['_quintile']]

        idx_xr = find_xr_bar_idx(session, entry_idx_in_session, entry_p, rs, ADD_TRIGGER_R)
        add_sized_pnl = 0.0
        flag_meta = {}
        if idx_xr is not None:
            flag_breakout_idx, flag_pattern = detect_flag_breakout_after(
                session, idx_xr + 1, detector, row['symbol'])
            if flag_breakout_idx is not None and flag_pattern is not None:
                breakout_close = float(session.iloc[flag_breakout_idx]['close'])
                breakout_bar = session.iloc[flag_breakout_idx]
                add_exit_price, add_exit_reason = sim_add_with_pattern(
                    session, flag_breakout_idx, breakout_close, flag_pattern,
                    entry_p, rs)
                add_entry_price = breakout_close * (1 + EXIT_SLIP_BPS/10000)
                add_pnl_per_share = add_exit_price - add_entry_price
                add_shares = max(1, int(OLD_POS * ADD_PCT / add_entry_price))
                add_raw_pnl = add_pnl_per_share * add_shares
                add_sized_pnl = add_raw_pnl * rp_pos / OLD_POS * mults[row['_quintile']]
                # Compute breakout volume ratio
                avg_flag_vol = float(flag_pattern.avg_flag_volume) if flag_pattern.avg_flag_volume else 1.0
                breakout_vol = float(breakout_bar['volume'])
                vol_ratio = breakout_vol / max(avg_flag_vol, 1.0)
                # Minutes since open at breakout
                minutes_since_open = flag_breakout_idx
                flag_meta = {
                    'flag_detected': True,
                    'flag_breakout_idx': flag_breakout_idx,
                    'flag_pole_gain_pct': float(flag_pattern.pole_gain_pct),
                    'flag_retracement_pct': float(flag_pattern.retracement_pct),
                    'flag_breakout_vol_ratio': vol_ratio,
                    'flag_minutes_since_open': minutes_since_open,
                    'add_exit_reason': add_exit_reason,
                    'add_was_winner': add_sized_pnl > 0,
                }
        if not flag_meta:
            flag_meta = {
                'flag_detected': False, 'flag_breakout_idx': None,
                'flag_pole_gain_pct': None, 'flag_retracement_pct': None,
                'flag_breakout_vol_ratio': None, 'flag_minutes_since_open': None,
                'add_exit_reason': None, 'add_was_winner': None,
            }
        records.append({
            'symbol': row['symbol'], 'date': row['date'],
            'quintile': row['_quintile'],
            'composite': float(row['_composite']),
            'range_size': rs, 'range_size_pct': float(row['range_size_pct']),
            'entry_price': entry_p, 'rp_position': rp_pos,
            'mult': mults[row['_quintile']],
            'gap_pct': float(row.get('gap_pct', 0) or 0),
            'avg_daily_volume_20d': float(row.get('avg_daily_volume_20d', 0) or 0),
            'v0_sized_pnl': v0_sized_pnl,
            'reached_xr': idx_xr is not None,
            'add_sized_pnl': add_sized_pnl,
            'combined_sized_pnl': v0_sized_pnl + add_sized_pnl,
            'detector_label': label,
            **flag_meta,
        })
    return pd.DataFrame(records)


def _walk_forward_summary(rdf, label):
    rdf['date'] = pd.to_datetime(rdf['date'])
    slices = [
        ('TRAIN',  '2025-01-01', TRAIN_END),
        ('VAL',    '2025-07-01', VAL_END),
        ('HOQ1+',  '2026-01-01', '2030-12-31'),
        ('FULL',   '2025-01-01', '2030-12-31'),
    ]
    print(f"\n{'='*100}")
    print(f"  {label}")
    print(f"{'='*100}")
    print(f"{'Slice':<7} {'Trades':>7} {'Fires':>7} {'WR':>6} "
          f"{'V0 P&L':>11} {'Δ Add':>10} {'rm3':>10} {'rm5':>10}")
    print('-' * 90)
    for slice_name, lo, hi in slices:
        sub = rdf[(rdf['date'] >= lo) & (rdf['date'] <= hi)]
        if len(sub) == 0: continue
        fires = sub[sub['flag_detected']]
        n_fires = len(fires)
        wr = (fires['add_was_winner'].sum() / n_fires * 100) if n_fires else 0
        v0 = sub.groupby('date')['v0_sized_pnl'].sum().sort_index().sum()
        delta_add = fires['add_sized_pnl'].sum()
        sorted_fires = fires.sort_values('add_sized_pnl', ascending=False)
        rm3 = sorted_fires.iloc[3:]['add_sized_pnl'].sum() if n_fires >= 3 else 0
        rm5 = sorted_fires.iloc[5:]['add_sized_pnl'].sum() if n_fires >= 5 else 0
        print(f"{slice_name:<7} {len(sub):>7} {n_fires:>7} {wr:>5.1f}% "
              f"${v0:>+9,.0f} ${delta_add:>+8,.0f} ${rm3:>+8,.0f} ${rm5:>+8,.0f}")


def _stratify(fires, dim_name, buckets, dim_col=None):
    """Stratify fires by `dim_col` into `buckets`, report WR + tail-dep per bucket."""
    if dim_col is None:
        dim_col = dim_name
    print(f"\n  Stratification by {dim_name}:")
    print(f"    {'Bucket':<25} {'N':>4} {'Win%':>6} {'Sum':>10} {'rm3':>10} "
          f"{'avg/fire':>10}")
    for bucket_label, predicate in buckets:
        sub = fires[predicate(fires)]
        n = len(sub)
        if n == 0:
            print(f"    {bucket_label:<25} {n:>4}  --     --        --        --")
            continue
        wr = (sub['add_was_winner'].sum() / n * 100) if n else 0
        s = sub['add_sized_pnl'].sum()
        rm3 = sub.sort_values('add_sized_pnl', ascending=False).iloc[3:]['add_sized_pnl'].sum() \
            if n >= 3 else 0
        avg = sub['add_sized_pnl'].mean()
        print(f"    {bucket_label:<25} {n:>4}  {wr:>5.1f}% ${s:>+8,.0f} ${rm3:>+8,.0f} ${avg:>+8,.0f}")


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
    print("Building V0 trade set...")
    sel, mults = _run_pipeline(df)
    print(f"Selected: {len(sel)} trades")

    # Detector configs
    detector_default = dict(
        min_pole_candles=3, min_pole_gain_pct=3.0,
        max_retracement_pct=50.0, max_pullback_candles=5,
        min_breakout_volume_ratio=1.5,
        max_green_in_flag=1, max_pole_bars=0,
    )
    detector_loose = dict(
        min_pole_candles=2, min_pole_gain_pct=2.0,
        max_retracement_pct=60.0, max_pullback_candles=8,
        min_breakout_volume_ratio=1.2,
        max_green_in_flag=2, max_pole_bars=0,
    )
    detector_strict = dict(
        min_pole_candles=4, min_pole_gain_pct=5.0,
        max_retracement_pct=30.0, max_pullback_candles=4,
        min_breakout_volume_ratio=2.0,
        max_green_in_flag=0, max_pole_bars=0,
    )

    runs = [
        ('loose', detector_loose),
        ('default', detector_default),
        ('strict', detector_strict),
    ]

    all_data = {}
    for label, cfg in runs:
        print(f"\n>>> Running detector: {label}")
        rdf = replay_with_detector(sel, bars_cache, mults, cfg, label)
        all_data[label] = rdf
        _walk_forward_summary(rdf, f"Detector: {label}")

    # Stratification on STRICT detector fires (combined TRAIN+VAL only — preserve HOQ1+ OOS)
    print(f"\n{'='*100}")
    print(f"  STRATIFICATION on STRICT detector fires (TRAIN+VAL combined)")
    print(f"  Goal: find sub-population with WR >= 45% AND positive top-3 removal")
    print(f"{'='*100}")
    strict_fires = all_data['strict'][all_data['strict']['flag_detected']].copy()
    strict_fires['date'] = pd.to_datetime(strict_fires['date'])
    train_val = strict_fires[strict_fires['date'] <= VAL_END].copy()
    hoq1 = strict_fires[strict_fires['date'] > VAL_END].copy()
    print(f"\n  TRAIN+VAL fires: {len(train_val)} | HOQ1+ fires: {len(hoq1)}")
    print(f"  Baseline TRAIN+VAL: WR={train_val['add_was_winner'].mean()*100:.1f}%, "
          f"sum=${train_val['add_sized_pnl'].sum():+,.0f}")

    # Dimension 1: price bucket
    _stratify(train_val, 'price_bucket', [
        ('< $5',     lambda f: f['entry_price'] < 5.0),
        ('$5-$10',   lambda f: (f['entry_price'] >= 5.0) & (f['entry_price'] < 10.0)),
        ('$10-$20',  lambda f: (f['entry_price'] >= 10.0) & (f['entry_price'] < 20.0)),
        ('$20+',     lambda f: f['entry_price'] >= 20.0),
    ])
    # Dimension 2: quintile
    _stratify(train_val, 'quintile', [
        ('Q5', lambda f: f['quintile'] == 'Q5'),
        ('Q4', lambda f: f['quintile'] == 'Q4'),
        ('Q3', lambda f: f['quintile'] == 'Q3'),
        ('Q2', lambda f: f['quintile'] == 'Q2'),
    ])
    # Dimension 3: pole_gain_pct of detected flag
    _stratify(train_val, 'pole_gain_pct', [
        ('< 5%',    lambda f: f['flag_pole_gain_pct'] < 5.0),
        ('5-8%',    lambda f: (f['flag_pole_gain_pct'] >= 5.0) & (f['flag_pole_gain_pct'] < 8.0)),
        ('8-12%',   lambda f: (f['flag_pole_gain_pct'] >= 8.0) & (f['flag_pole_gain_pct'] < 12.0)),
        ('12%+',    lambda f: f['flag_pole_gain_pct'] >= 12.0),
    ])
    # Dimension 4: breakout volume ratio
    _stratify(train_val, 'breakout_vol_ratio', [
        ('< 2x',    lambda f: f['flag_breakout_vol_ratio'] < 2.0),
        ('2-3x',    lambda f: (f['flag_breakout_vol_ratio'] >= 2.0) & (f['flag_breakout_vol_ratio'] < 3.0)),
        ('3-5x',    lambda f: (f['flag_breakout_vol_ratio'] >= 3.0) & (f['flag_breakout_vol_ratio'] < 5.0)),
        ('5x+',     lambda f: f['flag_breakout_vol_ratio'] >= 5.0),
    ])
    # Dimension 5: minutes since open at breakout
    _stratify(train_val, 'minutes_since_open', [
        ('< 30 min',    lambda f: f['flag_minutes_since_open'] < 30),
        ('30-60 min',   lambda f: (f['flag_minutes_since_open'] >= 30) & (f['flag_minutes_since_open'] < 60)),
        ('60-120 min',  lambda f: (f['flag_minutes_since_open'] >= 60) & (f['flag_minutes_since_open'] < 120)),
        ('120+ min',    lambda f: f['flag_minutes_since_open'] >= 120),
    ])
    # Dimension 6: gap_pct
    _stratify(train_val, 'gap_pct', [
        ('< 10%',    lambda f: f['gap_pct'] < 10),
        ('10-20%',   lambda f: (f['gap_pct'] >= 10) & (f['gap_pct'] < 20)),
        ('20-50%',   lambda f: (f['gap_pct'] >= 20) & (f['gap_pct'] < 50)),
        ('50%+',     lambda f: f['gap_pct'] >= 50),
    ])

    # Same stratification on LOOSE detector for comparison
    loose_fires = all_data['loose'][all_data['loose']['flag_detected']].copy()
    loose_fires['date'] = pd.to_datetime(loose_fires['date'])
    train_val_loose = loose_fires[loose_fires['date'] <= VAL_END].copy()
    print(f"\n{'='*100}")
    print(f"  STRATIFICATION on LOOSE detector fires (TRAIN+VAL combined) for comparison")
    print(f"  TRAIN+VAL fires: {len(train_val_loose)}")
    print(f"  Baseline: WR={train_val_loose['add_was_winner'].mean()*100:.1f}%, "
          f"sum=${train_val_loose['add_sized_pnl'].sum():+,.0f}")
    print(f"{'='*100}")
    _stratify(train_val_loose, 'price_bucket', [
        ('< $5',     lambda f: f['entry_price'] < 5.0),
        ('$5-$10',   lambda f: (f['entry_price'] >= 5.0) & (f['entry_price'] < 10.0)),
        ('$10-$20',  lambda f: (f['entry_price'] >= 10.0) & (f['entry_price'] < 20.0)),
        ('$20+',     lambda f: f['entry_price'] >= 20.0),
    ])
    _stratify(train_val_loose, 'quintile', [
        ('Q5', lambda f: f['quintile'] == 'Q5'),
        ('Q4', lambda f: f['quintile'] == 'Q4'),
        ('Q3', lambda f: f['quintile'] == 'Q3'),
        ('Q2', lambda f: f['quintile'] == 'Q2'),
    ])

    # Save outputs
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    for label, rdf in all_data.items():
        rdf.to_csv(f'analysis_results/orb_add_bullflag_seg_{label}_{ts}.csv', index=False)
    print(f"\nSaved analysis_results/orb_add_bullflag_seg_*_{ts}.csv")


if __name__ == '__main__':
    main()

"""Stage 2 (revised) — Bull-flag-gated add at +2.5R / +3R MFE.

Stage 1 oracle (always-add at +3R) was tail-dependent: 5 hero trades carry the
entire net P&L; tail-removal kills the edge. Per user direction, the right
gate is a STRUCTURAL pattern signal — a clean bull flag breakout — not a
naive R-multiple trigger.

Mechanism:
  1. Wait for trade to reach +X R MFE (X ∈ {2.5, 3.0})
  2. From that bar onward, scan each subsequent bar with BullFlagDetector.detect()
  3. detect() fires only when the CURRENT bar is a BREAKOUT completion above
     a tight flag formed on prior bars (the structural confirmation)
  4. Add at the breakout close (with slippage)
  5. Stop = max(flag_low, entry + 2R) — structural floor, with safety
  6. Ride to EOD or stop hit, whichever first
  7. Only one add per trade

Walk-forward TRAIN/VAL/HOQ1+ + tail-dependence check.

If THIS doesn't work cleanly with structural gating, the project dies — there's
nothing more to test on the exit/add side without L2 data.
"""
from __future__ import annotations

import glob
import os
import sys
from datetime import datetime, timedelta
from typing import Optional, Tuple

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

# Add params
ADD_PCT = 0.5     # add 50% of original _rp_position
SAFETY_STOP_R = 2.0  # safety floor for stop — never below entry + 2R


def _session_open_timestamp(bars):
    if bars.empty: return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning): return morning.iloc[0]['timestamp']
    return None


def sim_v0(bars, entry_price, range_high, range_low, entry_time):
    """V0 simulator returning (exit_price, exit_reason)."""
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


def find_xr_bar_idx(session_bars: pd.DataFrame, entry_idx: int,
                     entry_price: float, range_size: float, x_r: float
                     ) -> Optional[int]:
    """Find first bar at or after entry_idx where high >= entry + x_r * range_size."""
    target = entry_price + x_r * range_size
    for i in range(entry_idx, len(session_bars)):
        if float(session_bars.iloc[i]['high']) >= target:
            return i
    return None


def detect_flag_breakout_after(session_bars: pd.DataFrame, start_idx: int,
                                 detector: BullFlagDetector, symbol: str
                                 ) -> Tuple[Optional[int], Optional[BullFlagPattern]]:
    """Walk bars from start_idx onward, calling detector.detect(end_idx=i+1) at
    each bar to check if THIS bar is a breakout completion.

    Returns (breakout_bar_idx, pattern) or (None, None).
    """
    # detect() needs at least 6 completed bars (3 pole + 2 pullback + 1 breakout).
    # When we call with end_idx=i+1, completed = bars.iloc[:i+1]. So we need
    # at least 6 bars before this (i >= 5).
    for i in range(max(start_idx, 5), len(session_bars)):
        # Pass full session bars; detector slices by end_idx
        try:
            pattern = detector.detect(symbol, session_bars, end_idx=i + 1)
        except Exception:
            continue
        if pattern is not None:
            return i, pattern
    return None, None


def sim_add_with_pattern(session_bars: pd.DataFrame, breakout_idx: int,
                          breakout_close: float, pattern: BullFlagPattern,
                          entry_price: float, range_size: float
                          ) -> Tuple[float, str]:
    """Simulate add at breakout_close with stop = max(flag_low, entry + 2R).
    Ride to EOD if not stopped.

    Returns (add_exit_price, exit_reason).
    """
    add_entry_price = breakout_close * (1 + EXIT_SLIP_BPS/10000)  # buy slippage
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
    """Standard shipping pipeline (composite + quintile + Q1 filter + dedup)."""
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
    kept = kept[kept['_quintile'] != 'Q1'].copy()  # Q1 filter on
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


def replay_with_x_r_trigger(sel, bars_cache, mults, x_r: float, detector_cfg: dict):
    """For a given +X R MFE trigger, replay all selected trades and simulate
    bull-flag-gated add. Returns per-trade record DataFrame.
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
        # V0 outcome
        v0_exit_price, v0_reason = sim_v0(bars, entry_p, rh, rl,
                                            session.iloc[entry_idx_in_session]['timestamp'])
        v0_pnl_per_share = v0_exit_price - entry_p
        per_pos_cap = ACCOUNT / N
        stop_pct = max(float(row['range_size_pct']), MIN_STOP_PCT)
        rp_pos = min(RISK / (stop_pct / 100.0), per_pos_cap)
        v0_sized_pnl = (v0_pnl_per_share * shares) * rp_pos / OLD_POS * mults[row['_quintile']]

        # Find +X R bar
        idx_xr = find_xr_bar_idx(session, entry_idx_in_session, entry_p, rs, x_r)
        # Default add metrics
        add_sized_pnl = 0.0; add_exit_reason = None
        flag_breakout_idx = None; flag_pattern = None
        add_entry_price = None; add_exit_price = None; add_stop_price = None
        if idx_xr is not None:
            # Scan bars after +X R bar for flag breakout
            flag_breakout_idx, flag_pattern = detect_flag_breakout_after(
                session, idx_xr + 1, detector, row['symbol'])
            if flag_breakout_idx is not None and flag_pattern is not None:
                breakout_close = float(session.iloc[flag_breakout_idx]['close'])
                add_exit_price, add_exit_reason = sim_add_with_pattern(
                    session, flag_breakout_idx, breakout_close, flag_pattern,
                    entry_p, rs)
                add_entry_price = breakout_close * (1 + EXIT_SLIP_BPS/10000)
                add_stop_price = max(float(flag_pattern.flag_low),
                                      entry_p + SAFETY_STOP_R * rs)
                add_pnl_per_share = add_exit_price - add_entry_price
                add_shares = max(1, int(OLD_POS * ADD_PCT / add_entry_price))
                # Apply same scale factor as V0
                add_raw_pnl = add_pnl_per_share * add_shares
                add_sized_pnl = add_raw_pnl * rp_pos / OLD_POS * mults[row['_quintile']]
        records.append({
            'symbol': row['symbol'], 'date': row['date'],
            'quintile': row['_quintile'],
            'range_size': rs, 'entry_price': entry_p,
            'rp_position': rp_pos, 'mult': mults[row['_quintile']],
            'v0_sized_pnl': v0_sized_pnl, 'v0_reason': v0_reason,
            'reached_xr': idx_xr is not None, 'idx_xr': idx_xr,
            'flag_detected': flag_breakout_idx is not None,
            'flag_breakout_idx': flag_breakout_idx,
            'flag_low': float(flag_pattern.flag_low) if flag_pattern else None,
            'flag_high': float(flag_pattern.flag_high) if flag_pattern else None,
            'pole_gain_pct': float(flag_pattern.pole_gain_pct) if flag_pattern else None,
            'add_entry_price': add_entry_price,
            'add_stop_price': add_stop_price,
            'add_exit_price': add_exit_price,
            'add_exit_reason': add_exit_reason,
            'add_sized_pnl': add_sized_pnl,
            'combined_sized_pnl': v0_sized_pnl + add_sized_pnl,
        })
    return pd.DataFrame(records)


def report_walk_forward(rdf, label):
    """Walk-forward report + tail-dependence."""
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
    print(f"{'Slice':<7} {'Trades':>7} {'+XR hits':>9} {'Flag fires':>11} "
          f"{'V0 P&L':>11} {'Combined':>11} {'Δ Add':>10} {'Cal Comb':>9}")
    print('-' * 95)
    for slice_name, lo, hi in slices:
        sub = rdf[(rdf['date'] >= lo) & (rdf['date'] <= hi)]
        if len(sub) == 0: continue
        n_xr = int(sub['reached_xr'].sum())
        n_fires = int(sub['flag_detected'].sum())
        v0_daily = sub.groupby('date')['v0_sized_pnl'].sum().sort_index()
        c0_daily = sub.groupby('date')['combined_sized_pnl'].sum().sort_index()
        v0_tp, _, _ = _calmar(v0_daily)
        c0_tp, c0_dd, c0_cal = _calmar(c0_daily)
        delta = c0_tp - v0_tp
        print(f"{slice_name:<7} {len(sub):>7} {n_xr:>9} {n_fires:>11} "
              f"${v0_tp:>+9,.0f} ${c0_tp:>+9,.0f} ${delta:>+8,.0f} "
              f"{c0_cal:>7.2f}x")
    # Tail check on adds
    fires = rdf[rdf['flag_detected']].copy()
    if len(fires) == 0:
        print("  (No flag fires — detector found no breakouts)")
        return
    print(f"\n  Flag-fire add P&L distribution ({len(fires)} fires):")
    for pct in [10, 25, 50, 75, 90, 99]:
        print(f"    p{pct}: ${fires['add_sized_pnl'].quantile(pct/100):>+8,.0f}")
    pos = (fires['add_sized_pnl'] > 0).sum()
    print(f"    positive: {pos}/{len(fires)} ({pos/len(fires)*100:.1f}%)")
    print(f"    mean: ${fires['add_sized_pnl'].mean():>+8,.0f}")
    print(f"    sum:  ${fires['add_sized_pnl'].sum():>+8,.0f}")

    # Tail-dependence per slice
    print(f"\n  Tail-dependence (remove top-N add winners per slice):")
    for slice_name, lo, hi in slices:
        sub = fires[(fires['date'] >= lo) & (fires['date'] <= hi)].copy()
        if len(sub) == 0: continue
        sub_sorted = sub.sort_values('add_sized_pnl', ascending=False).reset_index(drop=True)
        line = f"    {slice_name:<7}"
        for n in [0, 3, 5, 10]:
            remaining = sub_sorted.iloc[n:]
            line += f"  rm{n}=${remaining['add_sized_pnl'].sum():>+8,.0f}"
        line += f"   (n_fires={len(sub)})"
        print(line)

    # Add reason breakdown
    print(f"\n  Add exit reason breakdown ({len(fires)} fires):")
    for r, c in fires['add_exit_reason'].value_counts().items():
        avg_pnl = fires[fires['add_exit_reason'] == r]['add_sized_pnl'].mean()
        print(f"    {r:<10} {c:>4}  (avg ${avg_pnl:>+7,.0f})")


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

    # Detector configs to test
    detector_default = dict(
        min_pole_candles=3, min_pole_gain_pct=3.0,
        max_retracement_pct=50.0, max_pullback_candles=5,
        min_breakout_volume_ratio=1.5,
        max_green_in_flag=1, max_pole_bars=0,  # 0 = disabled
    )
    detector_loose = dict(
        min_pole_candles=2, min_pole_gain_pct=2.0,
        max_retracement_pct=60.0, max_pullback_candles=8,
        min_breakout_volume_ratio=1.2,
        max_green_in_flag=2, max_pole_bars=0,
    )

    variants = [
        ('+3R trigger / default detector', 3.0, detector_default),
        ('+2.5R trigger / default detector', 2.5, detector_default),
        ('+3R trigger / loose detector',   3.0, detector_loose),
        ('+2.5R trigger / loose detector', 2.5, detector_loose),
    ]

    all_results = {}
    for label, x_r, det_cfg in variants:
        print(f"\n>>> Running variant: {label}")
        rdf = replay_with_x_r_trigger(sel, bars_cache, mults, x_r, det_cfg)
        all_results[label] = rdf
        report_walk_forward(rdf, label)

    # Save outputs
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    for label, rdf in all_results.items():
        safe = label.replace(' ', '_').replace('/', '_').replace('+', 'p')
        rdf.to_csv(f'analysis_results/orb_add_bullflag_{safe}_{ts}.csv', index=False)
    print(f"\nSaved analysis_results/orb_add_bullflag_*_{ts}.csv  ({len(variants)} variants)")


if __name__ == '__main__':
    main()

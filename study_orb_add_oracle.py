"""Stage 1 — Oracle add BT for ORB +3R add-to-winners.

Question: if we add 50% at every +3R cross with stop at +2R and ride to EOD,
what's the upper-bound P&L lift across walk-forward splits (TRAIN/VAL/HOQ1+)?

This is the upper bound — assumes we always-add. Stage 2 will gate adds via
features identified in Stage 0; Stage 2 / Stage 1 ratio = "capture rate".

Key sanity checks (from plan):
  1. TRAIN should show positive lift (selection bias on +3R trades).
  2. If HOQ1+ negative even with always-add, stop project.
  3. Tail-removal (top-3/5/10) must keep ALL slices positive.

Output: per-add CSV + walk-forward summary report.
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

# Stage 1 add params
ADD_TRIGGER_R = 3.0
ADD_STOP_R = 2.0      # stop at entry + 2R = 1R below add price
ADD_PCT = 0.5         # add 50% of original _rp_position


def _session_open_timestamp(bars):
    if bars.empty: return None
    bars_et = bars.copy()
    bars_et['et'] = bars_et['timestamp'].dt.tz_convert('America/New_York')
    morning = bars_et[bars_et['et'].dt.time == pd.Timestamp('09:30').time()]
    if len(morning): return morning.iloc[0]['timestamp']
    return None


def sim_v0_with_3r(bars, entry_price, range_high, range_low, entry_time
                    ) -> Tuple[float, str, Optional[float], Optional[pd.Timestamp]]:
    """V0 simulator + return the +3R bar price/timestamp if reached.

    Returns: (v0_exit_price, v0_exit_reason, three_r_close_or_None, three_r_ts_or_None).
    """
    rs = range_high - range_low
    if rs <= 0:
        last = float(bars.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000)
        return last, 'no_range', None, None
    trig_15 = entry_price + 1.5 * rs
    lock_1r = entry_price + 1.0 * rs
    target_3r = entry_price + ADD_TRIGGER_R * rs

    stop = range_low; armed = False
    three_r_close = None; three_r_ts = None
    v0_exit_price = None; v0_exit_reason = None

    post = bars[bars['timestamp'] >= entry_time].reset_index(drop=True)
    for _, row in post.iloc[1:].iterrows():
        h = float(row['high']); lo = float(row['low']); c = float(row['close'])
        # Track +3R cross (first bar where high >= target)
        if three_r_ts is None and h >= target_3r:
            three_r_close = c
            three_r_ts = row['timestamp']
        # V0 exit logic
        if v0_exit_price is None:
            if not armed and h >= trig_15:
                armed = True
                stop = max(stop, lock_1r)
            if lo <= stop:
                v0_exit_price = stop * (1 - EXIT_SLIP_BPS/10000)
                v0_exit_reason = 'lock' if armed else 'stop'
        # Continue loop to capture +3R cross even after V0 exit (informational)
        if v0_exit_price is not None and three_r_ts is not None:
            break

    if v0_exit_price is None:
        v0_exit_price = float(post.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000)
        v0_exit_reason = 'eod'
    return v0_exit_price, v0_exit_reason, three_r_close, three_r_ts


def sim_add_after_3r(bars, three_r_close: float, three_r_ts: pd.Timestamp,
                      entry_price: float, range_size: float
                      ) -> Tuple[float, str]:
    """Simulate the ADD position. Enter at three_r_close (+3R bar close, with
    slippage). Stop at entry + 2R. Force-close at EOD (15:45 ET handled by
    last bar). Return (add_exit_price, exit_reason).
    """
    add_entry_price = three_r_close * (1 + EXIT_SLIP_BPS/10000)  # buy slippage
    add_stop = entry_price + ADD_STOP_R * range_size
    # Bars after the +3R bar
    post_3r = bars[bars['timestamp'] > three_r_ts].reset_index(drop=True)
    if len(post_3r) == 0:
        # +3R was the last bar — exit at the close price (no add room)
        return add_entry_price * (1 - EXIT_SLIP_BPS/10000), 'no_room'
    for _, row in post_3r.iterrows():
        lo = float(row['low'])
        if lo <= add_stop:
            return add_stop * (1 - EXIT_SLIP_BPS/10000), 'stop'
    # Survived to EOD — exit at EOD close
    return float(post_3r.iloc[-1]['close']) * (1 - EXIT_SLIP_BPS/10000), 'eod'


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
    # Q1 filter (shipping default)
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

    print("Building V0 trade set with shipping Q1 filter...")
    sel, mults = _run_pipeline(df)
    print(f"Selected: {len(sel)} trades")
    print(f"Mults: { {q: round(v,3) for q,v in mults.items()} }")

    # Replay each trade — V0 exit + (if +3R) the add trade
    print("\nReplaying trades for V0 + always-add at +3R...")
    records = []
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
        rs = rh - rl
        if rs <= 0: continue
        search = bars[(bars['timestamp'] >= range_end) &
                      (bars['timestamp'] < range_end + timedelta(minutes=60))]
        entry_ts = None
        for _, b in search.iterrows():
            if float(b['high']) > rh: entry_ts = b['timestamp']; break
        if entry_ts is None: continue
        entry_p = float(row['entry_price'])
        shares = max(1, int(OLD_POS / entry_p))

        v0_exit_price, v0_exit_reason, three_r_close, three_r_ts = sim_v0_with_3r(
            bars, entry_p, rh, rl, entry_ts)

        # V0 sized P&L
        v0_pnl_per_share = v0_exit_price - entry_p
        v0_raw_pnl = v0_pnl_per_share * shares
        rp_position = float(row['_rp_position']) if '_rp_position' in row else (
            min(RISK / max(rs / entry_p, 0.01), ACCOUNT/N))
        # Recompute _rp_position to match pipeline (defensive)
        per_pos_cap = ACCOUNT / N
        stop_pct = max(float(row['range_size_pct']), MIN_STOP_PCT)
        rp_pos = min(RISK / (stop_pct / 100.0), per_pos_cap)
        v0_rp_pnl = v0_raw_pnl * rp_pos / OLD_POS
        v0_sized_pnl = v0_rp_pnl * mults[row['_quintile']]

        # Add trade (only if +3R was reached)
        add_pnl_per_share = 0.0; add_raw_pnl = 0.0; add_sized_pnl = 0.0
        add_exit_price = None; add_exit_reason = None; add_entry_price = None
        add_shares = 0
        if three_r_close is not None and three_r_ts is not None:
            add_exit_price, add_exit_reason = sim_add_after_3r(
                bars, three_r_close, three_r_ts, entry_p, rs)
            # Add entry includes slippage on buy side
            add_entry_price = three_r_close * (1 + EXIT_SLIP_BPS/10000)
            add_shares = max(1, int(OLD_POS * ADD_PCT / add_entry_price))
            # P&L based on shares × (exit - entry)
            add_pnl_per_share = add_exit_price - add_entry_price
            add_raw_pnl = add_pnl_per_share * add_shares
            # Apply same RP-position scaling and quintile mult as original (proportional)
            add_rp_pnl = add_raw_pnl * (rp_pos * ADD_PCT) / (OLD_POS * ADD_PCT)
            # Simplified: same scale factor as original
            add_rp_pnl = add_raw_pnl * rp_pos / OLD_POS
            add_sized_pnl = add_rp_pnl * mults[row['_quintile']]

        rec = {
            'symbol': row['symbol'],
            'date': row['date'],
            'quintile': row['_quintile'],
            'range_size': rs,
            'range_size_pct': float(row['range_size_pct']),
            'entry_price': entry_p,
            'rp_position': rp_pos,
            'mult': mults[row['_quintile']],
            'v0_exit_price': v0_exit_price,
            'v0_exit_reason': v0_exit_reason,
            'v0_sized_pnl': v0_sized_pnl,
            'reached_3r': three_r_close is not None,
            'three_r_close': three_r_close,
            'three_r_ts': three_r_ts,
            'add_entry_price': add_entry_price,
            'add_exit_price': add_exit_price,
            'add_exit_reason': add_exit_reason,
            'add_sized_pnl': add_sized_pnl,
            'combined_sized_pnl': v0_sized_pnl + add_sized_pnl,
        }
        records.append(rec)

    rdf = pd.DataFrame(records)
    rdf['date'] = pd.to_datetime(rdf['date'])
    print(f"\nReplayed {len(rdf)} trades")
    print(f"  Reached +3R: {int(rdf['reached_3r'].sum())} ({rdf['reached_3r'].mean()*100:.1f}%)")
    print(f"  Add exit reasons:")
    for r, c in rdf[rdf['reached_3r']]['add_exit_reason'].value_counts().items():
        print(f"    {r:<10} {c}")

    # Walk-forward report
    slices = [
        ('TRAIN',  '2025-01-01', TRAIN_END),
        ('VAL',    '2025-07-01', VAL_END),
        ('HOQ1+',  '2026-01-01', '2030-12-31'),
        ('FULL',   '2025-01-01', '2030-12-31'),
    ]

    print(f"\n{'='*100}")
    print("  WALK-FORWARD ORACLE ADD RESULTS — always-add at +3R, stop +2R, ride to EOD")
    print(f"{'='*100}")
    print(f"{'Slice':<7} {'Trades':>7} {'+3R hits':>9} {'V0 P&L':>11} "
          f"{'Combined':>11} {'Δ Add':>10} {'DD V0':>11} {'DD Comb':>11} {'Cal V0':>7} {'Cal Comb':>9}")
    print('-' * 110)

    summary = []
    for slice_name, lo, hi in slices:
        sub = rdf[(rdf['date'] >= lo) & (rdf['date'] <= hi)]
        if len(sub) == 0: continue
        n_3r = int(sub['reached_3r'].sum())
        # V0 daily
        v0_daily = sub.groupby('date')['v0_sized_pnl'].sum().sort_index()
        c0_daily = sub.groupby('date')['combined_sized_pnl'].sum().sort_index()
        v0_tp, v0_dd, v0_cal = _calmar(v0_daily)
        c0_tp, c0_dd, c0_cal = _calmar(c0_daily)
        delta = c0_tp - v0_tp
        summary.append({
            'slice': slice_name, 'trades': len(sub), 'three_r_hits': n_3r,
            'v0_pnl': v0_tp, 'combined_pnl': c0_tp, 'delta_add': delta,
            'v0_dd': v0_dd, 'combined_dd': c0_dd,
            'v0_calmar': v0_cal, 'combined_calmar': c0_cal,
        })
        print(f"{slice_name:<7} {len(sub):>7} {n_3r:>9} "
              f"${v0_tp:>+9,.0f} ${c0_tp:>+9,.0f} ${delta:>+8,.0f} "
              f"${v0_dd:>+8,.0f} ${c0_dd:>+8,.0f} {v0_cal:>5.2f}x {c0_cal:>7.2f}x")

    # Tail-dependence: remove top-N adds per slice
    print(f"\n{'='*100}")
    print("  TAIL-DEPENDENCE on ADDS (remove top-N add P&L wins per slice)")
    print(f"{'='*100}")
    for slice_name, lo, hi in slices:
        sub = rdf[(rdf['date'] >= lo) & (rdf['date'] <= hi) & rdf['reached_3r']].copy()
        if len(sub) == 0: continue
        print(f"\n  Slice: {slice_name}  (3R-hit trades: {len(sub)})")
        sub_sorted = sub.sort_values('add_sized_pnl', ascending=False).reset_index(drop=True)
        for n in [0, 3, 5, 10]:
            remaining = sub_sorted.iloc[n:]
            add_sum = remaining['add_sized_pnl'].sum()
            print(f"    remove top {n:>2}: add-only P&L sum = ${add_sum:>+9,.0f}")

    # Add-only P&L distribution (for tail concern)
    if rdf['reached_3r'].any():
        adds = rdf[rdf['reached_3r']].copy()
        print(f"\n{'='*100}")
        print("  ADD-ONLY P&L distribution")
        print(f"{'='*100}")
        for pct in [10, 25, 50, 75, 90, 99]:
            print(f"  p{pct}: ${adds['add_sized_pnl'].quantile(pct/100):>+8,.0f}")
        print(f"  mean: ${adds['add_sized_pnl'].mean():>+8,.0f}")
        print(f"  positive count: {(adds['add_sized_pnl']>0).sum()}/{len(adds)}")
        print(f"  negative count: {(adds['add_sized_pnl']<0).sum()}/{len(adds)}")

        # Top-10 winning adds
        print(f"\n  Top 10 add winners (by sized P&L):")
        top = adds.sort_values('add_sized_pnl', ascending=False).head(10)
        cols = ['symbol', 'date', 'quintile', 'add_sized_pnl', 'v0_sized_pnl', 'add_exit_reason']
        pd.set_option('display.width', 200); pd.set_option('display.float_format', '{:,.0f}'.format)
        # Format date for printing
        top_print = top[cols].copy()
        top_print['date'] = top_print['date'].dt.strftime('%Y-%m-%d')
        print(top_print.to_string(index=False))

    # Save outputs
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    rdf.to_csv(f'analysis_results/orb_add_oracle_trades_{ts}.csv', index=False)
    pd.DataFrame(summary).to_csv(f'analysis_results/orb_add_oracle_summary_{ts}.csv', index=False)
    print(f"\nSaved analysis_results/orb_add_oracle_trades_{ts}.csv")
    print(f"Saved analysis_results/orb_add_oracle_summary_{ts}.csv")


if __name__ == '__main__':
    main()
